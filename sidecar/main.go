package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	v3 "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// --- Session context tracking ---

type SessionEvent struct {
	Sequence  int       `json:"sequence"`
	Direction string    `json:"direction"` // "inbound" or "outbound"
	Phase     string    `json:"phase"`     // "request" or "response"
	Method    string    `json:"method"`
	Authority string    `json:"authority"`
	Path      string    `json:"path"`
	Body      string    `json:"body"`   // truncated to 500 chars
	Action    string    `json:"action"` // what the sidecar did: "captured intent", "logged (trusted)", "BLOCKED", etc.
	Timestamp time.Time `json:"timestamp"`
}

type SessionContext struct {
	OriginalIntent string         `json:"original_intent"`
	Events         []SessionEvent `json:"events"`
	mu             sync.Mutex
}

func (sc *SessionContext) AddEvent(direction, phase, method, authority, path, body string) int {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	// Truncate body to 500 chars
	if len(body) > 500 {
		body = body[:500]
	}
	idx := len(sc.Events)
	sc.Events = append(sc.Events, SessionEvent{
		Sequence:  idx + 1,
		Direction: direction,
		Phase:     phase,
		Method:    method,
		Authority: authority,
		Path:      path,
		Body:      body,
		Timestamp: time.Now(),
	})
	return idx
}

// SetEventBody updates the body of an existing event (e.g. when RequestBody arrives after RequestHeaders)
func (sc *SessionContext) SetEventBody(idx int, body string) {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	if idx < 0 || idx >= len(sc.Events) {
		return
	}
	if len(body) > 500 {
		body = body[:500]
	}
	sc.Events[idx].Body = body
}

// SetEventAction updates the action field of an existing event
func (sc *SessionContext) SetEventAction(idx int, action string) {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	if idx < 0 || idx >= len(sc.Events) {
		return
	}
	sc.Events[idx].Action = action
}

// sessionStore maps sessionID -> *SessionContext
var sessionStore sync.Map

// activeSessionID tracks the currently active session for correlating outbound traffic
// that lacks X-Session-Id headers (e.g., agent→ollama, agent→email-server)
var activeSessionID atomic.Value

// trustedDestinations are logged but not validated by the LLM
var trustedDestinations map[string]bool

func initTrustedDestinations() {
	trustedDestinations = make(map[string]bool)
	env := os.Getenv("TRUSTED_DESTINATIONS")
	if env == "" {
		return
	}
	for _, d := range strings.Split(env, ",") {
		d = strings.TrimSpace(d)
		if d != "" {
			trustedDestinations[d] = true
			log.Printf("[IBAC] Trusted destination: %s", d)
		}
	}
}

func isTrustedDestination(authority string) bool {
	return trustedDestinations[authority]
}

// streamState holds per-stream metadata accumulated across headers and body phases
type streamState struct {
	direction       string
	sessionID       string
	method          string
	path            string
	authority       string
	statusCode      int
	requestEventIdx int // index of request event in SessionContext.Events, -1 if none
}

type processor struct {
	v3.UnimplementedExternalProcessorServer
}

// --- Helper functions ---

func getHeaderValue(headers []*core.HeaderValue, key string) string {
	for _, header := range headers {
		if strings.EqualFold(header.Key, key) {
			return string(header.RawValue)
		}
	}
	return ""
}

// blockRequest returns a 403 Forbidden immediate response
func blockRequest(reason string) *v3.ProcessingResponse {
	body := fmt.Sprintf(`{"error":"blocked","reason":"%s"}`, reason)
	return &v3.ProcessingResponse{
		Response: &v3.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &v3.ImmediateResponse{
				Status: &typev3.HttpStatus{
					Code: typev3.StatusCode_Forbidden,
				},
				Body:    []byte(body),
				Details: "ibac_intent_violation",
			},
		},
	}
}

// allowBody returns a body response that passes through unchanged
func allowBody() *v3.ProcessingResponse {
	return &v3.ProcessingResponse{
		Response: &v3.ProcessingResponse_RequestBody{
			RequestBody: &v3.BodyResponse{},
		},
	}
}

// allowHeaders returns a headers response that passes through unchanged
func allowHeaders() *v3.ProcessingResponse {
	return &v3.ProcessingResponse{
		Response: &v3.ProcessingResponse_RequestHeaders{
			RequestHeaders: &v3.HeadersResponse{},
		},
	}
}

// allowResponseHeaders returns a response headers response that passes through unchanged
func allowResponseHeaders() *v3.ProcessingResponse {
	return &v3.ProcessingResponse{
		Response: &v3.ProcessingResponse_ResponseHeaders{
			ResponseHeaders: &v3.HeadersResponse{},
		},
	}
}

// allowResponseBody returns a response body response that passes through unchanged
func allowResponseBody() *v3.ProcessingResponse {
	return &v3.ProcessingResponse{
		Response: &v3.ProcessingResponse_ResponseBody{
			ResponseBody: &v3.BodyResponse{},
		},
	}
}

// getOrCreateSession returns the SessionContext for a session, creating it if needed
func getOrCreateSession(sessionID string) *SessionContext {
	if ctx, ok := sessionStore.Load(sessionID); ok {
		return ctx.(*SessionContext)
	}
	ctx := &SessionContext{}
	actual, _ := sessionStore.LoadOrStore(sessionID, ctx)
	return actual.(*SessionContext)
}

// resolveSessionID returns the session ID from the stream state, falling back to activeSessionID
func resolveSessionID(state *streamState) string {
	if state.sessionID != "" {
		return state.sessionID
	}
	if v := activeSessionID.Load(); v != nil {
		return v.(string)
	}
	return ""
}

// extractA2AIntent pulls the user's natural-language intent from an A2A JSON-RPC
// message/send request body. A2A nests the text at params.message.parts[*].text
// (where each part is {"kind":"text","text":"..."}). Returns (text, true) when a
// text part is found; concatenates all text parts with "\n" separators.
func extractA2AIntent(reqBody map[string]interface{}) (string, bool) {
	params, ok := reqBody["params"].(map[string]interface{})
	if !ok {
		return "", false
	}
	message, ok := params["message"].(map[string]interface{})
	if !ok {
		return "", false
	}
	parts, ok := message["parts"].([]interface{})
	if !ok {
		return "", false
	}
	var texts []string
	for _, p := range parts {
		part, ok := p.(map[string]interface{})
		if !ok {
			continue
		}
		// Skip non-text parts (file, data).
		if kind, _ := part["kind"].(string); kind != "" && kind != "text" {
			continue
		}
		if text, ok := part["text"].(string); ok && text != "" {
			texts = append(texts, text)
		}
	}
	if len(texts) == 0 {
		return "", false
	}
	return strings.Join(texts, "\n"), true
}

// isMCPProtocolCall returns true for MCP POSTs that are framing calls, not
// user-initiated tool invocations. These occur during agent startup before any
// user session exists and shouldn't be blocked for "no session ID".
//
// POST /mcp with method ∈ {initialize, notifications/*, ping, tools/list,
// resources/list, prompts/list} is considered framing; method=tools/call and
// anything unrecognized falls through to normal session+LLM validation.
//
// GET/DELETE /mcp (SSE stream, session close) don't emit RequestBody events
// and never reach this check.
func isMCPProtocolCall(method, path, body string) bool {
	if method != "POST" || path != "/mcp" || body == "" {
		return false
	}
	var req map[string]interface{}
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		return false
	}
	rpcMethod, _ := req["method"].(string)
	switch rpcMethod {
	case "initialize", "ping", "tools/list", "resources/list", "prompts/list":
		return true
	}
	return strings.HasPrefix(rpcMethod, "notifications/")
}

// formatSessionContext renders session events as a numbered list for the LLM prompt
func formatSessionContext(sc *SessionContext) string {
	sc.mu.Lock()
	defer sc.mu.Unlock()

	if len(sc.Events) == 0 {
		return "(no prior activity)"
	}

	var sb strings.Builder
	for _, e := range sc.Events {
		actionStr := ""
		if e.Action != "" {
			actionStr = fmt.Sprintf(" => %s", e.Action)
		}
		fmt.Fprintf(&sb, "%d. [%s %s] %s %s%s%s\n",
			e.Sequence, e.Direction, e.Phase, e.Method, e.Authority, e.Path, actionStr)
		if e.Body != "" {
			fmt.Fprintf(&sb, "   Body: %s\n", e.Body)
		}
	}
	return sb.String()
}

// --- LLM-based intent checking ---

type LLMRequest struct {
	Model       string       `json:"model"`
	Messages    []LLMMessage `json:"messages"`
	Temperature float64      `json:"temperature"`
}

type LLMMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type LLMResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

type IntentDecision struct {
	Decision string `json:"decision"`
	Reason   string `json:"reason"`
}

// loadPromptTemplate reads the intent prompt template from a file.
// It looks for intent_prompt.txt next to the binary first, then falls back to the current directory.
func loadPromptTemplate() string {
	paths := []string{"intent_prompt.txt", "sidecar/intent_prompt.txt"}
	for _, p := range paths {
		data, err := os.ReadFile(p)
		if err == nil {
			log.Printf("[IBAC] Loaded intent prompt from %s", p)
			return string(data)
		}
	}
	log.Fatal("[IBAC] Could not load intent_prompt.txt")
	return ""
}

var intentPromptTemplate = loadPromptTemplate()

// formatHTTPAction formats an HTTP request as an action description for the intent prompt
func formatHTTPAction(method, authority, path, body string) string {
	return fmt.Sprintf(`Type: Outbound HTTP request
- Method: %s
- Destination: %s%s
- Body (first 500 chars): %.500s`, method, authority, path, body)
}

// checkIntent uses an LLM to determine if an outbound action aligns with the original intent
func checkIntent(sc *SessionContext, method, authority, path, body string) (string, string) {
	action := formatHTTPAction(method, authority, path, body)
	sessionTrace := formatSessionContext(sc)
	prompt := fmt.Sprintf(intentPromptTemplate, sc.OriginalIntent, sessionTrace, action)

	llmReq := LLMRequest{
		Model: "llama3.2:3b",
		Messages: []LLMMessage{
			{Role: "user", Content: prompt},
		},
		Temperature: 0.1,
	}

	reqBody, err := json.Marshal(llmReq)
	if err != nil {
		log.Printf("[IBAC] Failed to marshal LLM request: %v", err)
		return "BLOCK", "failed to create LLM request"
	}

	// Call ollama directly (NOT through envoy proxy)
	ollamaURL := os.Getenv("OLLAMA_URL")
	if ollamaURL == "" {
		ollamaURL = "http://localhost:11434"
	}
	resp, err := http.Post(ollamaURL+"/v1/chat/completions", "application/json", bytes.NewReader(reqBody))
	if err != nil {
		log.Printf("[IBAC] Failed to call LLM: %v", err)
		return "BLOCK", "LLM unavailable, default deny"
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("[IBAC] Failed to read LLM response: %v", err)
		return "BLOCK", "failed to read LLM response"
	}

	if resp.StatusCode != http.StatusOK {
		log.Printf("[IBAC] LLM returned %d: %s", resp.StatusCode, string(respBody))
		return "BLOCK", "LLM error, default deny"
	}

	var llmResp LLMResponse
	if err := json.Unmarshal(respBody, &llmResp); err != nil {
		log.Printf("[IBAC] Failed to unmarshal LLM response: %v", err)
		return "BLOCK", "failed to parse LLM response"
	}

	if len(llmResp.Choices) == 0 {
		return "BLOCK", "empty LLM response"
	}

	content := llmResp.Choices[0].Message.Content
	log.Printf("[IBAC] LLM raw response: %s", content)

	// Try to parse the JSON decision
	// The LLM may wrap the JSON in markdown code blocks, so strip those
	content = strings.TrimSpace(content)
	content = strings.TrimPrefix(content, "```json")
	content = strings.TrimPrefix(content, "```")
	content = strings.TrimSuffix(content, "```")
	content = strings.TrimSpace(content)

	var decision IntentDecision
	if err := json.Unmarshal([]byte(content), &decision); err != nil {
		log.Printf("[IBAC] Failed to parse decision JSON: %v (content: %s)", err, content)
		return "BLOCK", "unparseable LLM response, default deny"
	}

	decision.Decision = strings.ToUpper(strings.TrimSpace(decision.Decision))
	if decision.Decision != "ALLOW" && decision.Decision != "BLOCK" {
		return "BLOCK", fmt.Sprintf("invalid decision '%s', default deny", decision.Decision)
	}

	return decision.Decision, decision.Reason
}

// --- ext_proc Process implementation ---

func (p *processor) Process(stream v3.ExternalProcessor_ProcessServer) error {
	ctx := stream.Context()
	state := &streamState{requestEventIdx: -1}

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		req, err := stream.Recv()
		if err != nil {
			return status.Errorf(codes.Unknown, "cannot receive stream request: %v", err)
		}

		var resp *v3.ProcessingResponse

		switch r := req.Request.(type) {
		case *v3.ProcessingRequest_RequestHeaders:
			headers := r.RequestHeaders.Headers

			state.direction = getHeaderValue(headers.Headers, "x-ibac-direction")
			state.sessionID = getHeaderValue(headers.Headers, "x-session-id")
			state.method = getHeaderValue(headers.Headers, ":method")
			state.path = getHeaderValue(headers.Headers, ":path")
			state.authority = getHeaderValue(headers.Headers, ":authority")

			// For outbound: if no session ID in headers, use activeSessionID fallback
			if state.direction == "outbound" && state.sessionID == "" {
				state.sessionID = resolveSessionID(state)
			}

			log.Printf("[IBAC] %s request: session=%s method=%s authority=%s path=%s",
				state.direction, state.sessionID, state.method, state.authority, state.path)

			// Log request event (body and action will be updated in RequestBody if it arrives)
			sessionID := resolveSessionID(state)
			if sessionID != "" {
				sc := getOrCreateSession(sessionID)
				state.requestEventIdx = sc.AddEvent(state.direction, "request", state.method, state.authority, state.path, "")
				// Set a default action; RequestBody will overwrite with the actual decision
				if state.direction == "outbound" && isTrustedDestination(state.authority) {
					sc.SetEventAction(state.requestEventIdx, "ALLOW (trusted)")
				}
			}

			resp = allowHeaders()

		case *v3.ProcessingRequest_RequestBody:
			body := string(r.RequestBody.Body)
			sessionID := resolveSessionID(state)

			// Update the request event's body (event was created in RequestHeaders)
			if sessionID != "" && state.requestEventIdx >= 0 {
				sc := getOrCreateSession(sessionID)
				sc.SetEventBody(state.requestEventIdx, body)
			}

			if state.direction == "inbound" {
				// Inbound: capture the user's intent from the request body
				var reqBody map[string]interface{}
				if err := json.Unmarshal([]byte(body), &reqBody); err == nil {
					query, ok := reqBody["query"].(string)
					if !ok {
						// Fall back to A2A JSON-RPC: look at params.message.parts[*].text
						query, ok = extractA2AIntent(reqBody)
					}
					if ok && sessionID != "" {
						sc := getOrCreateSession(sessionID)
						sc.mu.Lock()
						sc.OriginalIntent = query
						sc.mu.Unlock()
						activeSessionID.Store(sessionID)
						log.Printf("[IBAC] Captured intent for session %s: %s", sessionID, query)
						sc.SetEventAction(state.requestEventIdx, "captured intent")
					}
				}
				resp = allowBody()

			} else if state.direction == "outbound" {
				// Check if destination is trusted
				if isTrustedDestination(state.authority) {
					log.Printf("[IBAC] Trusted destination %s, logging only", state.authority)
					if sessionID != "" {
						getOrCreateSession(sessionID).SetEventAction(state.requestEventIdx, "ALLOW (trusted)")
					}
					resp = allowBody()
				} else if isMCPProtocolCall(state.method, state.path, body) {
					// MCP housekeeping (initialize, tools/list, session open/close, SSE
					// stream) happens before any user request establishes a session.
					// These don't invoke tools on the user's behalf, so they don't
					// need intent validation. Only method=tools/call proceeds to the
					// session+LLM check below.
					log.Printf("[IBAC] ALLOW MCP protocol: %s %s (no intent check)", state.method, state.path)
					resp = allowBody()
				} else {
					// Untrusted destination: validate against session context
					if sessionID == "" {
						log.Printf("[IBAC] BLOCK: no session ID on outbound request to untrusted %s", state.authority)
						resp = blockRequest("missing session ID")
					} else {
						sc := getOrCreateSession(sessionID)
						if sc.OriginalIntent == "" {
							log.Printf("[IBAC] BLOCK: no intent found for session %s", sessionID)
							sc.SetEventAction(state.requestEventIdx, "BLOCK (no intent)")
							resp = blockRequest("no intent registered for session")
						} else {
							log.Printf("[IBAC] Session context for %s:\n%s", sessionID, formatSessionContext(sc))
							decision, reason := checkIntent(sc, state.method, state.authority, state.path, body)
							log.Printf("[IBAC] Decision for session %s: %s - %s", sessionID, decision, reason)

							if decision == "ALLOW" {
								sc.SetEventAction(state.requestEventIdx, "ALLOW (validated)")
								resp = allowBody()
							} else {
								sc.SetEventAction(state.requestEventIdx, fmt.Sprintf("BLOCK: %s", reason))
								resp = blockRequest(reason)
							}
						}
					}
				}
			} else {
				// Unknown direction, pass through
				log.Printf("[IBAC] Unknown direction '%s', passing through", state.direction)
				resp = allowBody()
			}

		case *v3.ProcessingRequest_ResponseHeaders:
			resp = allowResponseHeaders()

		case *v3.ProcessingRequest_ResponseBody:
			body := string(r.ResponseBody.Body)
			sessionID := resolveSessionID(state)

			// Log response event
			if sessionID != "" {
				sc := getOrCreateSession(sessionID)
				action := "logged"
				if state.direction == "outbound" && isTrustedDestination(state.authority) {
					action = "logged (trusted)"
				}
				idx := sc.AddEvent(state.direction, "response", state.method, state.authority, state.path, body)
				sc.SetEventAction(idx, action)
				log.Printf("[IBAC] Logged %s response for session %s: %s%s (%d bytes)",
					state.direction, sessionID, state.authority, state.path, len(body))
			}

			// Clear activeSessionID when inbound response completes
			if state.direction == "inbound" {
				activeSessionID.Store("")
				log.Printf("[IBAC] Cleared active session (inbound response complete)")
			}

			resp = allowResponseBody()

		default:
			log.Printf("[IBAC] Unknown request type: %T", r)
			resp = &v3.ProcessingResponse{}
		}

		if err := stream.Send(resp); err != nil {
			return status.Errorf(codes.Unknown, "cannot send stream response: %v", err)
		}
	}
}

func main() {
	log.Println("[IBAC] Starting ext_proc sidecar on :9090")

	initTrustedDestinations()

	lis, err := net.Listen("tcp", ":9090")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	grpcServer := grpc.NewServer()
	v3.RegisterExternalProcessorServer(grpcServer, &processor{})

	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}
