// SPDX-License-Identifier: AGPL-3.0-or-later

// systemone-bifrost-shim exposes an OpenAI-compatible chat surface (/v1/models,
// /v1/chat/completions) in front of llama-server's native decision endpoint
// (/v1/systemone), so decision models such as LiquidAI d1 can be registered as a
// custom OpenAI provider in the bifrost gateway.
//
// A decision model does not generate text: it reads a state and typed questions
// (noul, choice, score) and returns calibrated answers in one forward pass. The
// shim carries that request inside an ordinary chat completion:
//
//   - the last user message is a JSON object {"state": ..., "questions": {...}}
//     (plus any other /v1/systemone field, e.g. "images");
//   - or the request body carries it as a top-level "systemone" object.
//
// If the object has no "state", the text of the system and earlier messages is
// used as the state. The answer comes back as the assistant message content: the
// upstream JSON, unchanged, so a client asking for response_format json_object
// can json.loads it directly.
//
// Each entry in models.json is one OpenAI "model" bound to one llama-server
// (each llama-server serves one model). A model of kind "embeddings" is a
// llama-server --embedding behind POST /v1/embeddings, passed through as is, so a
// GPU tier's embedder rides the same bifrost provider. Config is read at startup.
//
// stdlib only -> a single static binary. Env:
//
//	SHIM_ADDR    listen address           (default 0.0.0.0:8190)
//	SHIM_CONFIG  path to models.json       (default ./config/models.json)
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"
)

const httpTimeout = 120 * time.Second // a cold model load plus a long state

type modelCfg struct {
	Upstream string `json:"upstream"` // llama-server base URL, e.g. http://127.0.0.1:8080
	// Kind is "decision" (default: chat completions -> /v1/systemone) or
	// "embeddings" (/v1/embeddings passed through to a llama-server --embedding).
	Kind string `json:"kind"`
}

type config struct {
	Models map[string]modelCfg `json:"models"`
}

var (
	addr    = envOr("SHIM_ADDR", "0.0.0.0:8190")
	cfgPath = envOr("SHIM_CONFIG", "./config/models.json")
	cfg     config
	client  = &http.Client{Timeout: httpTimeout}
)

func envOr(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

func loadConfig() error {
	raw, err := os.ReadFile(cfgPath)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return fmt.Errorf("parse %s: %w", cfgPath, err)
	}
	if len(cfg.Models) == 0 {
		return fmt.Errorf("%s defines no models", cfgPath)
	}
	for name, m := range cfg.Models {
		if m.Upstream == "" {
			return fmt.Errorf("model %q has no upstream", name)
		}
		m.Upstream = strings.TrimRight(m.Upstream, "/")
		if m.Kind == "" {
			m.Kind = "decision"
		}
		if m.Kind != "decision" && m.Kind != "embeddings" {
			return fmt.Errorf("model %q: kind %q is not decision or embeddings", name, m.Kind)
		}
		cfg.Models[name] = m
	}
	return nil
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	b, _ := json.Marshal(v)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(b)
}

func writeErr(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]any{"error": map[string]any{"message": msg, "type": "invalid_request_error"}})
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

// systemone posts a decision request to the model's llama-server.
func systemone(name string, m modelCfg, body map[string]any) (json.RawMessage, int, error) {
	b, _ := json.Marshal(body)
	resp, err := client.Post(m.Upstream+"/v1/systemone", "application/json", bytes.NewReader(b))
	if err != nil {
		return nil, http.StatusBadGateway, err
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 300 {
		return nil, resp.StatusCode, fmt.Errorf("llama-server /v1/systemone -> %d: %s", resp.StatusCode, truncate(string(data), 400))
	}
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, http.StatusBadGateway, fmt.Errorf("llama-server returned non-JSON: %s", truncate(string(data), 200))
	}
	// llama-server echoes its local model path; answer with the public model ID.
	out["model"] = name
	data, _ = json.Marshal(out)
	return data, http.StatusOK, nil
}

func handleModels(w http.ResponseWriter, _ *http.Request) {
	names := make([]string, 0, len(cfg.Models))
	for n := range cfg.Models {
		names = append(names, n)
	}
	sort.Strings(names)
	data := make([]map[string]any, 0, len(names))
	for _, n := range names {
		data = append(data, map[string]any{"id": n, "object": "model", "created": 0, "owned_by": "systemone"})
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

type chatMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// text returns a message's content as plain text (a string, or the text parts of a
// content-part array).
func (m chatMessage) text() string {
	var s string
	if err := json.Unmarshal(m.Content, &s); err == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(m.Content, &parts); err == nil {
		var b strings.Builder
		for _, p := range parts {
			if p.Type == "text" {
				b.WriteString(p.Text)
			}
		}
		return b.String()
	}
	return ""
}

type chatRequest struct {
	Model     string         `json:"model"`
	Messages  []chatMessage  `json:"messages"`
	Stream    bool           `json:"stream"`
	SystemOne map[string]any `json:"systemone"`
}

// decisionBody extracts the /v1/systemone body from a chat request.
func decisionBody(req chatRequest) (map[string]any, error) {
	body := req.SystemOne
	last := -1
	if body == nil {
		for i := len(req.Messages) - 1; i >= 0; i-- {
			if req.Messages[i].Role == "user" {
				last = i
				break
			}
		}
		if last < 0 {
			return nil, fmt.Errorf("no user message; send {\"state\": ..., \"questions\": {...}} as the last user message, or a top-level \"systemone\" object")
		}
		txt := strings.TrimSpace(req.Messages[last].text())
		txt = strings.TrimSuffix(strings.TrimPrefix(strings.TrimPrefix(txt, "```json"), "```"), "```")
		if err := json.Unmarshal([]byte(strings.TrimSpace(txt)), &body); err != nil || body == nil {
			return nil, fmt.Errorf("the last user message must be a JSON object {\"state\": ..., \"questions\": {...}} for a decision model")
		}
	}
	if _, ok := body["questions"].(map[string]any); !ok {
		return nil, fmt.Errorf("the decision request needs a \"questions\" object (noul / choice / score questions)")
	}
	if _, ok := body["state"]; !ok {
		var b strings.Builder
		for i, m := range req.Messages {
			if i == last {
				continue
			}
			if t := strings.TrimSpace(m.text()); t != "" {
				if b.Len() > 0 {
					b.WriteString("\n\n")
				}
				b.WriteString(t)
			}
		}
		if b.Len() > 0 {
			body["state"] = b.String()
		} else {
			body["state"] = nil
		}
	}
	return body, nil
}

func handleChat(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	m, ok := cfg.Models[req.Model]
	if !ok {
		writeErr(w, http.StatusNotFound, fmt.Sprintf("model %q not found; see GET /v1/models", req.Model))
		return
	}
	if m.Kind != "decision" {
		writeErr(w, http.StatusBadRequest, fmt.Sprintf("model %q is an embeddings model; use POST /v1/embeddings", req.Model))
		return
	}
	body, err := decisionBody(req)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	answer, status, err := systemone(req.Model, m, body)
	if err != nil {
		log.Printf("decision failed (%s): %v", req.Model, err)
		writeErr(w, status, err.Error())
		return
	}
	id := fmt.Sprintf("chatcmpl-so-%d", time.Now().UnixNano())
	created := time.Now().Unix()
	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		chunk := func(delta map[string]any, finish any) {
			b, _ := json.Marshal(map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": req.Model,
				"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}}})
			fmt.Fprintf(w, "data: %s\n\n", b)
		}
		chunk(map[string]any{"role": "assistant", "content": string(answer)}, nil)
		chunk(map[string]any{}, "stop")
		fmt.Fprint(w, "data: [DONE]\n\n")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"id": id, "object": "chat.completion", "created": created, "model": req.Model,
		"choices": []any{map[string]any{"index": 0, "finish_reason": "stop",
			"message": map[string]any{"role": "assistant", "content": string(answer)}}},
		"usage": map[string]any{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
	})
}

// handleSystemOne passes a native /v1/systemone body straight through; the model
// is chosen by a "model" field (default: the only model, if there is one).
func handleSystemOne(w http.ResponseWriter, r *http.Request) {
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	name, _ := body["model"].(string)
	delete(body, "model")
	if name == "" && len(cfg.Models) == 1 {
		for n := range cfg.Models {
			name = n
		}
	}
	m, ok := cfg.Models[name]
	if !ok || m.Kind != "decision" {
		writeErr(w, http.StatusNotFound, fmt.Sprintf("decision model %q not found; see GET /v1/models", name))
		return
	}
	answer, status, err := systemone(name, m, body)
	if err != nil {
		writeErr(w, status, err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(answer)
}

// handleEmbeddings passes an OpenAI embeddings request to the model's llama-server
// and answers with the public model ID.
func handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	var body map[string]any
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	name, _ := body["model"].(string)
	m, ok := cfg.Models[name]
	if !ok || m.Kind != "embeddings" {
		writeErr(w, http.StatusNotFound, fmt.Sprintf("embeddings model %q not found; see GET /v1/models", name))
		return
	}
	b, _ := json.Marshal(body)
	resp, err := client.Post(m.Upstream+"/v1/embeddings", "application/json", bytes.NewReader(b))
	if err != nil {
		writeErr(w, http.StatusBadGateway, err.Error())
		return
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	var out map[string]any
	if resp.StatusCode >= 300 || json.Unmarshal(data, &out) != nil {
		writeErr(w, http.StatusBadGateway, fmt.Sprintf("llama-server /v1/embeddings -> %d: %s", resp.StatusCode, truncate(string(data), 300)))
		return
	}
	out["model"] = name
	writeJSON(w, http.StatusOK, out)
}

func main() {
	if err := loadConfig(); err != nil {
		log.Fatalf("config: %v", err)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "models": len(cfg.Models)})
	})
	mux.HandleFunc("GET /v1/models", handleModels)
	mux.HandleFunc("POST /v1/chat/completions", handleChat)
	mux.HandleFunc("POST /v1/systemone", handleSystemOne)
	mux.HandleFunc("POST /v1/embeddings", handleEmbeddings)

	names := make([]string, 0, len(cfg.Models))
	for n, m := range cfg.Models {
		names = append(names, n+"->"+m.Upstream)
	}
	sort.Strings(names)
	log.Printf("systemone-shim on %s (%d models: %s)", addr, len(cfg.Models), strings.Join(names, ", "))
	srv := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	log.Fatal(srv.ListenAndServe())
}
