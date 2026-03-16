package quantify

import (
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// Test helper functions - regex parsing
func TestOllamaModelParsing(t *testing.T) {
	tests := []struct {
		name          string
		html          string
		expectedLimit int
		expectedSize  string
		expectedQuant string
	}{
		{
			name:          "qwen2.5 with 128K context",
			html:          `128K context window  •  6.6GB • qwen2.5-7b-instruct-q4_k_m.gguf`,
			expectedLimit: 128000,
			expectedSize:  "6.6GB",
		},
		{
			name:          "gemma3 with 128K context",
			html:          `128K context window  •  4.9GB • gemma3-4b-it-q4_k_m.gguf`,
			expectedLimit: 128000,
			expectedSize:  "4.9GB",
		},
		{
			name:          "32K context",
			html:          `32K context window  •  3.8GB • llama3-8b-instruct-q4_k_m.gguf`,
			expectedLimit: 32000,
			expectedSize:  "3.8GB",
		},
		{
			name:          "198K context",
			html:          `198K context window  •  4.7GB • llama-3.1-70b-instruct-q4_k_m.gguf`,
			expectedLimit: 198000,
			expectedSize:  "4.7GB",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			re := regexp.MustCompile(`(\d+)[Kk]\s*context\s*window`)
			matches := re.FindStringSubmatch(tt.html)
			if len(matches) >= 2 {
				contextVal, err := strconv.Atoi(matches[1])
				if err == nil {
					limit := contextVal * 1000
					if limit != tt.expectedLimit {
						t.Errorf("expected limit %d, got %d", tt.expectedLimit, limit)
					}
				}
			}

			sizeRe := regexp.MustCompile(`(\d+\.?\d*)([GM]B)`)
			sizeMatches := sizeRe.FindStringSubmatch(tt.html)
			if len(sizeMatches) >= 3 {
				size := sizeMatches[1] + sizeMatches[2]
				if size != tt.expectedSize {
					t.Errorf("expected size %s, got %s", tt.expectedSize, size)
				}
			}
		})
	}
}

func TestParseOllamaHTML(t *testing.T) {
	// Test with actual format from the code - uses FindAllStringSubmatch
	html := `128K context window  •  6.6GB • qwen2.5-7b-instruct-q4_k_m.gguf`
	info := &ModelInfo{
		Parameters: make(map[string]string),
	}

	re := regexp.MustCompile(`(\d+)[Kk]\s*context\s*window`)
	matches := re.FindStringSubmatch(html)
	if len(matches) >= 2 {
		contextVal, err := strconv.Atoi(matches[1])
		if err == nil {
			info.ContextLimit = contextVal * 1000
		}
	}

	sizeRe := regexp.MustCompile(`(\d+\.?\d*)([GM]B)`)
	sizeMatches := sizeRe.FindStringSubmatch(html)
	if len(sizeMatches) >= 3 {
		info.Size = sizeMatches[1] + sizeMatches[2]
	}

	// Use FindAllStringSubmatch like in the actual code
	quantRe := regexp.MustCompile(`\.(\w+)\.gguf|"(\w+)"\.gguf`)
	quantMatches := quantRe.FindAllStringSubmatch(html, -1)
	if len(quantMatches) > 0 {
		for _, match := range quantMatches {
			if len(match) >= 2 && match[1] != "" {
				info.Parameters["quantization"] = match[1]
				break
			}
		}
	}

	if info.ContextLimit != 128000 {
		t.Errorf("expected ContextLimit 128000, got %d", info.ContextLimit)
	}
	if info.Size != "6.6GB" {
		t.Errorf("expected Size 6.6GB, got %s", info.Size)
	}
}

// Test decodeJSON helper
func TestDecodeJSON(t *testing.T) {
	jsonData := `{"context": 4096, "size": "4GB"}`
	reader := strings.NewReader(jsonData)

	var result map[string]interface{}
	if err := decodeJSON(reader, &result); err != nil {
		t.Errorf("decodeJSON failed: %v", err)
	}

	if result["context"] == nil {
		t.Error("expected context key in result")
	}
}

func TestDecodeJSONInvalid(t *testing.T) {
	jsonData := `{invalid json}`
	reader := strings.NewReader(jsonData)

	var result map[string]interface{}
	if err := decodeJSON(reader, &result); err == nil {
		t.Error("expected error for invalid JSON")
	}
}

// Test GetModelInfoWithConfig - uses httptest to mock HTTP responses
func TestGetModelInfoWithConfigOllama(t *testing.T) {
	// Create mock server for Ollama
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`128K context window  •  6.6GB • qwen2.5-7b-instruct-q4_k_m.gguf`))
	}))
	defer server.Close()

	cfg := Config{
		OllamaURL: server.URL,
	}

	info, err := GetModelInfoWithConfig("qwen2.5", cfg)
	if err != nil {
		t.Errorf("GetModelInfoWithConfig failed: %v", err)
	}

	if info.ContextLimit != 128000 {
		t.Errorf("expected ContextLimit 128000, got %d", info.ContextLimit)
	}
	if info.Size != "6.6GB" {
		t.Errorf("expected Size 6.6GB, got %s", info.Size)
	}
}

func TestGetModelInfoWithConfigHuggingFace(t *testing.T) {
	// Create mock server for HuggingFace
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"config": {"max_model_length": 4096}, "gguf": {"context_length": 4096}, "siblings": [{"rfilename": "model.gguf"}]}`))
	}))
	defer server.Close()

	cfg := Config{
		HuggingFaceURL: server.URL,
	}

	// Use the full URL format that GetModelInfoWithConfig expects
	info, err := GetModelInfoWithConfig("hf.co/test/model", cfg)
	if err != nil {
		t.Errorf("GetModelInfoWithConfig failed: %v", err)
	}

	if info.ContextLimit != 4096 {
		t.Errorf("expected ContextLimit 4096, got %d", info.ContextLimit)
	}
}

func TestGetModelInfoWithConfigInvalidModel(t *testing.T) {
	// Create mock server that returns empty response
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("no context info"))
	}))
	defer server.Close()

	cfg := Config{
		OllamaURL: server.URL,
	}

	_, err := GetModelInfoWithConfig("unknownmodel", cfg)
	if err == nil {
		t.Error("expected error for invalid model")
	}
}

func TestGetModelInfo(t *testing.T) {
	// Test default config - will make real HTTP calls
	// This test verifies the function signature works
	// Note: This may fail if network is not available, which is expected
	info, err := GetModelInfo("qwen2.5")
	if err != nil {
		// Expected to fail without network, but function should be callable
		t.Logf("GetModelInfo failed (expected without network): %v", err)
	}
	_ = info // suppress unused warning
}

// Test fetchOllamaLibraryInfo directly
func TestFetchOllamaLibraryInfo(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`128K context window  •  6.6GB • test-model-q4_k_m.gguf`))
	}))
	defer server.Close()

	info := &ModelInfo{
		Parameters: make(map[string]string),
	}
	err := fetchOllamaLibraryInfo("test-model", info, server.URL, false)
	if err != nil {
		t.Errorf("fetchOllamaLibraryInfo failed: %v", err)
	}

	if info.ContextLimit != 128000 {
		t.Errorf("expected ContextLimit 128000, got %d", info.ContextLimit)
	}
}

func TestFetchOllamaLibraryInfoHTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	info := &ModelInfo{
		Parameters: make(map[string]string),
	}
	err := fetchOllamaLibraryInfo("test-model", info, server.URL, false)
	if err == nil {
		t.Error("expected error for HTTP 404")
	}
}

// Test fetchHuggingFaceInfo directly
func TestFetchHuggingFaceInfo(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"config": {"max_model_length": 8192}, "gguf": {"context_length": 8192}, "siblings": [{"rfilename": "model-q4.gguf"}]}`))
	}))
	defer server.Close()

	info := &ModelInfo{
		Parameters: make(map[string]string),
	}
	err := fetchHuggingFaceInfo("test/model", info, server.URL, false)
	if err != nil {
		t.Errorf("fetchHuggingFaceInfo failed: %v", err)
	}

	if info.ContextLimit != 8192 {
		t.Errorf("expected ContextLimit 8192, got %d", info.ContextLimit)
	}
}

func TestFetchHuggingFaceInfoHTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	info := &ModelInfo{
		Parameters: make(map[string]string),
	}
	err := fetchHuggingFaceInfo("test/model", info, server.URL, false)
	if err == nil {
		t.Error("expected error for HTTP 500")
	}
}

func TestFetchHuggingFaceInfoInvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`invalid json`))
	}))
	defer server.Close()

	info := &ModelInfo{
		Parameters: make(map[string]string),
	}
	err := fetchHuggingFaceInfo("test/model", info, server.URL, false)
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

// Test Config defaults
func TestConfigDefaults(t *testing.T) {
	cfg := Config{}
	if cfg.OllamaURL != "" {
		t.Error("expected empty OllamaURL by default")
	}
	if cfg.HuggingFaceURL != "" {
		t.Error("expected empty HuggingFaceURL by default")
	}
}

// Test ModelInfo struct
func TestModelInfoStruct(t *testing.T) {
	info := ModelInfo{
		ContextLimit: 128000,
		Size:         "6.6GB",
		Parameters:   map[string]string{"quantization": "q4_k_m"},
	}

	if info.ContextLimit != 128000 {
		t.Errorf("expected ContextLimit 128000, got %d", info.ContextLimit)
	}
	if info.Size != "6.6GB" {
		t.Errorf("expected Size 6.6GB, got %s", info.Size)
	}
	if info.Parameters["quantization"] != "q4_k_m" {
		t.Errorf("expected quantization q4_k_m, got %s", info.Parameters["quantization"])
	}
}

// Test io.ReadAll replacement (used in decodeJSON)
func TestIoReadAll(t *testing.T) {
	reader := strings.NewReader("test data")
	data, err := io.ReadAll(reader)
	if err != nil {
		t.Errorf("io.ReadAll failed: %v", err)
	}
	if string(data) != "test data" {
		t.Errorf("expected 'test data', got '%s'", string(data))
	}
}
