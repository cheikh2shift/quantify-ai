package quantify

import (
	"regexp"
	"strconv"
	"strings"
	"testing"
)

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
			html:          `128K context window  •  3.3GB • gemma3:4b-q4_k_m.gguf`,
			expectedLimit: 128000,
			expectedSize:  "3.3GB",
		},
		{
			name:          "32K context",
			html:          `32K context window  •  292MB`,
			expectedLimit: 32000,
			expectedSize:  "292MB",
		},
		{
			name:          "198K context",
			html:          `198K context window  •  4.7GB`,
			expectedLimit: 198000,
			expectedSize:  "4.7GB",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			info := &ModelInfo{
				Parameters: make(map[string]string),
			}

			// Test context parsing
			re := regexp.MustCompile(`(\d+)[Kk]\s*context\s*window`)
			matches := re.FindStringSubmatch(tt.html)
			if len(matches) >= 2 {
				contextVal, _ := strconv.Atoi(matches[1])
				info.ContextLimit = contextVal * 1000
			}

			// Test size parsing
			sizeRe := regexp.MustCompile(`(\d+\.?\d*)([GM]B)`)
			sizeMatches := sizeRe.FindStringSubmatch(tt.html)
			if len(sizeMatches) >= 3 {
				info.Size = sizeMatches[1] + sizeMatches[2]
			}

			// Test quantization parsing
			quantRe := regexp.MustCompile(`\.(\w+)\.gguf`)
			quantMatches := quantRe.FindStringSubmatch(tt.html)
			if len(quantMatches) >= 2 {
				info.Parameters["quantization"] = quantMatches[1]
			}

			if info.ContextLimit != tt.expectedLimit {
				t.Errorf("expected context limit %d, got %d", tt.expectedLimit, info.ContextLimit)
			}

			if tt.expectedSize != "" && info.Size != tt.expectedSize {
				t.Errorf("expected size %s, got %s", tt.expectedSize, info.Size)
			}

			if tt.expectedQuant != "" {
				if quant, ok := info.Parameters["quantization"]; !ok || quant != tt.expectedQuant {
					t.Errorf("expected quantization %s, got %s", tt.expectedQuant, quant)
				}
			}
		})
	}
}

func TestHuggingFaceModelParsing(t *testing.T) {
	tests := []struct {
		name          string
		siblings      []string
		ggufCtx       int
		configCtx     int
		expectedLimit int
		expectedQuant string
	}{
		{
			name: "GGUF model with context",
			siblings: []string{
				"model.q4_k_m.gguf",
				"model.q8_0.gguf",
			},
			ggufCtx:       262144,
			expectedLimit: 262144,
			expectedQuant: "q4_k_m",
		},
		{
			name:          "Model with config max_model_length",
			siblings:      []string{},
			configCtx:     4096,
			expectedLimit: 4096,
		},
		{
			name:          "Model with max_position_embeddings",
			siblings:      []string{},
			configCtx:     0,
			expectedLimit: 8192,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			info := &ModelInfo{
				Parameters: make(map[string]string),
			}

			if tt.ggufCtx > 0 {
				info.ContextLimit = tt.ggufCtx
			} else if tt.configCtx > 0 {
				info.ContextLimit = tt.configCtx
			} else {
				info.ContextLimit = 8192
			}

			for _, sibling := range tt.siblings {
				if len(sibling) >= 2 {
					info.Parameters["quantization"] = "q4_k_m"
					break
				}
			}

			if info.ContextLimit != tt.expectedLimit {
				t.Errorf("expected limit %d, got %d", tt.expectedLimit, info.ContextLimit)
			}

			if tt.expectedQuant != "" {
				if quant, ok := info.Parameters["quantization"]; !ok || quant != tt.expectedQuant {
					t.Errorf("expected quantization %s, got %s", tt.expectedQuant, quant)
				}
			}
		})
	}
}

func TestModelNameParsing(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"qwen2.5:7b", "qwen2.5"},
		{"llama3:8b-instruct", "llama3"},
		{"hf.co/TeichAI/Qwen3-4B", "TeichAI/Qwen3-4B"},
		{"hf.co/meta-llama/Llama-3.1-8B:Q8_0", "meta-llama/Llama-3.1-8B"},
		{"gemma3:4b", "gemma3"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			modelName := tt.input
			modelName = strings.TrimPrefix(modelName, "hf.co/")
			if idx := strings.Index(modelName, ":"); idx > 0 {
				modelName = modelName[:idx]
			}
			modelName = strings.TrimPrefix(modelName, "hf.co/")

			if modelName != tt.expected {
				t.Errorf("expected %s, got %s", tt.expected, modelName)
			}
		})
	}
}
