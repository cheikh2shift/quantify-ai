# Quantify AI

A Go SDK and CLI to fetch model context limits from Ollama library and HuggingFace.

## Installation

```bash
go get github.com/cheikh2shift/quantify-ai
```

## CLI Usage

```bash
# Build the CLI
go install github.com/cheikh2shift/quantify-ai/cmd@latest

# Or run directly
go run github.com/cheikh2shift/quantify-ai/cmd <model-name>
```

### Examples

```bash
# Ollama model
quantify qwen2.5:7b
# Output:
# Model:          qwen2.5:7b
# Context Limit: 32000
# Size:           4.7GB

# HuggingFace model
quantify hf.co/TeichAI/Qwen3-4B-Thinking-2507-GPT-5.1-Codex-Max-Distill-GGUF:Q8_0
# Output:
# Model:          hf.co/TeichAI/Qwen3-4B-Thinking-2507-GPT-5.1-Codex-Max-Distill-GGUF:Q8_0
# Context Limit:  262144
# Parameters:
#   quantization: bf16
```

## Go SDK Usage

```go
package main

import (
	"fmt"
	"github.com/cheikh2shift/quantify-ai"
)

func main() {
	// Get context limit for an Ollama model
	info, err := quantify.GetModelInfo("qwen2.5:7b")
	if err != nil {
		panic(err)
	}
	fmt.Printf("Context Limit: %d\n", info.ContextLimit)

	// Or HuggingFace model
	info, err = quantify.GetModelInfo("hf.co/TeichAI/Qwen3-4B-Thinking")
	if err != nil {
		panic(err)
	}
	fmt.Printf("Context Limit: %d\n", info.ContextLimit)
}
```

## Supported Providers

- **Ollama**: Pass model name (e.g., `qwen2.5:7b`, `llama3:8b`)
- **HuggingFace**: Pass model with `hf.co/` prefix (e.g., `hf.co/TeichAI/Qwen3-4B`)

The SDK automatically detects the provider based on the model name.

## ModelInfo Structure

```go
type ModelInfo struct {
	ContextLimit int            // Context window size in tokens
	Size         string         // Model size (e.g., "4.7GB")
	Parameters   map[string]string  // Additional parameters (e.g., quantization)
}
```
