# Quantify AI

A Go SDK to fetch model context limits from Ollama library and HuggingFace.

## Installation

```bash
go get github.com/cheikh2shift/quantify-ai
```

## Usage

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
