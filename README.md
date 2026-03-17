# Quantify AI

A Go library and CLI to fetch model context limits from Ollama library and HuggingFace.

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
quantify meta-llama/Llama-3.2-1B-Instruct
# Output:
# Model:                    meta-llama/Llama-3.2-1B-Instruct
# Context Limit:            128000
# Size:                     1.3GB
```

## Library Usage

```go
package main

import (
	"fmt"
	"log"

	"github.com/cheikh2shift/quantify-ai"
)

func main() {
	// Fetch Ollama model info
	model, err := quantify.FetchOllama("qwen2.5:7b")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Model: %s, Context Limit: %d, Size: %s\n", model.Name, model.ContextLimit, model.Size)

	// Fetch HuggingFace model info
	model, err = quantify.FetchHuggingFace("meta-llama/Llama-3.2-1B-Instruct")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Model: %s, Context Limit: %d, Size: %s\n", model.Name, model.ContextLimit, model.Size)
}
```

## Supported Models

- **Ollama**: All models available on Ollama library
- **HuggingFace**: All models available on HuggingFace hub

## License

MIT
