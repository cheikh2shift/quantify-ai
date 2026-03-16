package main

import (
	"fmt"
	"os"

	"github.com/cheikh2shift/quantify-ai"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: quantify <model-name>")
		fmt.Println("Example: quantify qwen2.5:7b")
		fmt.Println("         quantify hf.co/TeichAI/Qwen3-4B")
		os.Exit(1)
	}

	model := os.Args[1]
	info, err := quantify.GetModelInfo(model)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Model:          %s\n", model)
	fmt.Printf("Context Limit:  %d\n", info.ContextLimit)
	if info.Size != "" {
		fmt.Printf("Size:           %s\n", info.Size)
	}
	if len(info.Parameters) > 0 {
		fmt.Println("Parameters:")
		for key, value := range info.Parameters {
			fmt.Printf("  %s: %s\n", key, value)
		}
	}
}
