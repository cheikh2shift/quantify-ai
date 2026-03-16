package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/cheikh2shift/quantify-ai"
)

func main() {
	debug := flag.Bool("debug", false, "Print debug information")
	ollamaURL := flag.String("ollama-url", "", "Custom Ollama URL")
	huggingfaceURL := flag.String("hf-url", "", "Custom HuggingFace URL")
	flag.Parse()

	if flag.NArg() < 1 {
		fmt.Println("Usage: quantify [options] <model-name>")
		fmt.Println("Options:")
		flag.PrintDefaults()
		fmt.Println("Example: quantify qwen2.5:7b")
		fmt.Println("         quantify hf.co/TeichAI/Qwen3-4B")
		os.Exit(1)
	}

	model := flag.Arg(0)

	cfg := quantify.Config{
		OllamaURL:      *ollamaURL,
		HuggingFaceURL: *huggingfaceURL,
		Debug:          *debug,
	}

	if *debug {
		fmt.Fprintf(os.Stderr, "[DEBUG] Fetching model: %s\n", model)
		if cfg.OllamaURL != "" {
			fmt.Fprintf(os.Stderr, "[DEBUG] Using Ollama URL: %s\n", cfg.OllamaURL)
		}
		if cfg.HuggingFaceURL != "" {
			fmt.Fprintf(os.Stderr, "[DEBUG] Using HuggingFace URL: %s\n", cfg.HuggingFaceURL)
		}
	}

	info, err := quantify.GetModelInfoWithConfig(model, cfg)
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
