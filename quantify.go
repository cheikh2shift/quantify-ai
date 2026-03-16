package quantify

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"
)

// ModelInfo contains information about a model.
type ModelInfo struct {
	ContextLimit int               // Context window size in tokens
	Size         string            // Model size (e.g., "4.7GB")
	Parameters   map[string]string // Additional parameters (e.g., quantization)
}

// GetModelInfo fetches model information from Ollama library or HuggingFace.
// For Ollama models, pass the model name (e.g., "qwen2.5:7b").
// For HuggingFace models, pass with "hf.co/" prefix (e.g., "hf.co/TeichAI/Qwen3-4B").
// Returns ModelInfo with ContextLimit, Size, and Parameters.
// Returns an error if the model is not found or context limit cannot be determined.
func GetModelInfo(model string) (*ModelInfo, error) {
	info := &ModelInfo{
		Parameters: make(map[string]string),
	}

	if strings.Contains(model, "hf.co") {
		if err := fetchHuggingFaceInfo(model, info); err != nil {
			return nil, err
		}
	} else {
		if err := fetchOllamaLibraryInfo(model, info); err != nil {
			return nil, err
		}
	}

	if info.ContextLimit == 0 {
		return nil, fmt.Errorf("could not determine context limit for model: %s", model)
	}

	return info, nil
}

// fetchOllamaLibraryInfo fetches model info from Ollama's website.
func fetchOllamaLibraryInfo(model string, info *ModelInfo) error {
	modelName := model
	if idx := strings.Index(modelName, ":"); idx > 0 {
		modelName = modelName[:idx]
	}
	modelName = strings.TrimPrefix(modelName, "hf.co/")

	url := fmt.Sprintf("https://ollama.com/library/%s/tags", modelName)
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return fmt.Errorf("library page returned status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}

	re := regexp.MustCompile(`(\d+)[Kk]\s*context\s*window`)
	matches := re.FindStringSubmatch(string(body))
	if len(matches) >= 2 {
		contextVal, err := strconv.Atoi(matches[1])
		if err == nil {
			info.ContextLimit = contextVal * 1000
		}
	}

	sizeRe := regexp.MustCompile(`(\d+\.?\d*)([GM]B)`)
	sizeMatches := sizeRe.FindStringSubmatch(string(body))
	if len(sizeMatches) >= 3 {
		info.Size = sizeMatches[1] + sizeMatches[2]
	}

	quantRe := regexp.MustCompile(`\.(\w+)\.gguf|"(\w+)"\.gguf`)
	quantMatches := quantRe.FindAllStringSubmatch(string(body), -1)
	if len(quantMatches) > 0 {
		for _, match := range quantMatches {
			if len(match) >= 2 && match[1] != "" {
				info.Parameters["quantization"] = match[1]
				break
			}
		}
	}

	return nil
}

// fetchHuggingFaceInfo fetches model info from HuggingFace API.
func fetchHuggingFaceInfo(model string, info *ModelInfo) error {
	model = strings.TrimPrefix(model, "hf.co/")
	if idx := strings.Index(model, ":"); idx > 0 {
		model = model[:idx]
	}

	url := fmt.Sprintf("https://huggingface.co/api/models/%s", model)
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return fmt.Errorf("HuggingFace API returned status %d", resp.StatusCode)
	}

	var hfResp struct {
		Config struct {
			MaxModelLength        int `json:"max_model_length"`
			MaxPositionEmbeddings int `json:"max_position_embeddings"`
		} `json:"config"`
		GGUF struct {
			ContextLength int `json:"context_length"`
		} `json:"gguf"`
		Siblings []struct {
			Rfilename string `json:"rfilename"`
		} `json:"siblings"`
	}

	if err := decodeJSON(resp.Body, &hfResp); err != nil {
		return err
	}

	if hfResp.GGUF.ContextLength > 0 {
		info.ContextLimit = hfResp.GGUF.ContextLength
	} else if hfResp.Config.MaxModelLength > 0 {
		info.ContextLimit = hfResp.Config.MaxModelLength
	} else if hfResp.Config.MaxPositionEmbeddings > 0 {
		info.ContextLimit = hfResp.Config.MaxPositionEmbeddings
	}

	for _, sibling := range hfResp.Siblings {
		if strings.Contains(sibling.Rfilename, ".gguf") {
			parts := strings.Split(sibling.Rfilename, ".")
			if len(parts) >= 2 {
				info.Parameters["quantization"] = parts[len(parts)-2]
				break
			}
		}
	}

	return nil
}

func decodeJSON(body io.Reader, v interface{}) error {
	return json.NewDecoder(body).Decode(v)
}
