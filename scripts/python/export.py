import argparse
import os
import json
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class ModelWrapper(nn.Module):
    """Wrapper class for model export to ONNX"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def convert_safetensors_to_onnx(model_path, output_path, opset_version=17):
    """
    Convert a model saved in SafeTensors format to ONNX format.
    
    Args:
        model_path: Path to the model directory containing SafeTensors files
        output_path: Path to save the ONNX model
        opset_version: ONNX opset version to use
    """
    print(f"Loading model from {model_path}")
    
    # Load the HuggingFace model and tokenizer
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,  # Use float16 by default
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Move model to evaluation mode
    model.eval()
    
    # Create a model wrapper
    wrapped_model = ModelWrapper(model)
    
    # Create dummy inputs for ONNX export
    batch_size = 1
    sequence_length = 8
    dummy_input_ids = torch.ones(batch_size, sequence_length, dtype=torch.long)
    dummy_attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.long)
    
    # Export the model to ONNX
    print(f"Exporting model to ONNX format (opset version: {opset_version})")
    
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.onnx.export(
        wrapped_model,
        (dummy_input_ids, dummy_attention_mask),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        }
    )
    
    print(f"Model successfully exported to: {output_path}")
    
    # Optionally save tokenizer alongside model
    tokenizer_path = os.path.join(os.path.dirname(output_path), "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    print(f"Tokenizer saved to: {tokenizer_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SafeTensors model to ONNX format")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory containing SafeTensors files")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the ONNX model")
    parser.add_argument("--opset_version", type=int, default=17, help="ONNX opset version")
    
    args = parser.parse_args()
    convert_safetensors_to_onnx(args.model_path, args.output_path, args.opset_version)