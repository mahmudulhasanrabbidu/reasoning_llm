import os
import json
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars
from safetensors.torch import load_file
from tqdm import tqdm

# Import your custom architecture from gpt.py
from base_model import Qwen3Config, Qween3Model

# Disable Hugging Face progress bars to prevent UI crashes
disable_progress_bars()

class PretrainedQweenModel(Qween3Model):
    """
    A wrapper class that inherits the Qween3Model architecture and adds 
    methods for downloading, parsing configs, loading pretrained weights, 
    and visualizing parameters.
    """

    @staticmethod
    def download_model(repo_id="Qwen/Qwen3-0.6B", local_dir="."):
        """Downloads the required config, safetensors, and tokenizer files."""
        print(f"Downloading {repo_id} directly to: {os.path.abspath(local_dir)}")
        snapshot_download(repo_id=repo_id, allow_patterns=["*.safetensors", "config.json", "tokenizer.json"], local_dir=local_dir)
        print("Download complete!\n")

    @classmethod
    def create_custom_config(cls, config_path="config.json") -> Qwen3Config:
        """Reads the Hugging Face JSON and returns a strictly typed Qwen3Config."""
        with open(config_path, "r") as f:
            hf_config = json.load(f)

        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        hf_dtype_string = hf_config.get("torch_dtype", "float16")
        torch_dtype = dtype_map.get(hf_dtype_string, torch.float16)

        return Qwen3Config(
            vocab_size=hf_config["vocab_size"],
            emb_dim=hf_config["hidden_size"],                 
            hidden_dim=hf_config["intermediate_size"],         
            num_trf_blocks=hf_config["num_hidden_layers"],     
            num_q_head=hf_config["num_attention_heads"],       
            num_kv_head=hf_config["num_key_value_heads"],     
            head_dim=hf_config.get("head_dim", 128),
            qk_norm=True,                                      
            dtype=torch_dtype,
            rms_norm_eps=hf_config.get("rms_norm_eps", 1e-6),
            rope_theta=hf_config.get("rope_theta", 1000000.0), 
            max_position_embeddings=hf_config.get("max_position_embeddings", 40960)
        )

    @staticmethod
    def visualize_huggingface_safetensors(file_path="model.safetensors"):
        """Utility to inspect the raw Hugging Face safetensors file."""
        if not os.path.exists(file_path):
            print(f"Error: Could not find '{file_path}'. Make sure it is in the same folder as this script.")
            return
        
        # Load the file into CPU memory
        state_dict = load_file(file_path)
    
        for name, param in state_dict.items():
            print(f"name: {name}, shape: {param.shape}")
    
        return state_dict

    def visualize_custom_parameters(self):
        """
        Instance method to print the names and shapes of the custom model's parameters.
        """
        print("\n--- Custom Model Parameters ---")
        state_dict = self.state_dict()
        total_params = 0
        
        for name, param in state_dict.items():
            print(f"name: {name}, shape: {param.shape}")
            total_params += param.numel()
            
        print(f"\nTotal Architecture Parameters: {total_params:,}")
        print("-------------------------------\n")

    @classmethod
    def from_pretrained(cls, config_path="config.json", weight_path="model.safetensors"):
        """
        The master initialization method. It builds the config, instantiates 
        the model architecture, and injects the pretrained weights in-place.
        """
        # Build Configuration
        print("Building configuration from config.json...")
        config = cls.create_custom_config(config_path)

        # Instantiate Model (Using the parent class __init__)
        print("Initializing empty architecture...")
        model = cls(config)

        # Load Safetensors
        print(f"Loading weights from {weight_path}...")
        pretrained_state_dict = load_file(weight_path)

        # Map and Inject Weights
        print("Mapping and loading weights...")
        
        def assign(target_param, hf_key):
            if hf_key not in pretrained_state_dict:
                print(f"Warning: '{hf_key}' not found in state_dict. Skipping.")
                return
            source_tensor = pretrained_state_dict[hf_key]
            if target_param.shape != source_tensor.shape:
                raise ValueError(f"Shape mismatch at {hf_key}. Model expected {target_param.shape}, got {source_tensor.shape}")
            with torch.no_grad():
                target_param.copy_(source_tensor)

        # Top-Level Variables
        assign(model.emb.weight, "model.embed_tokens.weight")
        assign(model.final_norm.scale, "model.norm.weight")
        assign(model.out.weight, "lm_head.weight")

        # Transformer Blocks
        for i in tqdm(range(model.config.num_trf_blocks), desc="Injecting Transformer Weights"):
            block = model.trf_blocks[i]
            hf_prefix = f"model.layers.{i}"

            assign(block.rms_norm1.scale, f"{hf_prefix}.input_layernorm.weight")
            assign(block.rms_norm2.scale, f"{hf_prefix}.post_attention_layernorm.weight")

            assign(block.attn.w_q.weight, f"{hf_prefix}.self_attn.q_proj.weight")
            assign(block.attn.w_k.weight, f"{hf_prefix}.self_attn.k_proj.weight")
            assign(block.attn.w_v.weight, f"{hf_prefix}.self_attn.v_proj.weight")
            assign(block.attn.out_proj.weight, f"{hf_prefix}.self_attn.o_proj.weight")

            if model.config.qk_norm:
                assign(block.attn.q_norm.scale, f"{hf_prefix}.self_attn.q_norm.weight")
                assign(block.attn.k_norm.scale, f"{hf_prefix}.self_attn.k_norm.weight")

            assign(block.ffd.fc1.weight, f"{hf_prefix}.mlp.gate_proj.weight")
            assign(block.ffd.fc2.weight, f"{hf_prefix}.mlp.up_proj.weight")
            assign(block.ffd.fc3.weight, f"{hf_prefix}.mlp.down_proj.weight")

        print("Pretrained weights loaded successfully!\n")
        
        # We return both the model and the dictionary so the main block can run the verification check
        return model, pretrained_state_dict


# Execution Block
if __name__ == "__main__":
    
    # Download the files
    repo_name = "Qwen/Qwen3-0.6B"
    PretrainedQweenModel.download_model(repo_id=repo_name)

    # Verify files and build the model
    if not os.path.exists("model.safetensors") or not os.path.exists("config.json"):
        print("Error: Could not find the required configuration or weight files.")
    else:
        # One clean call handles config creation, instantiation, and weight injection
        my_model, hf_state_dict = PretrainedQweenModel.from_pretrained(config_path="config.json", weight_path="model.safetensors")

        # Call the new instance method to print the custom model's layers and shapes
        my_model.visualize_custom_parameters()

        # Verify the parameter counts match exactly
        num_pretrained = sum(p.numel() for p in hf_state_dict.values())
        num_my_model = sum(p.numel() for p in my_model.parameters())
        
        print("--- Parameter Count Verification ---")
        print(f"Pretrained File: {num_pretrained:,}")
        print(f"Custom Model:    {num_my_model:,}")
        
        if num_pretrained == num_my_model:
            print("Status: PERFECT MATCH!")
        else:
            print("Status: MISMATCH DETECTED.")