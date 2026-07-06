import os
import sys
import json
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from llm.base_model import Qween3Model, Qwen3Config
from tokenizer.qween_3_tokenizer import Qwen3Tokenizer

def load_model_and_tokenizer(checkpoint_path, config_path, tokenizer_path, device):
    print("Loading Tokenizer...")
    tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

    print("Loading Architecture Config...")
    with open(config_path, "r") as f:
        hf_config = json.load(f)
        
    custom_config = Qwen3Config(
        vocab_size=hf_config["vocab_size"],
        emb_dim=hf_config["hidden_size"],
        hidden_dim=hf_config["intermediate_size"],
        num_trf_blocks=hf_config["num_hidden_layers"],
        num_q_head=hf_config["num_attention_heads"],
        num_kv_head=hf_config["num_key_value_heads"],
        head_dim=hf_config.get("head_dim", 128),
        qk_norm=True,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    
    print("Initializing Model and Injecting Weights...")
    model = Qween3Model(custom_config).to(device)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    
    model.eval()
    
    return model, tokenizer


@torch.no_grad()
def generate_text(model, tokenizer, prompt, device, max_new_tokens=50, temperature=0.8, top_k=10):
    input_ids = tokenizer.encode(prompt, chat_wrapped=False, show_progress=False)
    
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    print(f"\n[Prompt]: {prompt}", end="")
    
    for _ in range(max_new_tokens):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            logits = model(input_tensor)
        
        next_token_logits = logits[:, -1, :] 
        
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
            
        if top_k is not None:
            kth_val, _ = torch.topk(next_token_logits, top_k)
            kth_val = kth_val[:, -1].unsqueeze(-1)
            next_token_logits = torch.where(next_token_logits < kth_val, torch.tensor(-float('Inf')).to(device), next_token_logits)
        
        probs = F.softmax(next_token_logits, dim=-1)
        
        next_token = torch.multinomial(probs, num_samples=1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
            
        word = tokenizer.decode([next_token.item()])
        print(word, end="", flush=True)
        
        input_tensor = torch.cat((input_tensor, next_token), dim=1)
        
    print("\n")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Inference Engine on {device.type.upper()}...")
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    config_path = os.path.join(root_dir, "config.json")
    tokenizer_path = os.path.join(root_dir, "tokenizer.json")
    checkpoint_path = os.path.join(root_dir, "checkpoints", "pretrain", "qween3_step_final.pt")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Could not find trained weights at {checkpoint_path}. Did pretraining finish?")
        
    model, tokenizer = load_model_and_tokenizer(checkpoint_path, config_path, tokenizer_path, device)
    
    print("\n" + "="*50)
    print("Model loaded successfully! Type 'quit' or 'exit' to stop.")
    print("="*50 + "\n")
    
    while True:
        user_prompt = input("You: ")
        if user_prompt.lower() in ["quit", "exit"]:
            break
            
        generate_text(
            model=model, 
            tokenizer=tokenizer, 
            prompt=user_prompt, 
            device=device,
            max_new_tokens=100, 
            temperature=0.8, 
            top_k=40
        )