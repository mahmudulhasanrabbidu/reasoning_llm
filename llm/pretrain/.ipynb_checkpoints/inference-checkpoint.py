import os
import sys
import json
import torch
import torch.nn.functional as F

# Adjust paths to look two folders up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from llm.base_model import Qween3Model, Qwen3Config
from tokenizer.qween3_tokenizer import Qwen3Tokenizer

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
    
    # Load the trained weights (strict=True ensures every single layer matches perfectly)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    
    # CRITICAL: Put the model in evaluation mode to disable dropout and gradient tracking
    model.eval()
    
    return model, tokenizer


@torch.no_grad() # CRITICAL: Disables the memory-heavy gradient tracking for inference
def generate_text(model, tokenizer, prompt, device, max_new_tokens=50, temperature=0.8, top_k=10):
    # 1. Translate the prompt into integers
    input_ids = tokenizer.encode(prompt, chat_wrapped=False, show_progress=False)
    
    # Convert to a PyTorch tensor and add a Batch dimension: shape becomes (1, Sequence Length)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    print(f"\n[Prompt]: {prompt}", end="")
    
    # 2. The Autoregressive Loop
    for _ in range(max_new_tokens):
        # Forward pass: get the logits for the entire sequence
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            logits = model(input_tensor)
        
        # We only care about the predictions for the very LAST token in the sequence
        # logits shape is (Batch, Seq_Len, Vocab_Size) -> we grab (Batch, -1, Vocab_Size)
        next_token_logits = logits[:, -1, :] 
        
        # --- Sampling Logic ---
        # Apply Temperature (Higher = more random, Lower = more strict)
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
            
        # Apply Top-K (Ignore all words except the top K most likely ones)
        if top_k is not None:
            # Find the value of the Kth highest probability
            kth_val, _ = torch.topk(next_token_logits, top_k)
            kth_val = kth_val[:, -1].unsqueeze(-1)
            # Overwrite anything lower than that value with -infinity (so probability becomes 0)
            next_token_logits = torch.where(next_token_logits < kth_val, torch.tensor(-float('Inf')).to(device), next_token_logits)
        
        # Convert the raw math logits into percentages (0.0 to 1.0)
        probs = F.softmax(next_token_logits, dim=-1)
        
        # Pick the next token based on those percentages
        next_token = torch.multinomial(probs, num_samples=1)
        
        # 3. Output and Append
        # Stop generating if the model outputs the End-Of-Sequence token
        if next_token.item() == tokenizer.eos_token_id:
            break
            
        # Decode just the single new token to print it in real-time (typewriter effect)
        word = tokenizer.decode([next_token.item()])
        print(word, end="", flush=True)
        
        # Glue the new token to the end of the input tensor for the next loop iteration
        input_tensor = torch.cat((input_tensor, next_token), dim=1)
        
    print("\n")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Inference Engine on {device.type.upper()}...")
    
    # Define exact paths based on your training script
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
    
    # The Interactive CLI Loop
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