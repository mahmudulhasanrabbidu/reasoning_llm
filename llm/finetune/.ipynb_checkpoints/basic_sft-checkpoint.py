import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# Go up two levels ('..', '..') since we are inside llm/finetune/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from llm.pretraining import PretrainedQweenModel
from tokenizer.qween3_tokenizer import Qwen3Tokenizer
from llm.datasets import Qwen3FineTuningDataset

def train(model, dataloader, optimizer, device, epochs=1, save_dir="checkpoints"):
    """
    The core fine-tuning training loop.
    """
    model.train()
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n--- Starting Fine-Tuning on {device.upper()} ---")
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        # Wrap the dataloader in a tqdm progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            # 1. Move data to the correct device (GPU/CPU)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # 2. Forward Pass: Get the raw logits from the model
            optimizer.zero_grad()
            logits = model(input_ids)
            
            # 3. Shift the logits and labels for Causal Language Modeling
            # We want the logit at position `i` to predict the label at position `i+1`
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 4. Calculate Loss
            # Flatten the tensors: (Batch * Seq_Len, Vocab_Size) and (Batch * Seq_Len)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1),
                ignore_index=-100 # Ignores the user prompt padding we setup in Phase 2
            )
            
            # 5. Backward Pass: Calculate gradients
            loss.backward()
            
            # 6. Optimizer Step: Update the model weights
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        # Save a checkpoint after every epoch
        checkpoint_path = os.path.join(save_dir, f"qwen3_finetuned_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}\n")

if __name__ == "__main__":
    # --- Configuration ---
    BATCH_SIZE = 2      # Keep this small to avoid Out Of Memory (OOM) errors on local GPUs
    LEARNING_RATE = 5e-5
    EPOCHS = 1
    MAX_SEQ_LEN = 256   # Truncating to 256 tokens for faster training
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Tokenizer
    tokenizer_path = "tokenizer.json"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Please provide '{tokenizer_path}'.")
    tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
    
    # 2. Load Model & Pretrained Weights
    print("Loading Base Model...")
    model, _ = PretrainedQweenModel.from_pretrained(
        config_path="llm/config.json", 
        weight_path="llm/model.safetensors"
    )
    model.to(device)
    
    # 3. Load Dataset
    data_path = os.path.join("data", "alpaca_training_data.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found at {data_path}. Please run datasets.py first.")
        
    print("Loading Training Data...")
    with open(data_path, "r", encoding="utf-8") as f:
        training_data = json.load(f)
        
    # Optional: For testing, you can slice the data to train on a small subset first
    # training_data = training_data[:100] 
    
    dataset = Qwen3FineTuningDataset(training_data, tokenizer, max_seq_len=MAX_SEQ_LEN)
    
    # 4. Create DataLoader
    # This automatically batches your data and shuffles it every epoch
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 5. Initialize Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 6. Launch Training Loop
    train(model, dataloader, optimizer, device, epochs=EPOCHS)