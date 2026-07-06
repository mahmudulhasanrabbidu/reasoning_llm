import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# go up two levels ('..', '..'), we are inside llm/finetune/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from llm.pretrained_weight_loader import PretrainedQweenModel
from tokenizer.qween_3_tokenizer import Qwen3Tokenizer
from llm.finetune.dataset import Qwen3FineTuningDataset

def train(model, dataloader, optimizer, device, epochs=1, save_dir="checkpoints"):
    """
    The core fine-tuning training loop.
    """
    model.train()
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n--- Starting Fine-Tuning on {device.upper()} ---")
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids)
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1),
                ignore_index=-100
            )
            
            loss.backward()
            
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        checkpoint_path = os.path.join(save_dir, f"qwen3_finetuned_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}\n")

if __name__ == "__main__":
    BATCH_SIZE = 2
    LEARNING_RATE = 5e-5
    EPOCHS = 1
    MAX_SEQ_LEN = 256
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer_path = "tokenizer.json"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Please provide '{tokenizer_path}'.")
    tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
    
    print("Loading Base Model...")
    model, _ = PretrainedQweenModel.from_pretrained(
        config_path="llm/config.json", 
        weight_path="llm/model.safetensors"
    )
    model.to(device)
    
    data_path = os.path.join("data", "alpaca_training_data.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found at {data_path}. Please run datasets.py first.")
        
    print("Loading Training Data...")
    with open(data_path, "r", encoding="utf-8") as f:
        training_data = json.load(f)
        
    
    dataset = Qwen3FineTuningDataset(training_data, tokenizer, max_seq_len=MAX_SEQ_LEN)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    train(model, dataloader, optimizer, device, epochs=EPOCHS)