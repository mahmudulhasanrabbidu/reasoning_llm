import os
import sys
import math
import requests
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import os
import sys
import math
import requests
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Adjust paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from llm.pretrained_weight_loader import PretrainedQweenModel
from tokenizer.qween3_tokenizer import Qwen3Tokenizer
from llm.pretrain.continuous_text_dataset import ContinuousTextDataset

class ContinuousPretrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device, tokenizer, grad_accum_steps=8, max_grad_norm=1.0, save_dir="checkpoints"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler 
        self.device = device
        self.tokenizer = tokenizer
        self.grad_accum_steps = grad_accum_steps 
        self.max_grad_norm = max_grad_norm
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.scaler = torch.amp.GradScaler(enabled=(self.device.type == "cuda"))

    def _save_checkpoint(self, global_step, loss):
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_step_{global_step}.pt")
        checkpoint = {
            "global_step": global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "loss": loss
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"\n[Checkpoint] Saved state at step {global_step}")

    def _load_latest_checkpoint(self):
        checkpoints = [f for f in os.listdir(self.save_dir) if f.startswith("checkpoint_step_")]
        if not checkpoints:
            return 0
        
        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        latest_ckpt = os.path.join(self.save_dir, checkpoints[-1])
        
        print(f"\n[Checkpoint] Loading {latest_ckpt}...")
        ckpt = torch.load(latest_ckpt, map_location=self.device)
        
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        return ckpt["global_step"]

    def _calc_loss_loader(self, loader, eval_iter):
        total_loss = 0.0
        processed_batches = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(loader):
                if i >= eval_iter: break
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=(self.device.type == "cuda")):
                    logits = self.model(inputs)
                    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
                total_loss += loss.item()
                processed_batches += 1
        return total_loss / processed_batches if processed_batches > 0 else 0.0

    def evaluate(self, eval_iter):
        self.model.eval()
        t_loss = self._calc_loss_loader(self.train_loader, eval_iter)
        v_loss = self._calc_loss_loader(self.val_loader, eval_iter)
        self.model.train()
        return t_loss, v_loss

    def generate_sample(self, start_context="Every effort moves you", max_new_tokens=30):
        self.model.eval()
        encoded = torch.tensor([self.tokenizer.encode(start_context, chat_wrapped=False, show_progress=False)], device=self.device)
        print(f"\n--- Generating Sample ---\n{start_context}", end="")
        with torch.no_grad():
            for _ in range(max_new_tokens):
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=(self.device.type == "cuda")):
                    logits = self.model(encoded)
                next_token_id = torch.argmax(logits[0, -1, :]).item()
                if next_token_id == self.tokenizer.eos_token_id: break
                encoded = torch.cat((encoded, torch.tensor([[next_token_id]], device=self.device)), dim=1)
                print(self.tokenizer.decode([next_token_id]), end="", flush=True)
        print("\n-------------------------\n")
        self.model.train()

    def train(self, num_epochs, eval_freq, eval_iter=5, sample_context="Every effort moves you", auto_resume=True):
        global_step = 0
        if auto_resume and os.path.exists(self.save_dir):
            global_step = self._load_latest_checkpoint()
        
        self.optimizer.zero_grad(set_to_none=True)
        
        # CORRECTED: total_steps represents optimization steps, not batches
        steps_per_epoch = math.ceil(len(self.train_loader) / self.grad_accum_steps)
        total_steps = steps_per_epoch * num_epochs
        
        pbar = tqdm(total=total_steps, initial=global_step, desc="Pretraining")
        
        v_loss = 0.0 # Initialize to avoid NameError

        for epoch in range(num_epochs):
            self.model.train()
            for step, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=(self.device.type == "cuda")):
                    logits = self.model(inputs)
                    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten()) / self.grad_accum_steps
                
                self.scaler.scale(loss).backward()
                
                if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    
                    global_step += 1
                    pbar.update(1)
                    
                    # Update status
                    current_lr = self.optimizer.param_groups[0]['lr']
                    pbar.set_postfix({"loss": f"{loss.item() * self.grad_accum_steps:.4f}", "lr": f"{current_lr:.2e}", "v_loss": f"{v_loss:.3f}"})
                    
                    if global_step % eval_freq == 0:
                        t_loss, v_loss = self.evaluate(eval_iter)
                        self._save_checkpoint(global_step, t_loss)
            
            self.generate_sample(start_context=sample_context)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "the-verdict.txt")
    
    if not os.path.exists(data_path):
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(requests.get(url).text)
            
    with open(data_path, "r", encoding="utf-8") as f: text_data = f.read()

    tokenizer = Qwen3Tokenizer(tokenizer_file_path=os.path.join(base_dir, '..', '..', 'tokenizer.json'))
    model, _ = PretrainedQweenModel.from_pretrained(os.path.join(base_dir, '..', 'config.json'), os.path.join(base_dir, '..', 'model.safetensors'))
    model.to(device)

    train_dataset = ContinuousTextDataset(text_data[:int(0.85 * len(text_data))], tokenizer, max_length=64, stride=32)
    val_dataset = ContinuousTextDataset(text_data[int(0.85 * len(text_data)):], tokenizer, max_length=64, stride=32)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    epochs = 3
    grad_accum_steps = 4
    total_steps = epochs * math.ceil(len(train_loader) / grad_accum_steps)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=5e-6)
    
    pretrainer = ContinuousPretrainer(model, train_loader, val_loader, optimizer, scheduler, device, tokenizer, grad_accum_steps=grad_accum_steps, save_dir=os.path.join(base_dir, 'checkpoints'))
    
    pretrainer.train(num_epochs=epochs, eval_freq=10)