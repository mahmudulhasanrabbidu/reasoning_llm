import os
import sys
import json
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from llm.pretrained_weight_loader import PretrainedQweenModel
from tokenizer.qween_3_tokenizer import Qwen3Tokenizer
from llm.finetune.dataset import Qwen3FineTuningDataset

class SFTTrainer:
    """
    Industry-standard Supervised Fine-Tuning (SFT) Trainer.
    Supports Mixed Precision (AMP), Gradient Accumulation, Gradient Clipping, and LR Scheduling.
    """
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        self.epochs = config.get("epochs", 3)
        self.grad_accum_steps = config.get("grad_accum_steps", 4)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.save_dir = config.get("save_dir", "checkpoints")
        
        self.scaler = torch.amp.GradScaler(enabled=(self.device.type == "cuda"))
        
        self.best_val_loss = float("inf")
        os.makedirs(self.save_dir, exist_ok=True)

    def _calculate_loss(self, logits, labels):
        """Applies causal shift and cross-entropy with the -100 ignore index."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1),
            ignore_index=-100 
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]")
        
        self.optimizer.zero_grad(set_to_none=True)
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=(self.device.type == "cuda")):
                logits = self.model(input_ids)
                loss = self._calculate_loss(logits, labels)
                
                loss = loss / self.grad_accum_steps
                
            self.scaler.scale(loss).backward()
            
            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                
            total_loss += (loss.item() * self.grad_accum_steps)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({"loss": f"{loss.item() * self.grad_accum_steps:.4f}", "lr": f"{current_lr:.2e}"})
            
        return total_loss / len(self.train_loader)

    @torch.inference_mode()
    def evaluate(self, epoch):
        self.model.eval()
        total_val_loss = 0.0
        
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.epochs} [Val]")
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=(self.device.type == "cuda")):
                logits = self.model(input_ids)
                loss = self._calculate_loss(logits, labels)
                
            total_val_loss += loss.item()
            progress_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})
            
        avg_val_loss = total_val_loss / len(self.val_loader)
        return avg_val_loss

    def train(self):
        print(f"\nStarting Industry-Level SFT on {self.device.type.upper()}...")
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate(epoch)
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"   - Train Loss: {train_loss:.4f}")
            print(f"   - Val Loss:   {val_loss:.4f}")
            
            # Save the best model checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_path = os.path.join(self.save_dir, "qwen3_best_lora_adapters.pt")
                
                lora_weights = {k: v for k, v in self.model.state_dict().items() if "lora_" in k}
                torch.save(lora_weights, save_path)
                
                print(f" New best LoRA adapters saved to {save_path}!")


if __name__ == "__main__":
    config = {
        "batch_size": 2,
        "grad_accum_steps": 8,
        "learning_rate": 2e-5,
        "epochs": 3,
        "max_seq_len": 512,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0,
        "save_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints'))
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading Tokenizer and Pretrained Base Model...")
    tokenizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'tokenizer.json'))
    tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
    
    model_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.json'))
    model_weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model.safetensors'))
    
    model, _ = PretrainedQweenModel.from_pretrained(config_path=model_config_path, weight_path=model_weights_path)

    from llm.finetune.lora import inject_lora, freeze_base_model, print_trainable_parameters
    
    model = inject_lora(model, rank=8, alpha=16) 
    
    freeze_base_model(model)
    
    model.to(device)
    
    print_trainable_parameters(model)

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'alpaca_training_data.json'))
    
    with open(data_path, "r", encoding="utf-8") as f:
        training_data = json.load(f)
        
    full_dataset = Qwen3FineTuningDataset(training_data, tokenizer, max_seq_len=config["max_seq_len"])
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config["learning_rate"], weight_decay=0.01)
    
    steps_per_epoch = math.ceil(len(train_loader) / config["grad_accum_steps"])
    total_training_steps = steps_per_epoch * config["epochs"]
    
    warmup_steps = max(1, int(total_training_steps * config["warmup_ratio"]))
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(total_training_steps - warmup_steps))
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    
    trainer = SFTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    )
    
    trainer.train()