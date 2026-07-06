import os
import sys
import glob
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm

# adjust paths to look two folders up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from llm.base_model import Qween3Model, Qwen3Config
from tokenizer.qween3_tokenizer import Qwen3Tokenizer
from llm.pretrain.massive_pretrainer_dataset import StreamingPretrainDataset

class MassivePretrainer:
    def __init__(self, model, dataloader, optimizer, scheduler, device, config):
        self.device = device
        self.model = model.to(device)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        self.grad_accum_steps = config.get("grad_accum_steps", 8)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.save_dir = config.get("save_dir", "pretrained_checkpoint")
        self.log_freq = config.get("log_freq", 100) # FIXED key
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    def _get_last_checkpoint_path(self):
        checkpnt_paths = glob.glob(os.path.join(self.save_dir, "qween3_step_*.pt")) # list of strings: ['dir/qwen3_step_1000.pt', 'dir/qwen3_step_2000.pt']

        if not checkpnt_paths:
            return None  
        checkpnt_paths.sort(key=lambda x: int(x.split("_step_")[1].split(".pt")[0])) # ['dir/qwen3', '1000.pt'] --> 1000.pt --> ['1000', ''] --> '1000' --> 1000
        return checkpnt_paths[-1]

    def _auto_resume(self):
        last_checkpnt_path = self._get_last_checkpoint_path()

        if last_checkpnt_path:
            check_pnt = torch.load(last_checkpnt_path, map_location=self.device) # dict

            self.model.load_state_dict(check_pnt["model_state_dict"])
            self.optimizer.load_state_dict(check_pnt["optimizer_state_dict"])
            self.scheduler.load_state_dict(check_pnt["scheduler_state_dict"])
            self.scaler.load_state_dict(check_pnt["scaler_state_dict"])
            global_step = check_pnt["global_step"]

            return global_step
        return 0 # if no checkpoint exists, start at step 0

    def _save_checkpoint(self, global_step):
        checkpnt_path = os.path.join(self.save_dir, f"qween3_step_{global_step}.pt")
        checkpnt_dict = {
            "global_step": global_step, 
            "model_state_dict": self.model.state_dict(), 
            "optimizer_state_dict": self.optimizer.state_dict(), 
            "scaler_state_dict": self.scaler.state_dict(), 
            "scheduler_state_dict": self.scheduler.state_dict()
        }

        torch.save(checkpnt_dict, checkpnt_path)

    def train(self, total_training_steps, auto_resume=True):
        global_step = 0
        step_loss = 0.0

        if auto_resume:
            global_step = self._auto_resume()

        self.model.train()
        pbar = tqdm(total=total_training_steps, initial=global_step, desc="pretraining")
        data_itr = iter(self.dataloader)

        while global_step < total_training_steps:
            try:
                inputs, targets = next(data_itr)
            except StopIteration:
                # iterators do not automatically loop back to the beginning when they reach the end.
                # they run out of data, they intentionally crash the program by throwing a StopIteration error.
                data_itr = iter(self.dataloader)
                inputs, targets = next(data_itr)

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # forward Pass (AMP)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=(self.device.type=="cuda")):
                logits = self.model(inputs)
                loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten()) # logits: (Batch * Sequence, Vocab Size), targets: (Batch * Sequence)
                loss = loss / self.grad_accum_steps

            # backward Pass
            self.scaler.scale(loss).backward() # It scale, sums and stores the Gradients until reach to effective batch.

            step_loss += loss.item() * self.grad_accum_steps # this line sums and stores the actual loss for each micro-batch (For Humans). optimizers does not need this

            if (pbar.n + 1) % self.grad_accum_steps == 0:
                # optimizer Step
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                # set_to_none=True --> completely deletes the gradient tensors from the GPU and frees up that VRAM for the next forward pass
                # rather than just filling them with 0.0s, saving precious VRAM.
                self.scheduler.step()

                global_step += 1
                
                pbar.update(1)
                current_lr = self.optimizer.param_groups[0]["lr"]
                # Pass as dictionary
                pbar.set_postfix({"loss": f"{step_loss:.4f}", "lr": f"{current_lr:.2e}"})
                step_loss = 0.0

                # save completed checkpoint
                if global_step % self.log_freq == 0:
                    self._save_checkpoint(global_step)

        print("Pretraining completed!")
        # for the final output, we only need the model weights for inference
        final_model_weight_path = os.path.join(self.save_dir, "qween3_step_final.pt")
        torch.save(self.model.state_dict(), final_model_weight_path)


if __name__ == "__main__":
    config = {
        "batch_size": 2,          
        "grad_accum_steps": 16,   
        "learning_rate": 3e-4,    
        "max_seq_len": 512,       
        "total_steps": 10000,     
        "warmup_ratio": 0.05,     
        "max_grad_norm": 1.0,
        "log_freq": 1000,         
        "save_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints', 'pretrain'))
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Initializing Qwen3 Architecture...")
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.json'))
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
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    
    model = Qween3Model(custom_config)
    model.gradient_checkpointing = True

    if hasattr(model, "emb"): 
        model.emb.weight.requires_grad_(True)
    
    print("Connecting to Wikipedia Stream...")
    tokenizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'tokenizer.json'))
    tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)
    
    dataset = StreamingPretrainDataset(dataset_path="wikipedia", dataset_name="20220301.en", tokenizer=tokenizer, max_length=config["max_seq_len"])
    
    dataloader = DataLoader(dataset, batch_size=config["batch_size"])
    
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.1)
    
    warmup_steps = int(config["total_steps"] * config["warmup_ratio"])
    decay_steps = config["total_steps"] - warmup_steps
    
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=1e-5)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    
    trainer = MassivePretrainer(model, dataloader, optimizer, scheduler, device, config)
    
    # auto-resume is enabled by default!
    trainer.train(total_training_steps=config["total_steps"], auto_resume=True)

