import copy
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import requests
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

@dataclass
class Config:
    rollout_pool_size: int = 8192
    group_size: int = 16  
    minibatches: int = 16 
    ref_update_freq: int = 400
    
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.9
    
    lr: float = 1e-5
    kl_beta: float = 0.04
    epsilon: float = 0.2
    
    use_amp: bool = True 
    seed: int = 42
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: Path = Path("./checkpoints")
    log_dir: Path = Path("./logs")

    def __post_init__(self):
        assert self.rollout_pool_size % self.group_size == 0, "Pool size must be divisible by group_size."
        assert self.rollout_pool_size % self.minibatches == 0, "Pool size must be divisible by minibatches."
        assert self.group_size > 1, "group_size must be > 1 to compute advantage variance."
        assert self.minibatch_size > 0, "minibatch_size must be > 0."

    @property
    def num_distinct_prompts(self) -> int:
        return self.rollout_pool_size // self.group_size
        
    @property
    def minibatch_size(self) -> int:
        return self.rollout_pool_size // self.minibatches


class MathDataset:
    def __init__(self, local_path: str = "math_train.json"):
        self.local_path = Path(local_path)
        self.data = self._load_data()

    def _load_data(self) -> List[Dict]:
        if self.local_path.exists():
            with self.local_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        
        url = "https://raw.githubusercontent.com/rasbt/math_full_minus_math500/refs/heads/main/math_full_minus_math500.json"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        with self.local_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return data

    def sample_prompts(self, batch_size: int) -> List[Dict]:
        indices = torch.randint(0, len(self.data), (batch_size,)).tolist()
        return [self.data[i] for i in indices]


class BatchedRolloutGenerator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate_groups(self, model: nn.Module, prompts: List[Dict], cfg: Config) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        grouped_prompts = [p["problem"] for p in prompts for _ in range(cfg.group_size)]
        
        original_padding = self.tokenizer.padding_side
        try:
            self.tokenizer.padding_side = "left"
            inputs = self.tokenizer(grouped_prompts, return_tensors="pt", padding=True).to(cfg.device)
        finally:
            self.tokenizer.padding_side = original_padding
            
        input_ids = inputs.input_ids
        
        gen_start_idx = input_ids.shape[1]

        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        gen_only_ids = generated_ids[:, gen_start_idx:]
        response_texts = self.tokenizer.batch_decode(gen_only_ids, skip_special_tokens=True)
        
        prompt_lens = torch.full((len(prompts) * cfg.group_size,), gen_start_idx, dtype=torch.long)
        
        return generated_ids.cpu(), prompt_lens.cpu(), response_texts


class MathVerifier:
    def compute_rewards(self, texts: List[str], expected_answers: List[str]) -> torch.Tensor:
        rewards = []
        for text, answer in zip(texts, expected_answers):
            match = re.search(r"\\boxed\{([^}]*)\}", text)
            if match:
                extracted = match.group(1).strip()
                is_correct = (extracted == answer.strip())
                rewards.append(1.0 if is_correct else 0.0)
            else:
                rewards.append(0.0)
        return torch.tensor(rewards, dtype=torch.float32)


class GRPOLoss:
    
    @staticmethod
    def _sequence_logprob(logits: torch.Tensor, input_ids: torch.Tensor, prompt_lens: torch.Tensor) -> torch.Tensor:
        logprobs_all = torch.log_softmax(logits[:, :-1, :], dim=-1)
        target_ids = input_ids[:, 1:].unsqueeze(-1)
        token_logprobs = logprobs_all.gather(2, target_ids).squeeze(-1)
        
        seq_len = token_logprobs.shape[1]
        device = logits.device
        
        mask = torch.arange(seq_len, device=device).unsqueeze(0) >= (prompt_lens - 1).unsqueeze(1)
        
        return (token_logprobs * mask).sum(dim=-1)

    @classmethod
    def compute(cls, new_logits: torch.Tensor, old_logits: torch.Tensor, input_ids: torch.Tensor, advantages: torch.Tensor, prompt_lens: torch.Tensor, cfg: Config) -> Tuple[torch.Tensor, float, float]:
        
        new_logprobs = cls._sequence_logprob(new_logits, input_ids, prompt_lens)
        old_logprobs = cls._sequence_logprob(old_logits, input_ids, prompt_lens)

        ratio = torch.exp(new_logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - cfg.epsilon, 1.0 + cfg.epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        log_ratio = old_logprobs - new_logprobs
        kl_div = torch.exp(log_ratio) - log_ratio - 1.0
        kl_loss = kl_div.mean()

        loss = policy_loss + (cfg.kl_beta * kl_loss)
        return loss, policy_loss.item(), kl_loss.item()


class CSVLogger:
    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = log_dir / f"grpo_metrics_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        self.csv_path.write_text("step,mb_idx,loss,policy_loss,kl_loss\n", encoding="utf-8")
        self.buffer = []
        
    def log(self, step: int, mb_idx: int, loss: float, p_loss: float, kl_loss: float):
        self.buffer.append(f"{step},{mb_idx},{loss:.6f},{p_loss:.6f},{kl_loss:.6f}\n")
        
    def flush(self):
        if self.buffer:
            with self.csv_path.open("a", encoding="utf-8") as f:
                f.writelines(self.buffer)
            self.buffer.clear()

class CheckpointManager:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, scaler, step: int):
        ckpt_path = self.checkpoint_dir / f"r1-grpo-step{step:05d}.pth"
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler else None
        }, ckpt_path)
        print(f"[*] Checkpoint saved: {ckpt_path}")


class RLTrainer:
    def __init__(self, model: nn.Module, tokenizer, dataset: MathDataset, cfg: Config):
        torch.manual_seed(cfg.seed)
        
        self.current_model = model.to(cfg.device)
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.cfg = cfg
        
        self.generator = BatchedRolloutGenerator(tokenizer)
        self.verifier = MathVerifier()
        
        self.csv_logger = CSVLogger(cfg.log_dir)
        self.checkpoint_manager = CheckpointManager(cfg.checkpoint_dir)
        
        self.optimizer = torch.optim.AdamW(self.current_model.parameters(), lr=cfg.lr)
        
        self.device_type = 'cuda' if 'cuda' in cfg.device else 'cpu'
        self.use_amp = cfg.use_amp and self.device_type == 'cuda'
        self.scaler = torch.amp.GradScaler(self.device_type, enabled=self.use_amp)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        
        self.ref_model = copy.deepcopy(self.current_model).to(cfg.device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def _compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        advantages = torch.zeros_like(rewards)
        for i in range(0, self.cfg.rollout_pool_size, self.cfg.group_size):
            group_rewards = rewards[i : i + self.cfg.group_size]
            std = group_rewards.std()
            std = std if std > 1e-6 else 1.0 
            advantages[i : i + self.cfg.group_size] = (group_rewards - group_rewards.mean()) / std
        return advantages

    def train(self, total_steps: int):
        for step in range(total_steps):
            print(f"\n========== Step {step + 1}/{total_steps} ==========")
            
            prompts = self.dataset.sample_prompts(self.cfg.num_distinct_prompts)
            expected_answers = [p["answer"] for p in prompts for _ in range(self.cfg.group_size)]
            
            print(f"[*] Generating {self.cfg.rollout_pool_size} rollouts...")
            responses_ids, prompt_lens, response_texts = self.generator.generate_groups(
                self.ref_model, prompts, self.cfg
            )
            
            rewards = self.verifier.compute_rewards(response_texts, expected_answers)
            advantages = self._compute_group_advantages(rewards)
            
            dataset = TensorDataset(responses_ids, prompt_lens, advantages)
            dataloader = DataLoader(dataset, batch_size=self.cfg.minibatch_size, shuffle=True)
            
            self.current_model.train()
            
            for mb_idx, (mb_ids, mb_lens, mb_advs) in enumerate(dataloader):
                mb_ids = mb_ids.to(self.cfg.device)
                mb_lens = mb_lens.to(self.cfg.device)
                mb_advs = mb_advs.to(self.cfg.device)
                mb_attention_mask = (mb_ids != self.tokenizer.pad_token_id).long()
                
                with torch.amp.autocast(self.device_type, enabled=self.use_amp):
                    new_logits = self.current_model(mb_ids, attention_mask=mb_attention_mask).logits
                    with torch.no_grad():
                        old_logits = self.ref_model(mb_ids, attention_mask=mb_attention_mask).logits
                        
                    loss, p_loss, kl_loss = GRPOLoss.compute(
                        new_logits, old_logits, mb_ids, mb_advs, mb_lens, self.cfg
                    )
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.current_model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                self.csv_logger.log(step, mb_idx, loss.item(), p_loss, kl_loss)
                print(f"  [Minibatch {mb_idx + 1}/{self.cfg.minibatches}] Loss: {loss.item():.4f}")

            self.scheduler.step()
            self.csv_logger.flush()

            if (step + 1) % self.cfg.ref_update_freq == 0:
                print(f"[*] Syncing Reference Model Weights...")
                self.ref_model.load_state_dict(self.current_model.state_dict())
                self.checkpoint_manager.save(self.current_model, self.optimizer, self.scheduler, self.scaler, step + 1)






from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    print("Initializing DeepSeek-R1 GRPO Training Pipeline...")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"[*] Loading Tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[*] Loading Model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    config = Config(
        rollout_pool_size=32,
        group_size=4,
        minibatches=4,
        ref_update_freq=50,
        max_new_tokens=128,
        temperature=0.8,
        lr=1e-5,
        use_amp=True,
        checkpoint_dir=Path("./checkpoints_test"),
        log_dir=Path("./logs_test")
    )

    print("[*] Preparing Math Dataset...")
    dataset = MathDataset(local_path="math_train_local.json")

    print("[*] Initializing RLTrainer...")
    trainer = RLTrainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        cfg=config
    )

    total_training_steps = 1000
    try:
        print("\n=======================================================")
        print(f"🚀 Starting GRPO Training for {total_training_steps} steps")
        print("=======================================================")
        trainer.train(total_steps=total_training_steps)
        
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user (KeyboardInterrupt).")
        print("[*] Saving emergency checkpoint...")
        emergency_step = len(trainer.csv_logger.buffer) + 1 
        trainer.checkpoint_manager.save(
            trainer.current_model, 
            trainer.optimizer, 
            trainer.scheduler, 
            trainer.scaler, 
            emergency_step
        )
        print("[*] Emergency checkpoint saved safely. Exiting.")