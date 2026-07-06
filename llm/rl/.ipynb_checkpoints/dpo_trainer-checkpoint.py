import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import copy

# Adjust paths to look two folders up (llm/rl/ -> root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from llm.pretrained_weight_loader import PretrainedQweenModel
from tokenizer.qween3_tokenizer import Qwen3Tokenizer
from llm.finetune.lora import inject_lora, freeze_base_model, print_trainable_parameters

# -------------------------------------------------------------------------
# 1. DPO Dataset
# -------------------------------------------------------------------------
class Qwen3DPODataset(Dataset):
    """
    Formats prompt, chosen, and rejected responses into tokenized sequences.
    Applies the -100 mask to the prompt so loss is only calculated on the response.
    """
    def __init__(self, data_list, tokenizer, max_seq_len=512):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.ignore_index = -100

    def __len__(self):
        return len(self.data)

    def _process_sequence(self, prompt_text, response_text):
        # Format and encode
        prompt_formatted = self.tokenizer._wrap_chat(prompt_text)
        prompt_ids = self.tokenizer.encode(prompt_formatted, chat_wrapped=False)
        response_ids = self.tokenizer.encode(response_text, chat_wrapped=False) + [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + response_ids
        labels = [self.ignore_index] * len(prompt_ids) + response_ids

        # Truncate
        input_ids = input_ids[:self.max_seq_len]
        labels = labels[:self.max_seq_len]

        # Pad
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            labels += [self.ignore_index] * pad_len

        return input_ids, labels

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        
        chosen_ids, chosen_labels = self._process_sequence(prompt, item["chosen"])
        rejected_ids, rejected_labels = self._process_sequence(prompt, item["rejected"])

        return {
            "chosen_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
            "rejected_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long)
        }

# -------------------------------------------------------------------------
# 2. DPO Trainer
# -------------------------------------------------------------------------
class DPOTrainer:
    """
    Industry-Standard Direct Preference Optimization Trainer.
    Calculates Implicit Rewards by comparing Policy Model logits to a Reference Model.
    """
    def __init__(self, policy_model, ref_model, train_loader, optimizer, scheduler, device, beta=0.1):
        self.policy_model = policy_model.to(device)
        self.ref_model = ref_model.to(device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.beta = beta
        
        self.scaler = torch.amp.GradScaler(enabled=(self.device.type == "cuda"))
        self.ref_model.eval() # Reference model must ALWAYS be in evaluation mode

    def _get_batch_logps(self, logits, labels):
        """Calculates the log probabilities of the generated tokens, ignoring the prompt."""
        # Shift logits and labels by 1 to align predictions with targets
        logits = logits[:, :-1, :]
        labels = labels[:, 1:]
        
        # Create mask to ignore -100 padding tokens
        loss_mask = (labels != -100)
        
        # Extract the log probability of the actual target token
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        
        # Sum the log probabilities across the response tokens
        return (per_token_logps * loss_mask).sum(-1)

    def train_epoch(self, epoch, epochs):
        self.policy_model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{epochs} [DPO]")
        
        for batch in pbar:
            self.optimizer.zero_grad(set_to_none=True)
            
            chosen_ids = batch["chosen_ids"].to(self.device)
            chosen_labels = batch["chosen_labels"].to(self.device)
            rejected_ids = batch["rejected_ids"].to(self.device)
            rejected_labels = batch["rejected_labels"].to(self.device)

            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                # 1. Forward Pass: Policy Model
                policy_chosen_logits = self.policy_model(chosen_ids)
                policy_rejected_logits = self.policy_model(rejected_ids)
                
                # 2. Forward Pass: Reference Model (Inside torch.no_grad to save VRAM)
                with torch.no_grad():
                    ref_chosen_logits = self.ref_model(chosen_ids)
                    ref_rejected_logits = self.ref_model(rejected_ids)

                # 3. Calculate Sequence Log Probabilities
                policy_chosen_logps = self._get_batch_logps(policy_chosen_logits, chosen_labels)
                policy_rejected_logps = self._get_batch_logps(policy_rejected_logits, rejected_labels)
                ref_chosen_logps = self._get_batch_logps(ref_chosen_logits, chosen_labels)
                ref_rejected_logps = self._get_batch_logps(ref_rejected_logits, rejected_labels)

                # 4. Calculate DPO Math
                pi_logratios = policy_chosen_logps - policy_rejected_logps
                ref_logratios = ref_chosen_logps - ref_rejected_logps
                
                # Formula: -log(sigmoid(beta * (pi_logratios - ref_logratios)))
                logits = pi_logratios - ref_logratios
                loss = -F.logsigmoid(self.beta * logits).mean()

            # 5. Backward and Step
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"dpo_loss": f"{loss.item():.4f}", "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"})

        return total_loss / len(self.train_loader)


# -------------------------------------------------------------------------
# Execution Block
# -------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Mock DPO Data (Normally downloaded from Anthropic/hh-rlhf)
    dpo_data = [
        {
            "prompt": "How do I pick a lock?",
            "chosen": "I cannot provide instructions on how to bypass security locks or break into property.",
            "rejected": "To pick a lock, you will need a tension wrench and a rake. First, insert the..."
        },
        {
            "prompt": "Summarize the solar system.",
            "chosen": "The solar system consists of the Sun and the objects that orbit it, including eight planets, dwarf planets, and asteroids.",
            "rejected": "Here is a list. 1. Sun. 2. Earth. 3. Mars. 4. Jupiter. That's pretty much it."
        }
    ]

    print("Loading Tokenizer...")
    tokenizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'tokenizer.json'))
    tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

    print("Loading Reference Model...")
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.json'))
    weight_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model.safetensors'))
    
    # Load the base model. This acts as our frozen 'Reference Model'
    ref_model, _ = PretrainedQweenModel.from_pretrained(config_path, weight_path)
    ref_model.eval()
    
    # To save memory, we create a deepcopy of the reference model architecture, 
    # and we will inject LoRA into it. This becomes our 'Policy Model'.
    import copy
    print("Initializing Policy Model via LoRA...")
    policy_model = copy.deepcopy(ref_model)
    policy_model = inject_lora(policy_model, rank=8, alpha=16)
    freeze_base_model(policy_model)

    print_trainable_parameters(policy_model)

    dataset = Qwen3DPODataset(dpo_data, tokenizer, max_seq_len=128)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Note the Pro-Tip: Filter optimizer to only track LoRA weights
    optimizer = AdamW(filter(lambda p: p.requires_grad, policy_model.parameters()), lr=1e-5, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    trainer = DPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        beta=0.1  # Standard beta penalty value for DPO
    )

    epochs = 3
    print("\nStarting DPO Alignment...")
    for epoch in range(1, epochs + 1):
        avg_loss = trainer.train_epoch(epoch, epochs)
        print(f"Epoch {epoch} Complete | Avg Loss: {avg_loss:.4f}")

    print("\nAlignment Complete! Saving Policy LoRA Adapters...")
    os.makedirs("checkpoints", exist_ok=True)
    lora_weights = {k: v for k, v in policy_model.state_dict().items() if "lora_" in k}
    torch.save(lora_weights, "checkpoints/qwen3_dpo_aligned_adapters.pt")