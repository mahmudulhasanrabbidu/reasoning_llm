import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    A wrapper that replaces a standard nn.Linear layer.
    It freezes the original weights and injects two tiny trainable matrices (A and B).
    """
    def __init__(self, linear_layer, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        
        # 1. Store and freeze the original base weights
        self.linear = linear_layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
            
        # 2. Setup LoRA Hyperparameters
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout)
        
        # 3. Inherit the exact device and dtype from the base layer (CRITICAL FIX)
        factory_kwargs = {
            "device": linear_layer.weight.device,
            "dtype": linear_layer.weight.dtype
        }
        
        # 4. Create the tiny trainable bypass matrices on the correct device
        self.lora_A = nn.Parameter(torch.zeros((rank, self.in_features), **factory_kwargs))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, rank), **factory_kwargs))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # A is initialized with Kaiming uniform (standard for LoRA stability)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B is initialized to zero so the initial LoRA modification is exactly 0
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        # Base path (Frozen)
        base_out = self.linear(x)
        
        # Bypass path (Trainable) - Dropout correctly applied to the LoRA branch input
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        lora_out = lora_out * self.scaling
        
        # The outputs are summed together
        return base_out + lora_out


def inject_lora(model, rank=8, alpha=16, dropout=0.05, target_modules=["w_q", "w_k", "w_v", "out_proj", "fc1", "fc2", "fc3"]):
    """
    Recursively crawls the custom architecture and wraps targeted linear layers with LoRA.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name in target_modules:
            setattr(model, name, LoRALinear(module, rank, alpha, dropout))
        else:
            inject_lora(module, rank, alpha, dropout, target_modules)
    return model


def freeze_base_model(model):
    """
    Ensures all parameters are frozen EXCEPT the injected LoRA matrices.
    """
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


def print_trainable_parameters(model):
    """
    Prints a summary of how many parameters are actually being trained.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    reduction_pct = 100 - (100 * trainable_params / all_param)
    print(f"\n--- LoRA Parameter Summary ---")
    print(f"Total Parameters:     {all_param:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Param Reduction:      {reduction_pct:.2f}% (VRAM savings via bypassed optimizer states)\n")