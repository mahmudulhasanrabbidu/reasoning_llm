import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        
        self.linear = linear_layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
            
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout)
        
        factory_kwargs = {
            "device": linear_layer.weight.device,
            "dtype": linear_layer.weight.dtype
        }
        
        self.lora_A = nn.Parameter(torch.zeros((rank, self.in_features), **factory_kwargs))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, rank), **factory_kwargs))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        base_out = self.linear(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        lora_out = lora_out * self.scaling
        
        return base_out + lora_out


def inject_lora(model, rank=8, alpha=16, dropout=0.05, target_modules=["w_q", "w_k", "w_v", "out_proj", "fc1", "fc2", "fc3"]):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name in target_modules:
            setattr(model, name, LoRALinear(module, rank, alpha, dropout))
        else:
            inject_lora(module, rank, alpha, dropout, target_modules)
    return model


def freeze_base_model(model):
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


def print_trainable_parameters(model):
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