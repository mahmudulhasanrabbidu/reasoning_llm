import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.utils.checkpoint as checkpoint



@dataclass
class Qwen3Config:
    vocab_size: int = 151936
    emb_dim: int = 1024
    hidden_dim: int = 3072
    num_trf_blocks: int = 28
    num_q_head: int = 16
    num_kv_head: int = 8
    head_dim: int = 128
    qk_norm: bool = False
    dtype: torch.dtype = torch.bfloat16
    rms_norm_eps: float = 1e-6          
    rope_theta: float = 1000000         
    max_position_embeddings: int = 40960


    

class KVCache:
    def __init__(self, n_layers):
        self.cache = [None] * n_layers

    def update(self, idx, value):
        self.cache[idx] = value

    def get(self, idx):
        return self.cache[idx]

    def get_all(self):
        return self.cache

    def reset(self):
        for i in range(len(self.cache)):
            self.cache[i] = None

class RMSNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-5, bias=False, qwen3_compatible=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim)) if bias else None

    def forward(self, x):
        # x: (B, S, E)
        input_dtype = x.dtype
        
        if self.qwen3_compatible:
            x = x.to(torch.float32)

        ms = (x**2).mean(dim=-1, keepdim=True)       # (B, S, 1)
        reciprocal_rms = torch.rsqrt(ms + self.eps)  # (B, S, 1)
        rms_norm = reciprocal_rms * x                # (B, S, E)
        x = rms_norm * self.scale                    # (B, S, E)

        if self.bias is not None:
            x = x + self.bias                        # (B, S, E)

        return x.to(dtype=input_dtype)

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.emb_dim, cfg.hidden_dim, dtype=cfg.dtype, bias=False) # W: (H, E) --> x * W.T + b
        self.fc2 = nn.Linear(cfg.emb_dim, cfg.hidden_dim, dtype=cfg.dtype, bias=False) # W: (H, E) --> x * W.T + b
        self.fc3 = nn.Linear(cfg.hidden_dim, cfg.emb_dim, dtype=cfg.dtype, bias=False) # W: (E, H) --> x * W.T + b

    def forward(self, x):
        # x: (B, S, E)
        x1 = self.fc1(x) # (B, S, H)
        x2 = self.fc2(x) # (B, S, H)
        x = nn.functional.silu(x1) * x2 # (B, S, H)

        return self.fc3(x) # (B, S, E)
    

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048, theta_base=10000, dtype=torch.float32):
        super().__init__()
        
        assert head_dim % 2 == 0, "head_dim must be divisible by 2"

        # 1. Compute the parameters once during initialization
        w_i = 1 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype) / head_dim))
        p = torch.arange(max_seq_len, dtype=dtype)
        angle = p[:, None] * w_i[None, :]
        angle = torch.cat([angle, angle], dim=1)

        cos = torch.cos(angle)
        sin = torch.sin(angle)

        # GPU when the model moves to the GPU, but DO NOT update them with gradients."
        # persistent=False means they won't be saved to the model's weight file.
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, x, offset=0):
        # x: (batch_size, num_heads, seq_len, head_dim)
        batch, num_head, seq_len, head_dim = x.shape
        
        # ensure sequence isn't longer than what we precomputed
        assert offset + seq_len <= self.cos_cached.shape[0], "Sequence length exceeds precomputed max_seq_len"

        # 3. Apply the rotation
        x1 = x[..., :head_dim//2]
        x2 = x[..., head_dim//2:]
        rotated = torch.cat([-x2, x1], dim=-1)

        # slice the precomputed tables dynamically based on sequence length
        cos = self.cos_cached[offset:offset+seq_len, :]
        sin = self.sin_cached[offset:offset+seq_len, :]

        x_rotated = x * cos + rotated * sin

        return x_rotated.to(x.dtype)
    
class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_q_head = cfg.num_q_head
        self.num_kv_head = cfg.num_kv_head
        self.head_dim = cfg.head_dim
        self.group_size = cfg.num_q_head // cfg.num_kv_head
        self.d_out = self.head_dim * cfg.num_q_head

        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        else:
            self.q_norm = self.k_norm = None

        rope_theta = cfg.rope_theta
        max_seq_len = cfg.max_position_embeddings
        self.apply_rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len, theta_base=rope_theta)

        # bias=False to exactly match Hugging Face Qwen parameters
        self.w_q = nn.Linear(cfg.emb_dim, self.head_dim * cfg.num_q_head, bias=False, dtype=cfg.dtype) # w: (head_dim * num_q_head, d_in) 
        self.w_k = nn.Linear(cfg.emb_dim, self.head_dim * cfg.num_kv_head, bias=False, dtype=cfg.dtype) # w: (head_dim * num_kv_head, d_in) 
        self.w_v = nn.Linear(cfg.emb_dim, self.head_dim * cfg.num_kv_head, bias=False, dtype=cfg.dtype) # w: (head_dim * num_kv_head, d_in) 
        self.out_proj = nn.Linear(self.d_out, cfg.emb_dim, bias=False, dtype=cfg.dtype) # w: (d_in, d_out)

    def forward(self, x, mask, start=0, cache=None):
        # x: (B, S, d_in)
        batch, seq_len, d_in = x.shape
        
        q = self.w_q(x) # (B, S, head_dim * num_q_head)
        k = self.w_k(x) # (B, S, head_dim * num_kv_head)
        v = self.w_v(x) # (B, S, head_dim * num_kv_head)

        # split into head
        q = q.view(batch, seq_len, self.num_q_head, self.head_dim).transpose(1, 2) # (B, S, head_dim * num_q_head) -> (B, S, num_q_head, head_dim) -> (B, num_q_head, S, head_dim)
        k = k.view(batch, seq_len, self.num_kv_head, self.head_dim).transpose(1, 2) # (B, S, head_dim * num_kv_head) -> (B, S, num_kv_head, head_dim) -> (B, num_kv_head, S, head_dim)
        v = v.view(batch, seq_len, self.num_kv_head, self.head_dim).transpose(1, 2) # (B, S, head_dim * num_kv_head) -> (B, S, num_kv_head, head_dim) -> (B, num_kv_head, S, head_dim)

        if self.q_norm:
            q = self.q_norm(q) # (B, num_q_head, S, head_dim)
        if self.k_norm:
            k = self.k_norm(k) # (B, num_kv_head, S, head_dim)

        q = self.apply_rope(q, offset=start) # (B, num_q_head, S, head_dim)
        k = self.apply_rope(k, offset=start) # (B, num_kv_head, S, head_dim)

        if cache is not None:
            k_prev, v_prev = cache
            k = torch.cat([k_prev, k], dim = 2) # (B, num_kv_head, S', head_dim)
            v = torch.cat([v_prev, v], dim = 2) # (B, num_kv_head, S', head_dim)
            
        next_cache = (k, v)

        k = k.repeat_interleave(self.group_size, dim=1) # (B, num_q_head, S', head_dim)
        v = v.repeat_interleave(self.group_size, dim=1) # (B, num_q_head, S', head_dim)

        attention_score = q @ k.transpose(2, 3) # (B, num_q_head, S, head_dim) @ (B, num_q_head, head_dim, s') -> (B, num_q_head, S, S')
        attention_score = attention_score.masked_fill(mask, -torch.inf) # (B, num_q_head, S, S')
        attention_weight = torch.softmax(attention_score / self.head_dim**0.5, dim=-1) # (B, num_q_head, S, S')

        context = attention_weight @ v # (B, num_q_head, S, S') @ (B, num_q_head, S', head_dim) -> (B, num_q_head, S, head_dim)
        context = context.transpose(1, 2) # (B, S, num_q_head, head_dim)
        context = context.reshape(batch, seq_len, self.num_q_head * self.head_dim) # (B, S, d_out) -- d_out = num_q_head * head_dim

        return self.out_proj(context), next_cache # (B, S, d_in)
        
        

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = GroupedQueryAttention(cfg) # (B, S, emb_dim) -> (B, S, emb_dim)
        self.rms_norm1 = RMSNorm(cfg.emb_dim, eps=cfg.rms_norm_eps) # (B, S, emb_dim) -> (B, S, emb_dim)
        self.rms_norm2 = RMSNorm(cfg.emb_dim, eps=cfg.rms_norm_eps) # (B, S, emb_dim) -> (B, S, emb_dim)
        self.ffd = FeedForward(cfg) # (B, S, emb_dim) -> (B, S, emb_dim)

    def forward(self, x, mask, offset=0, cache=None):
        # x: (B, S, emb_dim)
        # mask: (1, 1, S, S') --> S': total seq len
        shortcut = x # (B, S, emb_dim)
        x = self.rms_norm1(x) # (B, S, emb_dim)
        x, next_cache = self.attn(x, mask, offset, cache) # (B, S, emb_dim)
        x = x + shortcut # (B, S, emb_dim)
        shortcut = x # (B, S, emb_dim)
        x = self.rms_norm2(x) # (B, S, emb_dim)
        x = self.ffd(x) # (B, S, emb_dim)
        x = x + shortcut # (B, S, emb_dim)

        return x, next_cache
        

class Qween3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.gradient_checkpointing = False
        self.current_pos = 0
        
        self.emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim, dtype=cfg.dtype) # w: (vocab_size, emb_dim)
        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_trf_blocks)])
        self.final_norm = RMSNorm(cfg.emb_dim, eps=cfg.rms_norm_eps)
        self.out = nn.Linear(cfg.emb_dim, cfg.vocab_size, dtype=cfg.dtype, bias=False) # w: (vocab_size, emb_dim)

    def forward(self, tkn_ids, cache=None):
        # tnk_ids: (B, seq_len)
        # mask: (1, 1, S, S')
        # cache: (B, num_kv_head, S', head_dim)
        x = self.emb(tkn_ids) # (B, S, emb_dim)

        num_tokens = x.shape[1]
        if cache is not None:
            strt_pos = self.current_pos
            end_pos = strt_pos + num_tokens
            mask = torch.triu(torch.ones(end_pos, end_pos, device=x.device, dtype=torch.bool), diagonal=1)[strt_pos:end_pos, : end_pos] # mask: (strt_pos: end_posm, : end_pos)
            self.current_pos = end_pos
        else:
            strt_pos = 0
            mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1) # mask: (num_tokens, num_tokens)

        mask = mask.unsqueeze(0).unsqueeze(0) # (1, 1, strt_pos: end_posm, : end_pos) or (1, 1, num_tokens, num_tokens)

        for i, block in enumerate(self.trf_blocks):
            old_cache = cache.get(i) if cache is not None else None
            
            if self.gradient_checkpointing and self.training:
                # wrap the block in PyTorch's checkpoint utility to save VRAM
                x, new_cache = checkpoint.checkpoint(block, x, mask, strt_pos, old_cache, use_reentrant=False) # x: (B, S, emb_dim)
            else:
                # Standard forward pass for inference
                x, new_cache = block(x, mask, offset=strt_pos, cache=old_cache) # x: (B, S, emb_dim)

            if cache is not None:
                cache.update(i, new_cache)

        x = self.final_norm(x) # (B, S, emb_dim)
        logits = self.out(x) # (B, S, vocab_size)

        return logits

    def reset_kv_cache(self):
        self.current_pos = 0

    
 