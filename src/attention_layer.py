import torch
import torch.nn.functional as F

class RoPE(torch.nn.Module):
    def __init__(self, head_dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)          # (max_seq_len, head_dim/2)
        freqs = torch.cat((freqs, freqs), dim=-1)  # (max_seq_len, head_dim)
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, seq_len):
        return self.cos[:seq_len], self.sin[:seq_len]

def apply_rope(x, cos, sin):
        # x is (B, heads, T, head_dim)
        # cos/sin are (T, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # rotate pairs: split into first half and second half
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)

        return (x * cos) + (rotated * sin)

class AttentionLayer(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super(AttentionLayer, self).__init__()
        self.dimension = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"
        self.K = torch.nn.Linear(d_model, d_model)
        self.Q = torch.nn.Linear(d_model, d_model)
        self.V = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)    
        self.rope = RoPE(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        K = self.K(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        Q = self.Q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.V(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # RoPE applied here — only Q and K, not V
        cos, sin = self.rope(T)
        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        attn_output = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output)