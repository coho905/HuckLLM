import torch
from attention_layer import AttentionLayer

class FeedForward(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # SwiGLU uses 8/3 * d_model hidden dim to match param count
        hidden = int(8 / 3 * d_model)
        self.w1 = torch.nn.Linear(d_model, hidden, bias=False)
        self.w2 = torch.nn.Linear(hidden, d_model, bias=False)
        self.w3 = torch.nn.Linear(d_model, hidden, bias=False)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = AttentionLayer(d_model, n_heads)
        self.ffn = FeedForward(d_model)
        self.norm1 = torch.nn.RMSNorm(d_model)
        self.norm2 = torch.nn.RMSNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))   # attention + residual
        x = x + self.ffn(self.norm2(x))    # FFN + residual
        return x