import torch
from transformer_block import TransformerBlock

class Model(torch.nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, n_embd):
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(vocab_size, n_embd)
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.attention_blocks = torch.nn.ModuleList([TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = torch.nn.RMSNorm(n_embd)
        self.head = torch.nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.embedding_layer.weight
        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {n_params:,}")

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        x = self.embedding_layer(x)
        for block in self.attention_blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss
    
if __name__ == "__main__":
    model = Model(vocab_size=50257, n_layer=16, n_head=12, n_embd=768)
