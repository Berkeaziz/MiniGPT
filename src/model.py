import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_embd: int):
        super().__init__()

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_embd = n_embd

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)              
        pos = torch.arange(T, device=idx.device)               
        pos_emb = self.position_embedding_table(pos)           

        x = tok_emb + pos_emb                                  
        logits = self.lm_head(x)                               

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)                
            targets_flat = targets.view(B * T)                 
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]              
            logits, _ = self(idx_cond)                        

            logits = logits[:, -1, :]                          
            probs = F.softmax(logits, dim=-1)                 

            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1)           

        return idx