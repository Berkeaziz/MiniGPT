import torch


class CharDataset:
    def __init__(self, text: str, block_size: int, stoi: dict, itos: dict):
        self.block_size = block_size
        self.text = text
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(stoi)

        self.data = torch.tensor(self.encode(text), dtype=torch.long)

        if len(self.data) <= self.block_size:
            raise ValueError(
                f"Text length ({len(self.data)}) must be greater than block_size ({self.block_size})."
            )

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, tokens):
        return "".join([self.itos[int(i)] for i in tokens])

    def get_batch(self, batch_size: int, device: str):
        max_start = len(self.data) - self.block_size - 1
        if max_start <= 0:
            raise ValueError(
                "Dataset is too small for the given block_size."
            )

        ix = torch.randint(0, max_start + 1, (batch_size,))

        x = torch.stack([self.data[i:i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1:i + self.block_size + 1] for i in ix])

        return x.to(device), y.to(device)