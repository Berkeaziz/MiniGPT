import torch


class CharDataset:
    def __init__(self, file_path: str, block_size: int = 8, train_ratio: float = 0.9):
        self.block_size = block_size

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        self.text = text

        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        self.data = torch.tensor(self.encode(text), dtype=torch.long)

        split_idx = int(len(self.data) * train_ratio)
        self.train_data = self.data[:split_idx]
        self.val_data = self.data[split_idx:]

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, tokens):
        return "".join([self.itos[int(i)] for i in tokens])

    def get_batch(self, split: str, batch_size: int, device: str):
        data_source = self.train_data if split == "train" else self.val_data

        ix = torch.randint(len(data_source) - self.block_size, (batch_size,))

        x = torch.stack([data_source[i:i + self.block_size] for i in ix])
        y = torch.stack([data_source[i + 1:i + self.block_size + 1] for i in ix])

        return x.to(device), y.to(device)