import torch

class CharDataset:
    def __init__(self, train_path: str, val_path: str, test_path: str, block_size: int = 8):
        self.block_size = block_size

        with open(train_path, "r", encoding="utf-8") as f:
            train_text = f.read()

        with open(val_path, "r", encoding="utf-8") as f:
            val_text = f.read()

        with open(test_path, "r", encoding="utf-8") as f:
            test_text = f.read()

        full_text = train_text + val_text + test_text
        
        self.full_text = full_text
        
        chars = sorted(list(set(full_text)))
        self.vocab_size = len(chars)

        self.stoi = {ch:i for i ,ch in enumerate(chars)}
        self.itos ={i: ch for i,ch in enumerate(chars)}

        self.data =torch.tensor(self.encode(text),dtype =torch.long)