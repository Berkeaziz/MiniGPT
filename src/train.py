import pandas as pd
import torch
import torch
print(torch.cuda.is_available())
from dataset import CharDataset


def load_text_from_csv(file_path: str, text_column: str = "text") -> str:
    df = pd.read_csv(file_path)

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in {file_path}. "
            f"Available columns: {list(df.columns)}"
        )

    text_list = df[text_column].fillna("").astype(str).tolist()

    return "\n".join(text_list)


def build_vocab(*texts):
    full_text = "".join(texts)
    chars = sorted(list(set(full_text)))

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    return stoi, itos


def main():
    block_size = 128
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_path = "data/train.csv"
    valid_path = "data/validation.csv"
    test_path = "data/test.csv"

    train_text = load_text_from_csv(train_path, text_column="text")
    valid_text = load_text_from_csv(valid_path, text_column="text")
    test_text = load_text_from_csv(test_path, text_column="text")

    stoi, itos = build_vocab(train_text, valid_text, test_text)

    train_dataset = CharDataset(train_text, block_size, stoi, itos)
    valid_dataset = CharDataset(valid_text, block_size, stoi, itos)
    test_dataset = CharDataset(test_text, block_size, stoi, itos)

    print("Datasets loaded successfully.")
    print(f"Vocab size: {train_dataset.vocab_size}")
    print(f"Train length: {len(train_dataset.data)}")
    print(f"Valid length: {len(valid_dataset.data)}")
    print(f"Test length: {len(test_dataset.data)}")

    x, y = train_dataset.get_batch(batch_size=batch_size, device=device)

    print(f"x shape: {x.shape}")  
    print(f"y shape: {y.shape}")  

    print("\nSample input:")
    print(train_dataset.decode(x[0].tolist()))

    print("\nSample target:")
    print(train_dataset.decode(y[0].tolist()))


if __name__ == "__main__":
    main()