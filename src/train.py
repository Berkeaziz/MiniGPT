import pandas as pd
import torch

from dataset import CharDataset
from model import BigramLanguageModel


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


@torch.no_grad()
def estimate_loss(model, train_dataset, valid_dataset, eval_iters, batch_size, device):
    out = {}
    model.eval()

    for split, dataset in [("train", train_dataset), ("val", valid_dataset)]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            xb, yb = dataset.get_batch(batch_size=batch_size, device=device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()
    return out


def main():
    block_size = 128
    batch_size = 32
    n_embd = 64
    max_iters = 3000
    eval_interval = 300
    eval_iters = 100
    learning_rate = 1e-3

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)

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

    print("\nDatasets loaded successfully.")
    print(f"Vocab size: {train_dataset.vocab_size}")
    print(f"Train length: {len(train_dataset.data)}")
    print(f"Valid length: {len(valid_dataset.data)}")
    print(f"Test length: {len(test_dataset.data)}")

    model = BigramLanguageModel(
        vocab_size=train_dataset.vocab_size,
        block_size=block_size,
        n_embd=n_embd
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("\nTraining started...\n")

    for step in range(max_iters):
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(
                model=model,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                eval_iters=eval_iters,
                batch_size=batch_size,
                device=device
            )
            print(
                f"step {step}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

        xb, yb = train_dataset.get_batch(batch_size=batch_size, device=device)

        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("\nTraining finished.")

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=500)[0].tolist()

    print("\nGenerated text after training:\n")
    print(train_dataset.decode(generated))


if __name__ == "__main__":
    main()