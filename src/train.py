import json
from pathlib import Path

import pandas as pd
import torch

from dataset import CharDataset
from model import GPTLanguageModel


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


@torch.no_grad()
def evaluate_dataset(model, dataset, eval_iters, batch_size, device):
    model.eval()
    losses = torch.zeros(eval_iters)

    for k in range(eval_iters):
        xb, yb = dataset.get_batch(batch_size=batch_size, device=device)
        _, loss = model(xb, yb)
        losses[k] = loss.item()

    model.train()
    return losses.mean().item()


def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def main():

    block_size = 128
    batch_size = 32
    n_embd = 96
    n_head = 4
    n_layer = 3
    max_iters = 6000
    eval_interval = 300
    eval_iters = 100
    learning_rate = 3e-4
    dropout = 0.2

    train_path = "data/train.csv"
    valid_path = "data/validation.csv"
    test_path = "data/test.csv"

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = artifacts_dir / "best_model.pt"
    results_json_path = artifacts_dir / "training_results.json"
    generated_text_path = artifacts_dir / "generated_text.txt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)

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

    model = GPTLanguageModel(
        vocab_size=train_dataset.vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_step = -1

    steps = []
    train_losses = []
    val_losses = []

    print("\nTraining started...\n")

    for step in range(max_iters):
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(
                model=model,
                train_dataset=train_dataset,
                valid_dataset=valid_dataset,
                eval_iters=eval_iters,
                batch_size=batch_size,
                device=device,
            )

            train_loss = losses["train"]
            val_loss = losses["val"]

            steps.append(step)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(
                f"step {step}: "
                f"train loss {train_loss:.4f}, "
                f"val loss {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step

                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "vocab_size": train_dataset.vocab_size,
                        "block_size": block_size,
                        "n_embd": n_embd,
                        "n_head": n_head,
                        "n_layer": n_layer,
                        "dropout": dropout,
                    },
                    "stoi": stoi,
                    "itos": itos,
                    "best_val_loss": best_val_loss,
                    "best_step": best_step,
                }

                torch.save(checkpoint, best_model_path)
                print(f"New best model saved. val loss: {best_val_loss:.4f}")

        xb, yb = train_dataset.get_batch(batch_size=batch_size, device=device)

        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("\nTraining finished.")

    print(f"\nLoading best model from: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device)

    model = GPTLanguageModel(
        vocab_size=checkpoint["config"]["vocab_size"],
        block_size=checkpoint["config"]["block_size"],
        n_embd=checkpoint["config"]["n_embd"],
        n_head=checkpoint["config"]["n_head"],
        n_layer=checkpoint["config"]["n_layer"],
        dropout=checkpoint["config"]["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loss = evaluate_dataset(
        model=model,
        dataset=test_dataset,
        eval_iters=eval_iters,
        batch_size=batch_size,
        device=device,
    )

    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best step: {best_step}")
    print(f"Test loss: {test_loss:.4f}")

    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    generated = model.generate(
        context,
        max_new_tokens=500,
        temperature=0.8,
        top_k=20,
    )[0].tolist()

    generated_text = train_dataset.decode(generated)

    print("\nGenerated text after training (best model):\n")
    print(generated_text)

    results = {
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "test_loss": test_loss,
        "steps": steps,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "device": device,
        "config": {
            "block_size": block_size,
            "batch_size": batch_size,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "max_iters": max_iters,
            "eval_interval": eval_interval,
            "eval_iters": eval_iters,
            "learning_rate": learning_rate,
            "dropout": dropout,
        },
    }

    save_json(results, results_json_path)

    with open(generated_text_path, "w", encoding="utf-8") as f:
        f.write(generated_text)

    print(f"\nTraining results saved to: {results_json_path}")
    print(f"Generated text saved to: {generated_text_path}")
    print(f"Best model checkpoint saved to: {best_model_path}")


if __name__ == "__main__":
    main()