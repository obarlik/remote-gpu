"""Predefined training task: small causal transformer language model.

Runs as a subprocess invoked by the job queue. All hyperparameters come from
a JSON params file, so the task is fixed but the configuration is fully
flexible (model size, dtype, dataset, steps, etc).
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


class CausalTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, seq_len: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device)
        h = self.tok_emb(x) + self.pos_emb(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        return self.head(h)


def load_token_ids(dataset_path: str) -> tuple[torch.Tensor, int]:
    path = Path(dataset_path)
    if path.suffix == ".npy":
        import numpy as np
        ids = torch.from_numpy(np.load(path)).long()
        vocab_size = int(ids.max().item()) + 1
        return ids, vocab_size

    text = path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    ids = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    return ids, len(chars)


def sample_batch(ids: torch.Tensor, batch_size: int, seq_len: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = ids.size(0) - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([ids[s:s + seq_len] for s in starts]).to(device)
    y = torch.stack([ids[s + 1:s + seq_len + 1] for s in starts]).to(device)
    return x, y


def train(params: dict, output_dir: Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = DTYPE_MAP[params.get("dtype", "bf16")]

    ids, vocab_size = load_token_ids(params["dataset_path"])
    vocab_size = params.get("vocab_size", vocab_size)
    seq_len = params.get("seq_len", 128)
    batch_size = params.get("batch_size", 32)
    steps = params.get("steps", 200)
    log_every = params.get("log_every", 20)

    model = CausalTransformerLM(
        vocab_size=vocab_size,
        d_model=params.get("d_model", 256),
        n_heads=params.get("n_heads", 4),
        n_layers=params.get("n_layers", 4),
        seq_len=seq_len,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=params.get("lr", 3e-4))

    model.train()
    for step in range(1, steps + 1):
        x, y = sample_batch(ids, batch_size, seq_len, device)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=device == "cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == steps:
            print(f"step {step}/{steps} loss {loss.item():.4f}", flush=True)

    checkpoint_path = output_dir / "model.pt"
    torch.save({"state_dict": model.state_dict(), "params": params}, checkpoint_path)
    print(f"saved checkpoint to {checkpoint_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    params = json.loads(Path(args.params).read_text())
    output_dir = Path(args.output_dir)

    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"device={torch.cuda.get_device_name(0)}", flush=True)

    try:
        train(params, output_dir)
    except Exception as exc:
        print(f"training failed: {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
