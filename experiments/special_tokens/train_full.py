"""True full fine-tune including embeddings.

The mlx-lm lora CLI freezes the whole model and unfreezes only transformer
blocks (`model.layers[-num_layers:]`) — embed_tokens is NEVER trained, so new
special-token rows stay at init and the model can't learn to emit them. This
script unfreezes everything and saves a standalone reloadable model dir.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.optimizers import clip_grad_norm
from mlx_lm.tuner.datasets import CacheDataset, load_dataset
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import print_trainable_parameters
from mlx_lm.utils import _download, load, load_config, save_config, save_model

SPECIAL_ID_START = 49152  # first new-token id; weight decision/structure tokens
SPECIAL_WEIGHT = 10.0  # 1 decision token vs ~40 body tokens dilutes its gradient


def weighted_loss(model, batch, lengths):
    """default_loss but with special-token targets upweighted (Breeze-2 style)."""
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
    weights = mask * (1 + (SPECIAL_WEIGHT - 1) * (targets >= SPECIAL_ID_START))
    ce = nn.losses.cross_entropy(logits, targets) * weights
    ntoks = mask.sum()
    ce = ce.astype(mx.float32).sum() / weights.sum()
    return ce, ntoks


class ClippedAdam(optim.Adam):
    """Adam with gradient-norm clipping — bf16 full FT diverged without it."""

    def __init__(self, learning_rate: float, max_norm: float = 1.0):
        super().__init__(learning_rate=learning_rate)
        self.max_norm = max_norm

    def apply_gradients(self, gradients, parameters):
        gradients, _ = clip_grad_norm(gradients, max_norm=self.max_norm)
        return super().apply_gradients(gradients, parameters)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--resume", help="checkpoint .safetensors to continue from")
    args = ap.parse_args()

    mx.random.seed(0)
    model, tok = load(args.model)
    if args.resume:
        model.load_weights(args.resume, strict=False)
        print(f"resumed weights from {args.resume}")
    model.unfreeze()
    print_trainable_parameters(model)

    ds_args = SimpleNamespace(data=args.data, train=True, test=False, mask_prompt=True)
    train_set, valid_set, _ = load_dataset(ds_args, tok)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    targs = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        val_batches=25,
        steps_per_report=50,
        steps_per_eval=200,
        steps_per_save=500,
        max_seq_length=1024,
        adapter_file=str(out / "adapters.safetensors"),
    )
    train(
        model,
        ClippedAdam(learning_rate=args.learning_rate),
        CacheDataset(train_set),
        CacheDataset(valid_set),
        args=targs,
        loss=weighted_loss,
    )

    # save a standalone model dir (weights + config + tokenizer) for mlx_lm.load
    save_model(out, model)
    src = Path(args.model)
    if not src.exists():
        src = _download(args.model, allow_patterns=["config.json"])
    config = load_config(src)
    config["vocab_size"] = model.model.embed_tokens.weight.shape[0]
    save_config(config, out / "config.json")
    tok._tokenizer.save_pretrained(out)
    print(f"saved standalone model to {out}")


if __name__ == "__main__":
    main()
