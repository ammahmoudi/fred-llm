"""Add 15 FRED special tokens to SmolLM2-360M-Instruct and resize embeddings.

Each new embedding row is initialized to the mean of the ORIGINAL tokenizer's
subword embeddings of a short description (GTI/R4 — never random, never
batch-mean). Vocab padded to a multiple of 64 for Metal; padding rows get the
overall embedding mean. Tied embeddings → one matrix to resize.
"""

from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import _download, load, load_config, save_config, save_model

HERE = Path(__file__).parent
MODEL_ID = "mlx-community/SmolLM2-360M-Instruct"
OUT = HERE / "base_resized"

# token -> description used for embedding init
NEW_TOKENS = {
    "<|fred|>": "Fredholm integral equation",
    "<|kernel|>": "the integral kernel K(x, t)",
    "<|lambda|>": "the parameter lambda",
    "<|rhs|>": "the right hand side function f(x)",
    "<|domain|>": "the integration domain from a to b",
    "<|solve|>": "solve the equation",
    "<|T_exact_symbolic|>": "exact closed form symbolic solution",
    "<|T_approx_coef|>": "approximate solution with numerical coefficients",
    "<|T_discrete_points|>": "solution given as discrete point values",
    "<|T_series|>": "truncated Neumann series solution",
    "<|T_family|>": "family of solutions with an arbitrary constant",
    "<|T_regularized|>": "ill-posed equation requiring regularization",
    "<|T_none|>": "no solution exists",
    "<|sol|>": "the solution is",
    "<|end|>": "end of the answer",
}


def main() -> None:
    model, tok = load(MODEL_ID)
    old_emb = model.model.embed_tokens.weight
    old_vocab, dim = old_emb.shape
    print(f"old embedding: {old_emb.shape}")

    hf_tok = tok._tokenizer
    n_added = hf_tok.add_special_tokens({"additional_special_tokens": list(NEW_TOKENS)})
    assert n_added == len(NEW_TOKENS), n_added
    new_ids = [hf_tok.convert_tokens_to_ids(t) for t in NEW_TOKENS]
    assert new_ids == list(range(old_vocab, old_vocab + len(NEW_TOKENS))), new_ids

    padded = -(-(old_vocab + len(NEW_TOKENS)) // 64) * 64
    mean_row = old_emb.mean(axis=0)
    desc_rows = []
    for desc in NEW_TOKENS.values():
        ids = hf_tok(desc, add_special_tokens=False)["input_ids"]
        desc_rows.append(old_emb[mx.array(ids)].mean(axis=0))
    pad_rows = mx.broadcast_to(mean_row, (padded - old_vocab - len(NEW_TOKENS), dim))
    new_emb = mx.concatenate([old_emb, mx.stack(desc_rows), pad_rows], axis=0)
    assert new_emb.shape == (padded, dim)

    model.model.embed_tokens.weight = new_emb
    model.args.vocab_size = padded

    OUT.mkdir(exist_ok=True)
    save_model(OUT, model)
    config = load_config(_download(MODEL_ID, allow_patterns=["config.json"]))
    config["vocab_size"] = padded
    save_config(config, OUT / "config.json")
    hf_tok.save_pretrained(OUT)
    print(f"saved to {OUT} (vocab {old_vocab} -> {padded})")

    # sanity: reload, every new token is exactly one id, model still generates
    m2, t2 = load(str(OUT))
    assert m2.model.embed_tokens.weight.shape == (padded, dim)
    for tok_str in NEW_TOKENS:
        ids = t2._tokenizer(tok_str, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, (tok_str, ids)
    from mlx_lm import generate

    out = generate(m2, t2, prompt="2+2=", max_tokens=5)
    print("reload OK, sample generation:", repr(out))


if __name__ == "__main__":
    main()
