# tpot-taste

Post-training a **small** language model to understand *tpot / tech-Twitter taste* — so it can:

1. **Recommend post ideas** (ideate),
2. **Decide whether a draft is worth posting** (score),
3. **Suggest how to improve a draft** for engagement/virality (rewrite).

We combine **SFT + preference optimization (DPO) + on-policy RL (GRPO)** on a single
12 GB RTX 3080 Ti, targeting near-SOTA on this narrow task — not a bare-minimum finetune.

## Architecture (two models)

| Model | What | Serves | Base |
|---|---|---|---|
| **Taste Scorer** | encoder, Bradley-Terry (same-author pairs) + regression head | capability #2 *and* the RL reward | ModernBERT-base / DeBERTa-v3 |
| **Taste Writer** | decoder, SFT → DPO → GRPO vs. the Scorer | capabilities #1, #3 | Qwen2.5-3B-Instruct (4-bit, Unsloth) |

Full rationale, alternatives considered, and the phased plan:
**[`docs/design/01-system-design.md`](docs/design/01-system-design.md)**.
Running decision log: **[`docs/design/00-decision-log.md`](docs/design/00-decision-log.md)**.

## Data

[The Community Archive](https://www.community-archive.org/) — public-domain, opt-in
full tweet histories from **tpot**. HF mirror
[`Rabrg/community-archive`](https://huggingface.co/datasets/Rabrg/community-archive):
~11.5M tweets with native `favorite_count` / `retweet_count` / `created_at` / author.

The core label is **not raw likes** (confounded by follower count + time): it is
**within-author normalized engagement** and **same-author pairwise preferences**,
which difference out the follower confound.

## Environment (NixOS + WSL2, CUDA via WSL passthrough)

```bash
nix develop                 # enters devShell, sets CUDA/WSL env, loads .env
uv sync                     # core data/eval deps (torch-free, fast)
uv sync --extra train       # + torch 2.10 / transformers 5.5 / unsloth / trl / peft / bnb
# uv sync --extra rl        # + vllm, when we reach GRPO (Phase 4)
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Heavy deps are pinned to the set proven working on this machine
(`negative-space-learning-v2/uv.lock`) so the build is a cache hit.

## Tracking

W&B project **`tpot-taste`** (key + project in `.env`). Runs grouped by phase:
`scorer`, `sft`, `dpo`, `grpo`.

## Layout

```
tpot_taste/      package: data/ scoring/ generation/ eval/ utils/
scripts/         entrypoints (download, build-dataset, train-*, eval-*)
configs/         experiment configs
docs/design/     design doc + decision log  <- start here
tests/           pytest (TDD for data transforms & metrics)
data/            raw/ interim/ processed/ splits/   (gitignored)
outputs/         checkpoints, reports          (gitignored)
```

## Status

See the **Status** section at the top of `docs/design/00-decision-log.md`.
