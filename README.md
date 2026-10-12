# reward-hacking-debate

Replicating and extending *Debate Training Reduces Reward Hacking in RLAIF*
([arXiv 2608.17776](https://arxiv.org/html/2608.17776v1)) on Tinker with open models:
GPT-OSS-20B (medium reasoning) as the trained policy and a frozen weaker judge (default
Qwen3-8B with thinking off).

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export TINKER_API_KEY=...
.venv/bin/python -m pytest -q tests   # offline; no API key needed
```

## Gate 0 pilot (no training, about $50)

Outputs go to `runs/pilot/`. Every stage resumes if interrupted, and spend is logged to
`runs/pilot/usage.json`.

```bash
.venv/bin/python -m scripts.pilot pool                       # free: 3k problems (DAPO / DeepScaleR / Omni-MATH >= 6)
.venv/bin/python -m scripts.pilot filter --yes --limit 200   # ~$3: measure real token counts first
.venv/bin/python -m scripts.pilot report
.venv/bin/python -m scripts.pilot filter --yes               # ~$30: rest of the pool
.venv/bin/python -m scripts.pilot judge --yes                # ~$10: qwen3-8b-nothink and gptoss20b-low
.venv/bin/python -m scripts.pilot report
```

The pilot passes when the filtered set has about 1-2k problems and one judge has step-0 MCC
between 0.4 and 0.7, with a solve rate well below the policy's.

## Layout

- `rhd/config.py`: models, judge candidates, Tinker prices
- `rhd/data.py`: dataset loading and normalization
- `rhd/sampling.py`: Tinker sampling, hidden-reasoning split, cost tracking
- `rhd/judge.py`: solver and judge prompts, verdict parsing
- `rhd/grading.py`: reference-answer checking (evaluation only)
- `rhd/metrics.py`: MCC and AUROC
- `scripts/pilot.py`: Gate 0 pilot
