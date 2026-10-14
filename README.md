# Debate training against reward hacking, with open-weight models

When a language model is trained with reinforcement learning against an AI judge (RLAIF), it can
learn to fool the judge rather than get the answer right. Kenton et al., *Debate Training Reduces
Reward Hacking in RLAIF* ([arXiv 2608.17776](https://arxiv.org/abs/2608.17776)), show that adding
a critic slows this down. The model plays both sides of a debate, and a second copy of itself is
rewarded for pointing out flaws to the judge. Their results use Gemini models and a proprietary
dataset.

This project **replicates that result with open-weight models and public data**, then **extends
it**. Extensions include using a small budget of labels, transferring to tasks without reference
answers, and alternative debate formats. It runs on [Tinker](https://thinkingmachines.ai/tinker/),
Thinking Machines' API for fine-tuning and sampling open-weight models.

**Status:** the pilot is done and the training loop is built. All four arms, cost tracking and
reports are in place, tested offline against a fake backend. Training runs are waiting on compute
credits.

| | |
|---|---|
| Policy (trained) | GPT-OSS-120B, medium reasoning effort, LoRA rank 32 |
| Judge (frozen) | GPT-OSS-20B, medium reasoning effort, 8k-token thinking budget, 8 votes per episode |
| Data | DAPO-Math-17k problems the policy solves 20–80% of the time: 771 in total, or 548 if the Chinese-language ones are left out (an open decision; see [DESIGN.md](docs/DESIGN.md#data)) |
| Main metric | The judge's MCC on held-out problems, which falls as the policy learns to fool the judge |

**Why GPT-OSS:**
- It is open-weight and available on Tinker.
- It reasons in a hidden channel before its visible answer, which matches the paper's setup where
  the judge sees only the visible text.
- The 120B/20B pair gave the right capability gap at the lowest cost of the pairings tested.

## Terms

| Term | Meaning |
|---|---|
| **Policy** | The model being trained. It plays both **Alice** (the solver) and **Bob** (the critic), with shared weights. |
| **Ply** | One turn by one player, e.g. Alice's solution or Bob's critique. Each ply has hidden reasoning and a visible response. |
| **AA, ABAB, ABAABA** | The paper's names for its protocols. AA is a single solver turn with no critic. ABAB adds Bob's critique. ABAABA adds Alice's reply to it. |
| **RLVR** | RL from verifiable rewards: the policy is rewarded for matching the reference answer directly, so there is no judge to fool. It is the *roofline*, the best the judge-based arms could hope to reach. |
| **MCC** | Matthews correlation coefficient: how well the judge's verdict agrees with whether Alice's answer is actually right. It ranges from −1 to 1, where 0 is chance. Reward hacking shows up as falling MCC while the training reward rises. |
| **Prefix-then-diverge** | The paper's way of training multi-turn debates. Pick one ply, keep the earlier plies fixed, sample several alternatives for that ply, and compare their rewards. |

## The experiment

```
RLVR    Alice ─────────────────────────────────▶ reference-answer check   (roofline: can't be hacked)
AA      Alice ─────────────────────────────────▶ judge                    (baseline: expected to be hacked)
ABAB    Alice ──▶ Bob critiques ───────────────▶ judge                    (debate)
ABAABA  Alice ──▶ Bob critiques ──▶ Alice replies ▶ judge                 (debate + rebuttal)
```

- Alice's reward is the share of judge votes saying her answer is correct. Bob's is the remainder.
- The judge never sees the reference answer or anyone's hidden reasoning.
- The reference answer is used only to *measure* reward hacking, except in RLVR, where it is the
  reward.

**What a successful replication shows:**
- AA's judge MCC falls during training while its reward rises.
- ABAB's MCC stays roughly flat.
- With three seeds per arm, ABAB's peak validation accuracy lands between AA and RLVR. The paper's
  accuracy gap was about 2 points, which is within the noise of a single seed. One seed can show
  the MCC result but not this one.

Absolute numbers aren't expected to match the paper's. The full design is in
[docs/DESIGN.md](docs/DESIGN.md).

## What's done

**Gate 0: pilot (no training, $90, plus $36 to build the dataset).** The pilot picked the models
and data, and found three bugs before they could cost real money.
- **Judge choice.** The judge has to be imperfect enough to be fooled, but accurate enough that
  its reward means something. It should also be weaker than the policy, as in the paper. Three
  settings of GPT-OSS-20B were measured on the same 1,057 policy solutions to 300 problems:

  | GPT-OSS-20B judge | MCC (95% CI) | Solves alone | Policy solves | Tokens per vote |
  |---|---|---|---|---|
  | Medium effort, 24k-token cap | 0.49 (0.42–0.55) | 61% | 60% | 4,540 |
  | Low effort, 8k cap | 0.18 (0.11–0.24) | 30% | 60% | 390 |
  | **Medium effort, 8k cap, forced verdict (chosen)** | **0.43 (0.36–0.50)** | **51%** | 60% | 3,800 |

  - **Medium effort at 24k** was a good judge but solved as many problems as the policy.
  - **Low effort** was clearly weaker, but barely better than chance as a judge.
  - **The chosen setting** gives the judge less thinking budget than the policy (8k against 12k
    tokens). When it runs out, *budget forcing* ends its reasoning and makes it give a verdict at
    once. This meets both pilot criteria:
    - MCC is inside the 0.4–0.7 range we set: good enough to give real signal, weak enough to be
      fooled.
    - The judge solves 9 points fewer problems than the policy. That is similar to the paper,
      where the judge solved about 66% and the policy peaked near 75%.
  - **It leans lenient:** it accepts 73% of solutions, while 68% are actually right. MCC is the
    same on the dataset's English and Chinese problems (0.43 vs 0.45).
  - **Known side effect:** 15% of votes hit the budget, mostly on short solutions with little
    working shown. Those forced verdicts lean INCORRECT, which pushes the policy to show checkable
    work. See [DESIGN.md](docs/DESIGN.md#the-judge).
  - All votes are committed (`runs/data_dapo/judges/`), and `python -m scripts.step0 --judge <name>`
    recomputes these numbers for free.
- **Dataset.** 771 problems were filtered from DAPO-Math-17k at the policy's 20–80% pass rate.
  Whether to keep the 223 Chinese-language problems is still open.
- **Bugs fixed:**
  - the Tinker Cookbook's math-verify wrapper, which always failed in threads;
  - truncated GPT-OSS output leaking hidden reasoning into the visible text;
  - a solver prompt that asked only for a boxed answer. The prompt now asks for a written
    solution, but 24% of solutions are still bare answers. The policy is not penalized for this,
    since leaning on bare answers is itself a hack worth watching, and the rate is tracked at every
    evaluation.

**Training loop.** A custom loop handles all four arms:
- prefix-then-diverge training, so Alice and Bob each train on their own turns;
- judge-vote rewards and word-limit and format penalties;
- evaluation every 20 steps, with MCC, accuracy and saved transcripts;
- resumable checkpoints and a hard spend cap.

It is covered by 30 offline tests, including every arm end to end against a fake Tinker backend.

## What credits buy

Runs are staged so each one decides whether the next is worth paying for. Costs are estimated from
the pilot's measured token counts and are rough (±30%).

| Gate | Run | Est. cost | Passes if |
|---|---|---|---|
| 0 | Pilot | done ($126) | Judge MCC 0.4–0.7 and clearly weaker than the policy at solving (both met: 0.43, and 51% vs 60%) |
| 1a | Smoke test (2 steps); LR sweep (2 × 30 RLVR steps); then RLVR, 200 steps | ~$14 + ~$140 + ~$420 | The real API behaves like the offline tests, and validation accuracy rises |
| 1b | AA, 200 steps | ~$1,230 | **Reward rises while MCC falls**: the hacking the paper studies, reproduced |
| 2 | ABAB + ABAABA, 200 steps each | ~$2,890 | Debate holds MCC flat where AA's falls |
| 3 | +2 seeds of all arms | ~$9,100 | Error bars on the curves, and the accuracy comparison |

One seed of the full replication comes to about **$4.7k, or about $5.9k with 25% for reruns**.
Each extension is roughly $3–9k. If 1b fails (AA doesn't visibly hack), the project stops there
for about $1.8k.

## Planned extensions

1. **A small label budget.** Mix a small fraction of ground-truth labels into the debate reward,
   or use them to tune the judge, and sweep the fraction.
2. **Generalization.** Train on one domain and test on another without reference answers,
   measuring success with a stronger model as grader.
3. **Debate variants:**
   - a variable number of rebuttal rounds, with the judge deciding when to stop;
   - several critics prompted to attack different aspects;
   - a moderator that comments on argument quality for the judge.

The reasoning behind these, and the risks for each, are in [docs/PLAN.md](docs/PLAN.md) §3.

## Running it

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q tests          # offline, no API key needed

.venv/bin/python -m scripts.train abab        # free: prints the cost estimate and exits
export TINKER_API_KEY=...
.venv/bin/python -m scripts.train abab --yes --max-usd 2000
.venv/bin/python -m scripts.report curves runs/aa-seed0 runs/abab-seed0 --out reports/replication
.venv/bin/python -m scripts.report transcripts runs/abab-seed0 -n 5
```

Tested with Python 3.12; plain `python3 -m venv` works too with a recent Python.

### Reproducing the pilot

The filtered dataset is committed (`runs/data_dapo/filtered.jsonl`), so training doesn't depend on
rerunning the pilot. To build a comparable dataset and re-measure the judge (about $55):

```bash
.venv/bin/python -m scripts.pilot pool --sources dapo --pool-size 2200 --out runs/pilot-repro
.venv/bin/python -m scripts.pilot filter --yes --out runs/pilot-repro
.venv/bin/python -m scripts.pilot judge --yes --judges gptoss20b-medium-8k-forced --out runs/pilot-repro
.venv/bin/python -m scripts.pilot report --out runs/pilot-repro
```

This is an approximate reproduction, not an exact one:
- The original pool of about 2,200 problems was drawn from DAPO problems that an earlier
  GPT-OSS-20B policy hadn't always solved. These commands draw from all of DAPO instead.
- The pilot's raw outputs (judge votes and per-stage reports) weren't kept, so the numbers are
  regenerated, not checked against saved ones.

## Repository layout

| Path | What it is |
|---|---|
| `rhd/debate.py` | Protocols, player and judge prompts, reward rules |
| `rhd/rollout.py` | Playing out episodes, including prefix-then-diverge groups |
| `rhd/trainer.py` | The training loop: steps, evaluation, checkpoints, spend cap, logs |
| `rhd/costs.py` | Cost estimates from pilot token counts |
| `rhd/sampling.py`, `rhd/judge.py`, `rhd/grading.py`, `rhd/metrics.py`, `rhd/data.py` | Tinker sampling, prompts, answer checking, MCC and AUROC, data |
| `scripts/train.py` | Train one arm |
| `scripts/report.py` | Curves, summary table and Markdown transcripts from run directories |
| `scripts/pilot.py` | The Gate 0 pilot (already run) |
| `scripts/step0.py` | Recomputes the judge's step-0 MCC from the committed votes |
| `runs/data_dapo/judges/` | The judge's votes on the dataset, and their summary |
| `runs/data_dapo/filtered.jsonl` | The filtered dataset, with the policy's pilot solutions |
| `docs/DESIGN.md` | How the training loop works and where it departs from the paper |
| `docs/PLAN.md` | The paper review, pilot results, budget and decision log, in chronological order |

## Reference

This project replicates:

```bibtex
@article{kenton2026debate,
  title   = {Debate Training Reduces Reward Hacking in {RLAIF}},
  author  = {Kenton, Zachary and Janzer, Lili and Greig, Rory and Teh, Tian Huey and
             Tyshchuk, Kirill and Brown-Cohen, Jonah and Edwards, Harri and
             Rajamanoharan, Senthooran and Siegel, Noah Y. and Jaques, Natasha and Shah, Rohin},
  journal = {arXiv preprint arXiv:2608.17776},
  year    = {2026}
}
```

## Contact

[Aditya Radhakrishnan](https://github.com/aditya-radhakrishnan). Mentor: Ansh Radhakrishnan.
Released under the [MIT License](LICENSE). The problems in `runs/data_dapo/` come from
[DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) (Apache-2.0).
