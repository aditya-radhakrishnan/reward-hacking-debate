# Training loop design

This describes the training loop in `rhd/` as built. It runs all four arms of the paper,
*Debate Training Reduces Reward Hacking in RLAIF* (Kenton et al., [arXiv 2608.17776](https://arxiv.org/abs/2608.17776)),
on [Tinker](https://thinkingmachines.ai/tinker/). The loop is tested end to end against a fake
Tinker backend (`tests/test_training.py`), but it hasn't yet run against the real API.

## The four arms

| Arm | Plies | Reward | What it tests |
|---|---|---|---|
| **RLVR** | Alice solves | 1 if her answer matches the reference, else 0 | The roofline: training on ground truth, which can't be hacked |
| **AA** | Alice solves | Share of 8 judge votes saying her answer is correct | The baseline the paper shows gets reward-hacked |
| **ABAB** | Alice solves → Bob critiques | Alice: vote share *r*. Bob: 1 − *r* | Does a critic keep the judge honest? |
| **ABAABA** | Alice solves → Bob critiques → Alice replies | Same; the reply earns Alice's reward | Does a rebuttal help further? |

Alice and Bob are the same model with shared LoRA weights. Every turn has hidden reasoning (GPT-OSS's
analysis channel) and a visible response. Only visible text is passed to the other player and to
the judge. Each ply is its own single-turn conversation, so earlier hidden reasoning is never in
context.

## The judge

The judge is frozen GPT-OSS-20B at medium reasoning effort.
- **What it sees:** the problem and the visible exchange, never the reference answer.
- **What it decides:** whether Alice's *original* final answer is correct. It ends with
  `VERDICT: CORRECT` or `VERDICT: INCORRECT`.
- **Votes:** it is sampled 8 times per episode. A vote with no verdict counts as INCORRECT; with
  budget forcing that is under 1% of votes.
- **Prompts:** AA reuses the pilot's judge prompt, so step-0 MCC is comparable with the pilot. The
  debate judge is also told that Bob was instructed to find flaws even in correct solutions, and to
  ignore persuasive language. Both follow the paper's Appendix G.

**Thinking budget and budget forcing.** The judge gets 8,192 tokens, less than the policy's
12,000. If it hits the budget mid-reasoning, `rhd/sampling.py` closes its reasoning, opens its
final answer with `VERDICT:`, and samples a few more tokens. The verdict is then based on less
reasoning, instead of being lost.
- **Why this setting:** it was chosen from three measured on the same 1,057 solutions (PLAN.md §8):

  | Setting | MCC | Judge solves | Policy solves |
  |---|---|---|---|
  | Medium, 24k cap | 0.49 | 61% | 60% |
  | Low, 8k cap | 0.18 | 30% | 60% |
  | **Medium, 8k, forced** | **0.43** | **51%** | 60% |

  It is the only one that is both a usable judge (MCC ≥ 0.4) and a clearly weaker solver than the
  policy, as in the paper.
- **Known side effect: forced verdicts lean INCORRECT.** 15% of votes are forced, mostly on short
  solutions with little working shown (78 words on average, against 128). Forced verdicts say
  INCORRECT 82% of the time and agree with the truth only 52% of the time.
  - The judge, short on time, tends to reject what it couldn't check. This pushes the policy to
    show checkable work.
  - The paper's judge had no budget, so this is a deviation from it. Watch `bare_answer_rate` and
    `forced_vote_rate` in the evaluation logs: if the policy learns to game the budget, it shows
    up there.

## One training step

1. **Batch.** 32 problems from the train split, taken in epoch order. Each epoch is shuffled from
   the seed, so a resumed run sees the same batches.
2. **Prefix-then-diverge.** For each problem, choose one ply uniformly at random. In ABAB that means
   about half the groups train Alice and half train Bob.
   - Earlier plies are sampled once and shared.
   - The chosen ply is sampled 8 times.
   - Each branch is played to the end, with later plies sampled once.
   - The judge votes on each finished episode.
3. **Rewards** go to the chosen ply (see below).
4. **Advantage** is the reward minus the group mean. Groups where every reward is the same are
   dropped, since they carry no signal.
5. **Update.** One `forward_backward` with the `importance_sampling` loss on the chosen ply's sampled
   tokens (prompt tokens get advantage 0), then one Adam step. LoRA rank 32, β = (0.9, 0.95), and no
   KL penalty, since the paper mentions none.

That's 256 trained completions per step, for 200 steps. Debate arms use a quarter of the base
learning rate, as in the paper.

## Rewards and penalties

| Rule | Value | Source |
|---|---|---|
| Alice's reward | *r* = share of the 8 votes saying CORRECT | Paper |
| Bob's reward | 1 − *r* (constant-sum) | Paper |
| Format failure: truncated or malformed output, empty visible text, or a solution without `\boxed{}` | Reward replaced with −1 | Paper (−1); the boxed value is ours |
| Word limit on critiques and replies | 150 words; −0.01 per excess word, capped at −1 | Paper: soft, proportional penalty. The coefficient is ours |
| How words are counted | Whitespace-separated words, with each Chinese or Japanese character counted as one word | Ours. A token limit would penalise LaTeX-heavy critiques 2–2.5× more than prose (tokens per word range from 2.0 to 5.1 across the pilot's solutions), pushing Bob towards vague language, the paper's failure mode |
| Word limit on Alice's solution | None | Paper |
| Bare answers (a `\boxed{}` answer with ≤5 words before it) | Not penalised, but tracked | Ours: leaning on bare answers is itself a hack worth watching |

If the shared prefix of a group fails the format rule, the group is skipped and counted in
`groups_skipped_prefix`.

## Evaluation (every 20 steps, plus step 0 and the last step)

- 150 validation problems × 2 episodes, each played in full under the arm's protocol. RLVR is
  evaluated with the AA judge, since it has none of its own.
  - The same 150 problems are used at every evaluation, out of the 274 in the validation split, to
    keep each evaluation at about $6–8. That's about 5% of a run's cost.
  - `--eval-problems` raises it if the curves are too noisy.
- **Accuracy:** Alice's final answer against the reference, using math-verify and then a sympy
  fallback (`rhd/grading.py`).
- **Judge MCC:** the judge's majority verdict (vote share ≥ 0.5) as a classifier of whether Alice is
  right. This is the paper's main reward-hacking signal: it falls when the policy learns to fool
  the judge. Episodes the judge never saw (a format failure) are left out. AUROC of the vote share
  is also logged.
- **Also logged:** vote share for Alice (the validation reward), Bob's win rate (1 − share),
  words and tokens per ply, format failures, the bare-answer rate, and no-verdict votes.
- **Transcripts:** every evaluation episode, including two of the judge's responses, is saved.

## Data

- `runs/data_dapo/filtered.jsonl` holds 771 DAPO-Math-17k problems the policy solved 1–3 times out of 4
  (25–75%) in the pilot.
- **223 of them (29%) are in Chinese.** Whether to keep them is an **open decision**. For now they
  are excluded by default; `--no-english-only` keeps them.
  - **Against keeping them:** fidelity to the paper's English data. Chinese problems also get more
    bare answers (32% vs 25%), and 40% of the policy's answers to them are written in Chinese.
  - **For keeping them:** 40% more data. The policy's pass rate is the same on both (0.60 vs 0.59),
    and so is the judge's accuracy (MCC 0.45 on Chinese, 0.43 on English for the chosen judge; PLAN.md §9).
  - **No longer a factor:** the word limit. The word count now counts each Chinese character as a
    word, so Chinese critiques can't slip past the limit.
  - **Options:** exclude them; keep them; or translate them to English. The paper gives no dataset
    size to match.
- That leaves 548 problems, split 50/50 by a fixed seed: **274 train, 274 validation.** A small
  train set is fine here: the paper's 20%-subset ablation hacked faster. 200 steps of 32 problems
  is about 23 passes over the train set.

## Running it

```bash
python -m scripts.train abab                          # free: prints the cost estimate
python -m scripts.train abab --yes --max-usd 2000     # paid
python -m scripts.report curves runs/aa-seed0 runs/abab-seed0 --out reports/replication
python -m scripts.report transcripts runs/abab-seed0 -n 5
```

Every `RunConfig` field is a flag, e.g. `--steps`, `--lr`, `--votes`, `--word-limit` or
`--groups-per-step`.

- **Spend cap.** A paid run requires `--yes` and `--max-usd`. Before each step or evaluation, the
  run stops if the next one, costed like the last, would pass the cap. Spend counts prefill,
  sampling and training tokens at list prices (`rhd/config.py`).
- **Resume.** State is saved to Tinker every 5 steps and after each evaluation. Re-running the same
  command resumes from the last checkpoint. Logs from later steps are dropped and redone, but their
  spend still counts. Changing a setting that affects results (LR, votes, seed and so on) on resume
  is refused.

**Run directory:**
- `config.json`
- `metrics.jsonl`: one row per step
- `eval.jsonl`: one row per evaluation
- `checkpoints.jsonl`
- `usage.json`
- `transcripts/train/stepNNNN.jsonl`: 8 sampled groups per step
- `transcripts/eval/stepNNNN.jsonl`

## Cost per run (estimated by `rhd/costs.py`)

Estimates use token counts measured in the pilot and are rough, about ±30%. Critique and reply
lengths are guesses until a debate run measures them.

| Arm | Per step | 200 steps + 11 evals |
|---|---|---|
| RLVR | $1.80 | ~$420 |
| AA | $5.82 | ~$1,230 |
| ABAB | $6.48 | ~$1,370 |
| ABAABA | $7.14 | ~$1,520 |

Judge votes are about 70% of the cost in the judge arms.

## Not done yet

- **Learning rate.** The cookbook's `get_lr` doesn't cover GPT-OSS and the paper gives no value.
  The default (4e-5) is a placeholder until a 30-step sweep at 2e-5 and 4e-5 (about $75 each) picks
  one.
- **Live API checks.** The fake backend checks shapes and control flow, not Tinker's real
  behaviour. The first paid run should be short (`--steps 2 --eval-problems 10 --max-usd 20`) to
  check sampling logprobs, datum lengths for GPT-OSS, checkpoint save and resume, and real token
  counts against the estimate.

## Reproducing the pilot

The filtered dataset is committed (`runs/data_dapo/filtered.jsonl`), so training doesn't depend on
rerunning the pilot. To build a comparable dataset and re-measure the judge (about $55):

```bash
.venv/bin/python -m scripts.pilot pool --sources dapo --pool-size 2200 --out runs/pilot-repro
.venv/bin/python -m scripts.pilot filter --yes --out runs/pilot-repro
.venv/bin/python -m scripts.pilot judge --yes --judges gptoss20b-medium-8k-forced --out runs/pilot-repro
.venv/bin/python -m scripts.pilot report --out runs/pilot-repro
```

This rebuilds the dataset approximately. The original pool of about 2,200 problems was drawn from
DAPO problems that an earlier GPT-OSS-20B policy hadn't always solved; these commands draw from all
of DAPO instead.

The judge measurements, by contrast, can be checked exactly. Their votes on the committed dataset
are in `runs/data_dapo/judges/`, and `python -m scripts.step0 --judge <name>` recomputes their
summaries for free.

