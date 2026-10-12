# Review: "Thoughts After Reading Paper" (Debate Training Reduces Reward Hacking in RLAIF, arXiv 2608.17776)

## Context
Aditya wants to replicate Kenton et al. (Gemini 2.5 Flash policy, Flash Lite frozen judge, proprietary AIME-like math) on Tinker with open models, then extend it. This is a review of the doc: things to fix, gaps, and what to do next. Nothing has been built yet.

## 1. Corrections to "My understanding"
- **Reward.** The judge doesn't output a probability. It is sampled **8 times** and each sample votes. Alice gets r, the fraction of votes for her, and Bob gets **1−r** (constant-sum). The penalties don't move reward from one player to the other. The word-limit penalty is a soft per-player cost proportional to the excess words. A formatting violation *replaces* that player's reward with −1.
- **Hidden CoT.** Each turn has hidden reasoning and then a visible response, and the judge only sees the visible part. This is central to the design and the doc should say it. In the replication you'll need to strip the thinking tokens before building the judge prompt.
- **Protocol names.** AA is the baseline. In ABAB, Bob critiques. In ABAABA, Alice also gets a rebuttal. RLVR (the same policy trained with the answer checker) is the **roofline**, and you need it to report "% of gap recovered". Debate recovered about 45%.
- **How hacking is measured.** Judge **MCC** falls while reward rises, where MCC treats the judge as a classifier of whether Alice's answer is correct against ground truth. Validation accuracy of Alice's final answer is a second signal. Ground truth is used for evaluation only, never for training.
- **Training.** Alice and Bob share weights. Prefix-then-diverge picks a random ply, keeps the earlier plies fixed, and samples several continuations. About half the batch trains Alice and half trains Bob. The paper cut the LR by 4× to keep multi-player training stable.

## 2. Limitations the doc misses
- **The critic can win by being loud.** Without word limits Bob dominates through verbosity bias. 150 words worked best, and even then Bob falls back on emphatic language and nitpicking the problem wording.
- **Debate plateaus below RLVR.** It slows hacking down but doesn't close the gap, and the paper doesn't know why.
- **Bob convinces the judge slowly.** Bob learns to *spot* errors (B-judge) but is slow to *convince* the frozen judge (B-convince). This bears on any critic-side extension.
- **Weak/strong gap.** The two models share a family and a training distribution, so this is a loose stand-in for humans vs. superhuman models. Your model pairing has the same problem.
- **Statistics.** There are few seeds (3 baseline, 1 RLVR), and trajectories within one question are correlated, which inflates the effective sample size. Plan for ≥3 seeds per arm.
- **Cost.** Every rollout needs 8 judge samples per episode, plus the extra turns. Judge inference, not training, may dominate the Tinker bill.
- "Only tasks with known answers" is a *measurement* limit, not a method limit. Moving to unverifiable tasks needs a new way to **detect** hacking (e.g., a held-out stronger grader or human labels), not only a new domain.

## 3. Feedback on the extensions table
| Idea | Comment |
|---|---|
| Multiple critics | Say how reward is split (constant-sum across A vs {B,C}?) and how you'll control total critique length. Otherwise you are just re-testing the verbosity effect. Different critic prompts collapse under shared weights plus RL, which the paper saw with misaligned prompts too. |
| Meta-critic C trained on judge performance | "Judge performance" needs ground truth, so this becomes partial RLVR. State that explicitly and tie it to Ansh's label-budget idea below. |
| More judge samples | The paper **already uses 8**. More votes cut variance but not *bias*, and hacking exploits bias. Cheap ablation (1/8/32 votes), probably a small effect. Low priority. |
| Pairwise over A's samples | The paper tested this for AA and it was hacked *faster*. Only worth doing as "pairwise + debate", with a hypothesis for why the critic would fix it. |
| Variable rounds gated by C | Simpler version: let the **judge** say "need another round" up to a max. Motivated by ABAABA mainly helping the *weak* judge. Needs prefix-then-diverge generalized to variable-length episodes. |
| Step-by-step output gated by C | The paper found step-by-step formatting *hurt*: lower peak accuracy, faster degradation. Needs a strong reason before spending compute. |

**Missing extensions worth adding:**
- **Ansh's comment: label budget (IID).** Give a small labeled fraction, mix RLVR into the debate reward or fine-tune the judge, and sweep the fraction. Realistic, and easy to plot.
- **Ansh's comment: OOD transfer.** Ground truth in domain X (e.g., algebra), judge-only in domain Y (e.g., combinatorics or code). Does debate transfer better than AA?
- **Judge-strength sweep.** Vary the judge (e.g., 4B → 9B → 27B) with the policy fixed and plot MCC decay rate against the capability gap. This extends §4.2 into a scaling curve.
- **Separate weights for Alice and Bob.** The doc lists shared weights as a limitation but has no experiment for it. Two LoRA adapters on Tinker make this cheap.
- **Automated hack taxonomy.** A classifier for judge impersonation, authoritative tone, and problem-wording nitpicks. It addresses the paper's own "mechanism understanding" limitation.

## 4. Gaps in "Experiment Setup"
The doc lists three requirements. It is missing the following:
1. **Model pairing.** Tinker does have Inkling / Inkling-Small. Check that (a) the small model is clearly weaker at *solving* the problems, and (b) its judge MCC starts reasonably high (~0.5+). If the gap is too small, nothing gets hacked. If it's too big, the judge is noise. Backup pairing: Qwen3.5-9B judge with a Qwen3.5-35B-A3B or Qwen3.8-27B policy. Tinker uses **LoRA only**. The frozen judge is just a sampling client on the base model.
2. **Dataset.** The paper's data is proprietary. Candidates: DAPO-Math-17k, Big-Math, or OmniMath, with past AIME years held out. Filter so the policy's starting accuracy is about 30–60%. You also need an answer checker such as `math-verify`, and a 50/50 IID train/val split.
3. **Inducing hacking in budget.** This is the biggest risk. If the AA baseline doesn't visibly hack, the replication shows nothing. The paper's 20%-dataset ablation hacked faster, so start with a small train set and long training.
4. **RL details.** Algorithm (GRPO-style from the Tinker cookbook), group size, LR (with the 4× reduction for debate), KL on/off, LoRA rank, and max tokens per turn.
5. **Reward details.** Judge prompt (adapt from Appendix G.2), 8 votes, word-limit penalty coefficient, and the −1 formatting rule.
6. **Metrics and logging.** Val accuracy, judge MCC, train/val reward, response and judge-reasoning lengths, and saved transcripts for qualitative review.
7. **Runs and budget.** Arms: AA, ABAB, ABAABA, RLVR, each with ≥3 seeds. Estimate tokens per step × steps against Tinker pricing *before* launching.

## 4b. Cost estimate (Tinker list prices, $/M tokens; rough, ±2×)
**Token assumptions:**
- Policy rollout: about 6k output tokens (reasoning plus answer).
- Judge vote: about 2k output tokens (judge with CoT), 8 votes per rollout.
- One RL step: 32 problems × 8 samples = 256 rollouts.
- Run length: about 200 steps per run. The paper doesn't report its step count, and training has to run past the peak for hacking to show.
- With CoT on, judge tokens (about 4M per step) are larger than policy tokens (about 1.5M per step).

**Prices:**
| Model | Sample | Train |
|---|---|---|
| Inkling | 9.36 | 11.22 |
| Inkling-Small | 2.88 | 3.46 |
| Qwen3.6-35B-A3B | 1.335 | 1.177 |
| Qwen3-8B | 0.60 | 0.44 |
| GPT-OSS-120B | 0.84 | 0.737 |
| GPT-OSS-20B | 0.45 | 0.396 |

Serverless inference, beta, Inkling models only (output price): Inkling-Small $1.20, Inkling $4.05.

**Pilot** (500 problems × 4 policy samples, judge solves the same problems, 8 judge votes per solution; about 12M policy + 44M judge tokens):
| Pairing | Cost |
|---|---|
| Inkling / Inkling-Small (training API) | ≈ $245 |
| Inkling / Inkling-Small (serverless) | ≈ $100 |
| Qwen3.6-35B-A3B / Qwen3-8B | ≈ $40 |
| GPT-OSS-120B / 20B | ≈ $30 |

**Replication, per 200-step run:**
| Pairing | AA | ABAB (~1.2×) | ABAABA (~1.4×) | RLVR (no judge) | 4 arms × 1 seed | 4 arms × 3 seeds |
|---|---|---|---|---|---|---|
| Inkling / Inkling-Small | ~$8.7k | ~$10.5k | ~$12k | ~$6.4k | ~$38k | ~$113k |
| Qwen3.6-35B-A3B / Qwen3-8B | ~$1.3k | ~$1.5k | ~$1.8k | ~$0.8k | ~$5.4k | ~$16k |
| GPT-OSS-120B / 20B | ~$0.85k | ~$1.0k | ~$1.2k | ~$0.5k | ~$3.6k | ~$11k |

Add 20–30% for debugging and failed runs.

**Ways to cut cost:**
- Cap the judge's CoT or use 4 votes (judge tokens are the biggest cost).
- Use 16 problems per step instead of 32.
- Use a smaller train set so hacking appears in fewer steps.
- Route frozen-judge calls to serverless if it serves base Inkling-Small (2.4× cheaper per output token).

**Recommendation on Inkling:**
- Don't use Inkling as the main workhorse. It is about 10× the cost of the cheap pairings, and all 4 arms × 3 seeds would be about $113k.
- Develop and replicate on a cheap pairing. Run the pilot on 2–3 pairings for about $100 total and pick the one with the right accuracy gap and starting MCC.
- Watch out: GPT-OSS-120B and Qwen3.6 are strong at math, so problems may need to be harder (e.g., OmniMath difficulty ≥7) to keep policy accuracy around 30–60%.
- Optional: one confirmatory ABAB-vs-AA run on Inkling at the end (about $20k).

## 4c. Decision: cost-minimal pairing
**Policy: GPT-OSS-20B.**
- Cheapest model on the list: $0.45 sample, $0.396 train.
- Reasoning-only, so it has a natural hidden CoT.
- Strong at math.
- 32K context. That's fine once reasoning is capped, e.g. at medium effort.

**Judge: Qwen3-8B with thinking off.**
- $0.60 per million tokens. With no CoT, a vote is about 300 tokens instead of about 2k.
- Judge tokens were the biggest cost, so this is the single largest saving.
- It also gives a clear capability gap against the policy.

**Cheaper models don't exist here.** Qwen3.5-4B is priced higher ($1.005), so this pairing is the price floor on Tinker.

**Pilot fallback judges, in order:**
1. GPT-OSS-20B at low effort (same family, similar cost).
2. Qwen3-8B with thinking on and a capped budget.

**Pilot gate:**
- Judge MCC at step 0 of about 0.4–0.7. Below about 0.3 the judge is noise and every arm fails.
- Judge solve-rate clearly below the policy's.

**Tradeoff to note:** a no-CoT judge is the paper's *weaker-judge* setting (§4.2), not its main one. Report the results as a replication of §4.2's ordering (ABAABA > ABAB > AA), and of §4.1 only if the CoT fallback judge is used. The "judge response length collapses" signal is also lost.

**Revised costs:** assumes 6k-token rollouts, 256 per step, 200 steps.
- Per step: about $1.3 for the policy plus about $0.45 for the judge, so about $1.8/step.
- Per run:
| Arm | Cost |
|---|---|
| AA | ~$360 |
| ABAB | ~$430 |
| ABAABA | ~$500 |
| RLVR | ~$260 |
- 4 arms: about **$1.55k per seed**, **~$4.7k for 3 seeds**.
- With the policy at low effort (about 3k tokens), roughly half that.
- Pilot plus dataset filtering: about **$50**.

**Dataset: filter a pool by the policy's own pass rate instead of choosing by name.**
- **Pool, about 3k problems:**
  - DAPO-Math-17k: integer answers, trivial to check.
  - DeepScaleR-40k: AIME, AMC and Omni-MATH.
  - Omni-MATH difficulty ≥ 6: keep only problems with numeric or expression answers.
- **Filter:** 4 GPT-OSS-20B samples per problem, about $30. Keep pass rates between 0.2 and 0.8. RL needs variance within groups anyway.
- **Target:** about 1–2k problems total, split 50/50.
- **Clean held-out check** (small, low contamination): BeyondAIME (100 problems) or MathArena's recent competition sets (HMMT/AIME 2025), for sanity only.

## 4d. Replication budget and go/no-go gates
**Evaluation cost, missing from the estimates above.**
- Validation runs every 20 steps: 300 problems × 2 samples, plus judge votes for MCC.
- About $2 per eval, about $20 per run.

**Staged spend.** You never commit more than the next gate.
| Gate | Runs | Spend | Cumulative | Pass criteria | If it fails |
|---|---|---|---|---|---|
| 0: Pilot | no training | ~$50 | ~$50 | See below | Switch judge, reasoning effort, or filter |
| 1a: Pipeline works | RLVR, 1 seed | ~$280 | ~$330 | Val accuracy rises within ~50 steps; reward std within groups > 0 | Bug in the RL loop, LR, or answer checker. Cheap to find here |
| 1b: Premise holds | AA, 1 seed | ~$380 | ~$710 | See below | Shrink the train set to 20% (the paper's accelerant), or weaken the judge further. If AA still doesn't hack, stop: there is nothing to replicate at this scale |
| 2: Core result | ABAB + ABAABA, 1 seed | ~$950 | ~$1.7k | See below | See below |
| 3: Confidence | +2 seeds of all 4 arms | ~$3.3k | ~$5.0k | Pooled curves with error bands | — |

Gate 0 pass criteria:
- Policy pass rate 0.2–0.8 on the filtered set.
- Judge MCC at step 0 of 0.4–0.7.
- Judge vote-share for Alice correlates with ground truth.

Gate 1b pass criteria:
- Train reward rises while **val MCC falls**.
- Val accuracy peaks, then declines.
- Transcripts show hack patterns such as judge impersonation or authoritative filler.

Gate 2 pass criteria:
- ABAB/ABAABA MCC stays roughly flat while AA's falls.
- Bob's win rate is not pinned near 0 or 1.

Gate 2 failure handling:
- If Bob always wins, the critic dominates: tighten the word limit.
- If Alice always wins, Bob isn't learning: check the prefix-then-diverge split and cross-play.
- If both look healthy but MCC still falls like AA, that's a genuine negative result.

Add about 25% for reruns and debugging. That puts **1 seed at ≈ $2.1k** and **3 seeds at ≈ $6.3k**.

**What one seed can and can't show.**
- **Can show:** MCC divergence between AA and debate. It's a large qualitative trend.
- **Can't show:** the paper's peak-accuracy gap of about 2 points (74.7 vs 72.6). That is within single-seed noise, since the val standard error is about 1.5–2 points at this size.
- **So:** claim the accuracy result only after gate 3.

**Extension budget (rough):**
- Each extension is roughly 2 arms × 3 seeds × about $400, so **$2.5–7k each**.
- Examples: a judge-strength sweep is about $7k; a label-budget sweep is about $5k.
- Replication plus 2–3 extensions: about **$15–25k**.
- Budget 50–100% rerun overhead, not 25%, for the first multi-agent RL build.

## 4e. Pilot results (2026-09-30, total spend $44.61). These supersede 4c.
**Main-model accuracy on GPT-OSS-20B was fine, but no cheap judge reached MCC 0.4.**
- The 20B policy solved 61% of the full pool.
- Keep rates: 40% on DAPO, 32% on DeepScaleR, 40% on Omni-MATH.
- Best cheap judge on DAPO, measured on 712 solutions:
| Judge | MCC | 95% CI |
|---|---|---|
| GPT-OSS-20B, low | 0.27 | 0.17–0.36 |
| Qwen3-8B, no-think | 0.16 | 0.07–0.24 |
- DeepScaleR and Omni-MATH were worse for every judge, probably partly because of label noise. **Use DAPO only.**

**Chosen setup: GPT-OSS-120B (medium effort) policy, GPT-OSS-20B (medium effort) judge.**
Measured on 77 DAPO problems, 288 solutions:
| Metric | Value |
|---|---|
| MCC | 0.44 (95% CI 0.31–0.59) |
| AUROC | 0.77 |
| Judge solve rate | 0.30 |
| Policy solve rate | 0.55 |
- Keep rate: 33% of DAPO problems the 20B didn't always solve.

**Bugs found and fixed:**
- The cookbook's math-verify wrapper fails in threads and reverses the argument order.
- Truncated GPT-OSS output leaked hidden reasoning into the visible text.
- The "no words" boxed prompt produced bare-answer solutions. The prompt now asks for a written solution.

**Open issues:**
- 20% of judge votes hit the 8,192-token cap and return no verdict. They count as abstentions, but they use about 37% of judge tokens.
- 24% of 120B solutions are still bare answers.

**Revised costs** (actual tokens: policy about 4.3k, judge about 4.3k per vote; 256 rollouts/step; 200 steps):
| Votes per rollout | Per step | AA run | 4 arms × 1 seed (with eval) | With 25% overhead | 3 seeds |
|---|---|---|---|---|---|
| 8 | ≈ $5.7 | ≈ $1.15k | ≈ $4.6k | ≈ $5.7k | ≈ $17k |
| 4 (MCC 0.39 vs 0.44) | ≈ $3.7 | ≈ $0.75k | ≈ $3.1k | ≈ $3.9k | ≈ $11.5k |

- **Dataset:** about 1k filtered problems needs about 3–4.5k DAPO problems through the 120B filter, **≈ $45–65**.

**Judge cap test (2026-10-01, $3.20; 4 votes × 288 solutions at a 24k cap).**
The paper sets no judge token limit; the 8,192 cap was ours.
| Measure | Value |
|---|---|
| Median tokens per vote | 3.8k |
| Mean tokens per vote | 5.4k |
| p90 / p95 / p99 | 12k / 16.6k / 23.5k |
| Votes still cut off at 24k | 1% |
| Votes with no verdict | 1.8% |
| MCC with 4 votes | 0.41 (CI 0.29–0.54) |
| MCC with 4 votes at the 8k cap | 0.40 |

Simulated caps on the same votes:
| Cap | Mean tokens per vote | Votes cut off |
|---|---|---|
| 8k | 4.4k | 20% |
| 12k | 4.9k | 10% |
| 16k | 5.2k | 5.6% |
| 24k | 5.4k | 1% |

- **Decision (2026-10-01): 24k cap, 8 votes for both training and evaluation, fewer seeds.**
  - A no-verdict vote counts as INCORRECT (our choice).
  - 4 vs 8 votes: MCC 0.40 vs 0.43 (95% CI of the difference −0.04 to +0.10). Reward reliability 0.80 vs 0.89.
  - Aditya prefers paper-fidelity votes over seeds.
- **Replication budget at 8 votes:**
  | Arm | Per run |
  |---|---|
  | AA | ~$1.37k |
  | ABAB | ~$1.64k |
  | ABAABA | ~$1.92k |
  | RLVR | ~$0.35k |
  - Step cost is about $6.85. Evals cost about $70 per run: 300 val solutions × 8 votes, every 20 steps.
  - **1 seed of all 4 arms is about $6.9k** with 25% overhead.
  - Then add AA + ABAB seeds at about $3.9k per extra seed pair, if the curves need it.
  - This excludes the pilot ($47.81), the dataset (about $45–65) and extensions (about $3–9k each).
- That's about $4.25 per step, about $3.5k per seed for 4 arms, and about $13k for 3 seeds with overhead.

## 5. Recommended next steps
1. **Pilot (cheap, no training).** Pick the models and dataset. Measure policy accuracy, judge accuracy, and zero-step judge MCC on about 500 problems.
2. **Show the problem exists.** Run AA against RLVR until AA's MCC visibly degrades. This is the go/no-go gate.
3. **Core replication.** ABAB, then ABAABA and the weak-judge variant.
4. **Extensions**, in order of value per compute:
   1. Label budget / OOD transfer.
   2. Judge-strength sweep.
   3. Separate weights.
   4. Multiple critics / adaptive rounds.

## Verification
The replication counts as successful if it shows the paper's qualitative pattern: AA's MCC falls while ABAB's stays flat, and ABAB's peak validation accuracy beats AA's and sits between AA and RLVR. Absolute numbers are not expected to match.

## 6. Training loop design (proposed 2026-10-01; not yet approved or built)
**Approach: a custom loop**, modelled on the cookbook's `recipes/rl_loop.py`. It reuses the cookbook's renderers, checkpointing and `train_step`.
- The stock `rl/train.py` handles AA, but the debate arms need several things it doesn't do cleanly:
  - training on one chosen turn, with the other turns sampled but not trained (prefix-then-diverge);
  - per-debate judge rewards;
  - word-limit penalties;
  - an MCC evaluation with 8 votes.

**Settings:**
| Setting | Proposed |
|---|---|
| Algorithm | Synchronous on-policy; `importance_sampling` loss; LoRA rank 32 |
| Batch | 32 problems × 8 samples = 256 rollouts per step; 200 steps |
| Advantage | Reward minus the group mean; drop groups where every reward is the same |
| KL penalty | None (the paper mentions none) |
| AA reward | Fraction of 8 judge votes saying CORRECT; a vote with no verdict counts as INCORRECT |
| RLVR reward | 1 if the answer is correct, else 0 |
| Format penalty | −1 if the output is truncated or malformed |
| Data | `runs/data_dapo/filtered.jsonl` (771 problems), split 50/50 |
| Evaluation, every 20 steps | 150 val problems × 2 samples: accuracy, judge MCC (8 votes), response and judge lengths, bare-answer rate, saved transcripts |
| Checkpoints | Saved at every eval, for resume |
| Spend cap | Every run tracks its Tinker spend, including training tokens, and stops at `--max-usd` |

**Learning rate:**
- The cookbook's `get_lr` does not support GPT-OSS, and the paper gives no LR.
- Plan: 30-step RLVR runs at 2e-5 and 4e-5 (about $105 total) before the full gate-1a run.
- Debate arms use LR/4, as in the paper.

**Bare answers are not penalised.** 24% of 120B solutions are a bare `\boxed{}` answer. The paper doesn't penalise this, and leaning on bare answers is itself a hack to observe. Track the rate at every eval.

**Tinker API notes:**
- `ServiceClient.create_lora_training_client_async(base_model, rank)`.
- `forward_backward_async(datums, "importance_sampling")`. Each Datum needs `target_tokens`, `logprobs` and `advantages`; prompt positions have advantage 0.
- `optim_step_async(AdamParams(learning_rate, beta1=0.9, beta2=0.95, eps=1e-8))`.
- Get a sampling client with `save_weights_and_get_sampling_client_async()`. Resume with `save_state` / `create_training_client_from_state_with_optimizer_async`.
- GPT-OSS trains on all sampled tokens, including the analysis channel. Re-rendered multi-turn history doesn't prefix-extend, so each debate turn becomes its own Datum.
