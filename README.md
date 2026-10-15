# Debate training against reward hacking, with open-weight models

Language models trained with reinforcement learning against an AI judge (RLAIF) can
learn to "reward hack": to give answers that fool the judge rather than answers that are actually correct. Kenton et al., *Debate Training Reduces
Reward Hacking in RLAIF* ([arXiv 2608.17776](https://arxiv.org/abs/2608.17776)), demonstrate that adding
a critic, a second copy of the model that points out flaws in the solver's answer, slows reward hacking. Their results use Gemini models and a proprietary dataset.

My goal is to **replicate that result with open-weight models and public data**, and then **extend
it**. The main extension I want to try is mixing a small amount of ground truth into the rewards:
can a little real supervision make debate even more robust? If there's room, I'd also like to see
whether debate still helps on tasks without reference answers, and try a few different debate
formats. Everything runs on [Tinker](https://thinkingmachines.ai/tinker/), Thinking Machines' API
for fine-tuning and sampling open-weight models.

**Where things stand:** I've finished a pilot and built the training loop, which is tested offline.
Training is waiting on compute. The replication plus the labels extension needs about **$19k**, and
the first big question, whether the baseline actually reward-hacks, can be answered for about
$1.8k. The details are in [Plan and budget](#plan-and-budget).

| | |
|---|---|
| Policy (trained) | GPT-OSS-120B, medium reasoning effort, LoRA rank 32 |
| Judge (frozen) | GPT-OSS-20B, medium reasoning effort, 8k-token budget per vote (reasoning and answer), 8 votes per episode |
| Data | 771 DAPO-Math-17k problems the policy solved 1–3 times out of 4. By default the 223 Chinese-language ones are left out, leaving 548, split 274 train / 274 validation; whether to keep them is still open ([DESIGN.md](docs/DESIGN.md#data)) |
| Main metric | The judge's MCC on held-out problems, which falls as the policy learns to fool the judge |

I chose GPT-OSS because it's open-weight, it's available on Tinker, and it reasons privately
before giving a visible answer. That matches the paper, where the judge only ever sees the visible
part.

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
  reward, and in the label-budget extension, where it is part of the reward.

**What I'm looking for.** If the replication works, I should see three things:
- In AA, the reward goes up during training while the judge's MCC goes down. That's reward hacking.
- In ABAB, MCC stays roughly flat.
- ABAB's best validation accuracy lands between AA and RLVR. The paper's gap here was only about 2
  points, which is close to the noise of a single run. That's why AA and ABAB get three seeds each,
  and even then it may show up as a consistent trend rather than a statistically significant one.

I don't expect the absolute numbers to match the paper's, just the pattern. The full design is in
[docs/DESIGN.md](docs/DESIGN.md).

## What's done

### The pilot ($126, including $36 to build the dataset)

Before spending thousands of dollars on training, I wanted to check that the experiment could
actually work. The pilot doesn't train anything. It only samples from the models and compares what
they say with the reference answers, to answer three questions. It also measured how many tokens
everything uses, which is where the cost estimates come from.

1. **Are the problems the right difficulty?** The policy wrote 4 solutions to each of about 2,200
   DAPO-Math-17k problems (already narrowed to ones a smaller model hadn't always solved), and I kept the **771** it got right 1, 2 or 3 times out of 4 (147, 182
   and 442 problems). A problem it always solves leaves nothing to learn or hack, and one it never
   solves gives it nothing to learn from.
2. **Can the judge tell right answers from wrong ones, but not perfectly?** The judge voted 8 times
   on each of the policy's solutions without seeing the reference answer, and I scored its majority
   verdict with MCC. I was aiming for 0.4–0.7. Any lower and its reward is mostly noise. Any higher
   and it's too hard to fool for reward hacking to show up.
3. **Is the judge weaker than the policy?** I also had the judge solve each problem itself. This
   matters because a judge that can just re-solve the problem is hard to fool. The whole point is to
   test a weaker overseer judging a stronger model, like in the paper.

I tried three settings of GPT-OSS-20B as the judge, all on the same 1,057 policy solutions to 300
problems. The policy wrote 1,200 solutions, but 143 were cut off or had no boxed answer, so I left
them out, the same way the training loop does.

| GPT-OSS-20B judge | MCC (95% CI) | Judge solves | Policy solves | Tokens per vote |
|---|---|---|---|---|
| Medium effort, 24k-token cap | 0.49 (0.42–0.55) | 61% | 60% | 4,540 |
| Low effort, 8k cap | 0.18 (0.11–0.24) | 30% | 60% | 390 |
| **Medium effort, 8k cap, forced verdict (chosen)** | **0.43 (0.36–0.50)** | **51%** | 60% | 3,800 |

- **Medium effort with 24k tokens** was a good judge, but it could solve the problems as well as the
  policy could. **Low effort** was clearly weaker, but also a weak judge (MCC 0.18).
- **I went with the middle option.** The judge gets fewer tokens than the policy, 8k against 12k.
  If it runs out mid-thought, I use *budget forcing*: I cut its reasoning off and make it give a
  verdict right away.
  - Its MCC lands in the target range, though the confidence interval dips just below 0.4.
  - It solves 9 points fewer problems than the policy. The policy's 60% comes from the same samples
    I used to pick the problems, so the gap is approximate.
- **One thing to keep an eye on:** about 15% of votes get forced, more often on short solutions,
  and forced verdicts lean INCORRECT. That should nudge the policy to show its work, which is
  probably fine, but it's a difference from the paper. More in [DESIGN.md](docs/DESIGN.md#the-judge).
- **Anyone can check these numbers.** All the votes are committed in `runs/data_dapo/judges/`, and
  `.venv/bin/python -m scripts.step0 --judge <name>` recomputes them for free.

### Training loop (built; waiting on compute)

I built a custom training loop that runs all four arms. It handles:
- prefix-then-diverge training, so Alice and Bob each learn from their own turns;
- rewards from the judge's votes, plus penalties for going over the word limit or breaking format;
- evaluation every 20 steps, tracking MCC and accuracy and saving transcripts to read through;
- checkpoints, so runs can resume, and a spend cap checked before every step.

It has 30 offline tests, including every arm run end to end against a fake version of Tinker.

## Plan and budget

I've broken the spending into stages, so each result tells me whether the next one is worth paying
for. Every full training run is 200 steps. I'm putting the seeds where they matter most: three each
for AA and ABAB, since that's the paper's main comparison. RLVR and ABAABA get one each. The paper
also ran RLVR only once.

| Stage | Runs | Est. cost | Running total | Go on if |
|---|---|---|---|---|
| Pilot | Sampling only | $126 (spent) | — | Judge MCC 0.4–0.7 and weaker than the policy (met: 0.43, CI reaching just below 0.4; 51% vs 60%) |
| 1a. Check the pipeline | 2-step smoke test; LR sweep (2 × 30 RLVR steps); RLVR | $580 | $580 | Real API matches the offline tests; accuracy rises |
| 1b. Does AA hack? | AA | $1,230 | $1,810 | **Reward rises while MCC falls** |
| 2. Does debate help? | ABAB, ABAABA | $2,890 | $4,700 | Debate holds MCC flat where AA's falls |
| 3. Error bars | AA and ABAB, 2 more seeds each | $5,200 | $9,900 | — |
| 4. Label budget | ABAB with 10% and 30% of training problems rewarded by the reference answer; 2 seeds each | $5,500 | $15,400 | — |
| **Total** | **12 full training runs** | **$15.4k** | | **About $19k with 25% for reruns** |

- **The estimates** use the token counts from the pilot and Tinker's current prices, so they're
  rough, give or take 30%.
- **The labels extension** asks: does a little ground truth make debate more robust? On a random 10%
  or 30% of training problems, the reference answer replaces the judge: Alice gets 1 if she's right
  and 0 if not, and Bob gets the opposite, as in the judged episodes. I can compare that with ABAB
  (no labels) and RLVR (all labels, no debate) from the replication. It reuses the
  replication's data and almost all of its code, so it's the cheapest and safest extension. Only
  the reward mixing is new.
- **If AA doesn't hack** at stage 1b, there's nothing to replicate, and I'd stop there, having spent
  about $1.8k.

**Stretch goals,** if runs come in under budget or I find more funding:
- **Generalization (≈ $6.2k before reruns):** does debate still help on a new kind of task without reference
  answers? I'd train AA and ABAB on it and use a stronger model to grade the results.
- **Debate variants (≈ $9.1k before reruns):** what if the judge decides how many rebuttal rounds to allow? What
  if there are several critics, each prompted to attack a different part of the answer? Or a
  moderator who comments on how reasonable each side's arguments are?

With both, the whole program would be about $38k including reruns. More on the reasoning and risks
behind each extension is in [docs/PLAN.md](docs/PLAN.md) §3, §7 and §10.

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

Tested with Python 3.12. To rebuild the dataset and re-measure the judge, see
[DESIGN.md](docs/DESIGN.md#reproducing-the-pilot).

## Repository layout

| Path | What it is |
|---|---|
| `rhd/` | The library: protocols and rewards (`debate.py`), episodes (`rollout.py`), the training loop (`trainer.py`), cost estimates (`costs.py`), Tinker sampling, grading, metrics, data |
| `scripts/` | `train.py` (train one arm), `report.py` (curves and transcripts), `pilot.py` and `step0.py` (the pilot) |
| `runs/data_dapo/` | The filtered dataset with the policy's solutions, and the judges' votes on them |
| `docs/DESIGN.md` | How the training loop works and where it departs from the paper |
| `docs/PLAN.md` | The paper review, pilot results and decision log, in chronological order |

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
