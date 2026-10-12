# reward-hacking-debate

Replication and extension of "Debate Training Reduces Reward Hacking in RLAIF" (arXiv 2608.17776)
on Tinker with open-weight models. The full review, pilot results, cost estimates and decision
history are in `docs/PLAN.md`; read it before proposing experiments or spending credits.

## Decisions so far
- **Models:** GPT-OSS-120B (medium reasoning effort) is the trained policy. GPT-OSS-20B (medium effort) is the frozen judge.
  - Cheaper judges reached step-0 MCC of only 0.16–0.27.
  - The chosen judge starts at MCC 0.44, and solves 30% of problems against the policy's 55%.
- **Data:** DAPO-Math-17k only.
  - `runs/data_dapo/filtered.jsonl` holds 771 problems the policy solves 20–80% of the time.
  - Split it 50/50 into train and validation.
- **Judge settings:** 24k-token cap; 8 votes in both training and evaluation; a vote with no verdict counts as INCORRECT.
- **Seeds:** the author prefers paper fidelity (8 votes) over more seeds. Start with 1 seed of all 4 arms (AA, ABAB, ABAABA, RLVR), about $6.9k.

## Working rules
- Ask before any Tinker spend and quote the estimated cost first. Paid stages require `--yes`.
- The API key is read from the `TINKER_API_KEY` environment variable. Never print it or commit it.
- The training loop is not built yet. Its proposed design is in `docs/PLAN.md` and the conversation history.

## Known pitfalls
- The cookbook's `safe_grade` runs math-verify in a thread, where it always fails, and it passes the answers in reverse order. Use `rhd/grading.py`.
- Truncated GPT-OSS output comes back as raw channel text that includes the hidden reasoning. `rhd/sampling.py` blanks the visible text whenever a completion isn't clean.
- Solver prompts must ask for a written solution, not just a boxed answer. Otherwise the judge has nothing to check.
- On Intel Macs, tinker pulls in torch 2.2.2, which needs `numpy<2`.

## Commands
- `.venv/bin/python -m pytest -q tests` runs the offline tests; no API key is needed.
- `.venv/bin/python -m scripts.pilot {pool,filter,judge,regrade,report} --out runs/<dir>` runs a pilot stage. See `scripts/pilot.py`.
