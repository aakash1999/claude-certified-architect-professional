# EP 09 — Code Walkthrough: what happens, step by step

This doc explains every file and every `STEP` in `run_eval.py` in the order things actually execute. Use it as your talking track while recording; the `STEP` banners in the code match the headings here.

**The demo in one sentence:** we take 10 hand-labelled documents (a *golden set*), have Claude extract invoice fields from each one, grade every extraction two ways — deterministic rules and an LLM judge — and print a stratified report that shows where the pipeline is weak, for about 2 cents.

```
golden.jsonl ──> extract() ──> grade_rules() ──┐
      (10 docs)   (Claude call)  (pure Python)  ├──> run() ──> report()
                          └────> grade_model() ──┘              (console)
                                 (Claude "judge")
```

## Files in this folder

| File | What it is | Maps to |
|---|---|---|
| `golden.jsonl` | 10 labelled docs + expected answers | Slide 11 |
| `run_eval.py` | The whole pipeline (extract → grade ×2 → report) | Slides 12–14 |
| `WALKTHROUGH.md` | This file | — |
| `ep09_snippets.md` | The original snippet sheet this code was built from | — |

## Before you record — setup (STEP 0)

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
cd "episode 9"
python3 run_eval.py
```

Do one practice run before recording. The accuracy numbers move a little run to run — that's not a bug, it's your **noise-floor** talking point (Slide 8).

---

## STEP 0 — Setup, line by line

```python
client = Anthropic()
```
The client reads `ANTHROPIC_API_KEY` from the environment automatically — no key handling in the code, nothing to leak on camera.

## STEP 1 — Pick the model (cost control)

```python
MODEL = os.environ.get("EVAL_MODEL", "claude-haiku-4-5")
```

- Default is **`claude-haiku-4-5`** — the cheapest current model: **$1 per million input tokens, $5 per million output tokens**.
- This demo makes **20 API calls total** (10 extractions + 10 judge calls), each only a few hundred tokens. Expected total: **~1–3 cents per full run**.
- `EVAL_MODEL` env var lets you switch models without touching code — but if you do, update the `PRICE` dict too, or the printed cost will be wrong.

`usage_cost(usage)` is a tiny helper: it turns one API call's `usage` object into an estimated dollar cost using the `PRICE` table. It's why the report can end with a "this whole eval cost X cents" line.

## STEP 2 — Extraction: system prompt + strict tool schema (Slide 12)

Three pieces:

1. **`SYSTEM` prompt** — the job description. Note its three rules: use `null` when a field is absent, all-null when there's no invoice at all, and *total payable, not subtotal*. Those last two rules are exactly what the trap rows in the golden set test.
2. **`EXTRACT_TOOL`** — a JSON schema the model must satisfy. `"strict": True` + every field in `required` + `"additionalProperties": False` means the API **enforces** the shape: extraction can never come back as prose or with a missing key. Each field is `["string", "null"]` / `["number", "null"]` so `null` is a legal, schema-valid answer.
3. **`extract()`** — one API call per document:
   - `tool_choice={"type": "tool", "name": "record_extraction"}` **forces** the model to call the tool. No chit-chat, guaranteed JSON.
   - The response's `content` is a list of blocks; we find the `tool_use` block and `.input` is the already-parsed dict (e.g. `{"vendor": "Northwind Traders", "invoice_number": "INV-2214", "total": 1240.5}`).
   - It also returns `resp.usage` — token counts, which feed the cost line.

Why tool use instead of "reply in JSON"? Parsing free-text JSON fails on camera eventually; a strict schema turns format compliance into something the API guarantees instead of something you prompt for and pray.

## STEP 3 — Rule-based grader (Slide 13)

`grade_rules(pred, expected)` — pure Python, no API calls, fully deterministic. It loops over the three fields and applies three different checks:

| Situation | Rule | Why |
|---|---|---|
| expected is `null` | prediction must **also** be `null` | the **fabrication tripwire** — inventing a vendor where none exists is a hard fail |
| field is `total` | `abs(got - want) <= 0.01` | float-safe numeric compare (never `==` on floats) |
| anything else (strings) | equal after `.strip().lower()` | "INV-2214" == "inv-2214" should pass |

Returns a list of `(field, ok)` pairs — per-field detail, not just pass/fail, so you can show *which* field broke.

## STEP 4 — Model-graded judge (Slide 14)

The same extraction gets a second opinion — from another Claude call:

- **`RUBRIC`** (the judge's system prompt) states the exact same rules the code grader uses: null means null, totals within 0.01, case/whitespace-insensitive strings. Same standard, different enforcement.
- **`GRADE_TOOL`** forces the verdict into `{"verdict": "pass"|"fail", "reason": "..."}`. The forced enum makes it machine-readable; the required `reason` makes disagreements explainable on camera.
- **`grade_model()`** packs document + extraction + expected answer into one JSON payload so the judge sees all three side by side, then forces one `record_grade` tool call.

**The on-camera point:** rules are exact but brittle — `"Granite Supply Co"` vs an OCR-mangled `"Gr@nite Supp1y C0"` is a hard rule-fail but arguably a correct extraction. A judge can be lenient *in the right ways*. That's why serious evals use both and measure where they disagree.

## STEP 5 — The backstage runner

`run()` ties it together, one golden row at a time:

1. `extract(r["text"])` — the extraction call.
2. `grade_rules(pred, r["expected"])` — deterministic grade.
3. `grade_model(...)` — the judge call.
4. Record the row: verdicts from both graders, latency, input/output tokens, cache reads, and **cost of both calls combined**.
5. Print a live progress line so viewers see results streaming in — each line shows the row id, doc type, rule verdict, judge verdict, latency, and the judge's *reason* (great to read one or two aloud).

## STEP 6 — Stratified report

`report()` prints four blocks:

- **Aggregate** — overall accuracy (`6/10 = 60%`-ish). The number everyone asks for, and the one that hides the most.
- **Stratified** — accuracy broken down by `doc_type` (email / fax / invoice). This is the payoff: clean emails score high, degraded scans and faxes drag the average down, and *now you know where to spend your fixing effort*. Aggregate says "60%"; stratified says "the fax pipeline is the problem."
- **judge vs rules** — how often the two graders agreed. A disagreement (classic: judge passes `fx-01`, rules fail it) is your segue into LLM-as-judge trade-offs (Slide 14 / Drill 2).
- **tokens + estimated cost** — total input/output/cache-read tokens and the run's estimated cost in dollars and cents. On Haiku this lands around **$0.01–0.03**.

---

## The golden set's booby traps (why each tricky row exists)

| Row | Trap | Tests |
|---|---|---|
| `em-04` | An email with **no invoice at all** — expected is all nulls | Fabrication: does the model invent fields? |
| `em-05` | Subtotal $890 − discount $90 → real total is **$800** | Does it grab the first number instead of the amount due? |
| `inv-02` | Subtotal $1,200 − credit $200 → balance **$1,000** | Same trap, invoice-shaped |
| `inv-03` | Degraded OCR scan: `Gr@nite Supp1y C0`, `4102,75` (comma decimal) | Can it recover vendor/number/total from noise? |
| `fx-01` | Terse fax shorthand: `TL 995 50` (= 995.50) | The row the judge and rules will disagree on |

The other five rows are clean wins so the demo isn't 0%.

## Why this costs cents (the math)

20 calls × a few hundred tokens each ≈ **~7K input + ~1.5K output tokens total**. At Haiku 4.5 pricing ($1/M input, $5/M output):

```
(7,000 × $1 + 1,500 × $5) / 1,000,000  ≈  $0.014  ≈  1.4 cents per full run
```

Re-running the whole demo live on camera is effectively free. (For contrast: the same run on a flagship-tier model would be roughly 20–40× that.)

## Two gotchas to remember on camera

- **No sampling parameters anywhere** (no temperature/top_p) — current models reject non-default values; the code intentionally sets none.
- **Model string is verified** as of 2026-09-29: `claude-haiku-4-5` is the latest Haiku alias. If the API ever 404s on it, check the models doc page for a newer alias.
