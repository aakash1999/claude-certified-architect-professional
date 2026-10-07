"""
EP 09 DEMO — push a golden dataset through a grading pipeline.

End-to-end flow (follow the STEP banners below while recording):
  golden.jsonl  (10 hand-labelled documents)
    -> Claude extracts {vendor, invoice_number, total} via a forced tool call
    -> a RULE-based grader checks each field against the expected values
    -> a MODEL-based judge (second Claude call) grades the same extraction
    -> console report: aggregate accuracy, per-doc-type breakdown,
       judge-vs-rules agreement, tokens used, and estimated cost in cents.

Slides 11-14 of the deck map to STEPs 2-5 below.

No sampling parameters (temperature etc.) are set anywhere —
current models reject non-default values.
"""

# ═══════════════════════════════════════════════════════════════════════
# STEP 0 — SETUP (do this before)
#   1. pip install anthropic
#   2. export ANTHROPIC_API_KEY=sk-ant-...     (your key)
#   3. Keep golden.jsonl in the SAME folder as this file.
#   4. Run:  python3 run_eval.py
# ═══════════════════════════════════════════════════════════════════════

import json
import os
import time

from anthropic import Anthropic

client = Anthropic()  # picks up ANTHROPIC_API_KEY from the environment

# ═══════════════════════════════════════════════════════════════════════
# STEP 1 — PICK THE MODEL (this is where the cost is controlled)
#
#   Default: claude-haiku-4-5 — the cheapest current model.
#   $1 per M input tokens, $5 per M output tokens.
#   The full 10-row demo run lands at roughly 1-3 CENTS total.
#   Override with:  export EVAL_MODEL=<model>   (then update PRICE below).
# ═══════════════════════════════════════════════════════════════════════

MODEL = os.environ.get("EVAL_MODEL", "claude-haiku-4-5")

# Haiku 4.5 list prices — USD per MILLION tokens.
# If you switch EVAL_MODEL, update these or the cost line will be wrong.
PRICE = {"input": 1.00, "output": 5.00, "cache_read": 0.10, "cache_write": 1.25}


def usage_cost(usage):
    """Estimated USD cost of one API call, from its usage object."""
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    return (
        usage.input_tokens * PRICE["input"]
        + usage.output_tokens * PRICE["output"]
        + cache_read * PRICE["cache_read"]
        + cache_write * PRICE["cache_write"]
    ) / 1_000_000


# ═══════════════════════════════════════════════════════════════════════
# STEP 2 — EXTRACT: system prompt + strict tool schema + forced tool call
#   (Slide 12)
#
#   Tool use is how we get GUARANTEED structured JSON out of the model:
#   the schema below is enforced, so extraction can never come back as
#   free text or with missing keys.
# ═══════════════════════════════════════════════════════════════════════

SYSTEM = """You extract invoice fields from vendor documents.
Extract exactly three fields: vendor, invoice_number, total.
- Use null for any field the document does not clearly contain.
- If the document contains no invoice at all, return null for every field.
- For totals, use the final amount payable, not a subtotal."""

EXTRACT_TOOL = {
    "name": "record_extraction",
    "description": "Record the invoice fields extracted from one document.",
    "input_schema": {
        "type": "object",
        "properties": {
            "vendor":         {"type": ["string", "null"], "description": "Vendor name, or null if absent"},
            "invoice_number": {"type": ["string", "null"], "description": "Invoice number, or null if absent"},
            "total":          {"type": ["number", "null"],  "description": "Invoice total, or null if absent"}
        },
        "required": ["vendor", "invoice_number", "total"],
        "additionalProperties": False,
        "strict": True
    }
}

def extract(doc_text):
    # One extraction = one API call, forced through the tool above.
    resp = client.messages.create(
        model=MODEL, max_tokens=512,
        system=SYSTEM,                       # stable prefix -> cacheable
        tools=[EXTRACT_TOOL],
        # "tool" + name = the model MUST call this tool, no prose allowed.
        tool_choice={"type": "tool", "name": "record_extraction"},
        messages=[{"role": "user", "content": doc_text}]  # variable part last
    )
    # Pull the tool_use block out of the response; .input is the parsed dict.
    block = next(b for b in resp.content if b.type == "tool_use")
    return block.input, resp.usage


# ═══════════════════════════════════════════════════════════════════════
# STEP 3 — RULE-BASED GRADER: exact, deterministic field checks
#   (Slide 13)
#
#   Three checks per field:
#     expected null  -> prediction must ALSO be null (fabrication tripwire)
#     total          -> numbers match within +/- 0.01 (float-safe compare)
#     strings        -> equal ignoring case and whitespace
# ═══════════════════════════════════════════════════════════════════════

def grade_rules(pred, expected):
    checks = []
    for field in ("vendor", "invoice_number", "total"):
        got, want = pred.get(field), expected.get(field)
        if want is None:
            ok = got is None                       # fabrication tripwire
        elif field == "total":
            ok = got is not None and abs(got - want) <= 0.01
        else:
            ok = isinstance(got, str) and got.strip().lower() == want.strip().lower()
        checks.append((field, ok))
    return checks


# ═══════════════════════════════════════════════════════════════════════
# STEP 4 — MODEL-GRADED JUDGE: a second Claude call with a rubric
#   (Slide 14)
#
#   Where rules are brittle, a judge can be lenient in the right ways
#   ("Granite Supply Co" vs a mangled OCR scan). We force it through a
#   pass/fail tool so the verdict is machine-readable, and it must give
#   a reason — that's what makes judge disagreements explainable.
# ═══════════════════════════════════════════════════════════════════════

GRADE_TOOL = {
    "name": "record_grade",
    "description": "Record the grading verdict for one extraction.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["pass", "fail"]},
            "reason":  {"type": "string"}
        },
        "required": ["verdict", "reason"],
        "additionalProperties": False,
        "strict": True
    }
}

RUBRIC = """You are grading an invoice-field extraction.
PASS only if every extracted field satisfies the expected value:
- expected null means the field must be null (any value is fabrication = FAIL);
- totals match within 0.01;
- names and numbers match ignoring case and whitespace.
If any field fails, verdict is fail and the reason names the field and the problem."""

def grade_model(doc_text, pred, expected):
    # Judge sees all three things at once: source doc, extraction, ground truth.
    payload = json.dumps({"document": doc_text, "extracted": pred, "expected": expected}, indent=2)
    resp = client.messages.create(
        model=MODEL, max_tokens=300,
        system=RUBRIC,                             # judge rubric also cacheable
        tools=[GRADE_TOOL],
        tool_choice={"type": "tool", "name": "record_grade"},
        messages=[{"role": "user", "content": payload}]
    )
    block = next(b for b in resp.content if b.type == "tool_use")
    # Returns usage too, so the cost line covers BOTH calls per row.
    return block.input, resp.usage


# ═══════════════════════════════════════════════════════════════════════
# STEP 5 — BACKSTAGE RUNNER: loop over the golden set, one row at a time
#   (referenced on Slide 14, not shown)
#
#   Per row: extract -> rule-grade -> judge -> record everything
#   (verdicts, latency, tokens, cost), then print a live progress line.
# ═══════════════════════════════════════════════════════════════════════

def run(golden_path="golden.jsonl"):
    rows = [json.loads(line) for line in open(golden_path)]
    results = []
    for r in rows:
        t0 = time.time()
        pred, usage = extract(r["text"])                 # 1) extraction call
        checks = grade_rules(pred, r["expected"])        # 2) deterministic grade
        judge, judge_usage = grade_model(r["text"], pred, r["expected"])  # 3) judge call
        passed = all(ok for _, ok in checks)
        latency = round(time.time() - t0, 1)
        results.append({
            "id": r["id"], "doc_type": r["doc_type"],
            "rule_pass": passed, "judge": judge["verdict"],
            "latency": latency,
            "in": usage.input_tokens + judge_usage.input_tokens,
            "out": usage.output_tokens + judge_usage.output_tokens,
            "cache_read": (getattr(usage, "cache_read_input_tokens", 0) or 0)
                          + (getattr(judge_usage, "cache_read_input_tokens", 0) or 0),
            "cost": usage_cost(usage) + usage_cost(judge_usage),
        })
        print(f'{r["id"]:8} {r["doc_type"]:8} rules={"pass" if passed else "FAIL"} '
              f'judge={judge["verdict"]:4} {latency}s  {judge["reason"]}')
    report(results)


# ═══════════════════════════════════════════════════════════════════════
# STEP 6 — STRATIFIED REPORT: the numbers you talk over on camera
#
#   Aggregate accuracy alone hides the story — the stratified breakdown
#   is what tells you WHERE the pipeline is weak (e.g. faxes vs emails).
#   "judge vs rules" agreement exposes grader disagreement to discuss.
# ═══════════════════════════════════════════════════════════════════════

def report(results):
    n = len(results)
    agg = sum(r["rule_pass"] for r in results)
    agree = sum(r["rule_pass"] == (r["judge"] == "pass") for r in results)
    cost = sum(r["cost"] for r in results)
    print("\n=== Aggregate ===")
    print(f"overall: {agg}/{n} = {agg / n:.0%}")
    print("\n=== Stratified ===")
    for doc_type in sorted({r["doc_type"] for r in results}):
        sub = [r for r in results if r["doc_type"] == doc_type]
        k = sum(r["rule_pass"] for r in sub)
        print(f"{doc_type:8}: {k}/{len(sub)} = {k / len(sub):.0%}")
    print(f"\njudge vs rules: {agree}/{n} agree")
    print(f'tokens: input {sum(r["in"] for r in results)} · '
          f'output {sum(r["out"] for r in results)} · '
          f'cache reads {sum(r["cache_read"] for r in results)}')
    print(f"estimated cost: ${cost:.4f}  (~{cost * 100:.1f} cents at {MODEL} pricing)")


if __name__ == "__main__":
    run()
