# CCAR-P Ep 09 — Demo snippets: a golden dataset through a grading pipeline

One-off demo for **Episode 09 — Evaluation Metrics & Frameworks**. Not a build-along: no cumulative project, no shared state with other episodes.

- Setup: `pip install anthropic` · `export ANTHROPIC_API_KEY=...`
- Model is read from `EVAL_MODEL` (default `claude-sonnet-5`) — **re-verify the model string before recording.**
- No sampling parameters are set anywhere — current models reject non-default values.
- Blocks 1–4 map to HTML slides 11–14; block 5 is the backstage runner (referenced on slide 14, not shown).

---

## 1) `golden.jsonl` — the golden set (Slide 11)

```json
{"id":"em-01","doc_type":"email","text":"Hi team, please process invoice INV-2214 from Northwind Traders, total $1,240.50. Thanks! — Dana","expected":{"vendor":"Northwind Traders","invoice_number":"INV-2214","total":1240.50}}
{"id":"em-02","doc_type":"email","text":"Forwarding the March statement from Blue Harbor Marine — invoice BH-8891, total USD 340.00. Please approve.","expected":{"vendor":"Blue Harbor Marine","invoice_number":"BH-8891","total":340.00}}
{"id":"em-03","doc_type":"email","text":"Invoice 55-C from Redwood Print Co. comes to 2,450.00 including tax. Can we get this scheduled?","expected":{"vendor":"Redwood Print Co.","invoice_number":"55-C","total":2450.00}}
{"id":"em-04","doc_type":"email","text":"Hi — just checking whether my March order has shipped yet. No invoice questions, thanks!","expected":{"vendor":null,"invoice_number":null,"total":null}}
{"id":"em-05","doc_type":"email","text":"Kestrel Foods invoice KF-0311: subtotal $890.00, less prompt-payment discount $90.00, total due $800.00.","expected":{"vendor":"Kestrel Foods","invoice_number":"KF-0311","total":800.00}}
{"id":"inv-01","doc_type":"invoice","text":"INVOICE No. 77120 — Vendor: Atlas Freight LLC — Date: 2026-08-14 — TOTAL DUE: $15,320.75","expected":{"vendor":"Atlas Freight LLC","invoice_number":"77120","total":15320.75}}
{"id":"inv-02","doc_type":"invoice","text":"Hollis & Vane Studio — Invoice HV-206: Subtotal $1,200.00 · Applied credit $200.00 · Balance due $1,000.00","expected":{"vendor":"Hollis & Vane Studio","invoice_number":"HV-206","total":1000.00}}
{"id":"inv-03","doc_type":"invoice","text":"SCAN (degraded) — V3ndor: Gr@nite Supp1y C0 — INVO1CE N0. 99127 — T0TAL: 4102,75","expected":{"vendor":"Granite Supply Co","invoice_number":"99127","total":4102.75}}
{"id":"fx-01","doc_type":"fax","text":"***FAX*** ...PURCHASE INV REF 4-1188 SUMMIT LOGISTICS TL 995 50 WE THANK YOU","expected":{"vendor":"Summit Logistics","invoice_number":"4-1188","total":995.50}}
{"id":"fx-02","doc_type":"fax","text":"FAXED INVOICE — Cedarline Nursery, no. CL-4470 — amount due 612.25 — remit within 30 days","expected":{"vendor":"Cedarline Nursery","invoice_number":"CL-4470","total":612.25}}
```

Design notes: `em-04` is the guard row (correct answer = all nulls); `em-05` and `inv-02` are subtotal traps; `inv-03` is the OCR nightmare; `fx-01` is the messy scan the judge will disagree on.

---

## 2) `run_eval.py` — config, schema, extraction (Slide 12)

```python
import json, os, time
from anthropic import Anthropic

client = Anthropic()
MODEL = os.environ.get("EVAL_MODEL", "claude-sonnet-5")  # ⚠ re-verify model string

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
    resp = client.messages.create(
        model=MODEL, max_tokens=512,
        system=SYSTEM,                       # stable prefix → cached
        tools=[EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": "record_extraction"},
        messages=[{"role": "user", "content": doc_text}]  # variable part last
    )
    block = next(b for b in resp.content if b.type == "tool_use")
    return block.input, resp.usage
```

---

## 3) `run_eval.py` — rule-based grader (Slide 13)

```python
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
```

---

## 4) `run_eval.py` — model-graded judge (Slide 14)

```python
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
    payload = json.dumps({"document": doc_text, "extracted": pred, "expected": expected}, indent=2)
    resp = client.messages.create(
        model=MODEL, max_tokens=300,
        system=RUBRIC,                             # judge rubric also caches
        tools=[GRADE_TOOL],
        tool_choice={"type": "tool", "name": "record_grade"},
        messages=[{"role": "user", "content": payload}]
    )
    block = next(b for b in resp.content if b.type == "tool_use")
    return block.input
```

---

## 5) `run_eval.py` — backstage runner + stratified report (referenced on Slide 14)

```python
def run(golden_path="golden.jsonl"):
    rows = [json.loads(line) for line in open(golden_path)]
    results = []
    for r in rows:
        t0 = time.time()
        pred, usage = extract(r["text"])
        checks = grade_rules(pred, r["expected"])
        judge = grade_model(r["text"], pred, r["expected"])
        passed = all(ok for _, ok in checks)
        results.append({
            "id": r["id"], "doc_type": r["doc_type"],
            "rule_pass": passed, "judge": judge["verdict"],
            "latency": round(time.time() - t0, 1),
            "tokens": usage.input_tokens + usage.output_tokens,
            "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
        })
        print(f'{r["id"]:8} {r["doc_type"]:8} rules={"pass" if passed else "FAIL"} '
              f'judge={judge["verdict"]:4} {results[-1]["latency"]}s  {judge["reason"]}')
    report(results)

def report(results):
    n = len(results)
    agg = sum(r["rule_pass"] for r in results)
    agree = sum(r["rule_pass"] == (r["judge"] == "pass") for r in results)
    print("\n=== Aggregate ===")
    print(f"overall: {agg}/{n} = {agg / n:.0%}")
    print("\n=== Stratified ===")
    for doc_type in sorted({r["doc_type"] for r in results}):
        sub = [r for r in results if r["doc_type"] == doc_type]
        k = sum(r["rule_pass"] for r in sub)
        print(f"{doc_type:8}: {k}/{len(sub)} = {k / len(sub):.0%}")
    print(f"\njudge vs rules: {agree}/{n} agree")
    print(f'tokens: input {sum(r["tokens"] for r in results)} · '
          f'cache reads {sum(r["cache_read"] for r in results)}')

if __name__ == "__main__":
    run()
```

Illustrative expected output (numbers vary run to run — that spread is the noise-floor talking point on Slide 8):

```
=== Aggregate ===
overall: 6/10 = 60%

=== Stratified ===
email   : 4/5 = 80%
fax     : 1/2 = 50%
invoice : 1/3 = 33%

judge vs rules: 9/10 agree    # judge passes fx-01, rules fail it → Drill 2
```
