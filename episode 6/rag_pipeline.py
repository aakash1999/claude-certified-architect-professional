"""
CCAR-P Ep 06 — RAG Pipeline Design (runnable demo)

Retrieval-Augmented Generation in 4 steps:

    DOCUMENTS ──chunk──> CHUNKS ──vectorize──> VECTORS (the index)
                                                  |
    QUESTION ──────────vectorize───────> query ───┤ retrieve top-k
                                                  v
                                    CONTEXT + QUESTION ──> Claude ──> ANSWER

Retrieval is local & free (TF-IDF). Only the final generation step
calls Claude — on Haiku, the cheapest current model.
"""

import anthropic
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- The only model we pay for: Haiku 4.5 ($1 in / $5 out per 1M tokens) ---
MODEL = "claude-haiku-4-5"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY automatically

# ---------------------------------------------------------------------------
# STEP 1 — CHUNK: split documents into small, overlapping pieces.
# Overlap (50 chars here) keeps sentences from being cut in half at the
# boundary, so a fact straddling two chunks survives in at least one.
# ---------------------------------------------------------------------------

DOCUMENTS = [
    {
        "title": "Employee Leave Policy",
        "text": """Employees are entitled to 20 days of paid annual leave per calendar year.
Leave must be requested at least 14 days in advance through the HR portal.
Unused leave cannot be carried over to the next year unless approved by
the department head in writing. Sick leave is separate and allows up to 10
days per year with a valid medical certificate. Parental leave follows
the statutory requirements of the employee's jurisdiction and is not
deducted from annual leave entitlement."""
    },
    {
        "title": "Remote Work Policy",
        "text": """All employees may work remotely up to 3 days per week with manager
approval. Remote work requires a stable internet connection of at least
25 Mbps. Employees must be available during core hours (10 AM to 4 PM
local time) and respond to messages within 30 minutes during these hours.
Remote employees are responsible for maintaining a secure workspace —
company data must not be visible to unauthorized individuals. VPN usage
is mandatory when accessing internal systems from outside the office."""
    },
    {
        "title": "Expense Reimbursement Policy",
        "text": """Business expenses must be submitted within 30 days of the expense date.
Receipts are required for any expense over $25. Travel expenses require
pre-approval from the department head for amounts exceeding $500.
Meal expenses during business travel are capped at $75 per day.
Reimbursement is processed within 10 business days of approved submission.
Personal expenses, even if incurred during business travel, are not
eligible for reimbursement."""
    }
]


def chunk_text(text, chunk_size=200, overlap=50):
    """Fixed-size chunker with character overlap."""
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap  # step forward, but keep the overlap
    return chunks


# Build the chunk list, remembering which document each chunk came from.
all_chunks = []
for doc in DOCUMENTS:
    for chunk in chunk_text(doc["text"]):
        all_chunks.append({"text": chunk, "source": doc["title"]})

# ---------------------------------------------------------------------------
# STEP 2 — INDEX: vectorize every chunk with TF-IDF.
# Each chunk becomes a sparse vector: one dimension per vocabulary word,
# valued by how distinctive that word is in this chunk vs. the whole corpus.
# Production swap-in: a real embedding model (e.g. Voyage AI's voyage-3).
# TF-IDF is our zero-cost stand-in — the pipeline shape is identical.
# ---------------------------------------------------------------------------

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform([c["text"] for c in all_chunks])

# ---------------------------------------------------------------------------
# STEP 3 — RETRIEVE: vectorize the question the same way, then rank every
# chunk by cosine similarity and keep the top-k. This is the "R" in RAG —
# no LLM involved yet, so it's instant and free.
# ---------------------------------------------------------------------------


def retrieve(query, top_k=3):
    """Return the top-k (chunk, similarity score) pairs for a query."""
    # Same vocabulary as the index — never re-fit on the query alone.
    query_vec = vectorizer.transform([query])

    # Similarity of the query against every chunk, in one shot.
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Rank descending, keep the best k.
    top_indices = similarities.argsort()[::-1][:top_k]
    return [(all_chunks[i], similarities[i]) for i in top_indices]


# ---------------------------------------------------------------------------
# STEP 4 — GENERATE: stuff the retrieved chunks into Claude's context and
# ask the question. Claude grounds its answer in the context — and the
# system prompt forbids using anything else (the anti-hallucination rule).
# ---------------------------------------------------------------------------


def ask(question, top_k=3):
    """Full RAG pipeline: retrieve -> build context -> generate."""
    # 1. Retrieve
    results = retrieve(question, top_k=top_k)

    # 2. Build the context string, citing each chunk's source document
    context = "\n\n".join(
        f"[Source: {chunk['source']}]\n{chunk['text']}" for chunk, _ in results
    )

    # 3. Generate with Claude Haiku
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=(  # the guardrail: answer ONLY from the provided context
            "You are a company policy assistant. Answer questions using "
            "ONLY the provided context. If the context doesn't contain "
            "the answer, say so — do not make up information."
        ),
        messages=[{
            "role": "user",
            "content": f"Context from company documents:\n\n{context}"
                       f"\n\nQuestion: {question}"
                       f"\n\nAnswer based only on the context above:"
        }],
    )

    answer = next(b.text for b in response.content if b.type == "text")
    return answer, results, response.usage


def cost_of(usage):
    """Estimated USD cost at Haiku rates: $1 / $5 per 1M tokens."""
    return (usage.input_tokens * 1.00 + usage.output_tokens * 5.00) / 1_000_000


def show(question):
    """Run one question end-to-end and print a camera-friendly transcript."""
    answer, results, usage = ask(question)

    print(f"\n{'=' * 70}\nQUESTION: {question}\n{'=' * 70}")

    print("\nRetrieved chunks (the 'R' in RAG — local, free, instant):")
    for chunk, score in results:
        print(f"  [{chunk['source']}] similarity={score:.3f}")
        print(f"    \"{chunk['text'][:70]}...\"")

    print(f"\nANSWER (Claude {MODEL}):\n{answer}")
    print(f"\nTokens: {usage.input_tokens} in / {usage.output_tokens} out"
          f" ≈ ${cost_of(usage):.5f}")  # Haiku keeps this near zero


if __name__ == "__main__":
    print(f"Indexed {tfidf_matrix.shape[0]} chunks "
          f"(vocabulary: {tfidf_matrix.shape[1]} terms) from "
          f"{len(DOCUMENTS)} documents\n")

    # Demo 1 — a question the docs CAN answer (retrieval + grounded answer)
    show("How many days can I work remotely per week?")

    # Demo 2 — a question the docs CANNOT answer: watch the guardrail
    # refuse to hallucinate instead of inventing an answer.
    show("What is the company's dress code?")

    # Live Q&A — type your own questions, 'quit' to exit.
    print(f"\n{'=' * 70}\nAsk your own questions about the policies "
          f"(type 'quit' to exit)\n{'=' * 70}")
    try:
        while True:
            question = input("\nask> ").strip()
            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                break
            show(question)
    except (KeyboardInterrupt, EOFError):
        pass
    print("\nDone.")
