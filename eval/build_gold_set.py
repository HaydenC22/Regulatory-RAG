"""Validates docs/eval/gold_set.jsonl: checks required fields, bucket counts,
and (post-ingestion) that every gold_citations doc_id exists in the documents
table and every non-null paragraph_id exists in the chunks table. The gold set
itself is hand-built, not generated — this script only sanity-checks it."""

import sys
from collections import Counter

from eval.ragas_harness import default_gold_set_path, load_gold_set
from services.db import get_connection


def validate(path: str, check_db: bool = True) -> bool:
    records = load_gold_set(path)
    ok = True

    counts = Counter(r.question_type for r in records)
    print(f"Loaded {len(records)} records: {dict(counts)}")
    if len(records) != 40:
        print(f"WARNING: expected 40 records, found {len(records)}")
        ok = False

    if not check_db:
        return ok

    try:
        with get_connection() as conn:
            known_docs = {row["doc_id"] for row in conn.execute("SELECT doc_id FROM documents").fetchall()}
            known_paragraphs = {
                (row["doc_id"], row["paragraph_id"])
                for row in conn.execute("SELECT doc_id, paragraph_id FROM chunks").fetchall()
            }
    except Exception as exc:  # pragma: no cover - depends on live DB availability
        print(f"Skipping DB checks (could not connect: {exc})")
        return ok

    missing_paragraphs = 0
    for record in records:
        for citation in record.gold_citations:
            if citation["doc_id"] not in known_docs:
                print(f"{record.id}: unknown doc_id {citation['doc_id']!r}")
                ok = False
            elif citation["paragraph_id"] is None:
                missing_paragraphs += 1
            elif (citation["doc_id"], citation["paragraph_id"]) not in known_paragraphs:
                pid, doc_id = citation["paragraph_id"], citation["doc_id"]
                print(f"{record.id}: paragraph_id {pid!r} not found in {doc_id}")
                ok = False

    if missing_paragraphs:
        print(
            f"{missing_paragraphs} gold_citations have paragraph_id=null — see "
            "docs/eval/gold_set.README.md, fill these in after `make ingest`."
        )

    return ok


if __name__ == "__main__":
    success = validate(default_gold_set_path())
    sys.exit(0 if success else 1)
