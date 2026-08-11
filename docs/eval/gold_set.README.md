# Gold set notes

`gold_set.jsonl` has 40 hand-built questions across four buckets (15 factual,
10 multi-hop, 10 out-of-scope/refusal, 5 ambiguous), grounded in the known
content of the 9 documents in `corpus/manifest.yaml`.

**`gold_citations[].paragraph_id` values are filled in and verified against
the real ingested corpus.** Every citation was looked up directly against the
`chunks` table (post-parsing, post-chunking — not just the raw PDF text) using
`docker compose exec db psql ...` keyword searches, then checked
programmatically with `eval/build_gold_set.py`, which confirms every
`(doc_id, paragraph_id)` pair actually exists in the ingested corpus.

Three citations are `paragraph_id: null` (F14, M01, M09 — all
`DTSP_LICENSING_GUIDELINES`) — that's a real, verified fact about this
corpus, not an unfilled placeholder: the parser's numbered-paragraph
regex didn't produce a numbered clause for that specific passage (it fell
inside an unnumbered introductory/table-of-contents-adjacent chunk), and
citing it that way is what the live system actually returns for the same
content. See `docs/adr` and the "known limitations" discussion around
paragraph coverage for why a minority of chunks — mostly in shorter
documents (`FEAT_PRINCIPLES`) or intro/TOC sections — don't carry a
paragraph number: multiple short numbered paragraphs sometimes get merged
into one chunk during chunking, and the chunk keeps only the *first*
paragraph's number.

If the corpus is re-ingested after any change to `services/ingestion/parse.py`
or `chunk.py`, re-run `eval/build_gold_set.py` — a parser change can shift
which paragraph_id a given passage ends up under, and a stale gold citation
would silently under-count `context_precision`.
