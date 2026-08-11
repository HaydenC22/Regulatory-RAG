# ADR-0004: LLM provider — Google Gemini (free tier) vs paid Anthropic/OpenAI

## Status
Accepted (supersedes the initial default of Anthropic Claude)

## Context
Generation (grounded Q&A answers, the compliance-checklist agent, and the
RAGAS judge) needs an LLM capable of reliable structured/tool-calling output.
The project was initially built against Anthropic Claude, but this is an
unpaid portfolio project with no ongoing API budget — a paid-only provider
would mean the live demo, the eval harness, and any reviewer poking at the
repo all depend on a credit card staying topped up indefinitely.

## Decision
Default to **Google Gemini's free tier** (`gemini-flash-lite-latest` — a
Google-maintained alias to the current recommended Flash-Lite model) via
`langchain-google-genai`, selected through `LLM_PROVIDER=gemini` in
`services/config.py`. Anthropic remains a fully supported alternative
(`LLM_PROVIDER=anthropic`) for anyone who does have API budget — `services/llm.py`
is a single provider-agnostic interface (`generate_structured`,
`generate_text`, `get_chat_model`) that every call site (`qa.py`,
`guardrails.py`, `checklist.py`, `ragas_harness.py`) depends on, so switching
providers is a config change, not a rewrite.

The model alias went through three live iterations before landing here,
each discovered by actually calling the API rather than assumed from
documentation, because Google's free-tier model lineup and quotas moved
faster than any written source (including this ADR's own first draft) could
track:
1. `gemini-2.0-flash` (dated snapshot) — retired: `404 This model ... is no
   longer available`.
2. `gemini-flash-latest` (alias) — resolved to `gemini-3.6-flash`, a
   brand-new flagship model with a **20 requests/day** free-tier cap
   (`ResourceExhausted: 429 ... quota_value: 20`) — nowhere near enough for
   a single 40-question eval run, let alone iterative development.
3. `gemini-2.5-flash-lite` (dated snapshot) — `404 ... no longer available
   to new users` (a new API key couldn't use it at all).
4. `gemini-flash-lite-latest` (alias) — works, and Flash-Lite tiers are
   consistently documented as getting the most generous free RPD of any
   Gemini tier, which is why this ADR settled on the *lite* alias rather
   than the plain *flash* alias.

**Lesson generalized into the code, not just this document:** `services/llm.py`
takes the model name from `LLM_MODEL_CHEAP`/`LLM_MODEL_QUALITY` env vars
rather than a hardcoded constant specifically so a future quota or
deprecation change is a `.env` edit, not a code change.

## Rationale
- **$0 cost for generation**, matching the $0 cost already achieved for
  embeddings (ADR-0003) — the entire pipeline, including the LLM-dependent
  paths, now runs on genuinely free infrastructure end to end.
- **No card required** to get a Gemini API key (aistudio.google.com), unlike
  Anthropic Console credits, which matters for anyone reproducing this repo
  without wanting to pay to try it.
- **Structured output support is equivalent** for this project's needs — both
  providers' LangChain integrations support `with_structured_output` via
  tool/function calling, so `GroundedAnswer`, `ComplianceChecklist`, and the
  scope-classification guardrail all work unchanged regardless of provider.

## Trade-off accepted
Gemini's free tier is genuinely rate-limited, and the limit is per-model,
not per-account — the flagship Flash model measured at just 20
requests/**day**, which a single 40-question eval run (roughly 100-300 LLM
calls once retrieval sub-queries, retries, and the RAGAS judge are counted)
would exhaust in minutes. The `tenacity` retry/backoff wrapper in
`services/llm.py` only helps with transient per-minute throttling — it
cannot fix a per-day cap, and naively retrying against one makes things
worse by spending retry attempts on a wait that won't resolve for hours.
The practical mitigation was choosing the Flash-**Lite** tier specifically
(higher documented free RPD than plain Flash), not the retry wrapper.
Running the full 40-question gold-set eval end to end may still need to be
paced across more than one calendar day, or accept that RAGAS's judge calls
and the checklist agent's multi-call-per-question pattern are the most
quota-hungry paths to budget around first.

Free-tier model quality/instruction-following may also trail Claude Sonnet
or GPT-4-class models on the hardest multi-hop questions in the gold set;
if that shows up as a real ceiling in `docs/eval/report_final.md`,
switching `LLM_PROVIDER=anthropic` for the `quality` tier only (keeping
Gemini for the cheap tier) is a one-line config change, not an architecture
change.

`langchain-google-genai` is pinned to `2.0.7` rather than the current
`4.x` line: `4.x` requires a newer `langchain-core` than this project's
pinned `langchain==0.3.13`/`langgraph==0.2.60`, and `2.0.7` still works
correctly against the live API (verified against real Gemini calls). It
does emit a `FutureWarning` that the underlying `google.generativeai`
SDK is sunset in favor of `google.genai` — functionally harmless today,
but a real future-maintenance item: upgrading past it means bumping the
whole LangChain/LangGraph stack together, not just this one package.
