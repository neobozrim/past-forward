# Search data

This directory contains rebuildable search artifacts. `digitized/` remains the archival source of truth.

## Files

- `normalized/articles.jsonl`: generated normalized articles; one JSON object per line.
- `index/archive.db`: SQLite pages, articles, retrieval passages, FTS, and optional embeddings.
- `cache/claims.jsonl`: evidence-backed extracted claims cached by the research layer.
- `evaluation/questions.jsonl`: evaluation questions supplied by researchers.

## Evaluation question contract

Only `question_id` and `question` are required. All other fields are optional.

```json
{"question_id":"q-001","question":"Кои от тези министър-председатели са учили в чужбина?","expected_answer":null,"expected_article_ids":[],"tags":["comparison","education"],"notes":null}
```

## Claim cache contract

The cache starts empty and is appended to only when a research query extracts a supported claim.

```json
{"claim_id":"...","subject":"...","predicate":"studied_at","object":"...","confidence":0.9,"explicit":true,"article_id":"...","passage_id":"...","evidence_text":"...","extractor":"...","created_at":"..."}
```

Absence from the cache never means a claim is false. The cache is an optimization, not an authority.

## Build

Local indexing without paid API calls:

```powershell
python -m sofia_harness.archive_index build
```

Add embeddings explicitly:

```powershell
python -m sofia_harness.archive_index build --embed
```

The embedding command requires `OPENAI_API_KEY`. It uses `text-embedding-3-small` by default and stores vectors against internal passages in `archive.db`.
