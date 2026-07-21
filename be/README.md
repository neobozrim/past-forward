# Past Forward

Making records from the past available in the future.

Past Forward converts photographs and scans of historical newspapers and dossiers into article-aware diplomatic transcriptions. The active workflow is intentionally lightweight:

1. Sol reads the untouched full page, identifies one semantic article, transcribes it directly from pixels, and checkpoints it immediately.
2. Sol continues article by article. A fresh context receives only the original page and saved article anchors—not prior body text—after several turns or after a failed request.
3. Terra independently inventories the untouched page and checks article presence, ownership, and order. Terra does not receive or transcribe article bodies. Its comparison pass rereads and returns a body-free visual locator for every saved article, including separately signed children inside an umbrella feature.
4. Each concrete Terra finding is sent to a fresh Sol vision request for targeted recovery or repair.
5. After inventory passes, a fresh Sol B request uses Terra's independent locator to reread every article from the source pixels without seeing Sol A's heading, source anchor, body text, confidence, or uncertainty hints. Up to three article reads run concurrently.
6. Insignificant horizontal spacing differences are ignored, but line and paragraph boundaries remain diplomatic evidence. Every heading, wording, punctuation, line break, hyphenation, omission, or uncertainty disagreement is sent to another fresh Sol request with the source image and both readings for pixel-based adjudication.
7. An adjudicator cannot directly overwrite text. A final fresh Sol request sees only each exact proposed before/after edit and the source page, verifies the glyph evidence, and selects reader A or the adjudicated candidate. Code applies only those selections, so this verifier cannot rewrite or omit the rest of the article.
8. Each accepted repair is checkpointed immediately. A failed, incomplete, or low-confidence verification leaves the primary transcription intact and prevents the page from being marked complete. The store rejects internal truncation markers, incomplete Responses API results, duplicate source anchors, and long verbatim overlap at article boundaries. During the existing blind A/B pass it also recognizes abrupt endings and substantial suffixes found only by reader B, streams `Possible truncation detected`, and routes the article through the normal source adjudication/fix path—without adding another model call or layout stage. The confidence floor is configurable and defaults to `0.80`.
9. The source image, diplomatic Markdown, article JSON, state, events, model attempts, failures, blind reads, comparisons, adjudications, source-change decisions, and audits remain together in one filesystem folder.

There is no Tesseract, Label Studio, mandatory polygon stage, page-wide layout plan, QC rejection gate, SQLite database, or vector index in this workflow. The source scan is never overwritten. Genuine uncertainty is marked instead of silently replaced with plausible prose.

## Start the UI

```powershell
Copy-Item .env.example .env
# Put your key in .env and set PAST_FORWARD_API_ENABLED=true when you want paid API execution.
python -m pip install -e ".[dev]"
python -m sofia_harness.system_cli serve --port 8000
```

Open `http://127.0.0.1:8000`.

Attach or drag one or more images into the chat and state the scope, for example `digitise the complete page` or `digitise only the article headed ...`. A file without a clear instruction causes the agent to ask one clarification. Status events stream while work is running. Each completed article survives later timeouts and an interrupted incremental run can resume from its saved anchors.

`PAST_FORWARD_API_ENABLED=true` enables paid model execution for authenticated administrators. Set it to `false` to leave the UI and Library available while disabling digitization calls.

## Output

Each page is saved under:

```text
digitized/<YYYYMMDD-HHMM>_<source-name>_incremental/
├── <YYYYMMDD-HHMM>_<source-file>
├── <YYYYMMDD-HHMM>_<source-name>.md
├── <YYYYMMDD-HHMM>_<source-name>.articles.json
└── audit/
    ├── incremental-state.json
    └── reads/
        ├── <article-id>.sol-b.json
        ├── <article-id>.sol-adjudication.json
        └── <article-id>.sol-change-verification.json
```

Markdown removes a duplicated leading heading for readability. The article JSON retains the full diplomatic model record. Files are written atomically after every article. No database ingest or vectorization occurs.

## Verification

```powershell
python -m pytest -q
```

Mocked tests validate the tool loop, atomic checkpoints, fresh-context isolation, blind Terra inputs, targeted Sol recovery, blind Sol B isolation, automatic A/B adjudication, source-only edit selection, verifier failure retention, truncation/ownership integrity gates, timeout retention, resume behavior, API routing, and UI contracts. Production acceptance additionally requires complete end-to-end inspection on real pages from `train-20260717T214057Z-1-001/train`; mocks alone are not treated as OCR evidence.
