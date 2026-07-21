# End-to-end audit report

Date: 2026-07-18

## Scope exercised

- All 28 supplied scans through deterministic QC.
- Two representative pages through agent-driven layout, reversible crops, and manual visual OCR.
- Immutable manifest ingestion into a new SQLite database.
- Region and article reconstruction, normalized text, FTS5, review queue, entity assertions, jobs, and interface memory.
- FastAPI through its real HTTP boundary and the browser UI through Playwright.
- Original scan and enhanced crop delivery, Bulgarian search, evidence navigation, review submission, correction persistence, article rebuild, FTS refresh, graph reindex, and process restart.

No paid OpenAI API call was made. “Actual OCR” in this audit means text read from the real supplied scan/crop by the driving agent and then verified visually. The live Luna/Terra/Sol adapter remains covered by structured-response and failure-path tests, not a live credentialed call.

## Defects found and fixed

1. Pending crop transcriptions were not creating review records. Nine pending regions now create nine review tasks during ingestion.
2. Manual pages received unstable `:0` and `:1` IDs. IDs now derive from source scan stems.
3. Search provenance was inaccessible from the UI. Results now open the stored source scan and page-region count.
4. A greedy page route intercepted image requests. Page images now use a separate `/api/page-images/` namespace and are verified by decoded dimensions.
5. Crop images were not exposed to reviewers. A provenance-backed region image route now serves the selected derivative.
6. Review corrections updated regions but left reconstructed articles and FTS stale. Corrections now rebuild articles, replace FTS entries, and refresh graph assertions transactionally.
7. Empty corrections could erase OCR. API validation and frontend validation now reject empty submissions.
8. Review cards made narrow newspaper columns too small. The reviewer now uses a responsive two-pane crop/transcription layout.
9. Raw JSON reasons and absolute source paths leaked into the UI. Reasons are humanized and only source filenames are shown.
10. Quoted all-caps Bulgarian names were absent from the graph. A conservative quoted-uppercase rule now extracts `ГЕОРГИ ДИМИТРОВ` with evidence offsets.
11. Extractor changes had no backfill path. The CLI now provides `reindex`.
12. `perspective_unassessed` was incorrectly a QC failure, sending every page to review. Unassessed capabilities are now separate from failure flags.
13. QC decoded and analyzed every scan at full resolution. Documented downsampled analysis reduced the 28-page pass from 482.7 s to 50.25 s (9.6×).
14. An OpenCV 5 Hough-line shape difference crashed real scans. The parser now accepts both flat and nested line arrays.
15. Windows CLI output failed on Bulgarian. The system CLI explicitly emits UTF-8.

## Final verified state

- Automated tests: 33 passing.
- Corpus QC: 28/28 processed; 11 pass, 17 review, 0 blocked.
- Pilot storage: 2 pages, 20 regions, 9 articles, 9 remaining open crop reviews.
- Verified OCR correction: the incorrect phrase `с книжарско съревнование` was replaced from the crop with `снижаване себестойността` in the complete headline.
- The corrected text is present in the region, reconstructed article, FTS result, API response, browser result, and after server restart; the obsolete phrase has zero FTS hits.
- Knowledge graph: one conservative entity, `ГЕОРГИ ДИМИТРОВ`, with exact evidence offsets and article/page provenance.
- Evidence scan decoded at 3024×4032; all nine review crops decoded successfully.
- Review UI rejected an empty correction and displayed the crop/transcription workspace correctly.

## Remaining improvements

1. Perform an explicitly authorized live 5–10 page run with Luna/Terra/Sol. Calibrate quality, retries, token telemetry, and review thresholds against locked librarian gold text.
2. Implement page-boundary/perspective detection and dewarping. It remains honestly marked unassessed.
3. Investigate the 17 edge-clipping flags; bound-volume edges may require collection-specific thresholds and sampled librarian adjudication.
4. Split large body regions into columns/paragraphs before human review. Several current crops are too large for efficient correction.
5. Add a draft/partial transcription state and uncertainty-span editing instead of requiring an all-or-nothing region correction.
6. Replace heuristic entity extraction with QA-gated structured extraction and authority reconciliation. Current graph output is intentionally conservative.
7. Add a true embedding adapter and hybrid reranker. Current retrieval is correctly labeled lexical FTS5.
8. Export ALTO XML, METS, and IIIF manifests and validate them against their schemas.
9. Add authentication, roles, CSRF protection, audit identity, and authorization before network deployment. The current service is localhost-only.
10. Move jobs to a production queue and object storage for multi-worker scaling; current SQLite jobs are suitable for a single-node pilot.
11. Add browser regression tests to CI using a standalone Playwright runner. This audit used the in-app Playwright surface interactively.
12. Resolve the Starlette TestClient deprecation warning by moving tests to the recommended newer client once the dependency ecosystem stabilizes.
