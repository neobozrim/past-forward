from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

from .metadata import metadata_from_filename
from .tracing import braintrust_logger

SEARCH_STOP_WORDS={"а","аз","в","във","да","до","за","и","как","каква","какви","какво","кой","кои","колко","къде","кога","ли","на","от","по","при","се","си","със","с","тези","този","то","тя","той",
                   "a","an","at","did","do","does","how","in","is","of","on","the","to","was","were","what","when","where","which","who"}
SEARCH_EXPANSIONS={
    "умира":["почина","смърт"],"умрял":["почина","смърт"],"учил":["образование","университет"],"учили":["образование","университет"],
    "die":["почина","смърт","умира"],"died":["почина","смърт","умира"],"death":["смърт","почина"],
    "old":["години","роден"],"age":["години","роден"],"born":["роден"],"studied":["учил","образование","университет"],
    "sick":["болен","боледуване","заболяване"],"ill":["болен","боледуване","заболяване"],"illness":["болест","боледуване","заболяване"],
    "joking":["шега","сатира","стършел"],"joke":["шега","сатира","стършел"],"mocking":["подигравка","сатира","стършел"],
    "contemptuous":["презрение","презрително","сатира"],"attitude":["отношение"],"attitudes":["отношение"],
    "usa":["съединените","щати","вашингтон"],"america":["америка","вашингтон"],"american":["американски","вашингтон"],
    "foreign":["чуждестранни","международни"],"partners":["партньори","съюзници"],
}


def transliterate_latin_to_bg(value: str) -> str:
    pairs=("sht","щ"),("zh","ж"),("ch","ч"),("sh","ш"),("ts","ц"),("yu","ю"),("ya","я")
    value=value.casefold()
    for latin,cyrillic in pairs:value=value.replace(latin,cyrillic)
    table=str.maketrans("abvgdeziyklmnoprstufh", "абвгдезийклмнопрстуфх")
    return value.translate(table)


SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS index_runs(
    run_id TEXT PRIMARY KEY, built_at TEXT NOT NULL, source_root TEXT NOT NULL,
    embedding_model TEXT, article_count INTEGER NOT NULL, passage_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS pages(
    page_id TEXT PRIMARY KEY, source_sha256 TEXT NOT NULL UNIQUE, source_path TEXT NOT NULL,
    articles_path TEXT NOT NULL, markdown_path TEXT, workflow TEXT,
    publication TEXT, publication_code TEXT, issue_date TEXT, page_number INTEGER,
    metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS articles(
    article_id TEXT PRIMARY KEY, page_id TEXT NOT NULL REFERENCES pages(page_id),
    source_article_id TEXT NOT NULL, article_order INTEGER, heading TEXT,
    verbatim_text TEXT NOT NULL, normalized_text TEXT NOT NULL, confidence REAL,
    uncertainties_json TEXT NOT NULL, provenance_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS passages(
    passage_id TEXT PRIMARY KEY, article_id TEXT NOT NULL REFERENCES articles(article_id),
    passage_order INTEGER NOT NULL, normalized_text TEXT NOT NULL,
    start_offset INTEGER NOT NULL, end_offset INTEGER NOT NULL,
    estimated_tokens INTEGER NOT NULL, embedding_model TEXT, embedding_json TEXT,
    UNIQUE(article_id, passage_order)
);
CREATE VIRTUAL TABLE IF NOT EXISTS passage_fts USING fts5(
    passage_id UNINDEXED, heading, normalized_text, tokenize='unicode61 remove_diacritics 0'
);
"""


@dataclass(frozen=True)
class Passage:
    text: str
    start: int
    end: int
    estimated_tokens: int


def normalize_for_search(text: str) -> str:
    """Make a search derivative without altering the diplomatic transcription."""
    text = unicodedata.normalize("NFC", text or "").replace("\r\n", "\n").replace("\r", "\n")
    # A hyphen immediately followed by a printed line break is a layout wrap in
    # the current diplomatic contract. The original remains available verbatim.
    text = re.sub(r"(?<=\w)-[ \t]*(?:\n[ \t]*)+(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    # A conservative tokenizer-free estimate that works for Cyrillic and Latin.
    # The embedding endpoint enforces the actual model limit.
    return max(1, math.ceil(len(text) / 3.2)) if text else 0


def _hard_split(text: str, start: int, maximum: int) -> list[tuple[str, int, int]]:
    words = list(re.finditer(r"\S+", text))
    if not words:
        return []
    output, first = [], 0
    for index in range(1, len(words) + 1):
        end = words[index - 1].end()
        if estimate_tokens(text[words[first].start():end]) > maximum and index - 1 > first:
            previous_end = words[index - 2].end()
            output.append((text[words[first].start():previous_end], start + words[first].start(), start + previous_end))
            first = index - 1
    output.append((text[words[first].start():words[-1].end()], start + words[first].start(), start + words[-1].end()))
    return output


def make_passages(text: str, minimum_tokens: int = 500, maximum_tokens: int = 900) -> list[Passage]:
    """Pack paragraphs into retrieval windows while retaining article offsets."""
    if not text.strip():
        return []
    if estimate_tokens(text) <= maximum_tokens:
        start = len(text) - len(text.lstrip())
        end = len(text.rstrip())
        clean = text[start:end]
        return [Passage(clean, start, end, estimate_tokens(clean))]

    units: list[tuple[str, int, int]] = []
    for match in re.finditer(r"(?:\A|\n\n)(.*?)(?=\n\n|\Z)", text, flags=re.DOTALL):
        raw = match.group(1)
        leading = len(raw) - len(raw.lstrip())
        paragraph = raw.strip()
        paragraph_start = match.start(1) + leading
        paragraph_end = paragraph_start + len(paragraph)
        if not paragraph:
            continue
        if estimate_tokens(paragraph) > maximum_tokens:
            units.extend(_hard_split(paragraph, paragraph_start, maximum_tokens))
        else:
            units.append((paragraph, paragraph_start, paragraph_end))

    groups: list[list[tuple[str, int, int]]] = []
    current: list[tuple[str, int, int]] = []
    for unit in units:
        candidate = "\n\n".join([item[0] for item in current] + [unit[0]])
        if current and estimate_tokens(candidate) > maximum_tokens:
            groups.append(current)
            current = [unit]
        else:
            current.append(unit)
    if current:
        groups.append(current)

    # Avoid a tiny final fragment when it fits safely in the preceding window.
    if len(groups) > 1:
        tail_text = "\n\n".join(item[0] for item in groups[-1])
        merged_text = "\n\n".join(item[0] for group in groups[-2:] for item in group)
        if estimate_tokens(tail_text) < minimum_tokens and estimate_tokens(merged_text) <= maximum_tokens:
            groups[-2].extend(groups.pop())

    passages = []
    for group in groups:
        value = "\n\n".join(item[0] for item in group)
        passages.append(Passage(value, group[0][1], group[-1][2], estimate_tokens(value)))
    return passages


def _article_text(article: dict) -> str:
    return article.get("verbatim_text") or article.get("text") or ""


def _heading(article: dict) -> str:
    return article.get("heading") or article.get("label") or ""


def _candidate_score(path: Path, payload: dict) -> tuple[int, int, float, float]:
    articles = [a for a in payload.get("articles", []) if _article_text(a).strip()]
    chars = sum(len(_article_text(a)) for a in articles)
    confidence = sum(float(a.get("confidence") or 0) for a in articles)
    return len(articles), chars, confidence, path.stat().st_mtime


def discover_pages(source_root: str | Path) -> list[tuple[Path, dict]]:
    """Choose the strongest transcription artifact for each immutable source hash."""
    selected: dict[str, tuple[Path, dict]] = {}
    for path in Path(source_root).rglob("*.articles.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source_hash = payload.get("source_sha256")
        if not source_hash:
            continue
        current = selected.get(source_hash)
        if current is None or _candidate_score(path, payload) > _candidate_score(*current):
            selected[source_hash] = (path, payload)
    return sorted(selected.values(), key=lambda item: str(item[0]).casefold())


def _atomic_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _embedding_batches(client, passages: list[tuple[str, str]], model: str, batch_size: int = 64):
    for offset in range(0, len(passages), batch_size):
        batch = passages[offset:offset + batch_size]
        response = client.embeddings.create(model=model, input=[text for _, text in batch], encoding_format="float")
        for (passage_id, _), item in zip(batch, response.data, strict=True):
            yield passage_id, item.embedding


def build_archive_index(source_root: str | Path = "digitized", workspace: str | Path = "search_data",
                        embed: bool = False, embedding_model: str = "text-embedding-3-small",
                        client=None) -> dict:
    workspace = Path(workspace)
    normalized_path = workspace / "normalized" / "articles.jsonl"
    db_path = workspace / "index" / "archive.db"
    claims_path = workspace / "cache" / "claims.jsonl"
    questions_path = workspace / "evaluation" / "questions.jsonl"
    for path in (claims_path, questions_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    pages = discover_pages(source_root)
    article_rows, passage_rows, page_rows = [], [], []
    for articles_path, payload in pages:
        source_hash = payload["source_sha256"]
        page_id = f"page:{source_hash}"
        source_name = payload.get("source", "")
        metadata = {**metadata_from_filename(source_name), **(payload.get("metadata") or {})}
        source_path = articles_path.parent / source_name
        markdowns = list(articles_path.parent.glob("*.md"))
        page_rows.append({
            "page_id": page_id, "source_sha256": source_hash,
            "source_path": str(source_path.resolve()), "articles_path": str(articles_path.resolve()),
            "markdown_path": str(markdowns[0].resolve()) if markdowns else None,
            "workflow": payload.get("workflow"), "metadata": metadata,
        })
        for article in payload.get("articles", []):
            verbatim = _article_text(article).strip()
            if not verbatim:
                continue
            source_article_id = str(article.get("article_id") or f"article-{len(article_rows) + 1}")
            article_id = f"{page_id}:{source_article_id}"
            normalized = normalize_for_search(verbatim)
            regions = article.get("regions") or []
            provenance = {
                "articles_path": str(articles_path.resolve()), "source_path": str(source_path.resolve()),
                "region_ids": [r.get("region_id") for r in regions if r.get("region_id")],
            }
            row = {
                "article_id": article_id, "page_id": page_id, "source_article_id": source_article_id,
                "article_order": article.get("article_order"), "heading": _heading(article),
                "verbatim_text": verbatim, "normalized_text": normalized,
                "confidence": article.get("confidence"),
                "uncertainties": article.get("uncertainties") or article.get("uncertain_region_ids") or [],
                "metadata": metadata, "provenance": provenance,
            }
            article_rows.append(row)
            for order, passage in enumerate(make_passages(normalized), start=1):
                passage_rows.append({
                    "passage_id": f"{article_id}:passage:{order}", "article_id": article_id,
                    "passage_order": order, "normalized_text": passage.text,
                    "start_offset": passage.start, "end_offset": passage.end,
                    "estimated_tokens": passage.estimated_tokens,
                })

    _atomic_jsonl(normalized_path, article_rows)
    with sqlite3.connect(db_path) as db:
        db.executescript(SCHEMA)
        db.execute("DELETE FROM passage_fts")
        for table in ("passages", "articles", "pages", "index_runs"):
            db.execute(f"DELETE FROM {table}")
        for page in page_rows:
            metadata = page["metadata"]
            db.execute("INSERT INTO pages VALUES(?,?,?,?,?,?,?,?,?,?,?)", (
                page["page_id"], page["source_sha256"], page["source_path"], page["articles_path"],
                page["markdown_path"], page["workflow"], metadata.get("publication"),
                metadata.get("publication_code"), metadata.get("issue_date"), metadata.get("page_number"),
                json.dumps(metadata, ensure_ascii=False),
            ))
        headings = {}
        for article in article_rows:
            headings[article["article_id"]] = article["heading"]
            db.execute("INSERT INTO articles VALUES(?,?,?,?,?,?,?,?,?,?)", (
                article["article_id"], article["page_id"], article["source_article_id"],
                article["article_order"], article["heading"], article["verbatim_text"],
                article["normalized_text"], article["confidence"],
                json.dumps(article["uncertainties"], ensure_ascii=False),
                json.dumps(article["provenance"], ensure_ascii=False),
            ))
        for passage in passage_rows:
            db.execute("INSERT INTO passages VALUES(?,?,?,?,?,?,?,?,?)", (
                passage["passage_id"], passage["article_id"], passage["passage_order"],
                passage["normalized_text"], passage["start_offset"], passage["end_offset"],
                passage["estimated_tokens"], None, None,
            ))
            db.execute("INSERT INTO passage_fts VALUES(?,?,?)", (
                passage["passage_id"], headings[passage["article_id"]], passage["normalized_text"],
            ))

        embedded = 0
        if embed and passage_rows:
            if client is None:
                load_dotenv()
                if not os.getenv("OPENAI_API_KEY"):
                    raise RuntimeError("OPENAI_API_KEY is required with --embed")
                from openai import OpenAI
                client = OpenAI()
            inputs = [(row["passage_id"], row["normalized_text"]) for row in passage_rows]
            for passage_id, vector in _embedding_batches(client, inputs, embedding_model):
                db.execute("UPDATE passages SET embedding_model=?, embedding_json=? WHERE passage_id=?",
                           (embedding_model, json.dumps(vector, separators=(",", ":")), passage_id))
                embedded += 1
        run_id = hashlib.sha256(f"{datetime.now(timezone.utc).isoformat()}:{len(article_rows)}".encode()).hexdigest()[:16]
        db.execute("INSERT INTO index_runs VALUES(?,?,?,?,?,?)", (
            run_id, datetime.now(timezone.utc).isoformat(), str(Path(source_root).resolve()),
            embedding_model if embed else None, len(article_rows), len(passage_rows),
        ))

    return {
        "pages": len(page_rows), "articles": len(article_rows), "passages": len(passage_rows),
        "embedded_passages": embedded, "normalized_path": str(normalized_path.resolve()),
        "database_path": str(db_path.resolve()), "claims_path": str(claims_path.resolve()),
        "questions_path": str(questions_path.resolve()),
    }


def _lexical_search(db_path: str | Path, query: str, limit: int = 10) -> list[dict]:
    raw_terms=re.findall(r"\w+", query.casefold(), flags=re.UNICODE)
    meaningful=[term for term in raw_terms if term not in SEARCH_STOP_WORDS and len(term)>1]
    terms=list(dict.fromkeys(meaningful or raw_terms))
    for term in tuple(terms):terms.extend(value for value in SEARCH_EXPANSIONS.get(term,[]) if value not in terms)
    for term in tuple(terms):
        if term.isascii() and term.isalpha() and term not in SEARCH_EXPANSIONS:
            transliterated=transliterate_latin_to_bg(term)
            if transliterated!=term and transliterated not in terms:terms.append(transliterated)
    if not terms:
        return []
    fts_query = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute("""
            SELECT p.passage_id, p.article_id, p.normalized_text AS relevant_excerpt,
                   a.heading, a.verbatim_text, pg.publication, pg.issue_date, pg.page_number,
                   pg.source_path, bm25(passage_fts) AS score
            FROM passage_fts
            JOIN passages p USING(passage_id)
            JOIN articles a ON a.article_id=p.article_id
            JOIN pages pg ON pg.page_id=a.page_id
            WHERE passage_fts MATCH ? ORDER BY score LIMIT ?
        """, (fts_query, limit)).fetchall()
        return [dict(row) for row in rows]


def lexical_search(db_path: str | Path, query: str, limit: int = 10) -> list[dict]:
    logger = braintrust_logger()
    if logger is None:
        return _lexical_search(db_path, query, limit)

    with logger.start_span(
        name="archive_search",
        type="task",
        input={"query": query, "limit": limit},
        metadata={"index": str(db_path)},
    ) as span:
        results = _lexical_search(db_path, query, limit)
        span.log(output=results, metadata={"result_count": len(results), "strategy": "fts5"})
        return results


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return -1.0
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return numerator / denominator if denominator else -1.0


def semantic_search(db_path: str | Path, query: str, limit: int = 10, client=None) -> list[dict]:
    """Embed a query and rank the cached passage vectors by cosine similarity."""
    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        model_row = db.execute("""
            SELECT embedding_model FROM passages
            WHERE embedding_model IS NOT NULL AND embedding_json IS NOT NULL
            GROUP BY embedding_model ORDER BY count(*) DESC LIMIT 1
        """).fetchone()
        if model_row is None:
            return []
        model = model_row["embedding_model"]
        rows = db.execute("""
            SELECT p.passage_id, p.article_id, p.normalized_text AS relevant_excerpt,
                   p.embedding_json, a.heading, a.verbatim_text, pg.publication,
                   pg.issue_date, pg.page_number, pg.source_path
            FROM passages p JOIN articles a ON a.article_id=p.article_id
            JOIN pages pg ON pg.page_id=a.page_id
            WHERE p.embedding_model=? AND p.embedding_json IS NOT NULL
        """, (model,)).fetchall()
    if client is None:
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            return []
        from openai import OpenAI
        client = OpenAI(timeout=30, max_retries=1)
    response = client.embeddings.create(model=model, input=[normalize_for_search(query)], encoding_format="float")
    query_vector = response.data[0].embedding
    ranked = []
    for row in rows:
        item = dict(row)
        item["score"] = _cosine_similarity(query_vector, json.loads(item.pop("embedding_json")))
        ranked.append(item)
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:limit]


def hybrid_search(db_path: str | Path, query: str, limit: int = 10, client=None) -> list[dict]:
    """Fuse keyword and semantic rankings with reciprocal-rank fusion."""
    candidate_limit = max(limit * 3, 20)
    lexical = lexical_search(db_path, query, candidate_limit)
    try:
        semantic = semantic_search(db_path, query, candidate_limit, client=client)
    except Exception:
        semantic = []
    combined: dict[str, dict] = {}
    for strategy, rows in (("lexical", lexical), ("semantic", semantic)):
        for rank, row in enumerate(rows, start=1):
            passage_id = row["passage_id"]
            entry = combined.setdefault(passage_id, {**row, "hybrid_score": 0.0, "match_strategies": []})
            entry["hybrid_score"] += 1.0 / (60 + rank)
            entry["match_strategies"].append(strategy)
            if strategy == "semantic":
                entry["semantic_score"] = row["score"]
            else:
                entry["lexical_score"] = row["score"]
    ranked = sorted(combined.values(), key=lambda item: item["hybrid_score"], reverse=True)
    for item in ranked:
        item["score"] = item.pop("hybrid_score")
        item["match_type"] = "+".join(item.pop("match_strategies"))
    return ranked[:limit]


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Build and inspect the Past Forward search index")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--source", default="digitized")
    build.add_argument("--workspace", default="search_data")
    build.add_argument("--embed", action="store_true", help="Make paid OpenAI embedding API calls")
    build.add_argument("--embedding-model", default="text-embedding-3-small")
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--db", default="search_data/index/archive.db")
    search.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build_archive_index(args.source, args.workspace, args.embed, args.embedding_model)
    else:
        result = lexical_search(args.db, args.query, args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
