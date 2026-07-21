import json
import sqlite3

from sofia_harness.archive_index import (
    build_archive_index,
    hybrid_search,
    lexical_search,
    make_passages,
    normalize_for_search,
    semantic_search,
)


def test_normalization_preserves_original_semantics_and_removes_layout_wraps():
    assert normalize_for_search("министър-пред-\nседател\n\nв София") == "министър-председател\n\nв София"


def test_short_article_is_one_passage():
    passages = make_passages("Кратък текст.")
    assert len(passages) == 1 and passages[0].text == "Кратък текст."


def test_long_article_uses_paragraph_aware_passages():
    paragraphs = [f"Абзац {i}. " + ("исторически текст " * 120) for i in range(8)]
    text = "\n\n".join(paragraphs)
    passages = make_passages(text, minimum_tokens=100, maximum_tokens=250)
    assert len(passages) > 1
    assert all(p.estimated_tokens <= 250 for p in passages)
    assert passages[0].start == 0 and passages[-1].end == len(text.rstrip())


def test_builds_jsonl_db_passages_and_deduplicates_source_hash(tmp_path):
    source = tmp_path / "digitized"
    first = source / "first"; second = source / "second"
    first.mkdir(parents=True); second.mkdir()
    payload = {
        "schema_version": 1, "workflow": "incremental-full-page", "source": "IMG_1_nm_07_04_1949_page1.png",
        "source_sha256": "abc123", "articles": [{
            "article_id": "news", "article_order": 1, "heading": "Новина",
            "verbatim_text": "Министър-пред-\nседателят пристигна в София.", "confidence": .9,
            "uncertainties": [],
        }],
    }
    (first / "page.articles.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    payload["articles"].append({"article_id": "more", "heading": "Още", "verbatim_text": "Друг текст."})
    (second / "page.articles.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    workspace = tmp_path / "search_data"

    result = build_archive_index(source, workspace)
    assert result == {**result, "pages": 1, "articles": 2, "passages": 2, "embedded_passages": 0}
    rows = [json.loads(line) for line in (workspace / "normalized/articles.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["metadata"]["publication"] == "Народна младеж"
    assert "министър-председателят" in rows[0]["normalized_text"].casefold()
    with sqlite3.connect(workspace / "index/archive.db") as db:
        assert db.execute("SELECT count(*) FROM pages").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM passages").fetchone()[0] == 2
    assert lexical_search(workspace / "index/archive.db", "София")[0]["heading"] == "Новина"


class FakeEmbeddings:
    def create(self, *, model, input, encoding_format):
        assert model == "test-embedding" and encoding_format == "float"
        return type("Response", (), {"data": [type("Item", (), {"embedding": [float(i), 1.0]}) for i, _ in enumerate(input)]})


class FakeClient:
    embeddings = FakeEmbeddings()


def test_embedding_vectors_are_cached_in_sqlite(tmp_path):
    source = tmp_path / "digitized"; page = source / "page"; page.mkdir(parents=True)
    payload = {"source": "scan.jpg", "source_sha256": "hash", "articles": [
        {"article_id": "a", "heading": "A", "verbatim_text": "Text for embedding."}
    ]}
    (page / "page.articles.json").write_text(json.dumps(payload), encoding="utf-8")
    workspace = tmp_path / "search_data"
    result = build_archive_index(source, workspace, embed=True, embedding_model="test-embedding", client=FakeClient())
    assert result["embedded_passages"] == 1
    with sqlite3.connect(workspace / "index/archive.db") as db:
        model, vector = db.execute("SELECT embedding_model, embedding_json FROM passages").fetchone()
    assert model == "test-embedding" and json.loads(vector) == [0.0, 1.0]


class MeaningEmbeddings:
    def create(self, *, model, input, encoding_format):
        vectors=[]
        for text in input:
            value=text.casefold()
            vectors.append([0.0,1.0] if "automobile" in value or "vehicle" in value else [1.0,0.0])
        return type("Response",(),{"data":[type("Item",(),{"embedding":vector}) for vector in vectors]})


class MeaningClient:
    embeddings=MeaningEmbeddings()


def test_hybrid_search_adds_semantic_matches_to_keyword_results(tmp_path):
    source=tmp_path/"digitized";page=source/"page";page.mkdir(parents=True)
    payload={"source":"scan.jpg","source_sha256":"hybrid","articles":[
        {"article_id":"literal","heading":"Literal","verbatim_text":"A train arrived at the station."},
        {"article_id":"meaning","heading":"Meaning","verbatim_text":"An automobile crossed the city."},
    ]}
    (page/"page.articles.json").write_text(json.dumps(payload),encoding="utf-8")
    workspace=tmp_path/"search_data"
    build_archive_index(source,workspace,embed=True,embedding_model="meaning-model",client=MeaningClient())
    semantic=semantic_search(workspace/"index/archive.db","vehicle",client=MeaningClient())
    assert semantic[0]["heading"]=="Meaning"
    hybrid=hybrid_search(workspace/"index/archive.db","train vehicle",client=MeaningClient())
    assert {row["heading"] for row in hybrid[:2]}=={"Literal","Meaning"}
    assert any("semantic" in row["match_type"] for row in hybrid)


def test_natural_language_query_removes_stop_words_and_expands_death_term(tmp_path):
    source=tmp_path/"digitized";page=source/"page";page.mkdir(parents=True)
    payload={"source":"scan.jpg","source_sha256":"query","articles":[
        {"article_id":"a","heading":"Biography","verbatim_text":"Георги Димитров почина след боледуване."}
    ]}
    (page/"page.articles.json").write_text(json.dumps(payload,ensure_ascii=False),encoding="utf-8")
    workspace=tmp_path/"search_data";build_archive_index(source,workspace)
    assert lexical_search(workspace/"index/archive.db","На колко години умира Георги Димитров?")[0]["heading"]=="Biography"
    assert lexical_search(workspace/"index/archive.db","how old did georgi dimitrov die")[0]["heading"]=="Biography"
