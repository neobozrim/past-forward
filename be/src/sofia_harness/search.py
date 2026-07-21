from __future__ import annotations
from .store import Store
from .tracing import braintrust_logger

def search(store:Store,query:str,limit=10):
    logger=braintrust_logger()

    def execute():
        with store.connect() as db:
            try:
                rows=db.execute("SELECT a.*, bm25(article_fts) score FROM article_fts JOIN articles a USING(article_id) WHERE article_fts MATCH ? ORDER BY score LIMIT ?",(query,limit)).fetchall()
                strategy="fts5"
            except Exception:
                rows=db.execute("SELECT *, 0 score FROM articles WHERE normalized_text LIKE ? OR title LIKE ? LIMIT ?",(f"%{query}%",f"%{query}%",limit)).fetchall()
                strategy="like_fallback"
            return [dict(r) for r in rows],strategy

    if logger is None:
        results,_=execute()
        return results

    with logger.start_span(name="archive_search", type="task", input={"query":query,"limit":limit}) as span:
        results,strategy=execute()
        span.log(output=results,metadata={"strategy":strategy,"result_count":len(results)})
        return results

def entity_neighborhood(store:Store,entity_id:str):
    with store.connect() as db:
        entity=db.execute("SELECT * FROM entities WHERE entity_id=?",(entity_id,)).fetchone()
        assertions=db.execute("SELECT * FROM assertions WHERE subject_id=?",(entity_id,)).fetchall()
        return {"entity":dict(entity) if entity else None,"assertions":[dict(a) for a in assertions]}
