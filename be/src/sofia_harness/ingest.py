from __future__ import annotations
import hashlib,json,re
from datetime import datetime,timezone
from pathlib import Path
from .store import Store
from .text import extract_entities, normalize_search, reconstruct_articles

def _slug(value):return re.sub(r"[^a-z0-9а-яѣ]+","-",value.casefold()).strip("-")
def ingest_manifest(store:Store,path:str|Path,publish:bool=True):
    path=Path(path); data=json.loads(path.read_text(encoding="utf-8")); store.init(); now=datetime.now(timezone.utc).isoformat()
    with store.connect() as db:
        if db.execute("SELECT 1 FROM runs WHERE run_id=?",(data["run_id"],)).fetchone():return {"status":"already_ingested","run_id":data["run_id"]}
        db.execute("INSERT INTO runs VALUES(?,?,?,?,?,?)",(data["run_id"],str(path.resolve()),data.get("schema_version"),data.get("status","manual"),data.get("created_at"),now))
        for page_index,page in enumerate(data.get("pages",[])):
            source=page.get("source") or page.get("qc",{}); stable_page=page.get("page_id") or Path(source.get("path",f"page-{page_index}")).stem
            page_id=f"{data['run_id']}:{stable_page}"
            db.execute("INSERT INTO pages VALUES(?,?,?,?,?,?,?,?)",(page_id,data["run_id"],source.get("path",""),source.get("sha256",""),source.get("width"),source.get("height"),source.get("qc_status"),json.dumps(source,ensure_ascii=False)))
            if publish: db.execute("INSERT INTO page_publications VALUES(?,?,?)",(page_id,now,"system-ingest"))
            full_read=page.get("full_page_read")
            if full_read and full_read.get("verbatim_text"):
                db.execute("INSERT INTO page_transcriptions VALUES(?,?,?,?,?)",(page_id,full_read["verbatim_text"],full_read.get("confidence"),"original_full_page",json.dumps({"manifest":str(path.resolve()),"read":"full_page_read"},ensure_ascii=False)))
            page={**page,"page_id":page_id}
            for item in page.get("regions",[]):
                region=item.get("region",item); region_id=f"{page_id}:{region['id']}"; article=region.get("article_id"); article_id=f"{page_id}:{article}" if article else None
                text=item.get("accepted_text") or item.get("text") or (item.get("read_a") or {}).get("verbatim_text")
                db.execute("INSERT INTO regions VALUES(?,?,?,?,?,?,?,?,?,?,?)",(region_id,page_id,article_id,region.get("type"),region.get("reading_order"),text,normalize_search(text or ""),region.get("confidence"),item.get("status","automatic"),json.dumps(region.get("polygon",[])),json.dumps(item.get("provenance",{}),ensure_ascii=False)))
                if item.get("status") == "needs_crop_transcription":
                    db.execute("INSERT INTO reviews(page_id,region_id,reasons_json,status) VALUES(?,?,?,'open')",(page_id,region_id,json.dumps(["needs_crop_transcription"])))
            for article in reconstruct_articles(page):
                aid=f"{page_id}:{article['article_id']}" if not article["article_id"].startswith(page_id) else article["article_id"]
                provenance={"manifest":str(path.resolve()),"page_id":page_id,"region_ids":article["region_ids"]}
                db.execute("INSERT INTO articles VALUES(?,?,?,?,?,?,?,?)",(aid,page_id,0,article["title"],article["verbatim_text"],article["normalized_text"],article["confidence"],json.dumps(provenance,ensure_ascii=False)))
                if publish: db.execute("INSERT INTO article_fts VALUES(?,?,?)",(aid,article["title"] or "",article["normalized_text"]))
                for kind,name,start,end in extract_entities(article["verbatim_text"]):
                    eid=f"{kind}:{_slug(name)}"; db.execute("INSERT OR IGNORE INTO entities VALUES(?,?,?, '[]')",(eid,kind,name))
                    db.execute("INSERT INTO assertions(subject_id,predicate,object_id,page_id,evidence,start_offset,end_offset,confidence,provenance_json) VALUES(?,?,?,?,?,?,?,?,?)",(eid,"mentioned_in",aid,page_id,name,start,end,.6,json.dumps(provenance,ensure_ascii=False)))
        for review in data.get("review_queue",[]):
            pid=f"{data['run_id']}:{review.get('page_id')}"; rid=f"{pid}:{review['region_id']}" if review.get("region_id") else None
            db.execute("INSERT INTO reviews(page_id,region_id,reasons_json,status) VALUES(?,?,?,'open')",(pid,rid,json.dumps(review.get("reasons",[]))))
    return {"status":"ingested","run_id":data["run_id"],**store.stats()}


def rebuild_article(db, article_id:str):
    rows=db.execute("SELECT * FROM regions WHERE article_id=? ORDER BY reading_order",(article_id,)).fetchall()
    if not rows:return
    title=next((r["verbatim_text"] for r in rows if r["type"]=="headline" and r["verbatim_text"]),None)
    verbatim="\n\n".join(r["verbatim_text"] for r in rows if r["verbatim_text"])
    normalized=normalize_search(verbatim); confidence=min((r["confidence"] for r in rows if r["confidence"] is not None),default=None)
    db.execute("UPDATE articles SET title=?,verbatim_text=?,normalized_text=?,confidence=? WHERE article_id=?",(title,verbatim,normalized,confidence,article_id))
    db.execute("DELETE FROM article_fts WHERE article_id=?",(article_id,))
    if db.execute("SELECT 1 FROM page_publications WHERE page_id=?",(rows[0]["page_id"],)).fetchone():
        db.execute("INSERT INTO article_fts VALUES(?,?,?)",(article_id,title or "",normalized))
    db.execute("DELETE FROM assertions WHERE object_id=?",(article_id,))
    page_id=rows[0]["page_id"]
    for kind,name,start,end in extract_entities(verbatim):
        eid=f"{kind}:{_slug(name)}";db.execute("INSERT OR IGNORE INTO entities VALUES(?,?,?, '[]')",(eid,kind,name))
        db.execute("INSERT INTO assertions(subject_id,predicate,object_id,page_id,evidence,start_offset,end_offset,confidence,provenance_json) VALUES(?,?,?,?,?,?,?,?,?)",(eid,"mentioned_in",article_id,page_id,name,start,end,.6,json.dumps({"source":"human_corrected_article"})))


def reindex_all(store:Store):
    with store.connect() as db:
        article_ids=[r[0] for r in db.execute("SELECT article_id FROM articles")]
        for article_id in article_ids: rebuild_article(db,article_id)
        pending=db.execute("SELECT region_id,page_id FROM regions WHERE status='needs_crop_transcription'").fetchall()
        for region_id,page_id in pending:
            if not db.execute("SELECT 1 FROM reviews WHERE region_id=? AND status='open'",(region_id,)).fetchone():
                db.execute("INSERT INTO reviews(page_id,region_id,reasons_json,status) VALUES(?,?,?,'open')",(page_id,region_id,json.dumps(["needs_crop_transcription"])))
    return {"status":"reindexed",**store.stats()}
