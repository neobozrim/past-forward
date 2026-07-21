from __future__ import annotations
import hashlib,json,os,shutil,uuid
from datetime import datetime,timezone
from pathlib import Path
from fastapi import BackgroundTasks,FastAPI,File,HTTPException,UploadFile
from fastapi.responses import FileResponse,HTMLResponse
from pydantic import BaseModel,Field
from dotenv import load_dotenv
from .search import entity_neighborhood,search
from .store import Store
from .text import normalize_search
from .ingest import ingest_manifest,rebuild_article
from .runner import run as run_pipeline
from .scans import discover_scans,inspect_scan
from .ui import HTML as UI_HTML
import yaml

load_dotenv()

class Correction(BaseModel): correction:str=Field(min_length=1); reviewer:str=Field(min_length=1)
class RunRequest(BaseModel): mode:str="qc"
class BatchCreate(BaseModel): name:str=Field(min_length=1,max_length=120)
class ProcessBatch(BaseModel): mode:str="qc";scan_ids:list[str]=[]
class ScanSelection(BaseModel): selected:bool
class Approval(BaseModel): status:str;reviewer:str=Field(min_length=1);note:str=""
def create_app(db_path="data/sofia.db",config_path="config.yaml"):
    app=FastAPI(title="Sofia Library Research System",version="0.1.0"); store=Store(db_path); store.init()
    config_path=Path(config_path).resolve()
    def load_config():
        config=yaml.safe_load(config_path.read_text(encoding="utf-8"));config["_path"]=str(config_path);return config
    def execute_job(job_id:int,mode:str,batch_id:str|None=None,scan_ids:list[str]|None=None):
        try:
            with store.connect() as db:db.execute("UPDATE jobs SET status='running',attempts=attempts+1,updated_at=datetime('now') WHERE job_id=?",(job_id,))
            paths=None
            if batch_id:
                with store.connect() as db:
                    rows=db.execute(f"SELECT scan_id,source_path FROM batch_scans WHERE batch_id=? AND scan_id IN ({','.join('?'*len(scan_ids))})",[batch_id,*scan_ids]).fetchall()
                    paths=[r["source_path"] for r in rows]; by_path={str(Path(r["source_path"]).resolve()):r["scan_id"] for r in rows}
                    db.execute("UPDATE batches SET status='processing',mode=?,started_at=datetime('now') WHERE batch_id=?",(mode,batch_id))
                def progress(path,stage,status,detail):
                    sid=by_path.get(str(Path(path).resolve()))
                    if sid:
                        with store.connect() as db:
                            db.execute("INSERT INTO stage_events(scan_id,stage,status,detail_json,created_at) VALUES(?,?,?,?,datetime('now'))",(sid,stage,status,json.dumps(detail,ensure_ascii=False)))
                            db.execute("UPDATE batch_scans SET status=? WHERE scan_id=?",(stage if status=='complete' else f'{stage}_{status}',sid))
            else: progress=None
            out=run_pipeline(load_config(),dry_run=mode=="qc",scan_paths=paths,progress=progress);result=ingest_manifest(store,out/"manifest.json",publish=not bool(batch_id))
            if batch_id:
                with store.connect() as db:
                    run_id=result["run_id"]
                    for row in db.execute("SELECT page_id,source_path FROM pages WHERE run_id=?",(run_id,)):
                        sid=by_path.get(str(Path(row["source_path"]).resolve()))
                        if sid: db.execute("INSERT INTO batch_pages VALUES(?,?,?)",(batch_id,sid,row["page_id"]))
                    db.execute("UPDATE batches SET status='complete',completed_at=datetime('now') WHERE batch_id=?",(batch_id,))
            with store.connect() as db:db.execute("UPDATE jobs SET status='complete',payload_json=?,updated_at=datetime('now') WHERE job_id=?",(json.dumps({"mode":mode,"run_dir":str(out.resolve()),"ingest":result},ensure_ascii=False),job_id))
        except Exception as exc:
            with store.connect() as db:
                db.execute("UPDATE jobs SET status='failed',error=?,updated_at=datetime('now') WHERE job_id=?",(f"{type(exc).__name__}: {exc}",job_id))
                if batch_id: db.execute("UPDATE batches SET status='failed',completed_at=datetime('now') WHERE batch_id=?",(batch_id,))
    @app.get("/health")
    def health():return {"status":"ok","counts":store.stats()}
    @app.get("/api/search")
    def query(q:str,limit:int=10):return {"query":q,"results":search(store,q,min(limit,100))}
    @app.get("/api/operations/summary")
    def operations_summary():
        config=load_config();scans=discover_scans(config["dataset"]["images"],config["dataset"].get("patterns",["*.png"]))
        with store.connect() as db:
            jobs=[dict(x) for x in db.execute("SELECT * FROM jobs ORDER BY job_id DESC LIMIT 30")];runs=[dict(x) for x in db.execute("SELECT * FROM runs ORDER BY ingested_at DESC LIMIT 30")]
            failures=db.execute("SELECT count(*) FROM jobs WHERE status='failed'").fetchone()[0];open_reviews=db.execute("SELECT count(*) FROM reviews WHERE status='open'").fetchone()[0]
        return {"scan_count":len(scans),"database":store.stats(),"jobs":jobs,"runs":runs,"failed_jobs":failures,"open_reviews":open_reviews,"api_key_configured":bool(os.getenv("OPENAI_API_KEY"))}
    @app.get("/api/operations/scans")
    def operations_scans(limit:int=100):
        config=load_config();scans=discover_scans(config["dataset"]["images"],config["dataset"].get("patterns",["*.png"]))
        return [{"name":p.name,"path":str(p),"bytes":p.stat().st_size} for p in scans[:min(limit,1000)]]
    @app.post("/api/operations/batches",status_code=201)
    def create_batch(body:BatchCreate):
        batch_id=uuid.uuid4().hex;now=datetime.now(timezone.utc).isoformat()
        with store.connect() as db:db.execute("INSERT INTO batches(batch_id,name,status,created_at) VALUES(?,?,'uploaded',?)",(batch_id,body.name,now))
        return {"batch_id":batch_id,"name":body.name,"status":"uploaded"}
    @app.get("/api/operations/batches")
    def list_batches():
        with store.connect() as db:return [dict(x) for x in db.execute("SELECT b.*,(SELECT count(*) FROM batch_scans s WHERE s.batch_id=b.batch_id) scan_count FROM batches b ORDER BY created_at DESC")]
    @app.get("/api/operations/batches/{batch_id}")
    def batch_detail(batch_id:str):
        with store.connect() as db:
            batch=db.execute("SELECT * FROM batches WHERE batch_id=?",(batch_id,)).fetchone()
            if not batch:raise HTTPException(404,"batch not found")
            scans=[dict(x) for x in db.execute("SELECT * FROM batch_scans WHERE batch_id=? ORDER BY created_at",(batch_id,))]
            for scan in scans:scan["events"]=[dict(x) for x in db.execute("SELECT * FROM stage_events WHERE scan_id=? ORDER BY event_id",(scan["scan_id"],))]
        return {"batch":dict(batch),"scans":scans}
    @app.post("/api/operations/batches/{batch_id}/upload",status_code=201)
    async def batch_upload(batch_id:str,file:UploadFile=File(...)):
        suffix=Path(file.filename or "").suffix.casefold()
        if suffix not in {".png",".jpg",".jpeg",".tif",".tiff"}:raise HTTPException(415,"unsupported scan format")
        with store.connect() as db:
            if not db.execute("SELECT 1 FROM batches WHERE batch_id=?",(batch_id,)).fetchone():raise HTTPException(404,"batch not found")
        safe=Path(file.filename or "scan").name;folder=Path("uploads")/batch_id;folder.mkdir(parents=True,exist_ok=True)
        target=folder/safe
        if target.exists():target=folder/f"{Path(safe).stem}-{uuid.uuid4().hex[:8]}{Path(safe).suffix}"
        with target.open("xb") as handle:shutil.copyfileobj(file.file,handle)
        try:inspect_scan(target,load_config().get("qc",{}))
        except Exception:target.unlink(missing_ok=True);raise HTTPException(422,"file is not a readable image")
        digest=hashlib.sha256(target.read_bytes()).hexdigest();sid=uuid.uuid4().hex;now=datetime.now(timezone.utc).isoformat()
        with store.connect() as db:
            db.execute("INSERT INTO batch_scans VALUES(?,?,?,?,?,'uploaded',1,?)",(sid,batch_id,safe,str(target.resolve()),digest,now))
            db.execute("INSERT INTO stage_events(scan_id,stage,status,created_at) VALUES(?,'uploaded','complete',?)",(sid,now))
        return {"scan_id":sid,"filename":safe,"selected":True,"status":"uploaded"}
    @app.patch("/api/operations/batches/{batch_id}/scans/{scan_id}")
    def select_scan(batch_id:str,scan_id:str,body:ScanSelection):
        with store.connect() as db:
            changed=db.execute("UPDATE batch_scans SET selected=? WHERE batch_id=? AND scan_id=?",(int(body.selected),batch_id,scan_id)).rowcount
            if not changed:raise HTTPException(404,"scan not found")
        return {"scan_id":scan_id,"selected":body.selected}
    @app.post("/api/operations/batches/{batch_id}/process",status_code=202)
    def process_batch(batch_id:str,body:ProcessBatch,background:BackgroundTasks):
        if body.mode not in {"qc","full"}:raise HTTPException(422,"mode must be qc or full")
        if body.mode=="full" and not os.getenv("OPENAI_API_KEY"):raise HTTPException(409,"OPENAI_API_KEY is required for full digitisation")
        with store.connect() as db:
            if not db.execute("SELECT 1 FROM batches WHERE batch_id=?",(batch_id,)).fetchone():raise HTTPException(404,"batch not found")
            ids=body.scan_ids or [r[0] for r in db.execute("SELECT scan_id FROM batch_scans WHERE batch_id=? AND selected=1",(batch_id,))]
            valid={r[0] for r in db.execute("SELECT scan_id FROM batch_scans WHERE batch_id=?",(batch_id,))}
        if not ids:raise HTTPException(422,"select at least one scan")
        if not set(ids)<=valid:raise HTTPException(422,"scan selection contains an item outside this batch")
        job_id=store.enqueue(f"batch:{batch_id}:{body.mode}:{uuid.uuid4().hex}","pipeline",{"batch_id":batch_id,"mode":body.mode,"scan_ids":ids})
        background.add_task(execute_job,job_id,body.mode,batch_id,ids)
        return {"job_id":job_id,"batch_id":batch_id,"mode":body.mode,"selected_count":len(ids)}
    @app.post("/api/operations/upload")
    async def upload_scan(file:UploadFile=File(...)):
        config=load_config();suffix=Path(file.filename or "").suffix.casefold()
        if suffix not in {".png",".jpg",".jpeg",".tif",".tiff"}:raise HTTPException(415,"unsupported scan format")
        safe=Path(file.filename or "scan").name;folder=Path(config["dataset"]["images"]);folder.mkdir(parents=True,exist_ok=True)
        target=folder/safe
        if target.exists():target=folder/f"{Path(safe).stem}-{uuid.uuid4().hex[:8]}{Path(safe).suffix}"
        with target.open("xb") as handle:shutil.copyfileobj(file.file,handle)
        try:qc=inspect_scan(target,config.get("qc",{}))
        except Exception:
            target.unlink(missing_ok=True);raise HTTPException(422,"file is not a readable image")
        return {"status":"imported","scan":target.name,"original_filename":safe,"qc":qc}
    @app.post("/api/operations/runs",status_code=202)
    def start_run(body:RunRequest,background:BackgroundTasks):
        if body.mode not in {"qc","full"}:raise HTTPException(422,"mode must be qc or full")
        if body.mode=="full" and not os.getenv("OPENAI_API_KEY"):raise HTTPException(409,"OPENAI_API_KEY is required for a full run")
        dedupe=f"{body.mode}:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}";job_id=store.enqueue(dedupe,"pipeline",{"mode":body.mode})
        background.add_task(execute_job,job_id,body.mode);return {"job_id":job_id,"status":"pending","mode":body.mode}
    @app.post("/api/operations/jobs/{job_id}/retry",status_code=202)
    def retry_job(job_id:int,background:BackgroundTasks):
        with store.connect() as db:
            row=db.execute("SELECT payload_json,status FROM jobs WHERE job_id=?",(job_id,)).fetchone()
            if not row:raise HTTPException(404,"job not found")
            if row["status"] not in {"failed","retry"}:raise HTTPException(409,"only failed jobs can be retried")
            mode=json.loads(row["payload_json"]).get("mode","qc");db.execute("UPDATE jobs SET status='pending',error=NULL,updated_at=datetime('now') WHERE job_id=?",(job_id,))
        background.add_task(execute_job,job_id,mode);return {"job_id":job_id,"status":"pending"}
    @app.get("/api/review/pages")
    def review_pages():
        with store.connect() as db:
            rows=db.execute("SELECT p.*,coalesce(a.status,'unreviewed') approval_status,(SELECT count(*) FROM reviews r WHERE r.page_id=p.page_id AND r.status='open') open_reviews FROM pages p LEFT JOIN page_approvals a USING(page_id) WHERE EXISTS(SELECT 1 FROM regions z WHERE z.page_id=p.page_id) OR NOT EXISTS(SELECT 1 FROM batch_pages bp WHERE bp.page_id=p.page_id) ORDER BY p.page_id").fetchall()
            return [dict(r) for r in rows]
    @app.post("/api/review/pages/{page_id:path}/approval")
    def approve_page(page_id:str,body:Approval):
        if body.status not in {"approved","needs_work"}:raise HTTPException(422,"invalid approval status")
        with store.connect() as db:
            if not db.execute("SELECT 1 FROM pages WHERE page_id=?",(page_id,)).fetchone():raise HTTPException(404,"page not found")
            db.execute("INSERT INTO page_approvals VALUES(?,?,?,?,datetime('now')) ON CONFLICT(page_id) DO UPDATE SET status=excluded.status,reviewer=excluded.reviewer,note=excluded.note,updated_at=excluded.updated_at",(page_id,body.status,body.reviewer,body.note))
            article_ids=[r[0] for r in db.execute("SELECT article_id FROM articles WHERE page_id=?",(page_id,))]
            if body.status=="approved":
                db.execute("INSERT INTO page_publications VALUES(?,datetime('now'),?) ON CONFLICT(page_id) DO UPDATE SET published_at=excluded.published_at,reviewer=excluded.reviewer",(page_id,body.reviewer))
                for aid in article_ids:rebuild_article(db,aid)
                db.execute("UPDATE batch_scans SET status='published' WHERE scan_id IN (SELECT scan_id FROM batch_pages WHERE page_id=?)",(page_id,))
                for sid in [r[0] for r in db.execute("SELECT scan_id FROM batch_pages WHERE page_id=?",(page_id,))]:db.execute("INSERT INTO stage_events(scan_id,stage,status,created_at) VALUES(?,'published','complete',datetime('now'))",(sid,))
            else:
                db.execute("DELETE FROM page_publications WHERE page_id=?",(page_id,))
                for aid in article_ids:db.execute("DELETE FROM article_fts WHERE article_id=?",(aid,))
        return {"page_id":page_id,"status":body.status}
    @app.get("/api/pages/{page_id:path}")
    def page(page_id:str):
        with store.connect() as db:
            row=db.execute("SELECT * FROM pages WHERE page_id=?",(page_id,)).fetchone()
            if not row:raise HTTPException(404,"page not found")
            full=db.execute("SELECT * FROM page_transcriptions WHERE page_id=?",(page_id,)).fetchone()
            return {"page":dict(row),"full_page_transcription":dict(full) if full else None,"regions":[dict(x) for x in db.execute("SELECT * FROM regions WHERE page_id=? ORDER BY reading_order",(page_id,))]}
    @app.get("/api/page-images/{page_id:path}")
    def page_image(page_id:str):
        with store.connect() as db: row=db.execute("SELECT source_path FROM pages WHERE page_id=?",(page_id,)).fetchone()
        if not row or not Path(row[0]).is_file():raise HTTPException(404,"source image not found")
        return FileResponse(row[0])
    @app.get("/api/regions/{region_id:path}/image")
    def region_image(region_id:str,variant:int=0):
        with store.connect() as db: row=db.execute("SELECT provenance_json FROM regions WHERE region_id=?",(region_id,)).fetchone()
        if not row:raise HTTPException(404,"region not found")
        variants=json.loads(row[0]).get("variants",[])
        if variant<0 or variant>=len(variants) or not Path(variants[variant]["path"]).is_file():raise HTTPException(404,"crop not found")
        return FileResponse(variants[variant]["path"])
    @app.get("/api/reviews")
    def reviews(status="open"):
        with store.connect() as db:return [dict(x) for x in db.execute("SELECT * FROM reviews WHERE status=? ORDER BY review_id",(status,))]
    @app.post("/api/reviews/{review_id}/resolve")
    def resolve(review_id:int,body:Correction):
        with store.connect() as db:
            row=db.execute("SELECT region_id FROM reviews WHERE review_id=?",(review_id,)).fetchone()
            if not row:raise HTTPException(404,"review not found")
            db.execute("UPDATE reviews SET status='resolved',correction=?,reviewer=?,updated_at=datetime('now') WHERE review_id=?",(body.correction,body.reviewer,review_id))
            if row[0]:
                db.execute("UPDATE regions SET verbatim_text=?,normalized_text=?,status='human_corrected' WHERE region_id=?",(body.correction,normalize_search(body.correction),row[0]))
                article=db.execute("SELECT article_id FROM regions WHERE region_id=?",(row[0],)).fetchone()
                if article and article[0]:rebuild_article(db,article[0])
        return {"status":"resolved"}
    @app.get("/api/graph/{entity_id:path}")
    def graph(entity_id:str):return entity_neighborhood(store,entity_id)
    @app.get("/",response_class=HTMLResponse)
    def home():return UI_HTML
    return app

HTML='''<!doctype html><html lang="bg"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Sofia Library Research</title><style>body{font:16px system-ui;max-width:1100px;margin:35px auto;padding:0 18px;background:#f4efe4;color:#211}nav button,input,textarea{padding:11px;font:inherit}input{width:min(70%,650px)}nav{display:flex;gap:8px;margin:18px 0}article,.panel{background:white;margin:15px 0;padding:18px;border-left:4px solid #8b1e1e}small{color:#765}.hidden{display:none}img{max-width:100%;max-height:600px}textarea{box-sizing:border-box;width:100%;height:220px}.review-card{display:grid;grid-template-columns:minmax(300px,1fr) minmax(300px,1fr);gap:22px}.crop-pane{background:#eee;display:flex;align-items:flex-start;justify-content:center;min-height:360px;overflow:auto}.crop-pane img{width:auto;height:auto;max-height:700px}.controls button{margin-top:10px}.error{color:#a00}@media(max-width:760px){.review-card{grid-template-columns:1fr}}</style><h1>Sofia Library Research</h1><p>Търсене и проверка с проследимост до оригиналното изображение.</p><nav><button data-testid="search-tab" onclick=show('search')>Search</button><button data-testid="review-tab" onclick=reviews()>Review</button></nav><section id=search><input id=q aria-label="Search query" placeholder="Търсене…"><button data-testid="search-button" onclick=s()>Search</button><main id=r></main></section><section id=review class=hidden><main id=queue></main></section><section id=detail></section><script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function show(id){search.classList.toggle('hidden',id!=='search');review.classList.toggle('hidden',id!=='review')}
async function s(){let d=await(await fetch('/api/search?q='+encodeURIComponent(q.value))).json();r.innerHTML=d.results.map(x=>`<article><h3>${esc(x.title||'Untitled')}</h3><p>${esc(x.normalized_text)}</p><small>${esc(x.page_id)} · confidence ${x.confidence??'unreviewed'}</small><p><button data-page="${esc(x.page_id)}" onclick="page('${esc(x.page_id)}')">View evidence</button></p></article>`).join('')||'<p>No results</p>'}
async function page(id){let d=await(await fetch('/api/pages/'+encodeURIComponent(id))).json(),name=d.page.source_path.split(/[\\/]/).pop();detail.innerHTML=`<div class=panel><h2>Evidence</h2><img alt="Original scan" src="/api/page-images/${encodeURIComponent(id)}"><p>${d.regions.length} regions · source ${esc(name)}</p></div>`;detail.scrollIntoView()}
async function reviews(){show('review');let d=await(await fetch('/api/reviews')).json();queue.innerHTML=d.map(x=>{let reasons=JSON.parse(x.reasons_json).join(', ').replaceAll('_',' ');return `<article class="review-card"><div class="crop-pane"><img alt="Region crop" src="/api/regions/${encodeURIComponent(x.region_id)}/image?variant=1"></div><div class="controls"><h3>Region review #${x.review_id}</h3><p>${esc(reasons)}</p><textarea aria-label="Correction ${x.review_id}" id="c${x.review_id}" placeholder="Enter the complete diplomatic transcription"></textarea><button data-review="${x.review_id}" onclick="resolveReview(${x.review_id})">Save correction</button><p class=error id="e${x.review_id}"></p></div></article>`}).join('')||'<p>No open reviews</p>'}
async function resolveReview(id){let correction=document.getElementById('c'+id).value.trim();if(!correction){document.getElementById('e'+id).textContent='Correction cannot be empty.';return}let res=await fetch('/api/reviews/'+id+'/resolve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({correction,reviewer:'local-reviewer'})});if(!res.ok){document.getElementById('e'+id).textContent=await res.text();return}await reviews()}
</script></html>'''

app=create_app()
