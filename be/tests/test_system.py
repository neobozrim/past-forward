import json
import yaml
from fastapi.testclient import TestClient
from PIL import Image
from sofia_harness.api import create_app
from sofia_harness.ingest import ingest_manifest
from sofia_harness.search import search
from sofia_harness.store import Store
from sofia_harness.text import extract_entities,normalize_search,reconstruct_articles


def manifest(tmp_path):
    Image.new("RGB",(20,30),"white").save(tmp_path/"page.png")
    data={"schema_version":"manual-1","run_id":"run1","status":"complete","created_at":"now","pages":[{
        "page_id":"p1","qc":{"path":str(tmp_path/"page.png"),"sha256":"abc","width":20,"height":30,"qc_status":"pass"},
        "regions":[{"id":"h","type":"headline","article_id":"a1","reading_order":0,"polygon":[],"text":"Важна новина","confidence":.9,"status":"transcribed","provenance":{}},
                   {"id":"b","type":"body_column","article_id":"a1","reading_order":1,"polygon":[],"text":"Георги Димитров говори през 1949 г.","confidence":.8,"status":"transcribed","provenance":{}}]}],"review_queue":[]}
    path=tmp_path/"manifest.json";path.write_text(json.dumps(data,ensure_ascii=False),encoding="utf-8");return path


def test_normalization_and_reconstruction():
    assert normalize_search("комунисти-\nческа") == "комунистическа"
    articles=reconstruct_articles({"page_id":"p","regions":[{"id":"h","type":"headline","article_id":"a","reading_order":0,"text":"Заглавие"}]})
    assert articles[0]["title"]=="Заглавие"
    assert ("named_entity","ГЕОРГИ ДИМИТРОВ",8,23) in extract_entities('Язовир „ГЕОРГИ ДИМИТРОВ“')


def test_ingest_is_idempotent_and_searchable(tmp_path):
    store=Store(tmp_path/"db.sqlite");result=ingest_manifest(store,manifest(tmp_path))
    assert result["articles"]==1 and search(store,"Димитров")[0]["title"]=="Важна новина"
    assert ingest_manifest(store,manifest(tmp_path))["status"]=="already_ingested"


def test_graph_has_evidence_offsets(tmp_path):
    store=Store(tmp_path/"db.sqlite");ingest_manifest(store,manifest(tmp_path))
    with store.connect() as db:
        row=db.execute("SELECT * FROM assertions LIMIT 1").fetchone()
        assert row["evidence"] and row["start_offset"] < row["end_offset"]


def test_job_resume_and_deduplication(tmp_path):
    store=Store(tmp_path/"db.sqlite");store.init();store.enqueue("page:qc","qc",{"page":1});store.enqueue("page:qc","qc",{"page":1})
    job=store.claim();assert job["stage"]=="qc";store.complete_job(job["job_id"],"boom",max_attempts=2)
    assert store.claim()["job_id"]==job["job_id"]


def test_interface_memory_roundtrip(tmp_path):
    store=Store(tmp_path/"db.sqlite");store.init();store.set_memory("saved_filters",{"publication":"Работническо дело"})
    assert store.get_memory("saved_filters")["publication"]=="Работническо дело"
    assert store.get_memory("missing",[])==[]


def test_api_health_and_search(tmp_path):
    db=tmp_path/"db.sqlite";store=Store(db);ingest_manifest(store,manifest(tmp_path));client=TestClient(create_app(db))
    assert client.get("/health").status_code==200
    assert client.get("/api/search",params={"q":"Димитров"}).json()["results"]
    assert client.get("/api/page-images/run1:p1").headers["content-type"]=="image/png"


def test_pending_region_creates_review_and_correction_reindexes(tmp_path):
    path=manifest(tmp_path);data=json.loads(path.read_text(encoding="utf-8"));data["pages"][0]["regions"][1]["text"]=None;data["pages"][0]["regions"][1]["status"]="needs_crop_transcription";path.write_text(json.dumps(data,ensure_ascii=False),encoding="utf-8")
    db=tmp_path/"db.sqlite";store=Store(db);ingest_manifest(store,path);client=TestClient(create_app(db))
    reviews=client.get("/api/reviews").json();assert len(reviews)==1
    assert client.post(f"/api/reviews/{reviews[0]['review_id']}/resolve",json={"correction":"","reviewer":"tester"}).status_code==422
    assert client.post(f"/api/reviews/{reviews[0]['review_id']}/resolve",json={"correction":"Нов проверен текст","reviewer":"tester"}).status_code==200
    assert client.get("/api/search",params={"q":"проверен"}).json()["results"]
    assert client.get("/api/reviews").json()==[]


def test_three_workspace_operations_and_approval(tmp_path,monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY",raising=False)
    scans=tmp_path/"scans";scans.mkdir();Image.new("RGB",(120,160),"white").save(scans/"scan.png")
    config={"dataset":{"images":str(scans),"patterns":["*.png"],"output":str(tmp_path/"runs")},
        "models":{"layout":"fake","layout_escalation":"fake","ocr_easy":"fake","ocr_hard":"fake"},
        "routing":{"layout_confidence_threshold":.8,"difficult_page_threshold":.7,"disagreement_threshold":.9,"confidence_threshold":.8},
        "run":{"max_pages":1},"qc":{"min_short_edge_px":10}}
    config_path=tmp_path/"config.yaml";config_path.write_text(yaml.safe_dump(config),encoding="utf-8")
    db=tmp_path/"ops.db";client=TestClient(create_app(db,config_path))
    summary=client.get("/api/operations/summary").json();assert summary["scan_count"]==1 and not summary["api_key_configured"]
    assert client.post("/api/operations/runs",json={"mode":"full"}).status_code==409
    started=client.post("/api/operations/runs",json={"mode":"qc"});assert started.status_code==202
    final=client.get("/api/operations/summary").json();assert final["jobs"][0]["status"]=="complete" and final["jobs"][0]["attempts"]==1 and final["database"]["pages"]==1
    page_id=client.get("/api/review/pages").json()[0]["page_id"]
    assert client.post(f"/api/review/pages/{page_id}/approval",json={"status":"approved","reviewer":"tester","note":"ok"}).status_code==200
    assert client.get("/api/review/pages").json()[0]["approval_status"]=="approved"
    home=client.get("/").text
    assert all(name in home for name in ["Operations","Review","Research"])


def test_batch_upload_selection_qc_stages_and_review_exclusion(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    scans=tmp_path/"source";scans.mkdir()
    config={"dataset":{"images":str(scans),"patterns":["*.png"],"output":str(tmp_path/"runs")},
        "models":{"layout":"fake","layout_escalation":"fake","ocr_easy":"fake","ocr_hard":"fake"},
        "routing":{"layout_confidence_threshold":.8,"difficult_page_threshold":.7,"disagreement_threshold":.9,"confidence_threshold":.8},
        "qc":{"min_short_edge_px":10}}
    config_path=tmp_path/"config.yaml";config_path.write_text(yaml.safe_dump(config),encoding="utf-8")
    image=tmp_path/"upload.png";Image.new("RGB",(120,160),"white").save(image)
    client=TestClient(create_app(tmp_path/"batch.db",config_path))
    batch=client.post("/api/operations/batches",json={"name":"July scans"}).json()
    with image.open("rb") as handle:
        uploaded=client.post(f"/api/operations/batches/{batch['batch_id']}/upload",files={"file":("upload.png",handle,"image/png")})
    assert uploaded.status_code==201
    with image.open("rb") as handle:
        repeated=client.post(f"/api/operations/batches/{batch['batch_id']}/upload",files={"file":("upload.png",handle,"image/png")})
    assert repeated.status_code==201 and repeated.json()["scan_id"]!=uploaded.json()["scan_id"]
    scan_id=uploaded.json()["scan_id"]
    client.patch(f"/api/operations/batches/{batch['batch_id']}/scans/{repeated.json()['scan_id']}",json={"selected":False})
    assert client.patch(f"/api/operations/batches/{batch['batch_id']}/scans/{scan_id}",json={"selected":False}).status_code==200
    assert client.post(f"/api/operations/batches/{batch['batch_id']}/process",json={"mode":"qc","scan_ids":[]}).status_code==422
    client.patch(f"/api/operations/batches/{batch['batch_id']}/scans/{scan_id}",json={"selected":True})
    result=client.post(f"/api/operations/batches/{batch['batch_id']}/process",json={"mode":"qc","scan_ids":[]})
    assert result.status_code==202 and result.json()["selected_count"]==1
    detail=client.get(f"/api/operations/batches/{batch['batch_id']}").json()
    assert detail["batch"]["status"]=="complete"
    assert [e["stage"] for e in detail["scans"][0]["events"]]==["uploaded","qc","qc","qc_only_complete"]
    assert client.get("/api/review/pages").json()==[]
    assert "Process selected scans" in client.get("/").text
