from __future__ import annotations
import json,os,re,shutil,threading,time,uuid
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request as UrlRequest,urlopen
from fastapi import BackgroundTasks,FastAPI,File,Form,HTTPException,Request,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse,HTMLResponse,JSONResponse,StreamingResponse
from openai import OpenAI
from PIL import Image
from .digitization_agent import DigitizationAgent
from .agent_ui import HTML
from .overlay_inspection_ui import HTML as OVERLAY_INSPECTION_HTML
from .overlay_agent import apply_typography,place_two_point_with_api,place_with_api,save_plan,validate_accepted_plan,without_inline_caption_duplicates
from .archive_index import hybrid_search

IMAGE_SUFFIXES={".png",".jpg",".jpeg",".tif",".tiff",".webp"}
PROJECT_ROOT=Path(os.getenv("PAST_FORWARD_PROJECT_ROOT",Path(__file__).resolve().parents[2])).resolve()
jobs={};conversations={};active_resumes=set();lock=threading.Lock()

def _normalized_transcription(folder:Path,stem:str)->dict:
    """Return readable prose while preserving the diplomatic transcript on disk."""
    records=folder/(stem+".articles.json")
    if not records.is_file():
        records=next(iter(sorted(folder.glob("*.articles.json"))),None)
    if not records:return {"sections":[],"text":"","filename":stem+".txt"}
    payload=json.loads(records.read_text(encoding="utf-8"));sections=[]
    for article in sorted(payload.get("articles",[]),key=lambda item:item.get("article_order",0)):
        paragraphs=[]
        for paragraph in re.split(r"\n\s*\n",(article.get("verbatim_text") or article.get("text") or "").strip()):
            # Newspaper line-end hyphens are layout artefacts. Join those and
            # collapse the remaining physical line breaks for reading view.
            paragraph=re.sub(r"(?<=\w)-\s*\n\s*(?=\w)","",paragraph)
            paragraph=re.sub(r"[ \t]*\n[ \t]*"," ",paragraph)
            paragraph=re.sub(r"[ \t]{2,}"," ",paragraph).strip()
            if paragraph:paragraphs.append(paragraph)
        heading=(article.get("heading") or article.get("label") or "").strip()
        if heading or paragraphs:sections.append({"article_id":article.get("article_id",""),"heading":heading,"paragraphs":paragraphs})
    chunks=[]
    for section in sections:
        if section["heading"]:chunks.append(section["heading"])
        chunks.extend(section["paragraphs"])
    return {"sections":sections,"text":"\n\n".join(chunks).strip()+("\n" if chunks else ""),"filename":stem+".txt"}

# The pipeline discovers article count while it works, so this is deliberately an
# estimate, not fabricated exact completion.  Fractions in event details become
# exact within their phase; the UI marks the result with ≈ until completion.
_STAGE_PROGRESS={
    "Thinking":2,"Using tool":4,"Resuming":4,"Resuming saved page":5,
    "Reading complete page":6,"Agent surveying page":6,"Preprocessing":9,
    "Preprocessing complete":13,"Discovering articles and headings":10,
    "Building article columns":18,"Detected reading regions":24,"Layout complete":96,
    "Terra checking article inventory":52,"Sol inspecting Terra findings":58,
    "Independent Sol verification":64,"Independent reads agree":80,
    "Sol resolving text disagreements":82,"Possible truncation detected":83,
    "Verifying proposed text edits":87,"Text disagreement resolved":91,
    "Reconstructing document":92,"Saved Markdown":95,"Agentic layout complete":96,
    "Agentic page complete":96,
}

def _detail_fraction(detail):
    match=re.search(r"\b(\d+)\s*/\s*(\d+)\b",detail or "")
    if not match or int(match.group(2))<=0:return None
    return min(1,max(0,int(match.group(1))/int(match.group(2))))

def _estimated_progress(job,stage,detail=""):
    """Return monotonic job progress while retaining page/phase provenance."""
    progress=job.setdefault("progress",{
        "percent":0.0,"page_index":1,"page_total":max(1,int(job.get("page_total",1))),
        "stage":"Queued","detail":"","estimated":True,"updated_at":time.time(),
    })
    if stage=="Page":
        match=re.search(r"\b(\d+)\s*/\s*(\d+)\b",detail or "")
        if match:
            next_index=max(1,int(match.group(1)))
            if next_index>int(progress.get("page_index",1)):progress["page_percent"]=0.0
            progress["page_index"]=next_index
            progress["page_total"]=max(progress["page_index"],int(match.group(2)))
        page_percent=2.0
    elif stage=="Article saved":
        count_match=re.match(r"\s*(\d+)",detail or "")
        count=int(count_match.group(1)) if count_match else 1
        page_percent=min(49.0,10.0+count*4.0)
    elif stage in {"Independent article read","Transcribing regions","Terra adjudicating regions",
                   "Sol resolving regions","Recovered region","Resolved line crop"}:
        fraction=_detail_fraction(detail)
        start,end=(64.0,79.0) if stage=="Independent article read" else (30.0,72.0)
        page_percent=start+(end-start)*(fraction if fraction is not None else 0)
    elif stage=="Complete":
        if not detail:
            progress.update(percent=100.0,stage=stage,detail=detail,estimated=False,updated_at=time.time())
            return 100.0
        page_percent=96.0
    else:
        page_percent=float(_STAGE_PROGRESS.get(stage,progress.get("page_percent",6.0)))
    progress["page_percent"]=max(float(progress.get("page_percent",0)),page_percent)
    page_index=max(1,int(progress.get("page_index",1)));page_total=max(page_index,int(progress.get("page_total",1)))
    overall=96.0*((page_index-1)+progress["page_percent"]/100.0)/page_total
    overall=max(float(progress.get("percent",0)),min(96.0,overall))
    progress.update(percent=round(overall,1),stage=stage,detail=detail,estimated=True,updated_at=time.time())
    return progress["percent"]
AGENT_PROMPT="""You are Past Forward, a conversational archival digitisation agent.
Always reply in the language used by the operator's latest message. Do not switch to the language of an attached document, source, tool output, or earlier message unless the operator explicitly asks for translation.
Understand the operator's request from the entire conversation, not keywords. You can answer questions
about capabilities and prior saved records without requiring an attachment. Attachments persist across
turns. If the user attaches images without enough scope, ask one concise clarification. A short follow-up
such as 'full', 'the whole page', or 'only the left article' answers that clarification and is valid in
context. When the requested scope is clear and images are attached, call digitize_attached_images. If the
operator asks to inspect, test, or run only layout/segmentation, use mode layout_only. Otherwise use
full_digitisation. Never continue into transcription when layout_only was requested.
When asked whether previous records exist or are accessible, you MUST call list_saved_outputs before
answering. When asked about the contents of a particular record, you MUST call read_saved_output. Never claim a tool ran
unless you called it. If the operator asks to continue or resume interrupted work, call resume_interrupted_run;
an empty run_folder means the most recently interrupted run. Keep user-facing answers concise."""
TOOLS=[
 {"type":"function","name":"digitize_attached_images","description":"Digitise attached images directly into incrementally saved semantic articles. Layout-only inspection remains available when explicitly requested.","parameters":{"type":"object","properties":{"instruction":{"type":"string","description":"Complete resolved instruction including scope inferred from conversation."},"mode":{"type":"string","enum":["layout_only","full_digitisation"],"description":"Inspect layout only, or directly transcribe and checkpoint articles without a polygon gate."}},"required":["instruction","mode"],"additionalProperties":False},"strict":True},
 {"type":"function","name":"list_saved_outputs","description":"List previously saved Markdown digitisation outputs so you can answer questions about prior records.","parameters":{"type":"object","properties":{},"required":[],"additionalProperties":False},"strict":True},
 {"type":"function","name":"read_saved_output","description":"Read one saved Markdown transcription by its path.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":False},"strict":True},
 {"type":"function","name":"resume_interrupted_run","description":"Resume a preserved digitisation run without repeating completed articles. Use an empty run_folder for the latest interrupted run.","parameters":{"type":"object","properties":{"run_folder":{"type":"string"},"instruction":{"type":"string"}},"required":["run_folder","instruction"],"additionalProperties":False},"strict":True},
]

SEARCH_ANSWER_PROMPT="""Answer the user's archival question using only the supplied resources.
Always answer in the same language as the user's question. Do not translate the answer into another language,
even when the supplied resources are written in a different language. Calculate values such as ages or durations
when the evidence supplies the required dates. Cite supporting resources as [R1], [R2], and so on.
Be OBJECTIVE AND NEUTRAL. it passes information as it has been stated by a given source - just as it has been given - nothing more, nothing less. Users draw their own conclusions about the past, what to learn from it and how to avoid repeating past mistakes in the future.
Do not treat missing evidence as proof of a negative. If the resources do not support an answer, say so
briefly and identify the missing evidence. Do not add a Resources section; the interface renders it."""
MONTHS_BG={"януари":1,"февруари":2,"март":3,"април":4,"май":5,"юни":6,"юли":7,"август":8,"септември":9,"октомври":10,"ноември":11,"декември":12}

def local_archive_answer(query,resources):
    """Deterministic answers for operations that should not require a language model."""
    lowered=query.casefold();english=bool(re.search(r"\b(?:how\s+old|what\s+age)\b",lowered))
    if not english and not re.search(r"\b(?:на\s+)?колко\s+години\b",lowered):return None
    births=[];deaths=[]
    months="|".join(MONTHS_BG)
    for resource in resources:
        text=resource.get("relevant_excerpt","")
        for match in re.finditer(rf"роден\s+на\s+(\d{{1,2}})\s+({months})\s+(\d{{4}})",text,re.I):
            births.append((int(match.group(3)),MONTHS_BG[match.group(2).casefold()],int(match.group(1)),resource["resource_id"]))
        for match in re.finditer(rf"[Нн]а\s+(\d{{1,2}})\s+({months})(?P<context>.{{0,180}}?)(?:почина|умира)",text,re.S):
            issue_year=int((resource.get("issue_date") or "0000")[:4] or 0)
            year_match=re.search(r"\b((?:18|19|20)\d{2})\b",match.group("context"))
            year=int(year_match.group(1)) if year_match else issue_year
            if year:deaths.append((year,MONTHS_BG[match.group(2).casefold()],int(match.group(1)),resource["resource_id"]))
    if not births or not deaths:return None
    birth=min(births);death=min(deaths,key=lambda value:abs(value[0]-birth[0]) if value[0]>=birth[0] else 9999)
    age=death[0]-birth[0]-((death[1],death[2])<(birth[1],birth[2]))
    citations=" ".join(f"[{item}]" for item in dict.fromkeys((birth[3],death[3])))
    if english:
        return f"Georgi Dimitrov died at age {age}. {citations}" if "georgi" in lowered and "dimitrov" in lowered else f"The person died at age {age}. {citations}"
    return f"Георги Димитров умира на {age} години. {citations}" if "георги" in lowered and "димитров" in lowered else f"Човекът умира на {age} години. {citations}"

def resource_budget(query):
    lowered=query.casefold()
    if re.search(r"\b(?:all|every|exhaustive|throughout|всички|всякъде|изчерпателно)\b",lowered):return 12
    if re.search(r"\b(?:examples?|compare|comparison|different|attitudes?|joking|mocking|примери?|сравни|сравнение|отношение|подиграв|шег|презрител)\b",lowered):return 8
    if re.search(r"\b(?:evidence|indications?|signs?|whether|were there|данни|доказателства|индикации|признаци|дали|имало ли)\b",lowered):return 5
    return 3

def cited_resources(answer,resources):
    cited=set(re.findall(r"\[(R\d+)\]",answer or ""))
    return [resource for resource in resources if resource["resource_id"] in cited] or resources

def search_answer_input(query,resources):
    """Keep source language from influencing the response language."""
    evidence=json.dumps(resources,ensure_ascii=False)
    return ("LANGUAGE REQUIREMENT: Reply only in the language of the QUESTION below. "
            "The resources may be in another language; do not adopt their language.\n\n"
            f"QUESTION:\n{query}\n\nRESOURCES:\n{evidence}")

def create_agent_app(output_root="digitized",api_enabled=None,search_db=None,search_answers_enabled=None,search_client=None,
                     auth_required=None,supabase_url=None,supabase_publishable_key=None,landing_enabled=None,
                     admin_emails=None):
    app=FastAPI(title="Past Forward")
    output_path=Path(output_root)
    if output_root=="digitized" and not output_path.is_absolute():output_path=PROJECT_ROOT/output_path
    root=output_path.resolve();root.mkdir(parents=True,exist_ok=True);inbox=(PROJECT_ROOT/"agent_inbox").resolve()
    frontend_urls=[url.strip().rstrip("/") for url in os.getenv("PAST_FORWARD_FRONTEND_URLS","http://localhost:5173").split(",") if url.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=frontend_urls,
        allow_credentials=True,
        allow_methods=["GET","POST","PATCH","OPTIONS"],
        allow_headers=["Authorization","Content-Type"],
    )
    configured_search_db=os.getenv("PAST_FORWARD_SEARCH_DB")
    if search_db is not None:search_db=Path(search_db).resolve()
    elif configured_search_db:
        search_db=Path(configured_search_db)
        if not search_db.is_absolute():search_db=PROJECT_ROOT/search_db
        search_db=search_db.resolve()
    else:search_db=(PROJECT_ROOT/"search_data/index/archive.db").resolve()
    if api_enabled is None:api_enabled=os.getenv("PAST_FORWARD_API_ENABLED","true").strip().casefold() in {"1","true","yes","on"}
    if search_answers_enabled is None:search_answers_enabled=os.getenv("PAST_FORWARD_SEARCH_ANSWERS_ENABLED","false").strip().casefold() in {"1","true","yes","on"}
    if auth_required is None:auth_required=os.getenv("PAST_FORWARD_AUTH_REQUIRED","false").strip().casefold() in {"1","true","yes","on"}
    if landing_enabled is None:landing_enabled=os.getenv("PAST_FORWARD_LANDING_ENABLED","true").strip().casefold() in {"1","true","yes","on"}
    supabase_url=(supabase_url or os.getenv("SUPABASE_URL","")).strip().rstrip("/")
    supabase_publishable_key=(supabase_publishable_key or os.getenv("SUPABASE_PUBLISHABLE_KEY","")).strip()
    if admin_emails is None:admin_emails=os.getenv("PAST_FORWARD_ADMIN_EMAILS","").split(",")
    elif isinstance(admin_emails,str):admin_emails=admin_emails.split(",")
    admin_emails={email.strip().casefold() for email in admin_emails if email and email.strip()}
    auth_cache={};auth_lock=threading.Lock()
    public_read_paths=("/api/library","/api/overlay/demos","/api/overlay/demo","/api/overlay/source","/overlay-inspection")
    protected_prefixes=("/api/","/outputs/")
    def supabase_user(token):
        if not token or not supabase_url or not supabase_publishable_key:return False
        with auth_lock:
            cached=auth_cache.get(token)
            if cached and cached[0]>time.time():return cached[1]
        try:
            request=UrlRequest(supabase_url+"/auth/v1/user",headers={"apikey":supabase_publishable_key,"Authorization":"Bearer "+token})
            with urlopen(request,timeout=8) as response:
                user=json.loads(response.read().decode("utf-8")) if response.status==200 else None
        except Exception:user=None
        if isinstance(user,dict) and user.get("id"):
            with auth_lock:
                auth_cache[token]=(time.time()+300,user)
                if len(auth_cache)>256:
                    for key,(expiry,_) in list(auth_cache.items()):
                        if expiry<=time.time():auth_cache.pop(key,None)
            return user
        return None
    def is_admin(user):return bool(user and str(user.get("email","")).strip().casefold() in admin_emails)
    def admin_only_request(request):
        path=request.url.path
        return (path=="/api/agent" or path.startswith("/api/agent/") or
                (request.method!="GET" and path.startswith("/api/overlay/sources/") and path.endswith("/place")))
    @app.middleware("http")
    async def require_supabase_session(request:Request,call_next):
        public_read=request.method=="GET" and (request.url.path.startswith("/outputs/") or request.url.path in public_read_paths)
        protected=not public_read and any(request.url.path==prefix or request.url.path.startswith(prefix) for prefix in protected_prefixes)
        if auth_required and protected:
            authorization=request.headers.get("authorization","")
            token=authorization[7:].strip() if authorization.casefold().startswith("bearer ") else request.cookies.get("pf_access_token","")
            user=supabase_user(token)
            if not user:return JSONResponse({"detail":"Sign in to access Past Forward."},status_code=401)
            request.state.pf_user=user
            if admin_only_request(request) and not is_admin(user):
                return JSONResponse({"detail":"Administrator access is required for digitization."},status_code=403)
        return await call_next(request)
    def emit(job_id,stage,detail=""):
        with lock:
            job=jobs[job_id];percent=_estimated_progress(job,stage,detail)
            job["events"].append({
                "stage":stage,"detail":detail,"percent":percent,
                "estimated":percent<100,"at":time.time(),
            })
    def safe_output(path):
        candidate=Path(path)
        if not candidate.is_absolute():candidate=root/candidate
        candidate=candidate.resolve()
        if candidate!=root and root not in candidate.parents:raise ValueError("output path is outside the archive")
        return candidate
    def attach_urls(item):
        for key,url_key in (("markdown_path","markdown_url"),("articles_path","articles_url"),("layout_path","layout_url"),("overlay_path","overlay_url"),("inspection_overlay_path","inspection_overlay_url"),
                            ("semantic_path","semantic_url"),("semantic_overlay_path","semantic_overlay_url")):
            if item.get(key):item[url_key]="/outputs/"+Path(item[key]).relative_to(root).as_posix()
        if item.get("source_path"):item["source_url"]="/outputs/"+Path(item["source_path"]).relative_to(root).as_posix()
        return item
    def prepare_inspection_overlay(item,progress,client=None):
        """Build and independently audit inspection geometry before a digitization result is published."""
        articles_path=item.get("articles_path");source_path=item.get("source_path")
        if not articles_path or not source_path:return item
        articles=Path(articles_path);source=Path(source_path)
        try:has_articles=bool(json.loads(articles.read_text(encoding="utf-8")).get("articles"))
        except (OSError,json.JSONDecodeError):has_articles=False
        if not has_articles:
            item["inspection_overlay_status"]="unavailable"
            return item
        target=articles.with_name(articles.name.removesuffix(".articles.json")+".overlay.json")
        if target.is_file():
            validate_accepted_plan(json.loads(target.read_text(encoding="utf-8")))
        elif api_enabled:
            progress("Placing inspection overlay","Multimodal placement and rendered-image audit")
            plan=place_two_point_with_api(source,articles,client or OpenAI(timeout=240,max_retries=1))
            save_plan(plan,target)
        else:
            item["inspection_overlay_status"]="pending_codex"
            return item
        item["inspection_overlay_path"]=str(target);item["inspection_overlay_status"]="accepted"
        return item
    def interrupted_state(requested=""):
        if requested:
            folder=safe_output(requested)
            candidates=[path for path in (folder/"audit"/"incremental-state.json",folder/"audit"/"pen-agent-state.json") if path.is_file()]
        else:
            candidates=[]
            for pattern in ("*/audit/incremental-state.json","*/audit/pen-agent-state.json"):
                for state_path in root.glob(pattern):
                    try:saved_state=json.loads(state_path.read_text(encoding="utf-8"))
                    except (OSError,json.JSONDecodeError):continue
                    if str(saved_state.get("status","")).startswith((
                        "interrupted","failed","text_review_required","transcribed_pending_review",
                        "review_required","repaired_pending_inventory_confirmation","inventory_review_required"
                    )):candidates.append(state_path)
            candidates.sort(key=lambda p:p.stat().st_mtime,reverse=True)
        if not candidates:raise FileNotFoundError("No interrupted run was found.")
        state_path=candidates[0];return state_path.parent.parent,json.loads(state_path.read_text(encoding="utf-8"))
    def claim_resume(folder):
        key=str(Path(folder).resolve()).casefold()
        with lock:
            if key in active_resumes:raise RuntimeError("This run is already being resumed by another job.")
            active_resumes.add(key)
        return key
    def release_resume(key):
        if key:
            with lock:active_resumes.discard(key)
    def run_resume(job_id,run_folder,instruction):
        resume_key=None
        try:
            folder,saved_state=interrupted_state(run_folder);resume_key=claim_resume(folder);emit(job_id,"Resuming",folder.name)
            def progress(stage,detail=""):emit(job_id,stage,detail)
            agent=DigitizationAgent(root)
            item=(agent.process_incremental(saved_state["source"],instruction,progress,resume_folder=folder)
                if saved_state.get("workflow")=="incremental-full-page" else
                agent.process_agentic(saved_state["source"],instruction,progress,resume_folder=folder))
            item=attach_urls(prepare_inspection_overlay(item,progress))
            with lock:jobs[job_id].update(status="complete",answer="The interrupted run is complete.",results=[item],resume_folder=None)
            emit(job_id,"Complete","")
        except Exception as exc:
            resume_folder=getattr(exc,"run_folder",run_folder or None)
            with lock:jobs[job_id].update(status="failed",error=f"{type(exc).__name__}: {exc}",resume_folder=resume_folder)
        finally:release_resume(resume_key)
    def run_chat(job_id,conversation_id,message):
        conv=conversations[conversation_id];client=OpenAI(timeout=240,max_retries=1)
        try:
            emit(job_id,"Thinking","Understanding your request from the conversation")
            history=conv["messages"][-20:]
            attachment_names=[Path(p).name for p in conv["attachments"]]
            current=f"{message.strip() or '[No text; attachment-only turn]'}\n\nCurrently attached images: {attachment_names or 'none'}"
            api_input=[*history,{"role":"user","content":current}]
            response=client.responses.create(model="gpt-5.6-sol",instructions=AGENT_PROMPT,input=api_input,tools=TOOLS)
            tool_rounds=0;results=[]
            while True:
                calls=[x for x in response.output if getattr(x,"type",None)=="function_call"]
                if not calls:break
                tool_rounds+=1
                if tool_rounds>6:raise RuntimeError("agent exceeded tool-call limit")
                outputs=[]
                for call in calls:
                    args=json.loads(call.arguments or "{}");emit(job_id,"Using tool",call.name)
                    if call.name=="digitize_attached_images":
                        if not conv["attachments"]:value={"error":"No images are attached. Ask the user to attach images or select a folder."}
                        else:
                            instruction=args["instruction"];mode=args["mode"];tool_results=[]
                            total=len(conv["attachments"])
                            for index,path in enumerate(conv["attachments"],1):
                                def progress(stage,detail=""):emit(job_id,stage,detail)
                                emit(job_id,"Page",f"{index}/{total} · {Path(path).name}")
                                agent=DigitizationAgent(root)
                                raw=agent.process_agentic(path,instruction,progress,stop_after_layout=True) if mode=="layout_only" else agent.process_incremental(path,instruction,progress)
                                item=attach_urls(raw if mode=="layout_only" else prepare_inspection_overlay(raw,progress,client))
                                tool_results.append(item)
                                with lock:jobs[job_id]["results"].append(item)
                            results.extend(tool_results);value={"mode":mode,"saved":[{"source":x["source_name"],"markdown":x.get("markdown_path"),"articles":x.get("articles_path"),"state":x.get("layout_path"),"layout_overlay":x.get("overlay_path"),"article_count":x.get("article_count"),"status":x.get("status"),"partial":x.get("partial",False),"confidence":x["confidence"],"omissions_repaired":x["omissions_found"],"text_verified_articles":len(x.get("text_verified_articles",[])),"text_verification_pending":x.get("text_verification_pending",[])} for x in tool_results]}
                    elif call.name=="list_saved_outputs":
                        value={"outputs":[str(p.relative_to(root)) for p in sorted(root.rglob("*.md"),reverse=True)[:100]]}
                    elif call.name=="read_saved_output":
                        path=safe_output(args["path"]);value={"path":str(path),"content":path.read_text(encoding="utf-8")[:30000]}
                    elif call.name=="resume_interrupted_run":
                        resume_key=None
                        try:
                            folder,saved_state=interrupted_state(args["run_folder"].strip())
                            resume_key=claim_resume(folder)
                            def progress(stage,detail=""):emit(job_id,stage,detail)
                            agent=DigitizationAgent(root)
                            item=(agent.process_incremental(saved_state["source"],args["instruction"],progress,resume_folder=folder)
                                if saved_state.get("workflow")=="incremental-full-page" else
                                agent.process_agentic(saved_state["source"],args["instruction"],progress,resume_folder=folder))
                            item=attach_urls(prepare_inspection_overlay(item,progress,client))
                            results.append(item);value={"resumed":str(folder),"saved":item}
                        except FileNotFoundError as exc:value={"error":str(exc)}
                        except RuntimeError as exc:value={"error":str(exc)}
                        finally:release_resume(resume_key)
                    else:value={"error":"unknown tool"}
                    outputs.append({"type":"function_call_output","call_id":call.call_id,"output":json.dumps(value,ensure_ascii=False)})
                emit(job_id,"Thinking","Reviewing tool results")
                response=client.responses.create(model="gpt-5.6-sol",instructions=AGENT_PROMPT,previous_response_id=response.id,input=outputs,tools=TOOLS)
            answer=response.output_text.strip() or "Done."
            with lock:
                conv["messages"].extend([{"role":"user","content":message or "[attachments]"},{"role":"assistant","content":answer}])
                jobs[job_id].update(status="complete",answer=answer,results=results)
            emit(job_id,"Complete","")
        except Exception as exc:
            resume_folder=getattr(exc,"run_folder",None)
            with lock:
                conv["messages"].extend([{"role":"user","content":message or "[attachments]"},{"role":"assistant","content":f"Task failed: {type(exc).__name__}: {exc}"}])
                jobs[job_id].update(status="failed",error=f"{type(exc).__name__}: {exc}",resume_folder=resume_folder)
    @app.get("/health")
    def health():return {"status":"ok","storage":"filesystem","output_root":str(root),"api_enabled":api_enabled,"search_ready":search_db.is_file(),"search_answers_enabled":search_answers_enabled,"mode":"production_api" if api_enabled else "codex_development"}
    @app.get("/api/auth/me")
    def auth_me(request:Request):
        user=getattr(request.state,"pf_user",None)
        return {"authenticated":bool(user),"email":user.get("email") if user else None,
                "is_admin":not auth_required or is_admin(user)}
    search_clients=[search_client] if search_client is not None else []
    def active_search_client():
        if not search_clients:search_clients.append(OpenAI(timeout=90,max_retries=1))
        return search_clients[0]
    def retrieve_resources(query):
        budget=resource_budget(query);rows=hybrid_search(search_db,query,max(12,min(budget*3,50)),client=active_search_client());results=[]
        for row in rows:
            source=Path(row.pop("source_path"));source_url=None
            if not source.is_file():source=root/source.parent.name/source.name
            try:source_url="/outputs/"+source.resolve().relative_to(root).as_posix()
            except ValueError:pass
            inspect_url=f"/overlay-inspection.html?source={quote(source.resolve().relative_to(root).as_posix())}" if source_url else None
            excerpt=row.pop("relevant_excerpt","");row.pop("verbatim_text",None)
            results.append({**row,"relevant_excerpt":excerpt[:1600],"source_url":source_url,"inspect_url":inspect_url})
        deduplicated=[];seen=set()
        for result in results:
            if result["article_id"] in seen:continue
            seen.add(result["article_id"]);deduplicated.append(result)
            if len(deduplicated)>=budget:break
        return [{**result,"resource_id":f"R{index}"} for index,result in enumerate(deduplicated,1)],budget
    def answer_evidence(resources):
        return [{"resource_id":r["resource_id"],"heading":r["heading"],"publication":r["publication"],
                 "issue_date":r["issue_date"],"page_number":r["page_number"],"text":r["relevant_excerpt"]}
                for r in resources]
    @app.get("/api/search")
    def archive_search(q:str="",limit:int=10):
        query=q.strip()
        if not query:return {"query":query,"results":[],"search_ready":search_db.is_file()}
        if not search_db.is_file():raise HTTPException(503,"Search index is not built yet.")
        resources,budget=retrieve_resources(query)
        answer=local_archive_answer(query,resources);answer_error=None
        if search_answers_enabled and resources and not answer:
            try:
                client=active_search_client()
                evidence=answer_evidence(resources)
                response=client.responses.create(model="gpt-5.6-terra",reasoning={"effort":"low"},
                    text={"verbosity":"low"},instructions=SEARCH_ANSWER_PROMPT,
                    input=search_answer_input(query,evidence))
                answer=response.output_text.strip() or None
            except Exception as exc:answer_error=f"{type(exc).__name__}: {exc}"
        visible_resources=cited_resources(answer,resources) if answer else resources
        return {"query":query,"answer":answer,"answer_error":answer_error,"resources":visible_resources,
                "results":visible_resources,"resource_budget":budget,"search_ready":True}
    @app.get("/api/search/stream")
    def archive_search_stream(q:str=""):
        query=q.strip()
        if not query:raise HTTPException(422,"Search query is required.")
        if not search_db.is_file():raise HTTPException(503,"Search index is not built yet.")
        def event(kind,**payload):return json.dumps({"type":kind,**payload},ensure_ascii=False,separators=(",",":"))+"\n"
        def generate():
            resources,budget=retrieve_resources(query)
            yield event("resources",resources=resources,resource_budget=budget)
            if not resources:
                yield event("done",resources=[]);return
            local_answer=local_archive_answer(query,resources)
            if local_answer:
                yield event("answer_delta",delta=local_answer)
                yield event("done",resources=cited_resources(local_answer,resources));return
            if not search_answers_enabled:
                yield event("answer_error",message="Answer generation is not enabled.")
                yield event("done",resources=resources);return
            try:
                client=active_search_client();answer_parts=[]
                with client.responses.stream(model="gpt-5.6-terra",reasoning={"effort":"low"},
                    text={"verbosity":"low"},instructions=SEARCH_ANSWER_PROMPT,
                    input=search_answer_input(query,answer_evidence(resources))) as stream:
                    for item in stream:
                        if getattr(item,"type",None)=="response.output_text.delta" and getattr(item,"delta",None):
                            answer_parts.append(item.delta);yield event("answer_delta",delta=item.delta)
                answer="".join(answer_parts)
                yield event("done",resources=cited_resources(answer,resources))
            except Exception as exc:
                yield event("answer_error",message=f"{type(exc).__name__}: {exc}")
                yield event("done",resources=resources)
        return StreamingResponse(generate(),media_type="application/x-ndjson",headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
    @app.post("/api/agent/chat",status_code=202)
    async def chat(background:BackgroundTasks,message:str=Form(""),conversation_id:str=Form(""),files:list[UploadFile]=File(default=[])):
        if not api_enabled:raise HTTPException(503,"Development mode: paid API execution is disabled. Use Codex to test and inspect pages until production readiness.")
        conversation_id=conversation_id or uuid.uuid4().hex
        with lock:conv=conversations.setdefault(conversation_id,{"messages":[],"attachments":[]})
        if files:
            folder=inbox/conversation_id;folder.mkdir(parents=True,exist_ok=True)
            for f in files:
                safe=Path(f.filename or "image").name
                if Path(safe).suffix.casefold() not in IMAGE_SUFFIXES:continue
                target=folder/f"{uuid.uuid4().hex[:8]}_{safe}"
                with target.open("wb") as h:shutil.copyfileobj(f.file,h)
                conv["attachments"].append(str(target))
        job_id=uuid.uuid4().hex;jobs[job_id]={"job_id":job_id,"conversation_id":conversation_id,"status":"queued","events":[],"answer":None,"results":[],"error":None,"resume_folder":None,"page_total":max(1,len(conv["attachments"]))}
        background.add_task(run_chat,job_id,conversation_id,message)
        return {"job_id":job_id,"conversation_id":conversation_id,"status":"queued"}
    @app.post("/api/agent/chat-folder",status_code=202)
    async def chat_folder(background:BackgroundTasks,message:str=Form(""),conversation_id:str=Form(""),path:str=Form(...)):
        if not api_enabled:raise HTTPException(503,"Development mode: paid API execution is disabled. Use Codex to test and inspect pages until production readiness.")
        folder=Path(path).expanduser().resolve()
        if not folder.is_dir():raise HTTPException(404,"Folder not found")
        conversation_id=conversation_id or uuid.uuid4().hex
        with lock:conv=conversations.setdefault(conversation_id,{"messages":[],"attachments":[]})
        conv["attachments"].extend(str(p) for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.casefold() in IMAGE_SUFFIXES)
        job_id=uuid.uuid4().hex;jobs[job_id]={"job_id":job_id,"conversation_id":conversation_id,"status":"queued","events":[],"answer":None,"results":[],"error":None,"resume_folder":None,"page_total":max(1,len(conv["attachments"]))}
        background.add_task(run_chat,job_id,conversation_id,message)
        return {"job_id":job_id,"conversation_id":conversation_id,"status":"queued"}
    @app.post("/api/agent/resume",status_code=202)
    async def resume(background:BackgroundTasks,run_folder:str=Form(""),instruction:str=Form("Continue the interrupted digitisation and repair every outstanding issue.")):
        if not api_enabled:raise HTTPException(503,"Development mode: paid API execution is disabled. Use Codex to continue inspection until production readiness.")
        job_id=uuid.uuid4().hex;jobs[job_id]={"job_id":job_id,"conversation_id":"","status":"queued","events":[],"answer":None,"results":[],"error":None,"resume_folder":run_folder or None,"page_total":1}
        background.add_task(run_resume,job_id,run_folder,instruction)
        return {"job_id":job_id,"status":"queued"}
    @app.get("/api/agent/jobs/{job_id}")
    def job(job_id:str):
        if job_id not in jobs:raise HTTPException(404,"Job not found")
        return jobs[job_id]
    @app.get("/api/agent/outputs")
    def outputs():return [{"name":p.name,"path":str(p),"url":"/outputs/"+p.relative_to(root).as_posix()} for p in sorted(root.rglob("*.md"),reverse=True)]
    @app.get("/api/library")
    def library():
        names={"rd":"Работническо дело","st":"Стършел","nm":"Народна младеж","ot":"Отечествен фронт"};groups={};seen=set()
        for plan in sorted(root.rglob("*.overlay.json"),key=lambda path:path.stat().st_mtime,reverse=True):
            stem=plan.name.removesuffix(".overlay.json");image=next((plan.parent/(stem+suffix) for suffix in IMAGE_SUFFIXES if (plan.parent/(stem+suffix)).is_file()),None)
            if not image:continue
            match=re.search(r"IMG_\d+_([a-z]+)_(\d{2})_(\d{2})_(\d{4})_page(\d+)",stem,re.I);code=match.group(1).casefold() if match else "other"
            identity=(code,match.group(2,3,4,5) if match else image.name)
            if identity in seen:continue
            try:validate_accepted_plan(json.loads(plan.read_text(encoding="utf-8")))
            except (OSError,json.JSONDecodeError,ValueError):continue
            seen.add(identity);relative=image.relative_to(root).as_posix();label=f"{match.group(2)}.{match.group(3)}.{match.group(4)} · Page {int(match.group(5))}" if match else image.stem
            groups.setdefault(names.get(code,code.upper()),[]).append({"label":label,"image_path":relative,"inspect_url":"/overlay-inspection.html?source="+quote(relative)})
        return [{"newspaper":name,"pages":sorted(pages,key=lambda page:page["label"])} for name,pages in sorted(groups.items())]
    @app.get("/api/overlay/sources")
    def overlay_sources():
        records=[]
        for markdown in root.rglob("*.md"):
            images=sorted((p for p in markdown.parent.iterdir() if p.is_file() and p.suffix.casefold() in IMAGE_SUFFIXES),key=lambda p:p.stat().st_mtime,reverse=True)
            if not images:continue
            image=next((p for p in images if p.stem==markdown.stem),images[0])
            placement=markdown.with_suffix(".overlay.json")
            records.append({"id":markdown.parent.relative_to(root).as_posix(),"name":image.stem,"run":markdown.parent.name,
                            "image_url":"/outputs/"+image.relative_to(root).as_posix(),"markdown_url":"/outputs/"+markdown.relative_to(root).as_posix(),
                            "placement_url":"/outputs/"+placement.relative_to(root).as_posix() if placement.is_file() else None,
                            "updated_at":max(image.stat().st_mtime,markdown.stat().st_mtime)})
        return sorted(records,key=lambda item:item["updated_at"],reverse=True)
    @app.get("/api/overlay/demos")
    def overlay_demos():return [
        {"id":"img-0933","label":"Работническо дело · issue 174 · vision audited"},
        {"id":"georgi","label":"Народна младеж · 4 July 1949 · vision audited"},
    ]
    @app.get("/api/overlay/source")
    def overlay_source(source_id:str):
        image=safe_output(source_id)
        if not image.is_file() or image.suffix.casefold() not in IMAGE_SUFFIXES:raise HTTPException(404,"Overlay source photograph not found")
        stem=image.stem;folder=image.parent;blocks=[];article_count=0
        with Image.open(image) as source_image:page_aspect=source_image.height/source_image.width
        plan=folder/(stem+".overlay.json")
        if plan.is_file():
            payload=json.loads(plan.read_text(encoding="utf-8"))
            try:validate_accepted_plan(payload)
            except ValueError as exc:raise HTTPException(409,str(exc))
            blocks=without_inline_caption_duplicates(payload.get("blocks",[]));article_count=len({block.get("article_id") for block in blocks});complete=True
        else:
            complete=False
        transcription=_normalized_transcription(folder,stem)
        return {"source_name":image.name,"image_url":"/outputs/"+image.relative_to(root).as_posix(),"blocks":blocks,"article_count":article_count,"coverage_complete":complete,"coverage_note":"Source loaded; overlay placement is not available yet" if not blocks else "Vision layout and saved transcript loaded","transcription_sections":transcription["sections"],"transcription_text":transcription["text"],"transcription_filename":transcription["filename"]}
    @app.get("/api/overlay/demo")
    def overlay_demo(demo_id:str="georgi"):
        accepted={
          "img-0933":"20260720-1624_IMG_0933_rd_07_04_1949_page3_incremental/20260720-1624_IMG_0933_rd_07_04_1949_page3.png",
          "georgi":"20260720-1505_IMG_0982_nm_07_04_1949_page1_codex-development/20260720-1505_IMG_0982_nm_07_04_1949_page1.png",
        }
        if demo_id in accepted:return overlay_source(accepted[demo_id])
        raise HTTPException(404,"Only independently accepted vision overlays are available")
        if demo_id=="georgi":
            demo=root/"20260720-1505_IMG_0982_nm_07_04_1949_page1_codex-development"
            articles_path=next(demo.glob("*.articles.json"),None);image=next((p for p in demo.iterdir() if p.suffix.casefold() in IMAGE_SUFFIXES),None) if demo.is_dir() else None
            if not articles_path or not image:raise HTTPException(404,"The prepared agent overlay source is unavailable")
            with Image.open(image) as source_image:page_aspect=source_image.height/source_image.width
            payload=json.loads(articles_path.read_text(encoding="utf-8"));blocks=[]
            for article in payload.get("articles",[]):
                for region in article.get("regions",[]):
                    points=region.get("points");text=region.get("verbatim_text","").strip()
                    if not points or not text:continue
                    blocks.append({"block_id":region["region_id"],"article_id":article["article_id"],"article_label":article.get("label",article["article_id"]),"text":text,"polygon":points,"rotation":0,"role":"heading" if "heading" in region.get("role","") else region.get("role","body"),"confidence":region.get("confidence",0)})
            return {"source_name":image.name,"image_url":"/outputs/"+image.relative_to(root).as_posix(),"blocks":[apply_typography(block,page_aspect) for block in blocks],"article_count":len(payload.get("articles",[])),"coverage_complete":True,"workflow":payload.get("workflow")}
        prepared={
          "repaired-0936":("20260721-0907_IMG_0936_rd_07_05_1949_page2_incremental",{
            "img-0936-umbrella":[(70,105,965,177)],"img-0936-fearless-fighter":[(48,180,177,640),(178,180,300,640)],
            "img-0936-inspiring-life":[(302,180,425,535),(426,180,548,535),(549,180,690,535)],
            "img-0936-meeting-with-dimitrov":[(694,180,825,760),(826,180,963,760)],"img-0936-fiery-fighter":[(48,642,177,760),(178,642,300,760)],
            "img-0936-banner-of-friendship":[(302,538,425,760),(426,538,548,760)],"img-0936-bright-image":[(550,538,690,760)],
            "img-0936-glory-and-pride":[(48,765,177,950),(178,765,300,950),(302,765,425,950),(426,765,548,950),(550,765,690,950),(694,765,825,950),(826,765,963,950)]}),
          "repaired-0967":("20260721-0904_IMG_0967_st_07_15_1949_page2_incremental",{
            "img_0967_a01":[(135,245,370,282)],"img_0967_a02":[(370,90,493,287),(494,90,617,287),(618,90,742,287)],
            "img_0967_a03":[(750,65,970,300)],"img_0967_a04":[(132,330,253,805),(254,330,370,805)],
            "img_0967_a05":[(495,310,742,458)],"img_0967_a07":[(750,345,860,960),(861,345,975,960)],
            "img_0967_a06":[(372,495,492,805),(493,495,615,805),(616,495,742,805)],"img_0967_a08":[(135,835,742,955)]}),
          "search-0965":("20260721-0814_IMG_0965_st_07_15_1949_page1_incremental",{
            "img0965_001_masthead":[(125,65,720,235)],"img0965_002_telephone_cartoon":[(760,75,930,235)],
            "img0965_003_chervenkov_quote":[(380,248,900,290)],"img0965_004_bkp_cartoon":[(130,285,920,465)],
            "img0965_005_agency_section":[(420,465,600,505)],"img0965_006_washington":[(128,465,292,590)],
            "img0965_007_london":[(292,465,452,590)],"img0965_008_paris":[(452,465,610,590)],
            "img0965_009_belgrade":[(610,465,770,590)],"img0965_010_venice":[(770,465,930,590)],
            "img0965_011_bad_encounter_cartoon":[(125,600,395,970)],"img0965_012_would_you_like":[(410,600,650,970)],
            "img0965_013_tito_cartoon":[(665,600,935,970)]})}
        selected=prepared.get(demo_id)
        if not selected:raise HTTPException(404,"Prepared agent overlay unavailable")
        folder=root/selected[0];articles_path=next(folder.glob("*.articles.json"),None);image=next((p for p in folder.iterdir() if p.suffix.casefold() in IMAGE_SUFFIXES),None) if folder.is_dir() else None
        if not articles_path or not image:raise HTTPException(404,"Repaired digitized source unavailable")
        with Image.open(image) as source_image:page_aspect=source_image.height/source_image.width
        payload=json.loads(articles_path.read_text(encoding="utf-8"));blocks=[]
        def split_lines(text,count):
            lines=text.strip().splitlines();total=sum(bool(line.strip()) for line in lines);base,extra=divmod(total,count);targets=[base+(extra>0 and index>=count-extra) for index in range(count)];parts=[];start=0
            for target in targets[:-1]:
                seen=0;end=start
                while end<len(lines) and seen<target:seen+=bool(lines[end].strip());end+=1
                while end<len(lines) and not lines[end].strip():end+=1
                parts.append("\n".join(lines[start:end]).strip());start=end
            parts.append("\n".join(lines[start:]).strip());return parts
        heading_boxes={
          "repaired-0936":{"img-0936-umbrella":(70,105,965,145),"img-0936-fearless-fighter":(48,175,300,205),"img-0936-inspiring-life":(302,175,690,205),"img-0936-meeting-with-dimitrov":(694,175,963,205),"img-0936-fiery-fighter":(48,625,300,650),"img-0936-banner-of-friendship":(302,520,548,545),"img-0936-bright-image":(550,520,690,545),"img-0936-glory-and-pride":(48,750,963,780)},
          "repaired-0967":{"img_0967_a02":(370,62,742,92),"img_0967_a03":(750,38,970,68),"img_0967_a04":(132,288,370,332),"img_0967_a05":(495,282,742,312),"img_0967_a07":(750,308,975,347),"img_0967_a06":(372,462,742,497),"img_0967_a08":(135,807,742,837)}}
        handoffs={"repaired-0967":{"img_0967_a02":["потриват ръце.","ни."]}}
        def split_at_handoffs(text,markers):
            lines=text.strip().splitlines();parts=[];start=0
            for marker in markers:
                matches=[index for index,line in enumerate(lines[start:],start) if line.strip()==marker]
                if len(matches)!=1:raise HTTPException(422,f"Placement handoff is not uniquely grounded: {marker}")
                end=matches[0]+1
                while end<len(lines) and not lines[end].strip():end+=1
                parts.append("\n".join(lines[start:end]).strip());start=end
            parts.append("\n".join(lines[start:]).strip());return parts
        for article in payload["articles"]:
            boxes=selected[1].get(article["article_id"],[]);markers=handoffs.get(demo_id,{}).get(article["article_id"]);parts=split_at_handoffs(article["verbatim_text"],markers) if markers else split_lines(article["verbatim_text"],len(boxes)) if boxes else []
            if len(parts)!=len(boxes):raise HTTPException(422,f'{article["article_id"]}: placement flow count does not match geometry')
            heading_box=heading_boxes.get(demo_id,{}).get(article["article_id"]);heading=article.get("heading","").strip()
            if heading_box and heading:
                x1,y1,x2,y2=heading_box;blocks.append({"block_id":f'{article["article_id"]}-heading',"article_id":article["article_id"],"article_label":heading,"text":heading,"polygon":[[x1,y1],[x2,y1],[x2,y2],[x1,y2]],"rotation":0,"role":"heading","confidence":article.get("confidence",0)})
            for index,(box,text) in enumerate(zip(boxes,parts),1):
                if not text:continue
                x1,y1,x2,y2=box;blocks.append({"block_id":f'{article["article_id"]}-flow-{index}',"article_id":article["article_id"],"article_label":article.get("heading") or article["article_id"],"text":text,
                    "polygon":[[x1,y1],[x2,y1],[x2,y2],[x1,y2]],"rotation":0,"role":"body","confidence":article.get("confidence",0)})
        return {"source_name":image.name,"image_url":"/outputs/"+image.relative_to(root).as_posix(),"blocks":[apply_typography(block,page_aspect) for block in blocks],"article_count":len(payload["articles"]),"coverage_complete":False,"workflow":"codex-overlay-development","coverage_note":"All articles placed; independent visual placement audit pending"}
    @app.post("/api/overlay/sources/{source_id:path}/place")
    def place_overlay(source_id:str):
        if not api_enabled:raise HTTPException(503,"API placement is disabled. Codex can prepare the .overlay.json plan during development.")
        folder=safe_output(source_id)
        if not folder.is_dir():raise HTTPException(404,"Digitized source not found")
        article_files=sorted(folder.glob("*.articles.json"));images=sorted(p for p in folder.iterdir() if p.suffix.casefold() in IMAGE_SUFFIXES)
        if not article_files or not images:raise HTTPException(422,"The digitized source needs both a photograph and an article transcript")
        articles=article_files[0];source_stem=articles.name.removesuffix(".articles.json")
        image=next((p for p in images if p.stem==source_stem),images[0])
        plan=place_two_point_with_api(image,articles,OpenAI(timeout=240,max_retries=1));target=articles.with_name(source_stem+".overlay.json");save_plan(plan,target)
        return {"status":"complete","placement_url":"/outputs/"+target.relative_to(root).as_posix(),"blocks":len(plan.blocks),"coverage_complete":plan.coverage_complete}
    @app.get("/outputs/{relative:path}")
    def output(relative:str):
        path=safe_output(relative)
        if not path.is_file():raise HTTPException(404,"Output not found")
        return FileResponse(path)
    @app.get("/assets/past-forward-hero.png",include_in_schema=False)
    def hero_asset():
        path=Path(__file__).with_name("static")/"past-forward-hero.png"
        if not path.is_file():raise HTTPException(404,"Hero asset not found")
        return FileResponse(path,media_type="image/png",headers={"Cache-Control":"public, max-age=86400"})
    @app.get("/assets/sturshel-page.png",include_in_schema=False)
    def sturshel_asset():
        path=Path(__file__).with_name("static")/"sturshel-page.png"
        if not path.is_file():raise HTTPException(404,"Sturshel page asset not found")
        return FileResponse(path,media_type="image/png",headers={"Cache-Control":"public, max-age=86400"})
    @app.get("/assets/newspaper-collage.png",include_in_schema=False)
    def newspaper_collage_asset():
        path=Path(__file__).with_name("static")/"newspaper-collage.png"
        if not path.is_file():raise HTTPException(404,"Newspaper collage asset not found")
        return FileResponse(path,media_type="image/png",headers={"Cache-Control":"public, max-age=86400"})
    @app.get("/assets/sturshel-shrug.png",include_in_schema=False)
    def sturshel_shrug_asset():
        path=Path(__file__).with_name("static")/"sturshel-shrug.png"
        if not path.is_file():raise HTTPException(404,"Cartoon asset not found")
        return FileResponse(path,media_type="image/png",headers={"Cache-Control":"public, max-age=86400"})
    @app.get("/assets/train-page/{filename}",include_in_schema=False)
    def train_page_asset(filename:str):
        allowed={"IMG_0930_rd_07_04_1949_page1.png","IMG_0965_st_07_15_1949_page1.png","IMG_0982_nm_07_04_1949_page1.png","IMG_1043_ot_07_04_1949_page1.png","IMG_0933_rd_07_04_1949_page3.png"}
        if filename not in allowed:raise HTTPException(404,"Training page asset not found")
        path=Path(__file__).parents[2]/"train-20260717T214057Z-1-001"/"train"/filename
        if not path.is_file():raise HTTPException(404,"Training page asset not found")
        return FileResponse(path,media_type="image/png",headers={"Cache-Control":"public, max-age=86400"})
    @app.get("/assets/landing-paper/{filename}",include_in_schema=False)
    def landing_paper_asset(filename:str):
        allowed={"rabotnichesko-delo.jpg","sturshel.jpg","narodna-mladezh.jpg","otechestven-front.jpg","rabotnichesko-delo-page3.jpg"}
        if filename not in allowed:raise HTTPException(404,"Landing paper asset not found")
        path=Path(__file__).with_name("static")/"landing-papers"/filename
        if not path.is_file():raise HTTPException(404,"Landing paper asset not found")
        return FileResponse(path,media_type="image/jpeg",headers={"Cache-Control":"public, max-age=86400"})
    @app.get("/",response_class=HTMLResponse)
    def home():
        configured=bool(supabase_url and supabase_publishable_key)
        rendered=HTML.replace("__SUPABASE_URL__",json.dumps(supabase_url)[1:-1]).replace("__SUPABASE_PUBLISHABLE_KEY__",json.dumps(supabase_publishable_key)[1:-1])
        rendered=rendered.replace("__AUTH_REQUIRED__","true" if auth_required else "false").replace("__AUTH_CONFIGURED__","true" if configured else "false")
        rendered=rendered.replace("__LANDING_ENABLED__","true" if landing_enabled else "false")
        return HTMLResponse(rendered,headers={"Cache-Control":"no-store"})
    @app.get("/overlay-inspection",response_class=HTMLResponse,include_in_schema=False)
    @app.get("/overlay-inspection.html",response_class=HTMLResponse,include_in_schema=False)
    def overlay_inspection():return HTMLResponse(OVERLAY_INSPECTION_HTML,headers={"Cache-Control":"no-store"})
    return app

app=create_agent_app()
