from fastapi.testclient import TestClient
from PIL import Image
from pathlib import Path
from types import SimpleNamespace
import json
import shutil
import pytest
import sofia_harness.agent_api as agent_api
from sofia_harness.archive_index import build_archive_index
from sofia_harness.agent_ui import HTML
from sofia_harness.overlay_inspection_ui import HTML as OVERLAY_HTML


@pytest.fixture(autouse=True)
def disable_environment_auth_by_default(monkeypatch):
    """Unit API tests opt into auth explicitly instead of inheriting a developer .env."""
    monkeypatch.setenv("PAST_FORWARD_AUTH_REQUIRED","false")


class FakeResponse:
    id="response-1";output=[]
    output_text="What part of the attached image should I digitize?"
class FakeResponses:
    calls=[]
    def create(self,**kwargs):self.calls.append(kwargs);return FakeResponse()
class FakeOpenAI:
    def __init__(self,**kwargs):self.responses=FakeResponses()


class FakeSearchResponses:
    def create(self,**kwargs):
        assert kwargs["model"]=="gpt-5.6-terra"
        assert "Resources:" in kwargs["input"] and "[R1]" not in kwargs["input"]
        return SimpleNamespace(output_text="Георги Димитров умира на 67 години. [R1]")
    def stream(self,**kwargs):
        assert kwargs["model"]=="gpt-5.6-terra"
        return FakeSearchStream()


class FakeSearchStream:
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def __iter__(self):
        yield SimpleNamespace(type="response.output_text.delta",delta="Supported answer. ")
        yield SimpleNamespace(type="response.output_text.delta",delta="[R1]")


class FakeSearchClient:
    responses=FakeSearchResponses()


class ToolCallingResponses:
    calls=[]
    def __init__(self):self.turn=0
    def create(self,**kwargs):
        self.calls.append(kwargs);self.turn+=1
        if self.turn==1:
            call=SimpleNamespace(type="function_call",name="digitize_attached_images",call_id="call-1",
                                 arguments=json.dumps({"instruction":"digitise the complete page","mode":"full_digitisation"}))
            return SimpleNamespace(id="tool-response-1",output=[call],output_text="")
        return SimpleNamespace(id="tool-response-2",output=[],output_text="Digitisation verified and saved.")


class ToolCallingOpenAI:
    def __init__(self,**kwargs):self.responses=ToolCallingResponses()


class FakeDigitizationAgent:
    calls=[]
    def __init__(self,root):self.root=Path(root)
    def process_incremental(self,source,instruction,progress=None,**kwargs):
        self.calls.append((str(source),instruction,kwargs))
        folder=self.root/"verified-run";folder.mkdir(parents=True,exist_ok=True)
        stored_source=folder/"scan.jpg";shutil.copy2(source,stored_source)
        markdown=folder/"scan.md";markdown.write_text("## ЗАГЛАВИЕ\n\nПЛАЧИ народът.",encoding="utf-8")
        articles=folder/"scan.articles.json";articles.write_text("{}",encoding="utf-8")
        audit=folder/"audit";audit.mkdir(exist_ok=True)
        state=audit/"incremental-state.json";state.write_text("{}",encoding="utf-8")
        if progress:
            progress("Independent Sol verification","1 article")
            progress("Text disagreement resolved","ЗАГЛАВИЕ")
        return {"source_name":"scan.jpg","source_path":str(stored_source),"folder":str(folder),
                "markdown_path":str(markdown),"articles_path":str(articles),"layout_path":str(state),
                "overlay_path":None,"semantic_path":None,"semantic_overlay_path":None,
                "article_count":1,"confidence":.99,"status":"complete","partial":False,
                "omissions_found":[],"text_verified_articles":["article-1"],
                "text_verification_pending":[]}


class FakeSupabaseUserResponse:
    status=200
    def __init__(self,email):self.email=email
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def read(self):return json.dumps({"id":"user-"+self.email,"email":self.email}).encode("utf-8")


def test_attachment_without_instruction_is_decided_by_llm(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(agent_api,"OpenAI",FakeOpenAI)
    image=tmp_path/"scan.jpg";Image.new("RGB",(20,20),"white").save(image)
    client=TestClient(agent_api.create_agent_app(tmp_path/"digitized",api_enabled=True))
    with image.open("rb") as handle:
        response=client.post("/api/agent/chat",data={"message":"","conversation_id":""},files={"files":("scan.jpg",handle,"image/jpeg")})
    assert response.status_code==202
    job=client.get("/api/agent/jobs/"+response.json()["job_id"]).json()
    assert job["status"]=="complete" and "attached image" in job["answer"]
    assert FakeResponses.calls and len(FakeResponses.calls[-1]["tools"])==4


def test_agent_tools_include_records_and_digitisation():
    names={tool["name"] for tool in agent_api.TOOLS}
    assert names=={"digitize_attached_images","list_saved_outputs","read_saved_output","resume_interrupted_run"}


def test_ui_renders_layout_inspection_artifacts():
    assert "class=layout-preview" in HTML
    assert "Open agent layout overlay" in HTML
    assert "View agent state" in HTML
    assert "Resume last interrupted run" in HTML
    assert "aria-live=polite" in HTML
    assert "r.layout_only" in HTML
    assert "Partial result saved" in HTML


def test_ui_shows_live_estimated_digitisation_progress():
    assert 'aria-label="Estimated digitisation progress"' in HTML
    assert "class=progress-fill" in HTML
    assert "Estimated progress" in HTML
    assert "setInterval(paintProgress,250)" in HTML
    assert "showWorking(e.detail||e.stage,e.percent)" in HTML
    assert "j.progress.percent" in HTML
    assert "≈${Math.round(Number(e.percent))}%" in HTML


def test_progress_estimate_is_monotonic_across_pages_and_exact_only_at_completion():
    job={"page_total":2}
    samples=[
        agent_api._estimated_progress(job,"Thinking","Understanding the request"),
        agent_api._estimated_progress(job,"Page","1/2 · first.png"),
        agent_api._estimated_progress(job,"Article saved","1 · First article"),
        agent_api._estimated_progress(job,"Independent article read","1/4 · First article"),
        agent_api._estimated_progress(job,"Independent article read","4/4 · Fourth article"),
        agent_api._estimated_progress(job,"Complete","4 articles"),
        agent_api._estimated_progress(job,"Page","2/2 · second.png"),
        agent_api._estimated_progress(job,"Article saved","1 · Another article"),
        agent_api._estimated_progress(job,"Saved Markdown","second.md"),
    ]
    assert samples==sorted(samples)
    assert all(value<100 for value in samples)
    assert job["progress"]["page_index"]==2
    assert job["progress"]["estimated"] is True
    assert agent_api._estimated_progress(job,"Complete","")==100
    assert job["progress"]["estimated"] is False


def test_ui_defaults_to_wired_search_and_keeps_digitize_available():
    assert "What do you want to search for?" in HTML
    assert ">Search</button>" in HTML and ">Digitize</button>" in HTML
    assert "id=searchMode class=active" in HTML
    assert "if(mode==='search')return submitSearch()" in HTML
    assert "pfFetch('/api/search/stream?q='" in HTML
    assert "headers.set('Authorization','Bearer '+data.session.access_token)" in HTML
    assert "Your session has expired. Please sign in again." in HTML
    assert "response.body.getReader()" in HTML and "new TextDecoder()" in HTML
    assert "answerNode.textContent+=item.delta" in HTML
    assert "`<h2>Resources</h2>`+resources" in HTML
    assert "<section class=result><h2>Resources</h2>" not in HTML
    assert ".app{width:min(1040px,calc(100vw - 330px))" in HTML
    assert "answerNode.className='bubble agent'" in HTML
    assert "Making records from the past available in the future." not in HTML
    assert "Continue with Google" in HTML
    assert "signInWithPassword" in HTML and "signUp(credentials)" in HTML
    assert "/assets/past-forward-hero.png" in HTML
    assert "/assets/landing-paper/rabotnichesko-delo.jpg" in HTML
    assert "pfSetupSparkle" in HTML
    assert "targetRadius=clamp(58+speed*92,58,230)" in HTML
    assert "function fadeTrail()" in HTML and "tailUntil=now+850" in HTML
    assert "requestAnimationFrame(draw)" in HTML and "pointerleave',leave" in HTML
    assert ".search-landing #hero{left:0!important;right:0!important;width:100vw!important}" in HTML
    assert "auth-copy.register-mode #emailSubmit" in HTML
    assert ".auth-copy.register-mode .auth-switch{border-color:#f4c0a4;background:#e8a07c" in HTML
    assert ">Library</button>" in HTML and "View Library for Free" in HTML
    assert "submitMode.insertAdjacentHTML('afterend','<button id=libraryMode" in HTML
    assert "class=library-divider>OR</div>" in HTML and "/assets/sturshel-shrug.png" in HTML
    assert "page.image_path" in HTML and "page.inspect_url" in HTML
    assert "libraryColumns.addEventListener('click'" in HTML and "location.href=link.href" in HTML
    assert "pastForwardSearchStateV1" in HTML
    assert "sessionStorage.setItem(SEARCH_STATE_KEY" in HTML
    assert "sessionStorage.getItem(SEARCH_STATE_KEY)" in HTML
    assert "copy.querySelector('#working')?.remove()" in HTML
    assert "const streamedSearch=submitSearch;submitSearch=async function()" in HTML
    assert "window.addEventListener('pagehide',persistSearchState);restoreSearchState()" in HTML
    assert "if(!copy.querySelector('.bubble,.result'))return" in HTML
    assert "function startNewSearch(event)" in HTML
    assert "startingNewSearch=true;sessionStorage.removeItem(SEARCH_STATE_KEY);location.href='/'" in HTML
    assert "document.querySelectorAll('.app .home-link')" in HTML
    assert "function showFreshDigitize()" in HTML
    assert "What do you want to digitize?" in HTML
    assert "Digitize. Make the past accessible going forward." in HTML
    assert "if(next==='digitize')persistSearchState()" in HTML
    assert "if(next==='search'&&mode==='search'&&!restoreSearchState())showFreshSearch()" in HTML


def test_home_injects_supabase_auth_configuration(tmp_path):
    client=TestClient(agent_api.create_agent_app(tmp_path/"digitized",auth_required=True,
        supabase_url="https://example.supabase.co",supabase_publishable_key="sb_publishable_test-key"))
    home=client.get("/")
    assert home.status_code==200
    assert "PF_AUTH_REQUIRED=true" in home.text and "PF_AUTH_CONFIGURED=true" in home.text
    assert "https://example.supabase.co" in home.text and "sb_publishable_test-key" in home.text
    hero=client.get("/assets/past-forward-hero.png")
    assert hero.status_code==200 and hero.headers["content-type"]=="image/png"
    sturshel=client.get("/assets/sturshel-page.png")
    assert sturshel.status_code==200 and sturshel.headers["content-type"]=="image/png"
    collage=client.get("/assets/newspaper-collage.png")
    assert collage.status_code==200 and collage.headers["content-type"]=="image/png"
    assert client.get("/api/health").status_code==401
    assert client.get("/api/library").status_code==200
    assert client.get("/overlay-inspection").status_code==200


def test_digitization_is_visible_to_configured_admin_and_forbidden_to_other_users(tmp_path,monkeypatch):
    emails={"admin-token":"yavor.b.popov@gmail.com","reader-token":"reader@example.com"}
    def fake_urlopen(request,timeout=0):
        token=request.get_header("Authorization").removeprefix("Bearer ")
        return FakeSupabaseUserResponse(emails[token])
    monkeypatch.setattr(agent_api,"urlopen",fake_urlopen)
    client=TestClient(agent_api.create_agent_app(
        tmp_path/"digitized",api_enabled=False,auth_required=True,
        supabase_url="https://example.supabase.co",
        supabase_publishable_key="sb_publishable_test-key",
        admin_emails=["Yavor.B.Popov@gmail.com"],
    ))
    admin={"Authorization":"Bearer admin-token"};reader={"Authorization":"Bearer reader-token"}
    assert client.get("/api/auth/me").status_code==401
    assert client.get("/api/auth/me",headers=admin).json()=={
        "authenticated":True,"email":"yavor.b.popov@gmail.com","is_admin":True,
    }
    assert client.get("/api/auth/me",headers=reader).json()["is_admin"] is False
    assert client.get("/api/agent/outputs",headers=reader).status_code==403
    assert client.post("/api/agent/resume",headers=reader).status_code==403
    assert client.post("/api/overlay/sources/run/place",headers=reader).status_code==403
    assert client.get("/api/agent/outputs",headers=admin).status_code==200
    assert client.post("/api/agent/resume",headers=admin).status_code==503


def test_ui_uses_server_admin_role_and_hides_digitization_for_non_admins():
    assert "fetch('/api/auth/me'" in HTML
    assert "profile.is_admin===true" in HTML
    assert "digitizeMode.hidden=!window.pfIsAdmin" in HTML
    assert "if(next==='digitize'&&!window.pfIsAdmin)return" in HTML
    assert "yavor.b.popov@gmail.com" not in HTML


def test_landing_can_run_without_authentication(tmp_path):
    client=TestClient(agent_api.create_agent_app(tmp_path/"digitized",auth_required=False,landing_enabled=True))
    home=client.get("/")
    assert home.status_code==200
    assert "PF_AUTH_REQUIRED=false" in home.text and "PF_LANDING_ENABLED=true" in home.text
    assert "Enter Past Forward" in home.text
    assert "Search the archive. Learn from the past to build a <strong>BETTER</strong> future." in home.text
    assert "box-shadow:10px 10px 0" not in home.text


def test_public_submission_ui_is_bulk_and_unwired():
    assert ">Submit for digitization</button>" in HTML
    assert "id=submitFiles" in HTML and "multiple hidden" in HTML
    assert "id=submitEmail type=email required" in HTML
    assert "id=submitConsent type=checkbox required" in HTML
    assert "Notes (optional)" in HTML
    assert "PF-${date}-" in HTML
    assert "No files have been uploaded yet." in HTML


def test_search_and_digitize_share_past_forward_title_with_mode_subtitle():
    assert '<h2 class=prompt><a class=home-link href="/" aria-label="Past Forward home">Past Forward</a></h2>' in HTML
    assert "<p id=prompt class=hero-subtitle>What do you want to search for?</p>" in HTML
    assert "What do you want to digitize?" in HTML
    assert HTML.count('aria-label="Past Forward home"')>=3


def test_search_starts_centered_then_moves_composer_to_bottom():
    assert "document.body.classList.add('search-landing')" in HTML
    assert ".search-landing #hero{top:50%;left:20px;right:20px" in HTML
    assert ".search-landing .composer{top:50%;left:0;right:0" in HTML
    assert ".composer,.search-landing .composer{left:280px;right:auto;width:min(1040px" in HTML
    assert ".box,.search-landing .box{width:100%;max-width:none;margin:0}" in HTML
    assert "document.body.classList.remove('search-landing')" in HTML


def test_mobile_uses_past_forward_artwork_and_desktop_notice():
    assert 'class=mobile-notice' in HTML
    assert 'src="/assets/past-forward-hero.png"' in HTML
    assert "Currently optimized for desktop" in HTML
    assert "@media(max-width:760px)" in HTML
    assert "let promptText=document.getElementById('prompt')" in HTML
    assert "promptText.textContent=searching?" in HTML


def test_archive_search_endpoint_returns_article_result_and_source_url(tmp_path):
    root=tmp_path/"digitized";run=root/"run";run.mkdir(parents=True)
    source=run/"IMG_1_nm_07_04_1949_page1.png";Image.new("RGB",(20,20),"white").save(source)
    payload={"source":source.name,"source_sha256":"abc","articles":[{
        "article_id":"news","heading":"Важна новина","verbatim_text":"Георги Димитров пристигна в София."
    }]}
    (run/"page.articles.json").write_text(json.dumps(payload,ensure_ascii=False),encoding="utf-8")
    workspace=tmp_path/"search_data";build_archive_index(root,workspace)
    client=TestClient(agent_api.create_agent_app(root,api_enabled=False,search_db=workspace/"index/archive.db",
        search_answers_enabled=True,search_client=FakeSearchClient()))
    response=client.get("/api/search",params={"q":"Георги София"})
    assert response.status_code==200
    result=response.json()["results"][0]
    assert result["heading"]=="Важна новина" and result["match_type"]=="lexical"
    assert result["source_url"].endswith(source.name)
    assert result["inspect_url"].startswith("/overlay-inspection.html?source=")
    assert "article=" not in result["inspect_url"]
    assert "verbatim_text" not in result
    assert response.json()["answer"]=="Георги Димитров умира на 67 години. [R1]"
    assert response.json()["resources"][0]["resource_id"]=="R1"


def test_archive_search_reports_missing_index(tmp_path):
    client=TestClient(agent_api.create_agent_app(tmp_path/"digitized",api_enabled=False,search_db=tmp_path/"missing.db"))
    assert client.get("/health").json()["search_ready"] is False
    assert client.get("/api/search",params={"q":"test"}).status_code==503


def test_default_paths_are_independent_of_working_directory(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    client=TestClient(agent_api.create_agent_app(api_enabled=False,auth_required=False))
    health=client.get("/health").json()
    assert health["search_ready"] is True
    assert Path(health["output_root"])==agent_api.PROJECT_ROOT/"digitized"


def test_archive_search_stream_emits_incremental_answer_and_final_resources(tmp_path):
    root=tmp_path/"digitized";run=root/"run";run.mkdir(parents=True)
    source=run/"IMG_1_nm_07_04_1949_page1.png";Image.new("RGB",(20,20),"white").save(source)
    payload={"source":source.name,"source_sha256":"stream","articles":[{
        "article_id":"news","heading":"Archive report","verbatim_text":"Georgi Dimitrov returned to Sofia with the delegation."
    }]}
    (run/"page.articles.json").write_text(json.dumps(payload,ensure_ascii=False),encoding="utf-8")
    workspace=tmp_path/"search_data";build_archive_index(root,workspace)
    client=TestClient(agent_api.create_agent_app(root,api_enabled=False,search_db=workspace/"index/archive.db",
        search_answers_enabled=True,search_client=FakeSearchClient()))
    response=client.get("/api/search/stream",params={"q":"What happened with the delegation?"})
    assert response.status_code==200
    events=[json.loads(line) for line in response.text.splitlines()]
    assert [event["type"] for event in events]==["resources","answer_delta","answer_delta","done"]
    assert "".join(event.get("delta","") for event in events)=="Supported answer. [R1]"
    assert events[-1]["resources"][0]["resource_id"]=="R1"
    assert response.headers["x-accel-buffering"]=="no"


def test_age_question_is_calculated_locally_from_cited_evidence(tmp_path):
    root=tmp_path/"digitized";run=root/"run";run.mkdir(parents=True)
    source=run/"IMG_1_nm_07_04_1949_page1.png";Image.new("RGB",(20,20),"white").save(source)
    text="На 2 юли след продължително боледуване почина Георги Димитров. Той е роден на 18 юни 1882 година."
    payload={"source":source.name,"source_sha256":"age","articles":[{"article_id":"bio","heading":"Биография","verbatim_text":text}]}
    (run/"page.articles.json").write_text(json.dumps(payload,ensure_ascii=False),encoding="utf-8")
    workspace=tmp_path/"search_data";build_archive_index(root,workspace)
    client=TestClient(agent_api.create_agent_app(root,api_enabled=False,search_db=workspace/"index/archive.db",search_answers_enabled=False))
    data=client.get("/api/search",params={"q":"На колко години умира Георги Димитров?"}).json()
    assert data["answer"]=="Георги Димитров умира на 67 години. [R1]"
    assert data["resources"][0]["heading"]=="Биография"
    english=client.get("/api/search",params={"q":"how old did georgi dimitrov die"}).json()
    assert english["answer"]=="Georgi Dimitrov died at age 67. [R1]"
    assert len(english["resources"])==1 and english["resource_budget"]==3


def test_resource_budget_adapts_to_question_scope():
    assert agent_api.resource_budget("How old was he?")==3
    assert agent_api.resource_budget("Were there indications that he was sick?")==5
    assert agent_api.resource_budget("Give me examples of mocking attitudes towards USA")==8
    assert agent_api.resource_budget("List all references throughout the archive")==12


def test_search_answer_prompt_matches_language_and_stays_source_neutral():
    assert "Always answer in the same language as the user's question" in agent_api.SEARCH_ANSWER_PROMPT
    assert "Be OBJECTIVE AND NEUTRAL. it passes information as it has been stated by a given source - just as it has been given - nothing more, nothing less. Users draw their own conclusions about the past, what to learn from it and how to avoid repeating past mistakes in the future." in agent_api.SEARCH_ANSWER_PROMPT


def test_overlay_inspection_is_isolated_and_exposes_layout_contract(tmp_path):
    client=TestClient(agent_api.create_agent_app(tmp_path/"digitized",api_enabled=True))
    response=client.get("/overlay-inspection")
    assert response.status_code==200
    assert response.headers["cache-control"]=="no-store"
    assert "Overlay Inspection" in response.text
    assert "/api/overlay/demo" in OVERLAY_HTML
    assert "blockStyle" in OVERLAY_HTML
    assert "polygon" in OVERLAY_HTML and "fitText" in OVERLAY_HTML
    assert "/place" not in OVERLAY_HTML and "still being prepared and audited" in OVERLAY_HTML
    home=client.get("/")
    assert home.status_code==200 and "PF_AUTH_REQUIRED=false" in home.text
    assert "__SUPABASE_URL__" not in home.text


def test_overlay_sources_pairs_saved_photograph_with_markdown(tmp_path):
    root=tmp_path/"digitized";run=root/"20260721-page-one";run.mkdir(parents=True)
    (run/"page-one.md").write_text("# Headline\n\nArticle text",encoding="utf-8")
    Image.new("RGB",(20,30),"white").save(run/"page-one.png")
    client=TestClient(agent_api.create_agent_app(root,api_enabled=True))
    response=client.get("/api/overlay/sources")
    assert response.status_code==200
    assert response.json()[0]["name"]=="page-one"
    assert response.json()[0]["image_url"].endswith("/page-one.png")
    assert response.json()[0]["markdown_url"].endswith("/page-one.md")
    assert client.get(response.json()[0]["image_url"]).status_code==200


def test_source_specific_overlay_view_loads_saved_image_and_regions(tmp_path):
    root=tmp_path/"digitized";run=root/"run";run.mkdir(parents=True)
    image=run/"page.png";Image.new("RGB",(20,30),"white").save(image)
    plan={"coverage_complete":True,"blocks":[{"block_id":"r1","article_id":"news","article_label":"Headline","polygon":[[0,0],[1000,0],[1000,1000],[0,1000]],"text":"Exact text","overlay_text":"Exact text","role":"body","confidence":.9,"font_size_ratio":.01,"line_height":1.0}],
          "audit":{"accepted":True,"coverage_complete":True,"article_identity_correct":True,"line_ownership_exact":True,"geometry_matches_source":True,"typography_matches_source":True,"regions_filled_to_source_bounds":True,"issues":[]}}
    (run/"page.overlay.json").write_text(json.dumps(plan),encoding="utf-8")
    (run/"page.articles.json").write_text(json.dumps({"articles":[{"article_id":"news","article_order":1,"heading":"Headline","verbatim_text":"First hyphen-\nated line.\nSecond line.\n\nNext paragraph."}]}),encoding="utf-8")
    client=TestClient(agent_api.create_agent_app(root,api_enabled=False))
    payload=client.get("/api/overlay/source",params={"source_id":"run/page.png"}).json()
    assert payload["source_name"]=="page.png" and payload["blocks"][0]["article_id"]=="news"
    assert payload["transcription_sections"]==[{"article_id":"news","heading":"Headline","paragraphs":["First hyphenated line. Second line.","Next paragraph."]}]
    assert payload["transcription_filename"]=="page.txt"
    assert payload["transcription_text"]=="Headline\n\nFirst hyphenated line. Second line.\n\nNext paragraph.\n"


def test_source_overlay_still_opens_full_page_without_vision_plan(tmp_path):
    root=tmp_path/"digitized";run=root/"run";run.mkdir(parents=True)
    Image.new("RGB",(20,30),"white").save(run/"page.png")
    (run/"page.articles.json").write_text(json.dumps({"articles":[]}),encoding="utf-8")
    response=TestClient(agent_api.create_agent_app(root,api_enabled=False)).get("/api/overlay/source",params={"source_id":"run/page.png"})
    assert response.status_code==200
    assert response.json()["image_url"].endswith("/run/page.png")
    assert response.json()["blocks"]==[] and response.json()["coverage_complete"] is False
    placement=TestClient(agent_api.create_agent_app(root,api_enabled=False)).post("/api/overlay/sources/run/page.png/place")
    assert placement.status_code==503
    assert "API placement is disabled" in placement.json()["detail"]


def test_search_source_recovers_agent_layout_and_sliven_text():
    project=Path(__file__).parents[1];root=project/"digitized";folder="20260720-1624_IMG_0933_rd_07_04_1949_page3_incremental";image=f"{folder}/20260720-1624_IMG_0933_rd_07_04_1949_page3.png"
    if not (root/image).is_file():pytest.skip("development search fixture is unavailable")
    payload=TestClient(agent_api.create_agent_app(root,api_enabled=False)).get("/api/overlay/source",params={"source_id":image}).json()
    sliven=[block for block in payload["blocks"] if block["article_id"]=="sliven_tightens_ranks_around_bkp"]
    assert [(block["role"],block["block_id"]) for block in sliven]==[("heading","sliven_heading"),("body","sliven_body")]
    assert "Сливен, 3 юли" in sliven[1]["text"] and payload["coverage_complete"] is True
    assert all(block["font_size_ratio"]>0 and block["line_height"]>0 for block in payload["blocks"])
    identities={block["block_id"]:block["article_id"] for block in payload["blocks"]}
    assert identities["dear_teacher_body"]=="dear_teacher_and_comrade"
    assert identities["burgas_c1"]=="burgas_mass_help_to_villagers"
    assert identities["popovo_c1"]=="popovo_resolve_to_continue_dimitrovs_work"
    assert identities["krumovgrad_body"]=="krumovgrad_pledges_to_fulfil_testament"
    assert identities["issue_number"]=="page_metadata_memorial_banner"
    assert identities["sofia_caption"]=="sofia_mourning_evenings"
    records=json.loads((root/folder/"20260720-1624_IMG_0933_rd_07_04_1949_page3.articles.json").read_text(encoding="utf-8"))["articles"]
    normalized=lambda value:"".join(character for character in value.casefold() if character.isalnum())
    for article in records:
        placed="\n".join(block["text"] for block in payload["blocks"] if block["article_id"]==article["article_id"])
        assert normalized(placed)==normalized(article["verbatim_text"]),article["article_id"]


def test_overlay_viewer_has_no_manual_build_or_upload_controls():
    assert "Build overlay" not in OVERLAY_HTML
    assert 'type="file"' not in OVERLAY_HTML
    assert 'type="range"' not in OVERLAY_HTML and "markdownInput" not in OVERLAY_HTML
    assert "/api/overlay/demo" in OVERLAY_HTML
    assert "inspectBlock" in OVERLAY_HTML and "overlaySwitch" in OVERLAY_HTML
    assert "<legend>Overlay</legend>" in OVERLAY_HTML and ">On</label>" in OVERLAY_HTML and ">Off</label>" in OVERLAY_HTML
    assert "Click to inspect an article block" in OVERLAY_HTML
    assert "Back to full page" in OVERLAY_HTML and "clearInspection" in OVERLAY_HTML and "exitInspection" in OVERLAY_HTML
    assert 'aria-label="Zoom in"' in OVERLAY_HTML and 'aria-label="Zoom out"' in OVERLAY_HTML
    assert "changeZoom" in OVERLAY_HTML and "canvas.style.width" in OVERLAY_HTML
    assert "class=viewer-column" in OVERLAY_HTML and ".zoom-controls{position:relative" in OVERLAY_HTML
    assert "onpointerdown" in OVERLAY_HTML and "stage.scrollLeft=scrollX-dx" in OVERLAY_HTML
    assert "if(!didPan&&Math.abs(dx)+Math.abs(dy)>4)" in OVERLAY_HTML
    assert "stage.classList.add('dragging');stage.setPointerCapture(e.pointerId)" in OVERLAY_HTML
    pointer_down=OVERLAY_HTML.split("onpointerdown=e=>",1)[1].split(";stage.onpointermove",1)[0]
    assert "setPointerCapture" not in pointer_down
    assert "overlaySwitch.checked=true" in OVERLAY_HTML and "stage.classList.remove('overlay-off'" in OVERLAY_HTML
    assert 'href="/">Back</a>' not in OVERLAY_HTML
    assert "That overlay block is hidden" not in OVERLAY_HTML
    assert "/api/overlay/demos" in OVERLAY_HTML
    assert "Past Forward" in OVERLAY_HTML
    assert "Compare the digitized text" in OVERLAY_HTML
    assert "new URLSearchParams(location.search)" in OVERLAY_HTML
    assert "3000-(Date.now()-started)" in OVERLAY_HTML and "preload.decode" in OVERLAY_HTML
    assert "animation:letter-rush 1.15s" in OVERLAY_HTML and "letter-rush 1.15s cubic-bezier(.2,.9,.3,1) infinite" not in OVERLAY_HTML
    assert "animation:light-pulse" in OVERLAY_HTML
    assert ">View transcription</button>" in OVERLAY_HTML
    assert "Source transcription" in OVERLAY_HTML and "transcriptionModal.showModal()" in OVERLAY_HTML
    assert 'aria-label="Download transcription as TXT"' in OVERLAY_HTML
    assert "transcription_filename||'transcription.txt'" in OVERLAY_HTML
    assert "position:sticky" in OVERLAY_HTML and "text/plain;charset=utf-8" in OVERLAY_HTML
    assert 'class=brand href="/" aria-label="Past Forward home"' in OVERLAY_HTML
    assert "goToPreviousPage()" in OVERLAY_HTML and "history.back()" in OVERLAY_HTML
    assert "class=history-back" in OVERLAY_HTML and "← Back" in OVERLAY_HTML
    assert "class=source-controls" in OVERLAY_HTML and "class=statusline hidden" in OVERLAY_HTML
    assert "<span id=pageName class=page-name>" in OVERLAY_HTML and "<select id=pageSelect hidden" in OVERLAY_HTML
    assert "class=inspection-actions" in OVERLAY_HTML and "pageName.textContent=current.source_name" in OVERLAY_HTML
    assert ".inspect-hint{width:max-content;max-width:100%;margin:0 auto" in OVERLAY_HTML
    assert "function centerBlock(el)" in OVERLAY_HTML
    assert "requestedArticle" not in OVERLAY_HTML and "articleId" not in OVERLAY_HTML
    assert "return loadPage(selected,requestedSource)" in OVERLAY_HTML
    assert "grid-template-columns:minmax(0,2fr) minmax(0,3fr)" in OVERLAY_HTML


def test_unaccepted_legacy_coordinate_demos_are_not_listed():
    root=Path(__file__).parents[1]/"digitized"
    client=TestClient(agent_api.create_agent_app(root,api_enabled=True))
    assert {item["id"] for item in client.get("/api/overlay/demos").json()}=={"img-0933","georgi"}


def test_completed_result_links_to_overlay_inspection():
    assert '>Inspect Source</a>' in HTML
    assert 'href="/overlay-inspection.html" target=_blank' in HTML


def test_library_inspection_back_link_preserves_guest_library():
    assert "page.inspect_url+'&from=library'" in HTML
    assert "get('from')==='library'" in agent_api.OVERLAY_INSPECTION_HTML
    assert "location.href='/?view=library'" in agent_api.OVERLAY_INSPECTION_HTML


def test_direct_resume_reports_missing_checkpoint_without_planning_call(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(agent_api,"OpenAI",FakeOpenAI)
    client=TestClient(agent_api.create_agent_app(tmp_path/"digitized",api_enabled=True))
    response=client.post("/api/agent/resume",data={"run_folder":"","instruction":"continue"})
    assert response.status_code==202
    job=client.get("/api/agent/jobs/"+response.json()["job_id"]).json()
    assert job["status"]=="failed" and "No interrupted run" in job["error"]


def test_full_digitisation_ui_tool_routes_to_verified_incremental_agent(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent_api,"OpenAI",ToolCallingOpenAI)
    monkeypatch.setattr(agent_api,"DigitizationAgent",FakeDigitizationAgent)
    FakeDigitizationAgent.calls.clear();ToolCallingResponses.calls.clear()
    image=tmp_path/"scan.jpg";Image.new("RGB",(20,20),"white").save(image)
    client=TestClient(agent_api.create_agent_app(tmp_path/"digitized",api_enabled=True))
    with image.open("rb") as handle:
        response=client.post("/api/agent/chat",data={"message":"digitise the complete page"},
                             files={"files":("scan.jpg",handle,"image/jpeg")})
    assert response.status_code==202
    job=client.get("/api/agent/jobs/"+response.json()["job_id"]).json()
    assert job["status"]=="complete" and FakeDigitizationAgent.calls
    assert job["results"][0]["text_verified_articles"]==["article-1"]
    assert job["results"][0]["text_verification_pending"]==[]
    assert any(event["stage"]=="Independent Sol verification" for event in job["events"])
    percentages=[event["percent"] for event in job["events"]]
    assert percentages==sorted(percentages)
    assert percentages[-1]==100
    assert all("at" in event and "estimated" in event for event in job["events"])
    assert job["progress"]["percent"]==100 and job["progress"]["estimated"] is False
    assert any(call.get("previous_response_id")=="tool-response-1" for call in ToolCallingResponses.calls)
