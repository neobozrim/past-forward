from __future__ import annotations

import base64, json, mimetypes, re, shutil, uuid, time
from concurrent.futures import ThreadPoolExecutor,as_completed
from threading import Lock
import cv2,numpy as np
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image,ImageDraw,ImageEnhance,ImageFilter,ImageFont,ImageOps
from .models import PageLayout,Transcription
from .pen_agent import PenAgent
from .incremental_agent import IncrementalArticleAgent

load_dotenv()


class Inspection(BaseModel):
    final_text: str
    omissions_found: list[str] = Field(default_factory=list)
    corrections_made: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class LayoutInspection(BaseModel):
    text_coverage: float = Field(ge=0, le=1)
    all_visible_text_covered: bool
    no_text_edges_clipped: bool
    headings_complete: bool
    subheadings_complete: bool
    article_grouping_coherent: bool
    semantic_plan_correct: bool
    column_structure_correct: bool
    reading_order_correct: bool
    missing_or_clipped_text: list[str] = Field(default_factory=list)
    heading_or_subheading_issues: list[str] = Field(default_factory=list)
    region_issues: list[str] = Field(default_factory=list)
    repair_instructions: list[str] = Field(default_factory=list)

    def failures(self) -> list[str]:
        failures=[]
        if self.text_coverage < .985: failures.append(f"visual text coverage {self.text_coverage:.3f} is below 0.985")
        if not self.all_visible_text_covered: failures.append("visible text is omitted")
        if not self.no_text_edges_clipped: failures.append("one or more polygon edges clip text")
        if not self.headings_complete: failures.append("one or more headings are missing or incomplete")
        if not self.subheadings_complete: failures.append("one or more subheadings are missing or incomplete")
        if not self.article_grouping_coherent: failures.append("article grouping is incoherent")
        if not self.semantic_plan_correct: failures.append("semantic heading/article plan is incorrect")
        if not self.column_structure_correct: failures.append("article column structure is incorrect")
        if not self.reading_order_correct: failures.append("article or region reading order is incorrect")
        failures.extend(self.missing_or_clipped_text)
        failures.extend(self.heading_or_subheading_issues)
        failures.extend(self.region_issues)
        return list(dict.fromkeys(failures))


class SemanticHeading(BaseModel):
    id: str
    verbatim_text: str
    level: str
    article_id: str
    polygon: list[list[float]]
    reading_order: int
    confidence: float = Field(ge=0,le=1)


class SemanticArticle(BaseModel):
    id: str
    descriptive_label: str
    polygon: list[list[float]]
    heading_ids: list[str] = Field(default_factory=list)
    body_column_count: int = Field(ge=0)
    reading_order: int
    confidence: float = Field(ge=0,le=1)


class SemanticPagePlan(BaseModel):
    headings: list[SemanticHeading]
    articles: list[SemanticArticle]
    notes: list[str] = Field(default_factory=list)

    def validate_plan(self) -> list[str]:
        errors=[];heading_ids=[x.id for x in self.headings];article_ids=[x.id for x in self.articles]
        if len(heading_ids)!=len(set(heading_ids)):errors.append("duplicate semantic heading id")
        if len(article_ids)!=len(set(article_ids)):errors.append("duplicate semantic article id")
        if len([x.reading_order for x in self.headings])!=len(set(x.reading_order for x in self.headings)):
            errors.append("duplicate semantic heading reading_order")
        if len([x.reading_order for x in self.articles])!=len(set(x.reading_order for x in self.articles)):
            errors.append("duplicate semantic article reading_order")
        for heading in self.headings:
            if heading.article_id not in article_ids:errors.append(f"{heading.id}: unknown article {heading.article_id}")
            if len(heading.polygon)!=4:errors.append(f"{heading.id}: heading polygon must have four corners")
        for article in self.articles:
            if len(article.polygon)<3:errors.append(f"{article.id}: article envelope must have at least three points")
            for heading_id in article.heading_ids:
                if heading_id not in heading_ids:errors.append(f"{article.id}: unknown heading {heading_id}")
        return errors


class PlannedLayout(BaseModel):
    semantic_plan: SemanticPagePlan
    page_layout: PageLayout


class ReadingRegion(BaseModel):
    id: str
    label: str
    x0: int = Field(ge=0, le=1000)
    y0: int = Field(ge=0, le=1000)
    x1: int = Field(ge=0, le=1000)
    y1: int = Field(ge=0, le=1000)
    reading_order: int


class ReadingPlan(BaseModel):
    regions: list[ReadingRegion]
    scope_description: str


TRANSCRIBE = """Digitize the complete visible document diplomatically. Read the entire original image,
not selected crops. Preserve Bulgarian historical spelling, ѣ/ѫ/ъ, capitalization, punctuation,
paragraphs, headings, signatures, stamps and line-wrap hyphens. Never omit a short heading or isolated
word. Do not modernize, summarize or invent. Transcribe only glyphs supported by pixels; never substitute
a contextually plausible Bulgarian word. Return the complete transcription."""

BLINDED_READ = """Independently transcribe every visible word in this document from the image alone.
You have not seen another transcription. Preserve columns, reading order, historical spelling and line
breaks. Never complete a sentence from context: when pixels do not support a reading, use [неясно]
instead of guessing. Pay special attention to text crossing crop, column, and line boundaries."""

INSPECT = """Act as an archival transcription adjudicator. Compare the attached ORIGINAL IMAGE
against TWO independently produced reads character by character and region by region. Treat neither
read as authoritative. Resolve every disagreement from visible pixels, not linguistic plausibility. Inspect all four edges,
headings, isolated letters/words, paragraph endings, signatures, stamps, columns, historical Bulgarian
characters, numbers and Latin medical text. Find omissions such as a partially captured heading (for
example АКТЪ becoming А), repair them in final_text, and preserve diplomatic spelling. Do not merely
comment: final_text must be the best complete corrected transcription. If the pixels cannot decide a
disagreement, retain an explicit [неясно: вариант 1 | вариант 2] marker; never choose the more fluent sentence.
You are performing OCR, not editing Bulgarian. Never normalize grammar, spelling, word endings, or
vocabulary. A visibly printed imperative such as ПЛАЧИ must never become the more expected ПЛАЧЕ.

READ A:
"""


class DigitizationAgent:
    def __init__(self, output_root: str | Path="digitized", client=None, model="gpt-5.6-luna",
                 inspector_model="gpt-5.6-sol", layout_model="gpt-5.6-sol", adjudicator_model="gpt-5.6-terra"):
        self.root=Path(output_root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.client=client or OpenAI(timeout=240,max_retries=1);self.model=model;self.inspector_model=inspector_model
        self.layout_model=layout_model;self.adjudicator_model=adjudicator_model

    def process_agentic(self,source:str|Path,instruction:str,progress=None,stop_after_layout=False,max_turns=120,resume_folder:str|Path|None=None):
        """Let a single visual agent mark, crop, transcribe, and revise the page with a pen-like tool loop."""
        return PenAgent(self.root,self.client,self.layout_model,self.model,self.adjudicator_model,self.inspector_model).run(
            source,instruction,progress,stop_after_layout,max_turns,resume_folder)

    def process_incremental(self,source:str|Path,instruction:str,progress=None,max_turns=40,
                            resume_folder:str|Path|None=None,verification_workers=3,
                            min_text_confidence=.80):
        """Digitise directly from the complete page, checkpointing semantic articles without a polygon gate."""
        return IncrementalArticleAgent(
            self.root,
            self.client,
            model=self.layout_model,
            inventory_model=self.adjudicator_model,
            inspector_model=self.inspector_model,
        ).run(
            source,instruction,progress,max_turns=max_turns,resume_folder=resume_folder,
            verification_workers=verification_workers,min_text_confidence=min_text_confidence)

    @staticmethod
    def _image(path:Path):
        mime=mimetypes.guess_type(path)[0] or "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"

    @staticmethod
    def _tokens(text:str)->set[str]:
        return {x.casefold() for x in re.findall(r"[^\W\d_]{3,}",text,flags=re.UNICODE)}

    @staticmethod
    def _atomic_json(path:Path,value:dict):
        temporary=path.with_suffix(path.suffix+".tmp")
        temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
        temporary.replace(path)

    def recover_accepted_layout(self, run_folder:str|Path, candidate_index:int) -> dict:
        """Finalize a fully audited candidate after interruption or a now-obsolete deterministic rejection."""
        folder=Path(run_folder).resolve();audit=folder/"audit";checkpoint_path=audit/"region-checkpoint.json"
        checkpoint=json.loads(checkpoint_path.read_text(encoding="utf-8"))
        history=checkpoint.get("layout_validation_history",[])
        accepted=next((x for x in history if x.get("attempt")==candidate_index+1),None)
        inspection=(accepted or {}).get("inspection",{})
        required=("all_visible_text_covered","no_text_edges_clipped","headings_complete","subheadings_complete",
                  "article_grouping_coherent","semantic_plan_correct","column_structure_correct","reading_order_correct")
        if not accepted or any(inspection.get(x) is not True for x in required):
            raise ValueError("candidate has not passed the complete hierarchical visual audit")
        layout=PageLayout.model_validate_json((audit/f"layout-candidate-{candidate_index}.json").read_text(encoding="utf-8"))
        semantic_plan=SemanticPagePlan.model_validate_json((audit/f"semantic-candidate-{candidate_index}.json").read_text(encoding="utf-8"))
        errors=layout.validate_layout()+semantic_plan.validate_plan()
        if errors:raise ValueError("candidate fails current structural validation: "+"; ".join(errors))
        source=Path(checkpoint["source"]);raw=Image.open(source).convert("RGB")
        layout_path=audit/"layout.json";semantic_path=audit/"semantic-plan.json"
        overlay_path=audit/"layout-regions.png";semantic_overlay_path=audit/"semantic-regions.png"
        self._atomic_json(layout_path,layout.model_dump());self._atomic_json(semantic_path,semantic_plan.model_dump())
        self._save_layout_overlay(raw,layout,{},overlay_path);self._save_semantic_overlay(raw,semantic_plan,semantic_overlay_path)
        checkpoint.update(status="layout_complete",layout_only=True,verified_layout=True,layout=layout.model_dump(),
            semantic_plan=semantic_plan.model_dump(),layout_overlay=str(overlay_path),semantic_overlay=str(semantic_overlay_path),
            recovered_candidate=candidate_index)
        self._atomic_json(checkpoint_path,checkpoint)
        return {"layout_path":str(layout_path),"semantic_path":str(semantic_path),"overlay_path":str(overlay_path),
                "semantic_overlay_path":str(semantic_overlay_path),"region_count":len(layout.regions),"article_count":len(semantic_plan.articles)}

    @staticmethod
    def _order_quad(points:np.ndarray) -> np.ndarray:
        """Return four polygon corners as top-left, top-right, bottom-right, bottom-left."""
        points=np.asarray(points,dtype=np.float32)
        ordered=np.zeros((4,2),dtype=np.float32)
        sums=points.sum(axis=1);differences=np.diff(points,axis=1).reshape(-1)
        ordered[0]=points[np.argmin(sums)];ordered[2]=points[np.argmax(sums)]
        ordered[1]=points[np.argmin(differences)];ordered[3]=points[np.argmax(differences)]
        return ordered

    @classmethod
    def _rectify_polygon(cls, image:Image.Image, polygon:list[list[float]], pad_pixels:int):
        """Perspective-rectify a padded four-corner text polygon from a photographed page."""
        width,height=image.size
        points=np.array([[p[0]*width/1000,p[1]*height/1000] for p in polygon],dtype=np.float32)
        source=cls._order_quad(points)
        center=source.mean(axis=0)
        top=np.linalg.norm(source[1]-source[0]);bottom=np.linalg.norm(source[2]-source[3])
        left=np.linalg.norm(source[3]-source[0]);right=np.linalg.norm(source[2]-source[1])
        factor=1+(2*pad_pixels/max(1,min(top,bottom,left,right)))
        padded=center+(source-center)*factor
        padded[:,0]=np.clip(padded[:,0],0,width-1);padded[:,1]=np.clip(padded[:,1],0,height-1)
        target_width=max(8,round(max(np.linalg.norm(padded[1]-padded[0]),np.linalg.norm(padded[2]-padded[3]))))
        target_height=max(8,round(max(np.linalg.norm(padded[3]-padded[0]),np.linalg.norm(padded[2]-padded[1]))))
        destination=np.array([[0,0],[target_width-1,0],[target_width-1,target_height-1],[0,target_height-1]],dtype=np.float32)
        matrix=cv2.getPerspectiveTransform(padded,destination)
        warped=cv2.warpPerspective(np.asarray(image),matrix,(target_width,target_height),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(warped),source,padded,matrix

    @staticmethod
    def _save_layout_overlay(image:Image.Image, layout:PageLayout, regions:dict, path:Path):
        """Save a non-destructive inspection derivative with polygons and reading order."""
        canvas=image.convert("RGBA");layer=Image.new("RGBA",canvas.size,(0,0,0,0));draw=ImageDraw.Draw(layer)
        line_width=max(3,round(image.width/900));font_size=max(14,round(image.width/150))
        try:font=ImageFont.truetype("arial.ttf",font_size)
        except OSError:font=ImageFont.load_default(size=font_size)
        colors={"verified":(35,105,210,235),"needs_review":(190,35,35,235),"failed":(220,125,20,235)}
        for region in sorted(layout.regions,key=lambda r:r.reading_order):
            points=[(round(p[0]*image.width/1000),round(p[1]*image.height/1000)) for p in region.polygon]
            status=regions.get(region.id,{}).get("status","prepared");color=colors.get(status,(35,105,210,235))
            draw.polygon(points,fill=(color[0],color[1],color[2],18),outline=color,width=line_width)
            x=min(p[0] for p in points);y=min(p[1] for p in points);label=f"{region.article_id or '-'}:{region.reading_order}"
            box=draw.textbbox((0,0),label,font=font);diameter=max(font_size+10,box[2]-box[0]+10,box[3]-box[1]+10)
            draw.ellipse((x,y,x+diameter,y+diameter),fill=color)
            draw.text((x+(diameter-(box[2]-box[0]))/2,y+(diameter-(box[3]-box[1]))/2),label,font=font,fill=(255,255,255,255))
        Image.alpha_composite(canvas,layer).convert("RGB").save(path,"PNG")

    @staticmethod
    def _save_semantic_overlay(image:Image.Image, plan:SemanticPagePlan, path:Path):
        """Render the LLM's article envelopes and typographic heading inventory separately."""
        canvas=image.convert("RGBA");layer=Image.new("RGBA",canvas.size,(0,0,0,0));draw=ImageDraw.Draw(layer)
        line_width=max(3,round(image.width/900));font_size=max(14,round(image.width/150))
        try:font=ImageFont.truetype("arial.ttf",font_size)
        except OSError:font=ImageFont.load_default(size=font_size)
        def points(polygon):return [(round(p[0]*image.width/1000),round(p[1]*image.height/1000)) for p in polygon]
        for article in sorted(plan.articles,key=lambda x:x.reading_order):
            pts=points(article.polygon);draw.polygon(pts,fill=(232,160,124,12),outline=(190,92,52,230),width=line_width)
            draw.text((min(x for x,_ in pts)+4,min(y for _,y in pts)+4),f"A:{article.id}",font=font,fill=(125,54,28,255),stroke_width=2,stroke_fill=(255,238,214,255))
        for heading in sorted(plan.headings,key=lambda x:x.reading_order):
            pts=points(heading.polygon);draw.polygon(pts,fill=(165,175,121,35),outline=(45,112,65,245),width=line_width+1)
            draw.text((min(x for x,_ in pts)+4,min(y for _,y in pts)+4),f"H:{heading.id}",font=font,fill=(22,80,42,255),stroke_width=2,stroke_fill=(255,238,214,255))
        Image.alpha_composite(canvas,layer).convert("RGB").save(path,"PNG")

    @staticmethod
    def _ordered_layout_regions(layout:PageLayout):
        """Flatten two-level order: page articles first, then local order inside each article."""
        by_id={r.id:r for r in layout.regions};ordered=[];seen=set()
        # Running issue metadata and other page-level text precede article streams.
        for region in sorted((r for r in layout.regions if r.article_id is None),key=lambda x:(x.reading_order,x.id)):
            ordered.append(region);seen.add(region.id)
        for article in sorted(layout.articles,key=lambda x:x.reading_order):
            members=[by_id[x] for x in article.region_ids if x in by_id]
            members.extend(r for r in layout.regions if r.article_id==article.id and r.id not in article.region_ids)
            for region in sorted(members,key=lambda x:(x.reading_order,x.id)):
                if region.id not in seen:ordered.append(region);seen.add(region.id)
        ordered.extend(sorted((r for r in layout.regions if r.id not in seen),key=lambda x:(x.reading_order,x.id)))
        return ordered

    def process_regions(self, source:str|Path, instruction:str, progress=None, max_workers:int=6, retries:int=2,
                        stop_after_layout:bool=False) -> dict:
        """Dynamic layout-aware pipeline with resumable region checkpoints."""
        def emit(stage,detail=""):
            if progress:
                try:progress(stage,detail)
                except (UnicodeError, OSError):pass
        source=Path(source).resolve();stamp=datetime.now().strftime("%Y%m%d-%H%M")
        base=f"{stamp}_{source.stem}";folder=self.root/base
        if folder.exists():folder=self.root/f"{base}_{uuid.uuid4().hex[:6]}"
        folder.mkdir(parents=True);audit=folder/"audit";audit.mkdir();crop_root=audit/"regions";crop_root.mkdir()
        saved_source=folder/f"{stamp}_{source.name}";shutil.copy2(source,saved_source)
        raw=Image.open(saved_source).convert("RGB");checkpoint_path=audit/"region-checkpoint.json";checkpoint_lock=Lock()
        checkpoint={"schema_version":1,"source":str(saved_source),"instruction":instruction,
                    "routing":{"layout":self.layout_model,"transcription":self.model,
                               "disagreement_adjudication":self.adjudicator_model,"final_escalation":self.inspector_model},
                    "layout":None,"regions":{},"calls":[],"retry_events":[],"status":"planning"}
        self._atomic_json(checkpoint_path,checkpoint)

        def call_parse(model,text_format,content,label):
            last=None
            for attempt in range(1,retries+2):
                started=time.perf_counter()
                try:
                    response=self.client.responses.parse(model=model,input=[{"role":"user","content":content}],text_format=text_format)
                    usage=getattr(response,"usage",None)
                    usage_data=usage.model_dump() if hasattr(usage,"model_dump") else (usage if isinstance(usage,dict) else None)
                    record={"label":label,"model":model,"attempt":attempt,"status":"success",
                            "latency_ms":round((time.perf_counter()-started)*1000),"request_id":getattr(response,"id",None),"usage":usage_data}
                    with checkpoint_lock:
                        checkpoint["calls"].append(record);self._atomic_json(checkpoint_path,checkpoint)
                    return response.output_parsed
                except Exception as exc:
                    last=exc
                    with checkpoint_lock:
                        checkpoint["calls"].append({"label":label,"model":model,"attempt":attempt,"status":"failed",
                            "latency_ms":round((time.perf_counter()-started)*1000),"error":f"{type(exc).__name__}: {exc}"})
                        checkpoint["retry_events"].append({"label":label,"attempt":attempt,"error":f"{type(exc).__name__}: {exc}"})
                        self._atomic_json(checkpoint_path,checkpoint)
                    if attempt>retries:break
                    emit("Retrying",f"{label} — attempt {attempt+1}/{retries+1}")
                    time.sleep(min(2**(attempt-1),4))
            raise last

        semantic_prompt=f"""Act as a newspaper editor and layout scholar. Analyze the attached archival page
semantically before any OCR crops are drawn. Do not assume fixed positions, a fixed number of articles, or
a fixed number or width of columns.
OPERATOR SCOPE: {instruction}

Perform only these first two tasks:
1. Inventory EVERY typographically distinct headline, subheading, deck, or section heading. Transcribe its
   visible wording verbatim so later passes have a semantic anchor. Typography—not linguistic plausibility—
   determines whether text is a heading: compare weight, size, spacing, rules, alignment, and whitespace.
2. Infer each independent article or reading stream and its complete rough extent. Associate its headings,
   estimate how many BODY COLUMNS belong to it, and determine article-level order. A headline spanning two
   columns normally owns both columns below it until another heading, rule, or clear article boundary.

Do NOT split body text into OCR regions yet. Do NOT treat an entire newspaper section as one article merely
because several stories share a broad theme. Multiple vertically stacked stories inside the same physical
page column remain separate articles. Conversely, do not split one two-column story into two articles.
Return SemanticPagePlan in normalized 0..1000 coordinates. Article polygons may be rough envelopes;
heading polygons must tightly but safely enclose the complete heading. Include small subheadings.

Use generic visual evidence:
- Article headings and subheadings are commonly larger, thicker, heavier, more widely spaced, or otherwise
  typographically distinct from preceding or following text. This is evidence, not an absolute rule.
- Associate smaller text with a likely heading using proximity, shared alignment, whitespace, printed rules,
  consistent column boundaries, and continuous reading flow.
- A heading or subheading may span one body column or several body columns.
- Stable vertical gutters or printed rules usually separate body columns.
- Newspaper pages are often photographed at an angle. Do not force text regions into axis-aligned rectangles.
  Return exactly four polygon corners that follow the visible skewed or perspective-distorted boundaries of
  the actual text block; trapezoids are expected. Place edges outside complete glyphs and words, not through
  them. Do not replace a trapezoid with a smaller rectangle merely because a rectangle is easier to draw.
- Determine article-level reading order without interleaving neighboring stories.
- Treat photographs, captions, lists, signatures, issue metadata, and other structurally distinct objects by
  their visual relationships. Exclude non-text photographs unless embedded writing is requested, but retain
  an associated caption when it is in scope.

Heading polygons use four corners and follow phone-camera perspective. Rough article envelopes may use as
many points as necessary for stepped, interrupted, or photo-wrapping flows; they are semantic inspection
guides, not OCR crops."""
        emit("Discovering articles and headings",source.name)
        semantic_plan=call_parse(self.layout_model,SemanticPagePlan,[{"type":"input_text","text":semantic_prompt},{"type":"input_image","image_url":self._image(saved_source),"detail":"high"}],"semantic layout planning")
        semantic_errors=semantic_plan.validate_plan()
        if semantic_errors:raise RuntimeError("Invalid semantic plan: "+"; ".join(semantic_errors))
        semantic_overlay=audit/"semantic-plan.png";self._save_semantic_overlay(raw,semantic_plan,semantic_overlay)
        checkpoint["semantic_plan"]=semantic_plan.model_dump();checkpoint["semantic_overlay"]=str(semantic_overlay);self._atomic_json(checkpoint_path,checkpoint)
        synthesis_prompt=f"""Turn this LLM-discovered semantic newspaper plan into OCR regions. Return PageLayout.
First respect the article and heading hierarchy; only then split each article body into its actual columns
and assign local reading order. Use the semantic article IDs exactly as PageLayout article IDs. Every semantic
heading must have its own complete LayoutRegion with semantic_heading_id set to its heading ID, detected_text
set to the verbatim anchor, and the correct article_id. Body regions must set column_index and column_count.
Never merge separate stacked articles. Never split a multi-column article into unrelated articles. Determine
reading order within each article (headline/subheading, then column 1, column 2, etc.) before page order.
Exclude photographs but include captions. Use four-corner trapezoids outside complete glyphs.
SEMANTIC PLAN: {semantic_plan.model_dump_json()}"""
        emit("Building article columns","Converting the semantic plan into OCR regions")
        layout=call_parse(self.layout_model,PageLayout,[{"type":"input_text","text":synthesis_prompt},{"type":"input_image","image_url":self._image(saved_source),"detail":"high"}],"layout synthesis")
        validation_history=[];layout_inspection=None
        candidate_overlay=audit/"layout-candidate-0.png";self._save_layout_overlay(raw,layout,{},candidate_overlay)
        candidate_semantic_overlay=audit/"semantic-candidate-0.png";self._save_semantic_overlay(raw,semantic_plan,candidate_semantic_overlay)
        for repair_attempt in range(1,6):
            candidate_layout_path=audit/f"layout-candidate-{repair_attempt-1}.json";self._atomic_json(candidate_layout_path,layout.model_dump())
            candidate_semantic_path=audit/f"semantic-candidate-{repair_attempt-1}.json";self._atomic_json(candidate_semantic_path,semantic_plan.model_dump())
            checkpoint["layout_candidate"]={"attempt":repair_attempt,"layout":str(candidate_layout_path),"overlay":str(candidate_overlay),
                                             "semantic_plan":str(candidate_semantic_path),"semantic_overlay":str(candidate_semantic_overlay)}
            checkpoint["layout_validation_history"]=validation_history;self._atomic_json(checkpoint_path,checkpoint)
            errors=semantic_plan.validate_plan()+layout.validate_layout()
            if not layout.regions:errors.append("layout contains no regions")
            planned_articles={x.id:x for x in semantic_plan.articles};layout_articles={x.id:x for x in layout.articles}
            for article_id in planned_articles:
                if article_id not in layout_articles:errors.append(f"semantic article {article_id} is absent from PageLayout")
            for heading in semantic_plan.headings:
                matches=[r for r in layout.regions if r.semantic_heading_id==heading.id]
                if not matches:errors.append(f"semantic heading {heading.id} ({heading.verbatim_text}) has no layout region")
                elif len(matches)>1:errors.append(f"semantic heading {heading.id} has multiple layout regions")
                elif matches[0].article_id!=heading.article_id:errors.append(f"semantic heading {heading.id} belongs to the wrong article")
            for article in semantic_plan.articles:
                if article.body_column_count:
                    indices={r.column_index for r in layout.regions if r.article_id==article.id and r.column_index is not None}
                    expected=set(range(1,article.body_column_count+1))
                    if indices!=expected:errors.append(f"{article.id}: expected body column indices {sorted(expected)}, found {sorted(indices)}")
            audit_prompt=f"""Audit this archival newspaper hierarchy against the untouched source image.
The second image shows semantic ARTICLE ENVELOPES and HEADING INVENTORY; the third shows final numbered OCR
regions. Judge semantic interpretation first, geometry second. Inspect systematically from top-left to
bottom-right. Every heading, SUBHEADING, deck, issue line, caption, article column, signature, short notice,
continuation and isolated text must be represented. Polygon edges must clear complete glyphs.

semantic_plan_correct requires every independent story to have its own article, every spanning headline to
own all and only its body columns, and vertically stacked stories not to be merged. column_structure_correct
requires the true number and extent of body columns for every article. reading_order_correct requires each
article's headline/subheadings before its bodies, followed by local column order without interleaving other
articles. Verify the complete transcribed heading anchors against the source. Do not reward mere pixel
coverage. Require text_coverage >= 0.985 and give concrete repair instructions using semantic IDs, numbered
regions, visible heading wording, and page landmarks.
SEMANTIC PLAN: {semantic_plan.model_dump_json()}
PROPOSED LAYOUT: {layout.model_dump_json()}"""
            layout_inspection=call_parse(self.layout_model,LayoutInspection,[
                {"type":"input_text","text":audit_prompt},
                {"type":"input_image","image_url":self._image(saved_source),"detail":"high"},
                {"type":"input_image","image_url":self._image(candidate_semantic_overlay),"detail":"high"},
                {"type":"input_image","image_url":self._image(candidate_overlay),"detail":"high"}],f"hierarchical layout audit {repair_attempt}")
            errors.extend(layout_inspection.failures())
            validation_history.append({"attempt":repair_attempt,"errors":errors,"inspection":layout_inspection.model_dump(),
                                       "semantic_plan":semantic_plan.model_dump(),"layout":layout.model_dump(),"overlay":str(candidate_overlay)})
            checkpoint["layout_validation_history"]=validation_history;self._atomic_json(checkpoint_path,checkpoint)
            if not errors or repair_attempt==5:break
            emit("Repairing article hierarchy",f"Semantic validation {repair_attempt}/5: {'; '.join(errors[:4])}")
            repair_prompt=f"""Repair both SemanticPagePlan and PageLayout from the source. Return PlannedLayout,
not a patch. Re-inventory typographically distinct headings first. You may split or merge semantic articles
and restore missing heading anchors before revising OCR regions. Then identify the complete extent and true
body-column count of each article, and only then assign local reading order. Use semantic article IDs in
PageLayout; map each heading via semantic_heading_id; set body column_index and column_count. Semantic article
envelopes may be multi-corner shapes for stepped or photo-wrapping flows; OCR regions remain four-corner.
Do not treat a
whole themed section as one article. Do not split a two-column story into unrelated articles. Preserve
perspective-aware trapezoids outside glyphs.
VALIDATION FAILURES: {errors}
VISUAL REPAIR INSTRUCTIONS: {layout_inspection.repair_instructions}
SEMANTIC PLAN: {semantic_plan.model_dump_json()}
PROPOSED LAYOUT: {layout.model_dump_json()}"""
            repaired=call_parse(self.layout_model,PlannedLayout,[{"type":"input_text","text":repair_prompt},{"type":"input_image","image_url":self._image(saved_source),"detail":"high"}],f"hierarchical layout repair {repair_attempt}")
            semantic_plan,layout=repaired.semantic_plan,repaired.page_layout
            candidate_overlay=audit/f"layout-candidate-{repair_attempt}.png";self._save_layout_overlay(raw,layout,{},candidate_overlay)
            candidate_semantic_overlay=audit/f"semantic-candidate-{repair_attempt}.png";self._save_semantic_overlay(raw,semantic_plan,candidate_semantic_overlay)
        errors=semantic_plan.validate_plan()+layout.validate_layout()
        if not layout.regions:errors.append("layout contains no regions")
        layout_article_ids={x.id for x in layout.articles}
        for article in semantic_plan.articles:
            if article.id not in layout_article_ids:errors.append(f"semantic article {article.id} is absent from PageLayout")
            if article.body_column_count:
                indices={r.column_index for r in layout.regions if r.article_id==article.id and r.column_index is not None}
                expected=set(range(1,article.body_column_count+1))
                if indices!=expected:errors.append(f"{article.id}: expected body column indices {sorted(expected)}, found {sorted(indices)}")
        for heading in semantic_plan.headings:
            matches=[r for r in layout.regions if r.semantic_heading_id==heading.id]
            if len(matches)!=1:errors.append(f"semantic heading {heading.id} must have exactly one layout region")
            elif matches[0].article_id!=heading.article_id:errors.append(f"semantic heading {heading.id} belongs to the wrong article")
        if layout_inspection:errors.extend(layout_inspection.failures())
        checkpoint["layout_validation_history"]=validation_history;checkpoint["semantic_plan"]=semantic_plan.model_dump()
        if errors:raise RuntimeError("Invalid layout: "+"; ".join(errors))
        ordered=self._ordered_layout_regions(layout)
        overlay_path=audit/"layout-regions.png"
        semantic_path=audit/"semantic-plan.json";self._atomic_json(semantic_path,semantic_plan.model_dump())
        semantic_overlay_path=audit/"semantic-regions.png";self._save_semantic_overlay(raw,semantic_plan,semantic_overlay_path)
        checkpoint["layout"]=layout.model_dump();checkpoint["layout_overlay"]=str(overlay_path)
        checkpoint["semantic_plan_path"]=str(semantic_path);checkpoint["semantic_overlay"]=str(semantic_overlay_path);checkpoint["status"]="preprocessing"
        self._save_layout_overlay(raw,layout,checkpoint["regions"],overlay_path);self._atomic_json(checkpoint_path,checkpoint)
        emit("Detected reading regions",f"{len(ordered)} regions")
        layout_path=audit/"layout.json";self._atomic_json(layout_path,layout.model_dump())
        if stop_after_layout:
            checkpoint.update(status="layout_complete",layout_only=True,verified_layout=True)
            self._atomic_json(checkpoint_path,checkpoint)
            emit("Layout complete",f"Saved {len(ordered)} regions; transcription not started")
            return {"source_name":source.name,"source_path":str(saved_source),"markdown_path":None,
                    "layout_path":str(layout_path),"overlay_path":str(overlay_path),"text":"",
                    "semantic_path":str(semantic_path),"semantic_overlay_path":str(semantic_overlay_path),
                    "omissions_found":[],"corrections_made":[],"confidence":layout.confidence,"verified":False,
                    "failed_regions":{},"region_count":len(ordered),"layout_only":True,"folder":str(folder)}

        prepared={}
        for region in ordered:
            # Preserve the inferred trapezoid, expand it geometrically so edge
            # glyphs survive, then rectify perspective for the OCR readers.
            pad=max(6,round(min(raw.width,raw.height)*.008))
            original,polygon_pixels,padded_polygon,matrix=self._rectify_polygon(raw,region.polygon,pad)
            region_dir=crop_root/region.id;region_dir.mkdir()
            scale=max(2,min(4,round(1600/max(original.width,original.height))))
            plain=ImageOps.grayscale(original).resize((original.width*scale,original.height*scale),Image.Resampling.LANCZOS)
            arr=np.asarray(plain);clahe_np=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(arr)
            paths=[region_dir/"original.png",region_dir/"plain-upscaled.png",region_dir/"clahe-upscaled.png"]
            original.save(paths[0]);plain.save(paths[1]);Image.fromarray(clahe_np).save(paths[2])
            xs=polygon_pixels[:,0];ys=polygon_pixels[:,1];bounds=[round(float(xs.min())),round(float(ys.min())),round(float(xs.max())),round(float(ys.max()))]
            provenance={"polygon_pixels":polygon_pixels.tolist(),"padded_polygon_pixels":padded_polygon.tolist(),
                        "perspective_transform":matrix.tolist(),"rectified_dimensions":[original.width,original.height]}
            prepared[region.id]={"region":region,"bounds":bounds,"paths":paths,"provenance":provenance}
            checkpoint["regions"][region.id]={"metadata":region.model_dump(),"bounds":bounds,**provenance,"status":"prepared","attempts":{}}
        checkpoint["status"]="transcribing";self._atomic_json(checkpoint_path,checkpoint)
        emit("Preprocessed regions",f"{len(ordered)}/{len(ordered)} ready")

        def save_region(region_id,**updates):
            with checkpoint_lock:
                checkpoint["regions"][region_id].update(updates);self._atomic_json(checkpoint_path,checkpoint)

        def read(region_id,reader):
            item=prepared[region_id];region=item["region"]
            prompt=f"""You are blinded OCR reader {reader}. Transcribe only this isolated archival region from
pixels. Region type: {region.type}. Operator scope: {instruction}
Preserve diplomatic spelling, capitalization, punctuation, paragraph and printed line-wrap hyphens. Never
modernize, paraphrase, or replace an unusual word with a plausible one. Include every visible line once.
Mark genuinely unsupported glyphs [неясно]. You have not seen another transcription."""
            # Each reader receives exactly one rendering of the region. Supplying
            # several variants in one OCR request can make a model transcribe the
            # same text once per image and silently duplicate it.
            selected=item["paths"][1] if reader=="A" else item["paths"][2]
            images=[{"type":"input_image","image_url":self._image(selected),"detail":"high"}]
            return call_parse(self.model,Transcription,[{"type":"input_text","text":prompt},*images],f"region {region.reading_order+1}/{len(ordered)} read {reader}")

        reads={rid:{} for rid in prepared};failed={}
        futures={}
        with ThreadPoolExecutor(max_workers=max(1,min(max_workers,len(ordered)*2))) as pool:
            for region in ordered:
                for reader in ("A","B"):futures[pool.submit(read,region.id,reader)]=(region.id,reader)
            completed_regions=set()
            for future in as_completed(futures):
                region_id,reader=futures[future]
                try:
                    result=future.result();reads[region_id][reader]=result
                    save_region(region_id,**{f"read_{reader.lower()}":result.model_dump(),"status":"read_partial"})
                except Exception as exc:
                    failed[region_id]=f"{type(exc).__name__}: {exc}";save_region(region_id,status="failed",error=failed[region_id])
                if len(reads[region_id])==2 or region_id in failed:
                    completed_regions.add(region_id)
                    emit("Transcribing regions",f"{len(completed_regions)}/{len(ordered)} complete")

        accepted={};disagreements=[]
        for region in ordered:
            pair=reads[region.id]
            if len(pair)<2:continue
            clean_agreement=(pair["A"].verbatim_text.strip()==pair["B"].verbatim_text.strip()
                and pair["A"].confidence>=.90 and pair["B"].confidence>=.90
                and not pair["A"].uncertain_spans and not pair["B"].uncertain_spans
                and "[неясно" not in pair["A"].verbatim_text.casefold()
                and not pair["A"].validate_offsets() and not pair["B"].validate_offsets())
            if clean_agreement:
                accepted[region.id]={"text":pair["A"].verbatim_text,"confidence":min(pair["A"].confidence,pair["B"].confidence),"method":"agreement","uncertain":[x.model_dump() for x in pair["A"].uncertain_spans]}
                save_region(region.id,status="verified",accepted=accepted[region.id])
            else:disagreements.append(region)
        emit("Comparing blinded Luna reads",f"{len(accepted)} clean agreements; {len(disagreements)} need Terra")

        def adjudicate(region):
            pair=reads[region.id];item=prepared[region.id]
            prompt=f"""Adjudicate two blinded diplomatic OCR reads against this isolated source crop. The three
attached images are different renderings of the SAME crop, not three consecutive sections; transcribe its
content exactly once. Resolve
differences character by character from pixels, never linguistic plausibility. Preserve printed line-wrap
hyphens and historical spelling. Include every visible line exactly once. If pixels cannot resolve a span,
retain [неясно: A | B]. Operator scope: {instruction}
READ A:\n{pair['A'].verbatim_text}\nREAD B:\n{pair['B'].verbatim_text}"""
            images=[{"type":"input_image","image_url":self._image(p),"detail":"high"} for p in item["paths"]]
            return call_parse(self.adjudicator_model,Inspection,[{"type":"input_text","text":prompt},*images],f"region {region.reading_order+1}/{len(ordered)} Terra adjudication")

        if disagreements:
            with checkpoint_lock:
                checkpoint["status"]="verifying";self._atomic_json(checkpoint_path,checkpoint)
            with ThreadPoolExecutor(max_workers=max(1,min(max_workers,len(disagreements)))) as pool:
                futures={pool.submit(adjudicate,r):r for r in disagreements};done=0
                for future in as_completed(futures):
                    region=futures[future]
                    try:
                        result=future.result();accepted[region.id]={"text":result.final_text,"confidence":result.confidence,"method":"terra_adjudication","uncertain":result.omissions_found}
                        save_region(region.id,status="terra_complete",adjudication=result.model_dump(),accepted=accepted[region.id])
                    except Exception as exc:
                        failed[region.id]=f"{type(exc).__name__}: {exc}";save_region(region.id,status="failed",error=failed[region.id])
                    done+=1;emit("Terra adjudicating regions",f"{done}/{len(disagreements)} complete")

        sol_regions=[r for r in disagreements if r.id in accepted and
            (accepted[r.id]["confidence"]<.90 or "[неясно" in accepted[r.id]["text"].casefold())]
        def final_adjudicate(region):
            pair=reads[region.id];terra=accepted[region.id];item=prepared[region.id]
            prompt=f"""Perform final source-grounded archival OCR adjudication for this isolated region. Luna
reads disagreed and Terra remained uncertain. Resolve only from visible pixels; never select fluent wording
without glyph evidence. Preserve historical spelling and printed line-wrap hyphens. Include every visible
line once. If the pixels genuinely cannot decide, retain [неясно: A | B]. Operator scope: {instruction}
LUNA A:\n{pair['A'].verbatim_text}\nLUNA B:\n{pair['B'].verbatim_text}\nTERRA DRAFT:\n{terra['text']}"""
            images=[{"type":"input_image","image_url":self._image(p),"detail":"high"} for p in item["paths"]]
            return call_parse(self.inspector_model,Inspection,[{"type":"input_text","text":prompt},*images],f"region {region.reading_order+1}/{len(ordered)} Sol escalation")
        if sol_regions:
            emit("Escalating unresolved regions",f"{len(sol_regions)} regions need Sol")
            with ThreadPoolExecutor(max_workers=max(1,min(max_workers,len(sol_regions)))) as pool:
                futures={pool.submit(final_adjudicate,r):r for r in sol_regions};done=0
                for future in as_completed(futures):
                    region=futures[future]
                    try:
                        result=future.result();accepted[region.id]={"text":result.final_text,"confidence":result.confidence,"method":"sol_escalation","uncertain":result.omissions_found}
                        save_region(region.id,sol_adjudication=result.model_dump(),accepted=accepted[region.id])
                    except Exception as exc:
                        failed[region.id]=f"{type(exc).__name__}: {exc}";accepted.pop(region.id,None);save_region(region.id,status="failed",error=failed[region.id])
                    done+=1;emit("Sol resolving regions",f"{done}/{len(sol_regions)} complete")

        for region in ordered:
            if region.id in accepted:
                result=accepted[region.id];status="needs_review" if result["confidence"]<.90 or "[неясно" in result["text"].casefold() else "verified"
                save_region(region.id,status=status,accepted=result)

        pieces=[];confidences=[];unresolved=[]
        for region in ordered:
            if region.id in accepted:
                pieces.append(accepted[region.id]["text"].strip());confidences.append(accepted[region.id]["confidence"])
                if "[неясно" in accepted[region.id]["text"].casefold():unresolved.append(region.id)
            else:
                pieces.append(f"[REGION {region.id} FAILED: {failed.get(region.id,'missing result')}]");unresolved.append(region.id)
        text="\n\n".join(x for x in pieces if x);confidence=min(confidences) if confidences else 0.0
        verified=not unresolved and len(accepted)==len(ordered) and confidence>=.90
        checkpoint.update(status="complete" if verified else "needs_review",verified=verified,failed_regions=failed,unresolved_regions=unresolved)
        self._save_layout_overlay(raw,layout,checkpoint["regions"],overlay_path)
        self._atomic_json(checkpoint_path,checkpoint);emit("Reconstructing document",f"{len(accepted)}/{len(ordered)} regions available")
        md=folder/f"{stamp}_{source.stem}.md";md.write_text(f'''---
source: "./{saved_source.name}"
digitized_at: "{datetime.now().astimezone().isoformat(timespec='minutes')}"
pipeline: "dynamic-regions-v1"
instruction: "{instruction.replace('"','').strip()}"
regions: {len(ordered)}
layout_model: "{self.layout_model}"
transcription_model: "{self.model}"
adjudicator_model: "{self.adjudicator_model}"
final_escalation_model: "{self.inspector_model}"
inspection_confidence: {confidence}
verification_status: "{'verified' if verified else 'needs_review'}"
---

# {source.stem}

{text}
''',encoding="utf-8")
        emit("Saved Markdown",md.name)
        return {"source_name":source.name,"source_path":str(saved_source),"markdown_path":str(md),"text":text,
                "omissions_found":unresolved,"corrections_made":[],"confidence":confidence,"verified":verified,
                "failed_regions":failed,"region_count":len(ordered),"layout_path":str(layout_path),
                "overlay_path":str(overlay_path),"semantic_path":str(semantic_path),
                "semantic_overlay_path":str(semantic_overlay_path),"layout_only":False,"folder":str(folder)}

    def process_direct(self, source:str|Path, instruction:str, progress=None) -> dict:
        """Fast path mirroring direct multimodal chat: one read and one grounded inspection."""
        def emit(stage,detail=""):
            if progress:progress(stage,detail)
        source=Path(source).resolve();stamp=datetime.now().strftime("%Y%m%d-%H%M")
        base=f"{stamp}_{source.stem}_direct";folder=self.root/base
        if folder.exists():folder=self.root/f"{base}_{uuid.uuid4().hex[:6]}"
        folder.mkdir(parents=True);audit=folder/"audit";audit.mkdir();derivatives=audit/"derivatives";derivatives.mkdir()
        saved_source=folder/f"{stamp}_{source.name}";shutil.copy2(source,saved_source)
        raw=Image.open(saved_source).convert("RGB");scale=max(2,min(4,round(1800/max(raw.width,raw.height))))
        plain=ImageOps.grayscale(raw).resize((raw.width*scale,raw.height*scale),Image.Resampling.LANCZOS)
        arr=np.asarray(plain);clahe_np=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(arr)
        plain_path=derivatives/"plain-upscaled.png";clahe_path=derivatives/"clahe-upscaled.png"
        plain.save(plain_path);Image.fromarray(clahe_np).save(clahe_path)
        images=[{"type":"input_image","image_url":self._image(p),"detail":"high"} for p in (saved_source,plain_path,clahe_path)]
        prompt=f"""{TRANSCRIBE}
OPERATOR SCOPE: {instruction}
Read the page spatially before writing. For multi-column text, finish each column before moving to the
next. Ignore text outside the requested scope. This is literal OCR: semantic plausibility is never evidence.
Pay special attention to short words and phrases previously vulnerable to hallucination, but do not assume
any particular wording. Use [неясно] rather than inventing text."""
        blinded=prompt+"""
You are the blinded second reader. You have not seen and must not infer another transcription. Start at
the source pixels and independently account for every requested line."""
        emit("Transcribing","Two blinded source-grounded reads in parallel")
        def call(text_prompt,image_inputs):
            return self.client.responses.parse(model=self.inspector_model,input=[{"role":"user","content":[{"type":"input_text","text":text_prompt},*image_inputs]}],text_format=Transcription).output_parsed
        with ThreadPoolExecutor(max_workers=2) as pool:
            fa=pool.submit(call,prompt,images);fb=pool.submit(call,blinded,list(reversed(images)))
            first=fa.result();second=fb.result()
        verify=f"""Adjudicate TWO blinded archival OCR reads against the attached source images from top
to bottom, character by character. Treat neither read as authoritative. Correct omissions, substitutions,
column-order mistakes, and fluent hallucinations only when pixels support the correction. Preserve exact
historical spelling, wording, and printed line-wrap hyphens. Do not add commentary inside final_text. Use
[неясно: A | B] where resolution cannot resolve a disagreement.
OPERATOR SCOPE: {instruction}
READ A:\n{first.verbatim_text}
READ B:\n{second.verbatim_text}"""
        emit("Inspecting","Adjudicating both blinded reads against pixels")
        final=self.client.responses.parse(model=self.inspector_model,input=[{"role":"user","content":[{"type":"input_text","text":verify},*images]}],text_format=Inspection).output_parsed
        unresolved="[неясно" in final.final_text.casefold() or "[нечетливо" in final.final_text.casefold()
        verified=final.confidence>=.95 and not unresolved
        (audit/"reads.json").write_text(json.dumps({"read_a":first.model_dump(),"read_b":second.model_dump(),"final":final.model_dump(),"verified":verified},ensure_ascii=False,indent=2),encoding="utf-8")
        md=folder/f"{stamp}_{source.stem}.md";md.write_text(f'''---
source: "./{saved_source.name}"
digitized_at: "{datetime.now().astimezone().isoformat(timespec='minutes')}"
pipeline: "direct-source-grounded-v1"
instruction: "{instruction.replace('"','').strip()}"
inspection_confidence: {final.confidence}
verification_status: "{'verified' if verified else 'needs_review'}"
---

# {source.stem}

{final.final_text.strip()}
''',encoding="utf-8")
        return {"source_name":source.name,"source_path":str(saved_source),"markdown_path":str(md),
                "text":final.final_text,"confidence":final.confidence,"verified":verified,"folder":str(folder)}

    def process_layout_first(self, source:str|Path, instruction:str, progress=None) -> dict:
        """Bounded high-accuracy path: isolate layout before OCR and never mix columns."""
        def emit(stage,detail=""):
            if progress:progress(stage,detail)
        source=Path(source).resolve();stamp=datetime.now().strftime("%Y%m%d-%H%M")
        base=f"{stamp}_{source.stem}_precise";folder=self.root/base
        if folder.exists():folder=self.root/f"{base}_{uuid.uuid4().hex[:6]}"
        folder.mkdir(parents=True);audit=folder/"audit";audit.mkdir();regions_dir=audit/"regions";regions_dir.mkdir()
        saved_source=folder/f"{stamp}_{source.name}";shutil.copy2(source,saved_source)
        raw=Image.open(saved_source).convert("RGB")
        emit("Planning layout","Isolating requested headings, columns, captions, and marginal text")
        plan_prompt=f"""Identify every rectangular reading region needed for this exact operator scope:
{instruction}
Return normalized coordinates 0..1000 relative to the complete image. Regions must follow reading
order, isolate columns so unrelated adjacent text never shares a crop, and include headings, bylines,
captions, signatures, and small-print lines within scope. For a complete-page request, cover every
visible text-bearing area. Regions may overlap slightly at boundaries but must not duplicate whole paragraphs."""
        plan=self.client.responses.parse(model=self.model,input=[{"role":"user","content":[
            {"type":"input_text","text":plan_prompt},{"type":"input_image","image_url":self._image(saved_source),"detail":"high"}]}],text_format=ReadingPlan).output_parsed
        valid=[r for r in sorted(plan.regions,key=lambda x:x.reading_order) if r.x1>r.x0 and r.y1>r.y0]
        if not valid:raise RuntimeError("Layout planner returned no usable reading regions")
        emit("Layout ready",f"{len(valid)} isolated reading regions")

        def read_region(region:ReadingRegion):
            pad=4;x0=max(0,round(raw.width*region.x0/1000)-pad);y0=max(0,round(raw.height*region.y0/1000)-pad)
            x1=min(raw.width,round(raw.width*region.x1/1000)+pad);y1=min(raw.height,round(raw.height*region.y1/1000)+pad)
            crop=raw.crop((x0,y0,x1,y1));gray=ImageOps.grayscale(crop).resize((crop.width*4,crop.height*4),Image.Resampling.LANCZOS)
            arr=np.asarray(gray);clahe_np=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(arr)
            threshold_np=cv2.adaptiveThreshold(clahe_np,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,11)
            stem=f"{region.reading_order:02d}-{region.id}";original_path=regions_dir/f"{stem}-original.png";plain_path=regions_dir/f"{stem}-plain.png"
            clahe_path=regions_dir/f"{stem}-clahe.png";threshold_path=regions_dir/f"{stem}-threshold.png"
            crop.save(original_path);gray.save(plain_path);Image.fromarray(clahe_np).save(clahe_path);Image.fromarray(threshold_np).save(threshold_path)
            images=[{"type":"input_image","image_url":self._image(p),"detail":"high"} for p in (original_path,plain_path,clahe_path,threshold_path)]
            base_prompt=f"""Literal archival OCR of isolated region {region.id} ({region.label}). Operator scope: {instruction}
Copy every visible glyph in natural reading order. Preserve historical Bulgarian spelling, letter endings,
capitalization, punctuation, line-wrap hyphens, numbers, and labels. Do not reconstruct meaning and never
substitute a plausible phrase. Use [неясно] for unsupported glyphs. Return only text visible in this crop."""
            a=self.client.responses.parse(model=self.model,input=[{"role":"user","content":[{"type":"input_text","text":base_prompt+"\nINDEPENDENT READ A"},*images]}],text_format=Transcription).output_parsed
            b=self.client.responses.parse(model=self.inspector_model,input=[{"role":"user","content":[{"type":"input_text","text":base_prompt+"\nINDEPENDENT BLINDED READ B"},*images]}],text_format=Transcription).output_parsed
            verify=f"""Adjudicate two independent literal OCR reads against the attached isolated source region.
Resolve character by character from pixels, never by fluency. Preserve exact unusual wording. Include all
visible text once. If pixels cannot resolve a disagreement, write [неясно: A | B].
READ A:\n{a.verbatim_text}\nREAD B:\n{b.verbatim_text}"""
            final=self.client.responses.parse(model=self.inspector_model,input=[{"role":"user","content":[{"type":"input_text","text":verify},*images]}],text_format=Inspection).output_parsed
            return {"region":region.model_dump(),"bounds":[x0,y0,x1,y1],"read_a":a.model_dump(),"read_b":b.model_dump(),"final":final.model_dump()}

        results=[];emit("Reading isolated regions",f"Processing {len(valid)} regions in parallel")
        with ThreadPoolExecutor(max_workers=min(6,len(valid))) as pool:
            futures={pool.submit(read_region,r):r for r in valid}
            for future in as_completed(futures):
                item=future.result();results.append(item);emit("Region verified",item["region"]["label"])
        results.sort(key=lambda x:x["region"]["reading_order"])
        text="\n\n".join(x["final"]["final_text"].strip() for x in results if x["final"]["final_text"].strip())
        unresolved="[неясно" in text.casefold();confidence=min(x["final"]["confidence"] for x in results)
        verified=confidence>=.95 and not unresolved
        (audit/"regions.json").write_text(json.dumps({"plan":plan.model_dump(),"results":results,"verified":verified},ensure_ascii=False,indent=2),encoding="utf-8")
        md=folder/f"{stamp}_{source.stem}.md";md.write_text(f'''---
source: "./{saved_source.name}"
digitized_at: "{datetime.now().astimezone().isoformat(timespec='minutes')}"
pipeline: "layout-first-region-verified-v1"
instruction: "{instruction.replace('"','').strip()}"
inspection_confidence: {confidence}
verification_status: "{'verified' if verified else 'needs_review'}"
---

# {source.stem}

{text}
''',encoding="utf-8")
        return {"source_name":source.name,"source_path":str(saved_source),"markdown_path":str(md),"text":text,
                "confidence":confidence,"verified":verified,"folder":str(folder),"regions":results}

    def _tile_reads(self, enhanced:Image.Image, clahe:Image.Image, threshold:Image.Image, audit:Path, instruction:str, emit, fine:bool=False) -> list[dict]:
        tile_dir=audit/("tiles-fine" if fine else "tiles-coarse");tile_dir.mkdir();height=enhanced.height
        # Four overlapping full-width bands preserve complete printed lines while
        # guaranteeing that no text disappears at a crop boundary.
        band=int(height*(.19 if fine else .34));step=int(height*(.115 if fine else .24));specs=[]
        for index,y0 in enumerate(range(0,max(1,height-band+1),step)):
            y1=min(height,y0+band)
            if y1-y0<height*(.10 if fine else .18):continue
            specs.append((index,y0,y1))
        if specs and specs[-1][2]<height:specs.append((len(specs),max(0,height-band),height))
        def read(spec):
            index,y0,y1=spec;path=tile_dir/f"tile-{index:02d}-{y0}-{y1}.png"
            enhanced.crop((0,y0,enhanced.width,y1)).save(path)
            clahe_path=path.with_name(path.stem+"-clahe.png");clahe.crop((0,y0,clahe.width,y1)).save(clahe_path)
            threshold_path=path.with_name(path.stem+"-threshold.png");threshold.crop((0,y0,threshold.width,y1)).save(threshold_path)
            prompt=f"""OCR this high-resolution overlapping page band from pixels only. Transcribe every
visible text fragment, including partial lines at the top/bottom, small print, headings and historical
spelling. Never paraphrase or normalize. Use [неясно] only after examining the letters. This band spans
vertical pixels {y0}..{y1} of {height}. Operator scope: {instruction}"""
            parsed=self.client.responses.parse(model=self.inspector_model,input=[{"role":"user","content":[
                {"type":"input_text","text":prompt+f"\nregion_id=tile_{index}"},
                {"type":"input_image","image_url":self._image(path),"detail":"high"},
                {"type":"input_image","image_url":self._image(clahe_path),"detail":"high"},
                {"type":"input_image","image_url":self._image(threshold_path),"detail":"high"}]}],text_format=Transcription).output_parsed
            return {"index":index,"y0":y0,"y1":y1,"path":str(path),"clahe_path":str(clahe_path),"threshold_path":str(threshold_path),"read":parsed.model_dump()}
        emit("Recovering regions",f"Reading {len(specs)} {'fine' if fine else 'coarse'} overlapping high-resolution bands")
        results=[]
        with ThreadPoolExecutor(max_workers=min(4,len(specs))) as pool:
            futures={pool.submit(read,s):s for s in specs}
            for future in as_completed(futures):
                result=future.result();results.append(result);emit("Recovered region",f"Band {result['index']+1}/{len(specs)}")
        return sorted(results,key=lambda x:x["index"])

    def _line_reads(self, enhanced:Image.Image, clahe:Image.Image, threshold:Image.Image, audit:Path, instruction:str, emit) -> list[dict]:
        line_dir=audit/"line-crops";line_dir.mkdir();height=enhanced.height
        # Fixed micro-bands outperform projection segmentation on stained pages,
        # whose background texture can connect otherwise separate printed lines.
        band=max(80,int(height*.11));step=max(55,int(height*.072));chunks=[]
        for y0 in range(0,max(1,height-band+1),step):chunks.append((len(chunks),y0,min(height,y0+band)))
        if chunks and chunks[-1][2]<height:chunks.append((len(chunks),height-band,height))
        def read(spec):
            index,y0,y1=spec;crop=enhanced.crop((0,y0,enhanced.width,y1));path=line_dir/f"lines-{index:02d}-{y0}-{y1}.png";crop.save(path)
            clahe_path=path.with_name(path.stem+"-clahe.png");clahe.crop((0,y0,clahe.width,y1)).save(clahe_path)
            threshold_path=path.with_name(path.stem+"-threshold.png");threshold.crop((0,y0,threshold.width,y1)).save(threshold_path)
            prompt=f"""This crop contains only one or two adjacent printed text lines. Perform literal
character OCR. Do not infer a sentence, modernize spelling, or replace an unusual word with a plausible
one. Preserve exact Bulgarian endings and punctuation. Include partial edge text with [неясно] markers.
Crop y={y0}..{y1}. Operator scope: {instruction}"""
            parsed=self.client.responses.parse(model=self.inspector_model,input=[{"role":"user","content":[
                {"type":"input_text","text":prompt+f"\nregion_id=line_chunk_{index}"},
                {"type":"input_image","image_url":self._image(path),"detail":"high"},
                {"type":"input_image","image_url":self._image(clahe_path),"detail":"high"},
                {"type":"input_image","image_url":self._image(threshold_path),"detail":"high"}]}],text_format=Transcription).output_parsed
            return {"index":index,"y0":y0,"y1":y1,"path":str(path),"clahe_path":str(clahe_path),"threshold_path":str(threshold_path),"read":parsed.model_dump()}
        emit("Resolving individual lines",f"Reading {len(chunks)} detected line crops")
        results=[]
        with ThreadPoolExecutor(max_workers=min(6,max(1,len(chunks)))) as pool:
            futures={pool.submit(read,s):s for s in chunks}
            for future in as_completed(futures):
                result=future.result();results.append(result);emit("Resolved line crop",f"{len(results)}/{len(chunks)}")
        return sorted(results,key=lambda x:x["index"])

    def process(self, source:str|Path, instruction:str, progress=None) -> dict:
        def emit(stage,detail=""):
            if progress:progress(stage,detail)
        source=Path(source).resolve();stamp=datetime.now().strftime("%Y%m%d-%H%M")
        base=f"{stamp}_{source.stem}";folder=self.root/base
        if folder.exists():folder=self.root/f"{base}_{uuid.uuid4().hex[:6]}"
        folder.mkdir(parents=True)
        saved_source=folder/f"{stamp}_{source.name}";shutil.copy2(source,saved_source)
        image=self._image(saved_source)
        audit=folder/"audit";audit.mkdir();derivatives=audit/"derivatives";derivatives.mkdir();emit("Preprocessing","Generating reversible reading variants")
        enhanced_path=derivatives/"original-upscaled.png";clahe_path=derivatives/"clahe-upscaled.png";threshold_path=derivatives/"adaptive-threshold.png";denoised_path=derivatives/"denoised-sharpened.png"
        raw=Image.open(saved_source).convert("RGB")
        up=np.asarray(ImageOps.grayscale(raw).resize((raw.width*4,raw.height*4),Image.Resampling.LANCZOS))
        clahe=cv2.createCLAHE(clipLimit=2.2,tileGridSize=(8,8)).apply(up)
        threshold_np=cv2.adaptiveThreshold(clahe,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,11)
        denoised=cv2.fastNlMeansDenoising(clahe,None,7,7,21);denoised=cv2.addWeighted(denoised,1.45,cv2.GaussianBlur(denoised,(0,0),1.2),-.45,0)
        enhanced=Image.fromarray(up);clahe_image=Image.fromarray(clahe);threshold=Image.fromarray(threshold_np);denoised_image=Image.fromarray(denoised)
        enhanced.save(enhanced_path);clahe_image.save(clahe_path);threshold.save(threshold_path);denoised_image.save(denoised_path)
        enhanced_image=self._image(enhanced_path);clahe_url=self._image(clahe_path);threshold_image=self._image(threshold_path);denoised_url=self._image(denoised_path)
        emit("Preprocessing complete","CLAHE, adaptive threshold, and denoised/sharpened variants ready")
        emit("Inspecting source",source.name)
        task=f"\n\nOPERATOR INSTRUCTION:\n{instruction.strip()}\nFollow this requested scope exactly."
        emit("Transcribing","Independent read A")
        first=self.client.responses.parse(model=self.model,input=[{"role":"user","content":[
            {"type":"input_text","text":TRANSCRIBE+task+"\nregion_id=full_page"},
            {"type":"input_image","image_url":image,"detail":"high"}]}],text_format=Transcription).output_parsed
        emit("Verifying","Blinded read B")
        second=self.client.responses.parse(model=self.model,input=[{"role":"user","content":[
            {"type":"input_text","text":BLINDED_READ+task+"\nregion_id=full_page_blinded"},
            {"type":"input_image","image_url":enhanced_image,"detail":"high"},
            {"type":"input_image","image_url":clahe_url,"detail":"high"},
            {"type":"input_image","image_url":threshold_image,"detail":"high"},
            {"type":"input_image","image_url":denoised_url,"detail":"high"}]}],text_format=Transcription).output_parsed
        emit("Adjudicating","Comparing both reads against the pixels")
        inspected=self.client.responses.parse(model=self.inspector_model,input=[{"role":"user","content":[
            {"type":"input_text","text":INSPECT+task+"\n\nREAD A:\n"+first.verbatim_text+"\n\nREAD B:\n"+second.verbatim_text},
            {"type":"input_image","image_url":image,"detail":"high"},
            {"type":"input_image","image_url":enhanced_image,"detail":"high"},
            {"type":"input_image","image_url":clahe_url,"detail":"high"},
            {"type":"input_image","image_url":threshold_image,"detail":"high"},
            {"type":"input_image","image_url":denoised_url,"detail":"high"}]}],text_format=Inspection).output_parsed
        if inspected.confidence < .90:
            emit("Escalating","Low-confidence adjudication; performing enhanced source-only recovery")
            recovery_prompt="""Perform a final source-grounded OCR recovery from the enhanced image. The prior
adjudication was low confidence. Copy visible characters, not meaning. Do not normalize Bulgarian or
complete sentences from context. Check every line against pixels. Return the complete corrected text,
explicit uncertainty markers for anything unresolved, and list every correction.\n\nLOW-CONFIDENCE DRAFT:\n"""
            recovered=self.client.responses.parse(model=self.inspector_model,input=[{"role":"user","content":[
                {"type":"input_text","text":recovery_prompt+inspected.final_text+task},
                {"type":"input_image","image_url":enhanced_image,"detail":"high"},
                {"type":"input_image","image_url":clahe_url,"detail":"high"},
                {"type":"input_image","image_url":threshold_image,"detail":"high"},
                {"type":"input_image","image_url":denoised_url,"detail":"high"}]}],text_format=Inspection).output_parsed
            if recovered.confidence >= inspected.confidence:inspected=recovered

        tiles=self._tile_reads(enhanced,clahe_image,threshold,audit,instruction,emit)
        tile_text="\n\n".join(f"BAND {t['index']} y={t['y0']}..{t['y1']}:\n{t['read']['verbatim_text']}" for t in tiles)
        emit("Assembling coverage","Reconciling regional reads with the full-page draft")
        assembly_prompt="""Reconstruct the complete diplomatic transcription using the full-page draft and
overlapping regional OCR evidence below. Regional evidence exists to restore omissions. Do not rewrite
correct text for fluency. Preserve every pixel-supported word, historical ending, heading and small-print
line. De-duplicate overlap between bands. If evidence conflicts and pixels cannot resolve it, retain an
explicit [неясно: A | B] marker. Return final_text plus exact omissions/corrections.\n\nFULL-PAGE DRAFT:\n"""
        assembled=self.client.responses.parse(model=self.inspector_model,input=[{"role":"user","content":[
            {"type":"input_text","text":assembly_prompt+inspected.final_text+"\n\nREGIONAL EVIDENCE:\n"+tile_text+task},
            {"type":"input_image","image_url":enhanced_image,"detail":"high"}]}],text_format=Inspection).output_parsed
        tile_tokens=set().union(*(self._tokens(t["read"]["verbatim_text"]) for t in tiles))
        final_tokens=self._tokens(assembled.final_text);coverage=len(tile_tokens&final_tokens)/max(1,len(tile_tokens))
        if coverage<.82 or "[неясно" in assembled.final_text.casefold() or "[нечетливо" in assembled.final_text.casefold():
            missing=sorted(tile_tokens-final_tokens)
            emit("Recursing on omissions",f"Coverage {coverage:.0%}; switching to tighter regional crops")
            fine_tiles=self._tile_reads(enhanced,clahe_image,threshold,audit,instruction,emit,True);tiles.extend(fine_tiles)
            fine_text="\n\n".join(f"FINE BAND {t['index']} y={t['y0']}..{t['y1']}:\n{t['read']['verbatim_text']}" for t in fine_tiles)
            repair_prompt=f"""The prior assembly omitted or could not resolve visible text. Reconstruct it
again using the much tighter overlapping crop reads below. Restore all pixel-supported lines and exact
word forms. Do not choose fluent substitutes. De-duplicate overlaps. Missing coarse candidates are only
diagnostic, not guaranteed correct: {missing[:250]}\n\nPRIOR ASSEMBLY:\n{assembled.final_text}
\nFINE REGIONAL EVIDENCE:\n{fine_text}{task}"""
            repaired=self.client.responses.parse(model=self.inspector_model,input=[{"role":"user","content":[
                {"type":"input_text","text":repair_prompt},{"type":"input_image","image_url":enhanced_image,"detail":"high"}]}],text_format=Inspection).output_parsed
            tile_tokens=set().union(*(self._tokens(t["read"]["verbatim_text"]) for t in tiles))
            repaired_tokens=self._tokens(repaired.final_text);repaired_coverage=len(tile_tokens&repaired_tokens)/max(1,len(tile_tokens))
            if repaired_coverage>coverage:assembled,coverage=repaired,repaired_coverage
        if coverage<.82 or "[неясно" in assembled.final_text.casefold() or "[нечетливо" in assembled.final_text.casefold():
            emit("Escalating to line OCR",f"Regional recovery remains incomplete at {coverage:.0%}")
            lines=self._line_reads(enhanced,clahe_image,threshold,audit,instruction,emit)
            line_text="\n\n".join(f"LINE GROUP {x['index']} y={x['y0']}..{x['y1']}:\n{x['read']['verbatim_text']}" for x in lines)
            line_prompt=f"""Reconstruct the complete document from literal OCR of one-to-two-line crops.
These line reads are the strongest evidence and override fluent but unsupported wording in the draft.
Preserve their vertical order, remove overlap duplicates, and retain explicit uncertainty. Do not omit
any line group.\n\nPRIOR DRAFT:\n{assembled.final_text}\n\nLINE EVIDENCE:\n{line_text}{task}"""
            line_final=self.client.responses.parse(model=self.inspector_model,input=[{"role":"user","content":[
                {"type":"input_text","text":line_prompt},{"type":"input_image","image_url":enhanced_image,"detail":"high"}]}],text_format=Inspection).output_parsed
            line_tokens=set().union(*(self._tokens(x["read"]["verbatim_text"]) for x in lines))
            new_tokens=self._tokens(line_final.final_text);line_coverage=len(line_tokens&new_tokens)/max(1,len(line_tokens))
            assembled,coverage=line_final,line_coverage;tiles.extend({**x,"kind":"line"} for x in lines)
        inspected=assembled
        verified=inspected.confidence>=.90 and coverage>=.82 and "[нечетливо]" not in inspected.final_text.casefold()
        (audit/"reads.json").write_text(json.dumps({"read_a":first.model_dump(),"read_b":second.model_dump(),"tiles":tiles,"final":inspected.model_dump(),"coverage":coverage,"verified":verified},ensure_ascii=False,indent=2),encoding="utf-8")
        md=folder/f"{stamp}_{source.stem}.md"
        notes="\n".join(f"  - {x}" for x in inspected.corrections_made) or "  - none"
        emit("Saving","Writing paired source and Markdown")
        md.write_text(f'''---
source: "./{saved_source.name}"
digitized_at: "{datetime.now().astimezone().isoformat(timespec='minutes')}"
transcriber_model: "{self.model}"
inspector_model: "{self.inspector_model}"
instruction: "{instruction.replace('"','').strip()}"
inspection_confidence: {inspected.confidence}
regional_coverage: {coverage:.4f}
verification_status: "{'verified' if verified else 'needs_review'}"
corrections:
{notes}
---

# {source.stem}

{inspected.final_text.strip()}
''',encoding="utf-8")
        return {"source_name":source.name,"source_path":str(saved_source),"markdown_path":str(md),
                "text":inspected.final_text,"omissions_found":inspected.omissions_found,
                "corrections_made":inspected.corrections_made,"confidence":inspected.confidence,
                "coverage":coverage,"verified":verified,"folder":str(folder)}
