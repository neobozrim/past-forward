from __future__ import annotations

import base64,hashlib,json,mimetypes,shutil,time,uuid
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
from pathlib import Path
from .metadata import metadata_from_filename

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageEnhance,ImageFont,ImageOps
from pydantic import BaseModel,Field
from .models import Transcription


class RegionReview(BaseModel):
    crop_complete: bool
    role_matches_pixels: bool
    contains_unmarked_heading: bool
    mixed_articles: bool
    column_flow_complete: bool
    issues: list[str] = Field(default_factory=list)
    required_action: str


class PageReview(BaseModel):
    passed: bool
    no_unmarked_text: bool
    all_headings_have_heading_regions: bool
    articles_separated_correctly: bool
    column_structure_correct: bool
    issues: list[str] = Field(default_factory=list)
    required_actions: list[str] = Field(default_factory=list)


class OCRDecision(BaseModel):
    final_text: str
    confidence: float = Field(ge=0,le=1)
    unresolved: bool
    crop_complete: bool
    role_matches_pixels: bool
    issues: list[str] = Field(default_factory=list)


PEN_PROMPT="""You are Past Forward's autonomous archival digitisation operator. The attached source is
a phone photograph of a historical page. You hold the pen: inspect, mark, crop, transcribe, reconsider, and
redraw your own work. Do not create and freeze an exhaustive page plan before reading.

Work like a careful human, one article at a time:
- Survey the page for typographic anchors first: larger or heavier headings, subheadings, italic decks,
  rules, and whitespace boundaries. Use those anchors to hypothesize article ownership; never begin from a
  generic page-wide column grid.
- Choose one article, declare it, and inspect closer when needed. Trace its body away from its heading using
  whitespace/rules, typography, and sentence continuity to determine its full extent, column count, and
  column order. Geometry records that evidence; geometry does not decide the article.
- Mark its heading/deck/body/caption/signature regions with perspective-aware four-point quadrilaterals.
  Phone photographs are skewed: never force axis-aligned rectangles. Use several overlapping regions for
  stepped, curved, interrupted, or photo-wrapping text rather than cutting text to fit one neat shape.
- mark_and_crop and revise_and_crop return the exact marked crop and an original-image safety-padded crop.
  Inspect the exact pixels first. Preprocessing is demand-driven only after blur/degradation or OCR
  disagreement is observed; never send enhanced variants for every clear region.
  A crop is invalid when a claimed heading is not visibly inside it, a glyph touches an edge, a line is
  partial, or neighboring text has been absorbed. Redraw it before accepting transcription.
- For full digitisation, call transcribe_region after inspecting a crop. It launches two blinded Luna reads
  in fresh requests, then routes disagreements to Terra and unresolved cases to Sol. Never write or approve
  OCR text from this layout conversation itself.
- If transcription reaches an unmarked heading, stop: mark it, reconsider article ownership, and revise the
  affected body crop. If a column starts or ends mid-sentence, inspect the source and correct its boundary,
  ownership, or local order. Structural continuity is evidence; linguistic plausibility is not.
- A deck or italic introduction may span fewer columns than its article. Infer each region from its own
  typography and pixels, not from the article's overall column count.
- Finish and visually inspect the current article before moving to the next. Headless signed contributions
  can be articles; never invent a heading for them.
- Use render_overlay repeatedly. Near the end, compare the overlay to the untouched source from top-left to
  bottom-right and find unmarked text. Establish article order only after all articles are complete.

You must use the tools. Do not merely describe intended work. You may issue multiple independent crop calls
in one turn. finish_page is accepted only when every marked region has a crop-grounded transcription, the
latest overlay reflects all edits, all visible text is accounted for, headings are grounded in their actual
crops, and article column ownership/order has been checked."""


POINTS={"type":"array","minItems":4,"maxItems":4,"items":{"type":"array","minItems":2,"maxItems":2,"items":{"type":"number","minimum":0,"maximum":1000}}}
TOOLS=[
 {"type":"function","name":"declare_article","description":"Create or revise the current article while evidence develops.","parameters":{"type":"object","properties":{
     "article_id":{"type":"string"},"label":{"type":"string"},"column_count":{"type":"integer","minimum":0},
     "column_order":{"type":"array","items":{"type":"string"}},"notes":{"type":"string"}},
     "required":["article_id","label","column_count","column_order","notes"],"additionalProperties":False},"strict":True},
 {"type":"function","name":"look_closer","description":"Inspect an unmarked perspective-aware area before deciding what it is.","parameters":{"type":"object","properties":{
     "points":POINTS,"purpose":{"type":"string"}},"required":["points","purpose"],"additionalProperties":False},"strict":True},
 {"type":"function","name":"mark_and_crop","description":"Mark a transcribable region and immediately receive exact plus original safety-padded crops for visual verification.","parameters":{"type":"object","properties":{
     "region_id":{"type":"string"},"article_id":{"type":"string"},"role":{"type":"string","enum":["heading","subheading","deck","body","caption","signature","metadata","notice"]},
     "label":{"type":"string"},"points":POINTS,"local_order":{"type":"integer","minimum":0},
     "column_index":{"anyOf":[{"type":"integer","minimum":1},{"type":"null"}]},"notes":{"type":"string"}},
     "required":["region_id","article_id","role","label","points","local_order","column_index","notes"],"additionalProperties":False},"strict":True},
 {"type":"function","name":"revise_and_crop","description":"Replace a bad mark and receive fresh exact plus original safety-padded crops.","parameters":{"type":"object","properties":{
     "region_id":{"type":"string"},"article_id":{"type":"string"},"role":{"type":"string","enum":["heading","subheading","deck","body","caption","signature","metadata","notice"]},
     "label":{"type":"string"},"points":POINTS,"local_order":{"type":"integer","minimum":0},
     "column_index":{"anyOf":[{"type":"integer","minimum":1},{"type":"null"}]},"reason":{"type":"string"}},
     "required":["region_id","article_id","role","label","points","local_order","column_index","reason"],"additionalProperties":False},"strict":True},
 {"type":"function","name":"remove_region","description":"Delete a mark that was false, duplicated, or assigned incorrectly.","parameters":{"type":"object","properties":{
     "region_id":{"type":"string"},"reason":{"type":"string"}},"required":["region_id","reason"],"additionalProperties":False},"strict":True},
 {"type":"function","name":"remove_article","description":"Delete an article hypothesis after all of its regions have been removed or reassigned.","parameters":{"type":"object","properties":{
     "article_id":{"type":"string"},"reason":{"type":"string"}},"required":["article_id","reason"],"additionalProperties":False},"strict":True},
 {"type":"function","name":"transcribe_region","description":"Run two isolated blinded Luna OCR reads for a verified crop, route disagreements to Terra and unresolved cases to Sol, and persist every read.","parameters":{"type":"object","properties":{
     "region_id":{"type":"string"}},"required":["region_id"],"additionalProperties":False},"strict":True},
 {"type":"function","name":"render_overlay","description":"Render every current mark over the untouched source and inspect it.","parameters":{"type":"object","properties":{},"required":[],"additionalProperties":False},"strict":True},
 {"type":"function","name":"finish_layout","description":"Finish a layout-only run after a final source-versus-overlay inspection; no transcription is required.","parameters":{"type":"object","properties":{
     "article_order":{"type":"array","items":{"type":"string"}},"source_checked":{"type":"boolean"},
     "no_unmarked_text":{"type":"boolean"},"headings_grounded":{"type":"boolean"},"columns_verified":{"type":"boolean"},
     "final_notes":{"type":"string"}},"required":["article_order","source_checked","no_unmarked_text","headings_grounded","columns_verified","final_notes"],"additionalProperties":False},"strict":True},
 {"type":"function","name":"finish_page","description":"Finish only after a final source-versus-overlay inspection and establish article order.","parameters":{"type":"object","properties":{
     "article_order":{"type":"array","items":{"type":"string"}},"source_checked":{"type":"boolean"},
     "no_unmarked_text":{"type":"boolean"},"headings_grounded":{"type":"boolean"},"columns_verified":{"type":"boolean"},
     "final_notes":{"type":"string"}},"required":["article_order","source_checked","no_unmarked_text","headings_grounded","columns_verified","final_notes"],"additionalProperties":False},"strict":True},
]


class PenAgent:
    def __init__(self,root:Path,client,model="gpt-5.6-sol",transcription_model="gpt-5.6-luna",adjudicator_model="gpt-5.6-terra",inspector_model="gpt-5.6-sol"):
        self.root=Path(root).resolve();self.client=client;self.model=model
        self.transcription_model=transcription_model;self.adjudicator_model=adjudicator_model;self.inspector_model=inspector_model

    @staticmethod
    def _data(path:Path):
        mime=mimetypes.guess_type(path)[0] or "image/png"
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"

    @staticmethod
    def _atomic(path:Path,value):
        temporary=path.with_suffix(path.suffix+".tmp")
        temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8");temporary.replace(path)

    @staticmethod
    def _order_quad(points):
        points=np.asarray(points,dtype=np.float32);center=points.mean(axis=0)
        angles=np.arctan2(points[:,1]-center[1],points[:,0]-center[0]);ordered=points[np.argsort(angles)]
        # The top-right corner can be a little higher than the top-left on a
        # handheld photograph.  Starting at the smallest y therefore rotates
        # an otherwise valid crop by 90 degrees.  x+y identifies the visual
        # top-left while retaining the clockwise hull order for trapezoids.
        start=min(range(4),key=lambda i:(float(ordered[i,0]+ordered[i,1]),float(ordered[i,0])))
        ordered=np.roll(ordered,-start,axis=0)
        first=ordered[1]-ordered[0];second=ordered[2]-ordered[1]
        if first[0]*second[1]-first[1]*second[0]<0:ordered=ordered[[0,3,2,1]]
        return ordered.astype(np.float32)

    @classmethod
    def _validate_quad(cls,points):
        array=np.asarray(points,dtype=np.float32)
        if array.shape!=(4,2) or not np.isfinite(array).all():raise ValueError("quadrilateral must contain four finite points")
        if len(np.unique(array,axis=0))!=4:raise ValueError("quadrilateral points must be unique")
        hull=cv2.convexHull(array).reshape(-1,2)
        if len(hull)!=4 or abs(cv2.contourArea(hull))<25:raise ValueError("quadrilateral must be convex and have non-trivial area")
        return cls._order_quad(array)

    @classmethod
    def _overlap_fraction(cls,first,second):
        a=cls._validate_quad(first);b=cls._validate_quad(second)
        intersection,_=cv2.intersectConvexConvex(a,b);denominator=min(abs(cv2.contourArea(a)),abs(cv2.contourArea(b)))
        return float(intersection/max(denominator,1))

    @classmethod
    def _crop(cls,image:Image.Image,points,pad=0):
        normalized=cls._validate_quad(points);width,height=image.size
        source=np.asarray([[x*width/1000,y*height/1000] for x,y in normalized],dtype=np.float32)
        center=source.mean(axis=0);top=np.linalg.norm(source[1]-source[0]);bottom=np.linalg.norm(source[2]-source[3])
        left=np.linalg.norm(source[3]-source[0]);right=np.linalg.norm(source[2]-source[1])
        factor=1+(2*pad/max(1,min(top,bottom,left,right)));source=center+(source-center)*factor
        source[:,0]=np.clip(source[:,0],0,width-1);source[:,1]=np.clip(source[:,1],0,height-1)
        target_width=max(8,round(max(np.linalg.norm(source[1]-source[0]),np.linalg.norm(source[2]-source[3]))))
        target_height=max(8,round(max(np.linalg.norm(source[3]-source[0]),np.linalg.norm(source[2]-source[1]))))
        target=np.array([[0,0],[target_width-1,0],[target_width-1,target_height-1],[0,target_height-1]],dtype=np.float32)
        matrix=cv2.getPerspectiveTransform(source,target)
        warped=cv2.warpPerspective(np.asarray(image),matrix,(target_width,target_height),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(warped),matrix

    @staticmethod
    def _enhance(image:Image.Image):
        scale=max(1,min(3,round(1800/max(image.size))))
        gray=ImageOps.grayscale(image).resize((image.width*scale,image.height*scale),Image.Resampling.LANCZOS)
        enhanced=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(np.asarray(gray))
        return Image.fromarray(enhanced)

    @staticmethod
    def _overlay(image:Image.Image,state:dict,path:Path):
        canvas=image.convert("RGBA");layer=Image.new("RGBA",canvas.size,(0,0,0,0));draw=ImageDraw.Draw(layer)
        palette=[(35,105,210),(190,70,55),(45,135,70),(155,90,185),(220,125,20),(40,145,155)]
        try:font=ImageFont.truetype("arial.ttf",max(13,round(image.width/180)))
        except OSError:font=ImageFont.load_default(size=max(11,round(image.width/180)))
        article_colors={key:palette[i%len(palette)] for i,key in enumerate(state["articles"])}
        article_numbers={key:i+1 for i,key in enumerate(state["articles"])}
        for region in state["regions"].values():
            color=article_colors.get(region["article_id"],(80,80,80));pts=[(round(x*image.width/1000),round(y*image.height/1000)) for x,y in region["points"]]
            draw.polygon(pts,fill=(*color,10),outline=(*color,245),width=max(3,round(image.width/900)))
            role={"heading":"H","subheading":"SH","deck":"D","body":f'C{region.get("column_index") or ""}',"caption":"CAP","signature":"S","metadata":"M","notice":"N"}.get(region["role"],region["role"][:3].upper())
            label=f'A{article_numbers.get(region["article_id"],0):02}/{region["local_order"]}:{role}';x=min(p[0] for p in pts);y=min(p[1] for p in pts)
            draw.text((x+3,y+3),label,font=font,fill=(15,15,15,255),stroke_width=2,stroke_fill=(255,245,225,255))
        Image.alpha_composite(canvas,layer).convert("RGB").save(path,"PNG")

    def _review_region(self,region:dict):
        prompt=f"""Independently inspect this exact marked newspaper crop before it is accepted.
Claimed role: {region['role']}; claimed label: {region['label']}; article: {region['article_id']}.
Candidate transcription follows:
{region.get('pending_text','')}

Judge pixels, not the label or fluent text. crop_complete is false if any glyph/line is cut or if the crop
contains only part of the claimed region. role_matches_pixels is false when a claimed heading crop contains
body text or a claimed body crop is actually a heading. For a BODY crop, actively scan top-to-bottom for
larger, heavier, centered, spaced, ruled-off, italic-deck, or otherwise distinct typography. If any such
heading/subheading begins inside the body crop, contains_unmarked_heading must be true: body crops may not
swallow headings. mixed_articles is true if a heading, signature/closure, rule, whitespace boundary, or flow
change shows that more than one independent story is present. column_flow_complete is false when the crop
begins/ends at the wrong place, switches articles, or follows the wrong column. State a concrete required
action such as expand, shrink, split at visible heading wording, reassign, or accept."""
        response=self.client.responses.parse(model=self.model,input=[{"role":"user","content":[
            {"type":"input_text","text":prompt},{"type":"input_image","image_url":self._data(Path(region["exact_crop"])),"detail":"original"}]}],text_format=RegionReview)
        return response.output_parsed,response

    def _review_page(self,page_path:Path,overlay_path:Path,state:dict):
        summary={"articles":state["articles"],"regions":[{k:v for k,v in r.items() if k in ("region_id","article_id","role","label","local_order","column_index")} for r in state["regions"].values()]}
        prompt=f"""Independently audit the untouched phone photograph against the proposed overlay. The first
image is the source; the second is the overlay. Do not reward pixel coverage. Search the source line by line
for every larger, heavier, centered, italic, widely spaced, or ruled-off heading/subheading/deck. Every one
must have its own correctly aligned heading/subheading/deck polygon—not merely sit inside a body polygon.
Reject any body region that contains a visually distinct heading or multiple stacked stories. Verify each
article's full extent, column count, and column order; a deck may span fewer columns than its article. Reject
neat axis-aligned crops that cut skewed phone-photo text. Identify issues using visible heading wording and
overlay article/region labels so the operating agent can redraw them.
CURRENT STATE: {json.dumps(summary,ensure_ascii=False)}"""
        response=self.client.responses.parse(model=self.model,input=[{"role":"user","content":[
            {"type":"input_text","text":prompt},{"type":"input_image","image_url":self._data(page_path),"detail":"original"},
            {"type":"input_image","image_url":self._data(overlay_path),"detail":"original"}]}],text_format=PageReview)
        return response.output_parsed,response

    def _transcribe_blinded(self,region:dict,instruction:str,emit):
        region_dir=Path(region["exact_crop"]).parent;exact=Path(region["exact_crop"])
        enhanced=Path(region["enhanced_crop"]) if region.get("enhanced_crop") else None
        safety=Path(region.get("safety_crop") or region.get("context_crop") or region.get("padded_crop") or region["exact_crop"])
        calls=[]
        def parse(model,text_format,content,label):
            started=time.perf_counter()
            try:
                response=self.client.responses.parse(model=model,input=[{"role":"user","content":content}],text_format=text_format)
                usage=getattr(response,"usage",None);calls.append({"label":label,"model":model,"status":"success",
                    "latency_ms":round((time.perf_counter()-started)*1000),"request_id":getattr(response,"id",None),
                    "usage":usage.model_dump() if hasattr(usage,"model_dump") else None})
                return response.output_parsed
            except Exception as exc:
                calls.append({"label":label,"model":model,"status":"failed","latency_ms":round((time.perf_counter()-started)*1000),
                              "error":f"{type(exc).__name__}: {exc}"});raise
        def read(reader,path):
            scope=("The whole image is the owned core." if reader=="A" else
                   "The untinted central polygon is the owned article core. Tinted neighboring text is accidental safety padding; never transcribe it.")
            prompt=f"""You are blinded archival OCR reader {reader}. This is a fresh isolated request. Transcribe
every visible glyph inside this single perspective-rectified crop exactly once. Preserve Bulgarian historical
spelling, capitalization, punctuation, line breaks, and printed line-end hyphens. Never modernize, paraphrase,
complete from linguistic expectation, or choose a more fluent word. Use [неясно] only where pixels do not support a reading.
{scope} Set region_id to {region['region_id']}. You have not seen and must not infer another reader's answer."""
            parsed=parse(self.transcription_model,Transcription,[{"type":"input_text","text":prompt},
                {"type":"input_image","image_url":self._data(path),"detail":"original"}],f"{region['region_id']} Luna {reader}")
            self._atomic(region_dir/f"read-{reader.casefold()}.json",parsed.model_dump());return parsed
        emit("Blinded OCR",f"{region['region_id']} · Luna A/B")
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_a=pool.submit(read,"A",exact);future_b=pool.submit(read,"B",safety)
            first=future_a.result();second=future_b.result()
        invalid=first.validate_offsets()+second.validate_offsets()
        agreement=(not invalid and first.verbatim_text.strip()==second.verbatim_text.strip()
                   and min(first.confidence,second.confidence)>=.92 and not first.uncertain_spans and not second.uncertain_spans
                   and "[неясно" not in first.verbatim_text.casefold())
        high_impact=region["role"] in {"heading","subheading","deck","caption","signature","metadata","notice"}
        if agreement and not high_impact:
            decision=OCRDecision(final_text=first.verbatim_text,confidence=min(first.confidence,second.confidence),unresolved=False,
                                 crop_complete=True,role_matches_pixels=True,issues=[]);route="luna_agreement"
        else:
            if enhanced is None:
                enhanced=region_dir/"selected-enhanced.png"
                self._enhance(Image.open(exact).convert("RGB")).save(enhanced)
                region["enhanced_crop"]=str(enhanced)
                region["preprocessing"]={"applied":True,"reason":"OCR disagreement or high-impact typography required closer inspection","recipe":"CLAHE grayscale upscaling"}
                region.setdefault("crop_hashes",{})["enhanced_sha256"]=hashlib.sha256(enhanced.read_bytes()).hexdigest()
            emit("OCR adjudication",f"{region['region_id']} · Terra")
            prompt=f"""Adjudicate two blinded archival OCR reads against renderings of the SAME isolated crop.
The first is exact color, the second is enhanced exact, and the third adds safety padding where accidental
neighboring text is tinted outside the owned core. Return only the owned core text exactly once.
Resolve characters only from pixels, not linguistic plausibility. Preserve unusual
Bulgarian spelling, capitalization, punctuation, line breaks, and printed line-end hyphens. If pixels cannot
resolve a difference, set unresolved true and retain [неясно: A | B]. Verify that the complete claimed
{region['role']} is inside the crop and no line is cut. OPERATOR SCOPE: {instruction}
READ A:\n{first.verbatim_text}\nREAD B:\n{second.verbatim_text}"""
            decision=parse(self.adjudicator_model,OCRDecision,[{"type":"input_text","text":prompt},
                {"type":"input_image","image_url":self._data(exact),"detail":"original"},
                {"type":"input_image","image_url":self._data(enhanced),"detail":"original"},
                {"type":"input_image","image_url":self._data(safety),"detail":"original"}],f"{region['region_id']} Terra adjudication")
            route="terra_adjudication"
            if decision.unresolved or decision.confidence<.90:
                emit("OCR escalation",f"{region['region_id']} · Sol")
                prompt=f"""Final source-grounded OCR escalation. Resolve the crop from glyph evidence, never fluent
word prediction. Preserve diplomatic Bulgarian text and printed hyphenation. Return the crop exactly once;
retain [неясно: A | B] and unresolved true if pixels genuinely cannot decide.
LUNA A:\n{first.verbatim_text}\nLUNA B:\n{second.verbatim_text}\nTERRA:\n{decision.final_text}"""
                decision=parse(self.inspector_model,OCRDecision,[{"type":"input_text","text":prompt},
                    {"type":"input_image","image_url":self._data(exact),"detail":"original"},
                    {"type":"input_image","image_url":self._data(enhanced),"detail":"original"},
                    {"type":"input_image","image_url":self._data(safety),"detail":"original"}],f"{region['region_id']} Sol escalation")
                route="sol_escalation"
        record={"route":route,"read_a":first.model_dump(),"read_b":second.model_dump(),"decision":decision.model_dump(),"calls":calls}
        self._atomic(region_dir/"ocr-routing.json",record);return decision,record

    def run(self,source:str|Path,instruction:str,progress=None,stop_after_layout=False,max_turns=120,resume_folder:str|Path|None=None):
        def emit(stage,detail=""):
            if progress:
                try:progress(stage,detail)
                except (UnicodeError,OSError):pass
        source=Path(source).resolve();stamp=datetime.now().strftime("%Y%m%d-%H%M")
        if resume_folder:
            folder=Path(resume_folder).resolve();audit=folder/"audit";crop_root=audit/"regions";look_root=audit/"looks"
            state_path=audit/"pen-agent-state.json";state=json.loads(state_path.read_text(encoding="utf-8"));saved_source=Path(state["source"])
            page=Image.open(saved_source).convert("RGB");overlay_path=audit/"agent-overlay.png";state.update(status="working",article_order=[])
            state["instruction"]=instruction;state.setdefault("verification_calls",[]);state.setdefault("ocr_calls",[])
            state.setdefault("source_sha256",hashlib.sha256(saved_source.read_bytes()).hexdigest());state.setdefault("source_dimensions",list(page.size))
            state.setdefault("routing",{"layout":self.model,"transcription":self.transcription_model,"adjudication":self.adjudicator_model,"escalation":self.inspector_model})
            self._atomic(state_path,state)
            look_index=len(list(look_root.glob("look-*.png")))
        else:
            folder=self.root/f"{stamp}_{source.stem}_agentic"
            if folder.exists():folder=self.root/f"{folder.name}_{uuid.uuid4().hex[:6]}"
            audit=folder/"audit";crop_root=audit/"regions";look_root=audit/"looks"
            crop_root.mkdir(parents=True);look_root.mkdir();saved_source=folder/f"{stamp}_{source.name}";shutil.copy2(source,saved_source)
            page=Image.open(saved_source).convert("RGB");state_path=audit/"pen-agent-state.json";overlay_path=audit/"agent-overlay.png"
            state={"schema_version":2,"source":str(saved_source),"source_sha256":hashlib.sha256(saved_source.read_bytes()).hexdigest(),
                   "source_dimensions":list(page.size),"instruction":instruction,"model":self.model,
                   "routing":{"layout":self.model,"transcription":self.transcription_model,"adjudication":self.adjudicator_model,"escalation":self.inspector_model},
                   "status":"working","articles":{},"regions":{},"article_order":[],"actions":[],"calls":[],"verification_calls":[],"ocr_calls":[],"version":0,"overlay_version":-1,"overlay_turn":-1}
            self._atomic(state_path,state);look_index=0
        finished=False

        def checkpoint_interruption(exc):
            try:
                self._overlay(page,state,overlay_path);state["overlay_version"]=state["version"]
            except (OSError,ValueError):
                pass
            state.update(status="interrupted",error=f"{type(exc).__name__}: {exc}");self._atomic(state_path,state)
            try:setattr(exc,"run_folder",str(folder))
            except (AttributeError,TypeError):pass

        def record(name,args,result):
            state["actions"].append({"index":len(state["actions"])+1,"tool":name,"args":args,"result":result,"timestamp":datetime.now().isoformat()})
            self._atomic(state_path,state)

        def crop_result(region_id,points,revision):
            exact,matrix=self._crop(page,points,0);padded,padded_matrix=self._crop(page,points,max(10,round(min(page.size)*.008)))
            region_dir=crop_root/region_id/f"revision-{revision:03d}";region_dir.mkdir(parents=True,exist_ok=False)
            exact_path=region_dir/"exact.png";padded_path=region_dir/"safety-padded.png";safety_path=region_dir/"padded-safety-boundary.png"
            normalized=self._validate_quad(points);source_points=np.asarray([[[x*page.width/1000,y*page.height/1000] for x,y in normalized]],dtype=np.float32)
            core=cv2.perspectiveTransform(source_points,padded_matrix)[0]
            base=padded.convert("RGBA");tint=Image.new("RGBA",base.size,(130,113,72,100));shaded=Image.alpha_composite(base,tint)
            mask=Image.new("L",base.size,0);ImageDraw.Draw(mask).polygon([tuple(map(float,p)) for p in core],fill=255)
            safety=Image.composite(base,shaded,mask).convert("RGB");ImageDraw.Draw(safety).line([tuple(map(float,p)) for p in [*core,core[0]]],fill=(232,160,124),width=max(2,round(min(safety.size)/300)))
            exact.save(exact_path);padded.save(padded_path);safety.save(safety_path)
            sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
            return {"crop_revision":revision,"exact_crop":str(exact_path),"enhanced_crop":None,"padded_crop":str(padded_path),"safety_crop":str(safety_path),
                    "perspective_transform":matrix.tolist(),"padded_perspective_transform":padded_matrix.tolist(),
                    "safety_core_polygon":[[round(float(x),2),round(float(y),2)] for x,y in core],
                    "exact_dimensions":list(exact.size),"enhanced_dimensions":None,"padded_dimensions":list(padded.size),"safety_dimensions":list(safety.size),
                    "crop_hashes":{"exact_sha256":sha(exact_path),"padded_sha256":sha(padded_path),"safety_sha256":sha(safety_path)},
                    "preprocessing":{"applied":False,"reason":None,"recipe":None}}

        def image_messages(title,paths):
            content=[{"type":"input_text","text":title}]
            for path in paths:content.append({"type":"input_image","image_url":self._data(path),"detail":"original"})
            return {"role":"user","content":content}

        def handle(name,args):
            nonlocal finished,look_index
            images=[];result={}
            if name=="declare_article":
                state["articles"][args["article_id"]]={**args,"status":"working"};state["version"]+=1
                result={"ok":True,"article":state["articles"][args["article_id"]]};emit("Working article",args["label"])
            elif name=="look_closer":
                try:crop,_=self._crop(page,args["points"],max(8,round(min(page.size)*.006)))
                except ValueError as exc:result={"ok":False,"error":str(exc)}
                else:
                    look_index+=1;path=look_root/f"look-{look_index}.png";self._enhance(crop).save(path);images=[path]
                    result={"ok":True,"look":look_index,"purpose":args["purpose"]};emit("Looking closer",args["purpose"])
            elif name in ("mark_and_crop","revise_and_crop"):
                if args["article_id"] not in state["articles"]:result={"ok":False,"error":"declare the article first"}
                else:
                    region_id=args["region_id"];previous=state["regions"].get(region_id,{});revision=previous.get("crop_revision",0)+1
                    while (crop_root/region_id/f"revision-{revision:03d}").exists():revision+=1
                    try:
                        conflict=next((other for key,other in state["regions"].items() if key!=region_id and other["article_id"]!=args["article_id"]
                                       and self._overlap_fraction(args["points"],other["points"])>.12),None)
                        if conflict:raise ValueError(f"owned core substantially overlaps region {conflict['region_id']} from article {conflict['article_id']}")
                        provenance=crop_result(region_id,args["points"],revision)
                    except ValueError as exc:result={"ok":False,"error":str(exc),"instruction":"Redraw a convex core around only this article. Safety padding is added separately and remains outside article ownership."}
                    else:
                        state["regions"][region_id]={"region_id":region_id,"article_id":args["article_id"],"role":args["role"],"label":args["label"],
                            "points":args["points"],"local_order":args["local_order"],"column_index":args["column_index"],
                            "notes":args.get("notes",args.get("reason","")),"status":"awaiting_transcription","verbatim_text":"","confidence":0,"evidence_turn":turn,**provenance}
                        images=[Path(provenance["exact_crop"]),Path(provenance["safety_crop"])]
                        state["version"]+=1;result={"ok":True,"region_id":region_id,"crop_revision":revision,"instruction":"Inspect exact pixels first and the original safety-padded crop second. Redraw clipping or mixed ownership; enhance only when degradation actually prevents reading."}
                        emit("Marked region",f'{args["article_id"]} · {args["role"]} · {region_id}')
            elif name=="remove_region":
                existed=state["regions"].pop(args["region_id"],None);state["version"]+=1
                result={"ok":bool(existed),"removed":args["region_id"]};emit("Removed bad mark",args["region_id"])
            elif name=="remove_article":
                article_id=args["article_id"]
                attached=[key for key,value in state["regions"].items() if value["article_id"]==article_id]
                if article_id not in state["articles"]:
                    result={"ok":False,"error":"unknown article"}
                elif attached:
                    result={"ok":False,"error":"remove or reassign the article's regions first","region_ids":attached}
                else:
                    state["articles"].pop(article_id);state["version"]+=1
                    result={"ok":True,"removed":article_id,"reason":args["reason"]};emit("Removed false article",article_id)
            elif name=="transcribe_region":
                region=state["regions"].get(args["region_id"])
                if stop_after_layout:result={"ok":False,"error":"this is a layout-only run; inspect geometry and use finish_layout"}
                elif not region:result={"ok":False,"error":"unknown region"}
                elif region.get("evidence_turn",turn)>=turn:result={"ok":False,"error":"crop evidence has not been returned and inspected yet; continue on the next turn"}
                else:
                    decision,routing=self._transcribe_blinded(region,instruction,emit);state["ocr_calls"].extend(routing["calls"])
                    review={"source":"blinded_routing","crop_complete":decision.crop_complete,"role_matches_pixels":decision.role_matches_pixels,
                            "issues":decision.issues,"route":routing["route"]}
                    if not decision.crop_complete or not decision.role_matches_pixels:
                        region.update(status="needs_revision",structural_notes="; ".join(decision.issues),region_review=review,ocr_route=routing["route"])
                        result={"ok":False,"must_revise":True,"route":routing["route"],"issues":decision.issues};emit("OCR exposed a crop problem",args["region_id"])
                    elif not decision.final_text.strip():result={"ok":False,"error":"blinded OCR returned empty text"}
                    else:
                        region.update(status="transcribed",verbatim_text=decision.final_text,confidence=decision.confidence,
                                      structural_notes="; ".join(decision.issues),region_review=review,ocr_route=routing["route"],ocr_unresolved=decision.unresolved)
                        state["version"]+=1;done=sum(x["status"]=="transcribed" for x in state["regions"].values())
                        result={"ok":True,"saved":args["region_id"],"text":decision.final_text,"confidence":decision.confidence,
                                "unresolved":decision.unresolved,"route":routing["route"]};emit("Transcribed regions",f'{done}/{len(state["regions"])} · {args["region_id"]}')
            elif name=="render_overlay":
                self._overlay(page,state,overlay_path);state["overlay_version"]=state["version"];state["overlay_turn"]=turn;images=[overlay_path]
                result={"ok":True,"regions":len(state["regions"]),"articles":len(state["articles"]),"instruction":"Compare this overlay to the untouched source. Revise omissions, false marks, clipping, and ownership before finishing."};emit("Inspecting current overlay",f'{len(state["regions"])} regions')
            elif name in ("finish_layout","finish_page"):
                layout_finish=name=="finish_layout"
                missing=[] if layout_finish else [key for key,value in state["regions"].items() if value["status"]!="transcribed"]
                expected=set(state["articles"]);provided=args["article_order"]
                structure_errors=[]
                for article_id,article in state["articles"].items():
                    article_regions=[value for value in state["regions"].values() if value["article_id"]==article_id]
                    orders=[value["local_order"] for value in article_regions]
                    if len(orders)!=len(set(orders)):structure_errors.append(f"{article_id}: local_order values must be unique")
                    declared=article["column_count"]
                    indices={value["column_index"] for value in article_regions if value["role"]=="body" and value["column_index"] is not None}
                    if declared and indices!=set(range(1,declared+1)):structure_errors.append(f"{article_id}: expected body column indices 1..{declared}, found {sorted(indices)}")
                    if any(index is not None and (not declared or index>declared) for index in indices):structure_errors.append(f"{article_id}: body column_index exceeds declared column_count")
                attest=all(args[x] for x in ("source_checked","no_unmarked_text","headings_grounded","columns_verified"))
                if layout_finish!=stop_after_layout:result={"ok":False,"error":"use finish_layout for layout-only runs and finish_page for full digitisation"}
                elif missing:result={"ok":False,"error":"regions not transcribed","region_ids":missing}
                elif structure_errors:result={"ok":False,"error":"article region ordering/columns are inconsistent","issues":structure_errors}
                elif set(provided)!=expected or len(provided)!=len(expected):result={"ok":False,"error":"article_order must contain every declared article exactly once"}
                elif state["overlay_version"]!=state["version"]:result={"ok":False,"error":"render and inspect a fresh overlay after the last edit"}
                elif state["overlay_turn"]>=turn:result={"ok":False,"error":"the fresh overlay has not been returned and inspected yet; finish on the next turn"}
                elif not attest:result={"ok":False,"error":"all final visual attestations must be true"}
                else:
                    started=time.perf_counter();review,response=self._review_page(saved_source,overlay_path,state)
                    usage=getattr(response,"usage",None);state["verification_calls"].append({"kind":"page","request_id":getattr(response,"id",None),
                        "latency_ms":round((time.perf_counter()-started)*1000),"usage":usage.model_dump() if hasattr(usage,"model_dump") else None,"review":review.model_dump()})
                    state["page_review"]=review.model_dump()
                    if not (review.passed and review.no_unmarked_text and review.all_headings_have_heading_regions and review.articles_separated_correctly and review.column_structure_correct):
                        result={"ok":False,"independent_page_review":review.model_dump(),"instruction":"Continue with look_closer, declare/split articles, and revise marks. Render a new overlay afterward."};images=[overlay_path]
                        emit("Independent page review rejected finish",f'{len(review.issues)} issues')
                    else:
                        status="complete_layout" if layout_finish else "complete"
                        state.update(article_order=provided,status=status,final_notes=args["final_notes"]);finished=True
                        result={"ok":True,"finished":True,"mode":"layout_only" if layout_finish else "full_digitisation","independent_page_review":review.model_dump()}
                        emit("Agentic layout complete" if layout_finish else "Agentic page complete",f'{len(state["articles"])} articles · {len(state["regions"])} regions')
            else:result={"ok":False,"error":"unknown tool"}
            record(name,args,result);return result,images

        resume_findings=[]
        if resume_folder:
            occupied={region["article_id"] for region in state["regions"].values()}
            resume_findings.extend({"empty_article":article_id,"required_action":"remove_article"}
                                   for article_id in state["articles"] if article_id not in occupied)
            unchecked=[r for r in state["regions"].values() if r.get("status")=="transcribed" and not r.get("region_review")]
            emit("Reviewing existing crops",f"{len(unchecked)} crop-grounding checks")
            def review_existing(region):
                region["pending_text"]=region.get("verbatim_text","");started=time.perf_counter();review,response=self._review_region(region)
                usage=getattr(response,"usage",None)
                verification_record={"kind":"region","region_id":region["region_id"],"request_id":getattr(response,"id",None),
                    "latency_ms":round((time.perf_counter()-started)*1000),"usage":usage.model_dump() if hasattr(usage,"model_dump") else None,"review":review.model_dump()}
                return region["region_id"],review,verification_record
            with ThreadPoolExecutor(max_workers=min(4,max(1,len(unchecked)))) as pool:
                futures=[pool.submit(review_existing,r) for r in unchecked]
                for future in as_completed(futures):
                    try:region_id,review,verification_record=future.result()
                    except Exception as exc:
                        checkpoint_interruption(exc);raise
                    region=state["regions"][region_id];state["verification_calls"].append(verification_record)
                    acceptable=review.crop_complete and review.role_matches_pixels and not review.contains_unmarked_heading and not review.mixed_articles and (region["role"]!="body" or review.column_flow_complete)
                    region["region_review"]=review.model_dump();region.pop("pending_text",None)
                    if not acceptable:
                        region["status"]="needs_revision";region["structural_notes"]="; ".join(review.issues)
                        resume_findings.append({"region_id":region_id,"issues":review.issues,"required_action":review.required_action})
            self._overlay(page,state,overlay_path);state["overlay_version"]=state["version"]
            emit("Auditing resumed page","Searching for headings trapped inside body regions")
            try:review,response=self._review_page(saved_source,overlay_path,state)
            except Exception as exc:
                checkpoint_interruption(exc);raise
            usage=getattr(response,"usage",None)
            state["verification_calls"].append({"kind":"page","request_id":getattr(response,"id",None),"latency_ms":None,
                "usage":usage.model_dump() if hasattr(usage,"model_dump") else None,"review":review.model_dump()})
            state["page_review"]=review.model_dump();resume_findings.extend({"page_issue":x} for x in review.issues);self._atomic(state_path,state)
        emit("Agent surveying page",saved_source.name)
        resume_text=("A prior agent stopped too early. Continue from the persisted state, inspect the overlay, and repair it; do not trust its completion attestations. "
                     f"Independent pixel-grounding findings: {json.dumps(resume_findings,ensure_ascii=False)}") if resume_folder else "Begin with the page survey, then take the pen and work article by article."
        initial_content=[{"type":"input_text","text":f"OPERATOR REQUEST: {instruction}\n{resume_text}"},{"type":"input_image","image_url":self._data(saved_source),"detail":"original"}]
        if resume_folder and overlay_path.is_file():initial_content.append({"type":"input_image","image_url":self._data(overlay_path),"detail":"original"})
        initial=[{"role":"user","content":initial_content}]
        response=None
        for turn in range(1,max_turns+1):
            started=time.perf_counter()
            mode_prompt=("\n\nMODE: LAYOUT ONLY. Do not call transcribe_region and do not spend turns reading body text. "
                         "Use exact/padded crops only to verify geometry, article ownership, headings, column count and column order. "
                         "After a fresh overlay is inspected, call finish_layout."
                         if stop_after_layout else
                         "\n\nMODE: FULL DIGITISATION. After visually checking each returned crop, call transcribe_region. "
                         "Revise any crop rejected by OCR routing, then call finish_page.")
            kwargs={"model":self.model,"instructions":PEN_PROMPT+mode_prompt,"tools":TOOLS,"parallel_tool_calls":True}
            if response is None:kwargs["input"]=initial
            else:kwargs.update(previous_response_id=response.id,input=pending)
            try:response=self.client.responses.create(**kwargs)
            except Exception as exc:
                checkpoint_interruption(exc);raise
            usage=getattr(response,"usage",None)
            state["calls"].append({"turn":turn,"request_id":getattr(response,"id",None),"latency_ms":round((time.perf_counter()-started)*1000),
                "usage":usage.model_dump() if hasattr(usage,"model_dump") else None});self._atomic(state_path,state)
            calls=[x for x in response.output if getattr(x,"type",None)=="function_call"]
            if not calls:
                pending=[{"role":"user","content":[{"type":"input_text","text":"Continue operating the page with tools. Do not stop at commentary; render and finish only after the visual work is complete."}]}]
                continue
            pending=[]
            priority={"declare_article":0,"look_closer":1,"mark_and_crop":2,"revise_and_crop":2,"remove_region":2,"remove_article":3,"transcribe_region":4,"render_overlay":5,"finish_layout":6,"finish_page":6}
            for call in sorted(calls,key=lambda x:priority.get(x.name,9)):
                args=json.loads(call.arguments or "{}")
                try:result,images=handle(call.name,args)
                except Exception as exc:
                    checkpoint_interruption(exc);raise
                pending.append({"type":"function_call_output","call_id":call.call_id,"output":json.dumps(result,ensure_ascii=False)})
                if images:pending.append(image_messages(f"Visual evidence returned by {call.name} for {args.get('region_id',args.get('purpose','overlay'))}. Image 1 is exact/source-aligned; image 2, when present, is padded and enhanced.",images))
            if finished:break
        if not finished:
            state["status"]="failed";self._atomic(state_path,state);raise RuntimeError(f"pen agent did not finish within {max_turns} turns")

        self._overlay(page,state,overlay_path);ordered=[]
        for article_id in state["article_order"]:
            ordered.extend(sorted((r for r in state["regions"].values() if r["article_id"]==article_id),key=lambda r:(r["local_order"],r["region_id"])))
        text="\n\n".join(r["verbatim_text"].strip() for r in ordered if r["verbatim_text"].strip())
        uncertain_regions=[r["region_id"] for r in ordered if r.get("ocr_unresolved") or "[неясно" in r.get("verbatim_text","").casefold()]
        article_rows=[]
        for article_index,article_id in enumerate(state["article_order"],1):
            article=state["articles"][article_id];regions=sorted((r for r in state["regions"].values() if r["article_id"]==article_id),key=lambda r:(r["local_order"],r["region_id"]))
            article_text="\n\n".join(r["verbatim_text"].strip() for r in regions if r["verbatim_text"].strip())
            article_uncertain=[r["region_id"] for r in regions if r["region_id"] in uncertain_regions]
            article_rows.append({"article_id":article_id,"article_order":article_index,"label":article["label"],
                "column_count":article["column_count"],"column_order":article["column_order"],"text":article_text,
                "confidence":min((r["confidence"] for r in regions if r.get("verbatim_text")),default=0),
                "uncertain":bool(article_uncertain),"uncertain_region_ids":article_uncertain,
                "regions":[{key:r.get(key) for key in ("region_id","role","label","local_order","column_index","points","verbatim_text","confidence","ocr_unresolved","ocr_route","crop_revision")} for r in regions]})
        articles_path=folder/f"{stamp}_{source.stem}.articles.json"
        self._atomic(articles_path,{"schema_version":1,"source":saved_source.name,"source_sha256":state["source_sha256"],
                                   "metadata":metadata_from_filename(source.name),
                                   "workflow":"agentic-pen","article_count":len(article_rows),"articles":article_rows})
        markdown_path=None
        if not stop_after_layout:
            markdown_path=folder/f"{stamp}_{source.stem}.md"
            sections=[]
            for article_id in state["article_order"]:
                article=state["articles"][article_id];parts=[r["verbatim_text"].strip() for r in sorted((x for x in state["regions"].values() if x["article_id"]==article_id),key=lambda x:(x["local_order"],x["region_id"])) if r["verbatim_text"].strip()]
                sections.append(f'## {article["label"]}\n\n'+"\n\n".join(parts))
            markdown_path.write_text(f"---\nsource: {saved_source.name}\nlayout_model: {self.model}\ntranscription_model: {self.transcription_model}\nadjudicator_model: {self.adjudicator_model}\nworkflow: agentic-pen\narticles: {len(article_rows)}\nuncertain_regions: {len(uncertain_regions)}\n---\n\n"+"\n\n".join(sections)+"\n",encoding="utf-8")
        if uncertain_regions and not stop_after_layout:state["status"]="complete_with_uncertainty"
        state["uncertain_regions"]=uncertain_regions;state["articles_path"]=str(articles_path);self._atomic(state_path,state)
        return {"source_name":source.name,"source_path":str(saved_source),"markdown_path":str(markdown_path) if markdown_path else None,
            "layout_path":str(state_path),"articles_path":str(articles_path),"overlay_path":str(overlay_path),"semantic_path":None,"semantic_overlay_path":None,
            "text":text,"omissions_found":[],"uncertain_regions":uncertain_regions,"corrections_made":[],"confidence":1.0 if stop_after_layout else min((r["confidence"] for r in ordered),default=0),
            "verified":not uncertain_regions,"failed_regions":{},"region_count":len(ordered),"layout_only":stop_after_layout,"folder":str(folder),
            "article_count":len(state["articles"]),"workflow":"agentic_pen"}
