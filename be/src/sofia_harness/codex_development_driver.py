from __future__ import annotations

import argparse,hashlib,json,shutil,uuid
from datetime import datetime
from pathlib import Path
from .metadata import metadata_from_filename

import cv2
import numpy as np
from PIL import Image,ImageDraw
from pydantic import BaseModel,Field,model_validator

from .pen_agent import PenAgent


class DevelopmentRegion(BaseModel):
    region_id:str
    role:str
    local_order:int=Field(ge=0)
    column_index:int|None=None
    polygon:list[list[float]]
    transcribe:bool=True
    text:str=""
    confidence:float=Field(default=0,ge=0,le=1)
    notes:str=""
    preprocess:bool=False
    preprocess_reason:str=""


class DevelopmentArticle(BaseModel):
    article_id:str
    label:str
    kind:str="article"
    article_order:int=Field(ge=1)
    column_count:int=Field(ge=0)
    column_order:list[str]=Field(default_factory=list)
    regions:list[DevelopmentRegion]


class DevelopmentPlan(BaseModel):
    source:str
    instruction:str="Digitise the complete newspaper page article by article."
    inspected_by:list[str]=Field(default_factory=list)
    articles:list[DevelopmentArticle]

    @model_validator(mode="after")
    def validate_identity_and_order(self):
        article_ids=[article.article_id for article in self.articles]
        article_orders=[article.article_order for article in self.articles]
        region_ids=[region.region_id for article in self.articles for region in article.regions]
        if len(article_ids)!=len(set(article_ids)):raise ValueError("article_id values must be unique")
        if len(article_orders)!=len(set(article_orders)):raise ValueError("article_order values must be unique")
        if len(region_ids)!=len(set(region_ids)):raise ValueError("region_id values must be unique across articles")
        for article in self.articles:
            orders=[region.local_order for region in article.regions]
            if len(orders)!=len(set(orders)):raise ValueError(f"{article.article_id}: local_order values must be unique")
            for region in article.regions:PenAgent._validate_quad(region.polygon)
        return self


class ReviewedRegion(BaseModel):
    region_id:str
    verbatim_text:str
    confidence:float=Field(ge=0,le=1)
    unresolved:list[str]=Field(default_factory=list)
    corrections:list[str]=Field(default_factory=list)


class ReviewedTranscript(BaseModel):
    source_sha256:str
    reviewed_by:str
    regions:list[ReviewedRegion]


def materialize(plan_file:str|Path,output_root:str|Path="digitized",transcription_files:list[str|Path]|None=None):
    plan_path=Path(plan_file).resolve();plan=DevelopmentPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    source=Path(plan.source).resolve()
    if not source.is_file():raise FileNotFoundError(source)
    stamp=datetime.now().strftime("%Y%m%d-%H%M");root=Path(output_root).resolve();root.mkdir(parents=True,exist_ok=True)
    folder=root/f"{stamp}_{source.stem}_codex-development"
    if folder.exists():folder=root/f"{folder.name}_{uuid.uuid4().hex[:6]}"
    audit=folder/"audit";crop_root=audit/"articles";crop_root.mkdir(parents=True)
    saved_source=folder/f"{stamp}_{source.name}";shutil.copy2(source,saved_source);page=Image.open(saved_source).convert("RGB")
    source_hash=hashlib.sha256(saved_source.read_bytes()).hexdigest();reviewed={};review_sources=[]
    known_region_ids={region.region_id for article in plan.articles for region in article.regions}
    for value in transcription_files or []:
        review_path=Path(value).resolve();review=ReviewedTranscript.model_validate_json(review_path.read_text(encoding="utf-8"))
        if review.source_sha256.casefold()!=source_hash.casefold():raise ValueError(f"review source hash does not match scan: {review_path}")
        for region in review.regions:
            if region.region_id not in known_region_ids:raise ValueError(f"review contains unknown region: {region.region_id}")
            if region.region_id in reviewed:raise ValueError(f"duplicate reviewed region: {region.region_id}")
            reviewed[region.region_id]=(region,review.reviewed_by)
        review_sources.append({"path":str(review_path),"reviewed_by":review.reviewed_by,"sha256":hashlib.sha256(review_path.read_bytes()).hexdigest()})
    state={"schema_version":1,"workflow":"codex-development-real-scan",
        "source":str(saved_source),"source_sha256":source_hash,"source_dimensions":list(page.size),"instruction":plan.instruction,
        "inspected_by":plan.inspected_by,"transcription_reviews":review_sources,"status":"layout_complete","articles":{},"regions":{},"article_order":[]}
    complete=True
    for article in sorted(plan.articles,key=lambda value:value.article_order):
        state["article_order"].append(article.article_id);state["articles"][article.article_id]={
            "article_id":article.article_id,"label":article.label,"kind":article.kind,"article_order":article.article_order,
            "column_count":article.column_count,"column_order":article.column_order}
        for region in sorted(article.regions,key=lambda value:value.local_order):
            exact,matrix=PenAgent._crop(page,region.polygon,0)
            pad=max(10,round(min(page.size)*.008));padded,padded_matrix=PenAgent._crop(page,region.polygon,pad)
            region_dir=crop_root/article.article_id/region.region_id;region_dir.mkdir(parents=True)
            exact_path=region_dir/"exact.png";padded_path=region_dir/"safety-padded.png";safety_path=region_dir/"safety-boundary.png"
            exact.save(exact_path);padded.save(padded_path)
            normalized=PenAgent._validate_quad(region.polygon)
            source_points=np.asarray([[[x*page.width/1000,y*page.height/1000] for x,y in normalized]],dtype=np.float32)
            core=cv2.perspectiveTransform(source_points,padded_matrix)[0]
            base=padded.convert("RGBA");tint=Image.new("RGBA",base.size,(130,113,72,100));shaded=Image.alpha_composite(base,tint)
            mask=Image.new("L",base.size,0);ImageDraw.Draw(mask).polygon([tuple(map(float,point)) for point in core],fill=255)
            safety=Image.composite(base,shaded,mask).convert("RGB")
            ImageDraw.Draw(safety).line([tuple(map(float,point)) for point in [*core,core[0]]],fill=(232,160,124),width=max(2,round(min(safety.size)/300)))
            safety.save(safety_path)
            enhanced_path=None
            if region.preprocess:
                enhanced_path=region_dir/"selected-enhanced.png";PenAgent._enhance(exact).save(enhanced_path)
            accepted_entry=reviewed.get(region.region_id);accepted=accepted_entry[0] if accepted_entry else None
            text=accepted.verbatim_text if accepted else region.text
            confidence=accepted.confidence if accepted else region.confidence
            has_text=bool(text.strip()) or not region.transcribe;complete=complete and has_text
            artifact_paths=[exact_path,padded_path,safety_path]+([enhanced_path] if enhanced_path else [])
            state["regions"][region.region_id]={"region_id":region.region_id,"article_id":article.article_id,"role":region.role,
                "label":region.notes or region.role,"points":region.polygon,"local_order":region.local_order,"column_index":region.column_index,
                "transcribe":region.transcribe,"verbatim_text":text,"confidence":confidence,"status":"transcribed" if has_text else "awaiting_transcription",
                "exact_crop":str(exact_path),"padded_crop":str(padded_path),"safety_crop":str(safety_path),
                "enhanced_crop":str(enhanced_path) if enhanced_path else None,"perspective_transform":matrix.tolist(),
                "padded_perspective_transform":padded_matrix.tolist(),"safety_core_polygon":[[round(float(x),2),round(float(y),2)] for x,y in core],
                "preprocessing":{"applied":region.preprocess,"reason":region.preprocess_reason,"recipe":"CLAHE grayscale upscaling" if region.preprocess else None},
                "review":{"reviewed_by":accepted_entry[1] if accepted_entry else None,
                          "unresolved":accepted.unresolved if accepted else [],"corrections":accepted.corrections if accepted else []},
                "crop_hashes":{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in artifact_paths}}
    overlay_path=audit/"article-overlay.png";PenAgent._overlay(page,state,overlay_path)
    article_rows=[];uncertain=[]
    for article_id in state["article_order"]:
        article=state["articles"][article_id];regions=sorted((value for value in state["regions"].values() if value["article_id"]==article_id),key=lambda value:value["local_order"])
        text="\n\n".join(value["verbatim_text"].strip() for value in regions if value["transcribe"] and value["verbatim_text"].strip())
        uncertain_ids=[value["region_id"] for value in regions if "[неясно" in value["verbatim_text"].casefold() or value.get("review",{}).get("unresolved")];uncertain.extend(uncertain_ids)
        article_rows.append({**article,"text":text,"uncertain":bool(uncertain_ids),"uncertain_region_ids":uncertain_ids,
            "confidence":min((value["confidence"] for value in regions if value["transcribe"]),default=0),"regions":regions})
    articles_path=folder/f"{stamp}_{source.stem}.articles.json"
    articles_path.write_text(json.dumps({"schema_version":1,"source":saved_source.name,"source_sha256":source_hash,
        "metadata":metadata_from_filename(source.name),
        "workflow":"codex-development-real-scan","articles":article_rows},ensure_ascii=False,indent=2),encoding="utf-8")
    markdown_path=None
    if complete:
        markdown_path=folder/f"{stamp}_{source.stem}.md"
        sections=[]
        for article in article_rows:
            if not article["text"]:continue
            unresolved=[f"{region['region_id']}: {value}" for region in article["regions"] for value in region.get("review",{}).get("unresolved",[])]
            note=("\n\n> Неясно в източника: "+"; ".join(unresolved)) if unresolved else ""
            sections.append(f"## {article['label']}\n\n{article['text']}{note}")
        markdown_path.write_text(f"---\nsource: {saved_source.name}\nworkflow: codex-development-real-scan\narticles: {len(article_rows)}\nuncertain_regions: {len(uncertain)}\n---\n\n"+"\n\n".join(sections)+"\n",encoding="utf-8")
        state["status"]="complete_with_uncertainty" if uncertain else "complete"
    state["uncertain_regions"]=uncertain;state["overlay_path"]=str(overlay_path);state["articles_path"]=str(articles_path)
    state_path=audit/"development-state.json";state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding="utf-8")
    return {"folder":str(folder),"source_path":str(saved_source),"overlay_path":str(overlay_path),"articles_path":str(articles_path),
            "markdown_path":str(markdown_path) if markdown_path else None,"article_count":len(article_rows),"region_count":len(state["regions"]),
            "complete":complete,"uncertain_regions":uncertain,"state_path":str(state_path)}


def main():
    parser=argparse.ArgumentParser(description="Materialize a real-scan run inspected and transcribed by Codex without paid API calls.")
    parser.add_argument("plan");parser.add_argument("--output",default="digitized")
    parser.add_argument("--transcription",action="append",default=[],help="Reviewed region JSON; may be repeated")
    args=parser.parse_args()
    print(json.dumps(materialize(args.plan,args.output,args.transcription),ensure_ascii=False,indent=2))


if __name__=="__main__":main()
