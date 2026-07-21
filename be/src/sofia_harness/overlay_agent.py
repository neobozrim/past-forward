from __future__ import annotations

import base64,hashlib,html,io,json
from pathlib import Path
from typing import Literal
from PIL import Image,ImageDraw,ImageFont
from pydantic import BaseModel,Field,model_validator


class DraftBlock(BaseModel):
    block_id:str
    article_id:str
    role:Literal["heading","body","caption","byline","metadata"]
    start_line:int|None=None
    end_line:int|None=None
    start_quote:str|None=None
    end_quote:str|None=None
    polygon:list[list[float]]
    confidence:float=Field(ge=0,le=1)
    font_size_ratio:float=Field(ge=.0005,le=.08)
    line_height:float=Field(default=1.04,ge=.85,le=1.8)
    font_family:Literal["serif","sans","condensed","display"]="serif"
    font_weight:int=Field(default=400,ge=300,le=900)
    letter_spacing_em:float=Field(default=0,ge=-.2,le=.5)
    font_width_scale:float=Field(default=1,ge=.35,le=1.3)
    visual_text:str|None=None

    @model_validator(mode="after")
    def valid(self):
        if len(self.polygon)!=4 or any(len(point)!=2 for point in self.polygon):raise ValueError("polygon must contain four [x,y] points")
        if any(value<0 or value>1000 for point in self.polygon for value in point):raise ValueError("polygon coordinates must be normalized from 0 to 1000")
        if self.role=="heading" and any(value is not None for value in (self.start_line,self.end_line,self.start_quote,self.end_quote)):raise ValueError("heading blocks use the saved heading, not text anchors")
        if self.role!="heading":
            line_mode=self.start_line is not None or self.end_line is not None
            quote_mode=self.start_quote is not None or self.end_quote is not None
            if line_mode==quote_mode:raise ValueError("text blocks require exactly one selector: legacy line range or exact start/end quotes")
            if line_mode and (self.start_line is None or self.end_line is None or self.start_line<0 or self.end_line<self.start_line):raise ValueError("invalid inclusive source line range")
            if quote_mode and (not self.start_quote or not self.end_quote):raise ValueError("both exact text anchors are required")
        return self


class PlacementDraft(BaseModel):
    inspected_by:str
    blocks:list[DraftBlock]
    notes:list[str]=Field(default_factory=list)


class TwoPointDraftBlock(BaseModel):
    """Vision-authored source rectangle; no inferred or calculated geometry."""
    block_id:str
    article_id:str
    role:Literal["heading","body","caption","byline","metadata"]
    start_line:int|None=None
    end_line:int|None=None
    start_quote:str|None=None
    end_quote:str|None=None
    top_left:list[float]
    bottom_right:list[float]
    confidence:float=Field(ge=0,le=1)
    font_size_ratio:float=Field(ge=.0005,le=.08)
    line_height:float=Field(default=1.04,ge=.85,le=1.8)
    font_family:Literal["serif","sans","condensed","display"]="serif"
    font_weight:int=Field(default=400,ge=300,le=900)
    letter_spacing_em:float=Field(default=0,ge=-.2,le=.5)
    font_width_scale:float=Field(default=1,ge=.35,le=1.3)
    visual_text:str|None=None

    @model_validator(mode="after")
    def valid_points(self):
        if len(self.top_left)!=2 or len(self.bottom_right)!=2:raise ValueError("two image points are required")
        if any(value<0 or value>1000 for value in [*self.top_left,*self.bottom_right]):raise ValueError("points must be normalized 0..1000")
        if self.bottom_right[0]<=self.top_left[0] or self.bottom_right[1]<=self.top_left[1]:raise ValueError("bottom_right must be below and right of top_left")
        return self


class TwoPointPlacementDraft(BaseModel):
    inspected_by:str
    blocks:list[TwoPointDraftBlock]
    notes:list[str]=Field(default_factory=list)


class AuditIssue(BaseModel):
    block_id:str
    issue:str
    required_action:str


class PlacementAudit(BaseModel):
    accepted:bool
    coverage_complete:bool
    article_identity_correct:bool
    line_ownership_exact:bool
    geometry_matches_source:bool
    typography_matches_source:bool
    regions_filled_to_source_bounds:bool
    checked_block_ids:list[str]=Field(default_factory=list)
    issues:list[AuditIssue]=Field(default_factory=list)
    notes:list[str]=Field(default_factory=list)


class OverlayBlock(BaseModel):
    block_id:str
    article_id:str="article"
    article_label:str=""
    text:str=Field(min_length=1)
    overlay_text:str|None=None
    start_line:int|None=None
    end_line:int|None=None
    start_offset:int|None=None
    end_offset:int|None=None
    polygon:list[list[float]]
    rotation:float=Field(default=0,ge=-15,le=15)
    role:str="body"
    confidence:float=Field(ge=0,le=1)
    font_size_ratio:float=Field(ge=.0005,le=.08)
    line_height:float=Field(default=1.04,ge=.85,le=1.8)
    font_family:Literal["serif","sans","condensed","display"]="serif"
    font_weight:int=Field(default=400,ge=300,le=900)
    letter_spacing_em:float=Field(default=0,ge=-.2,le=.5)
    font_width_scale:float=Field(default=1,ge=.35,le=1.3)

    @model_validator(mode="after")
    def valid_polygon(self):
        if len(self.polygon)!=4 or any(len(point)!=2 for point in self.polygon):raise ValueError("polygon must contain four [x,y] points")
        if any(value<0 or value>1000 for point in self.polygon for value in point):raise ValueError("polygon coordinates must be normalized from 0 to 1000")
        return self


class OverlayPlan(BaseModel):
    schema_version:int=3
    source_sha256:str
    transcript_sha256:str
    inspected_by:str
    blocks:list[OverlayBlock]
    coverage_complete:bool
    audit:PlacementAudit
    notes:list[str]=Field(default_factory=list)


PLACEMENT_PROMPT="""You are an archival visual-placement agent. The transcription is already verified: never OCR, rewrite, normalize, or invent text. Inspect the untouched photograph and return geometry plus source-line ownership only.

For every saved article:
- locate its heading separately only when the photograph actually typesets a separate heading. A saved label that is merely the opening phrase of a continuous caption sentence is inline and must not become a duplicate heading block;
- identify the real number of body columns and their visual reading order;
- make one body block per physical column or genuine continuation region;
- for each block, first identify the top-left pixel of its first photographed glyph and the bottom-right pixel of its last photographed glyph. Use those as experimental extent anchors, then inspect the other two corners and refine all four points for camera skew and irregular column edges. The anchors are visual observations, never coordinates calculated from text length;
- use exact start_quote/end_quote copied from the supplied diplomatic text to identify each region. Quotes must be long enough to be unique and may end at ANY word visible at a physical column handoff; never inherit Markdown paragraph boundaries or divide by character count;
- follow page skew using clockwise normalized 0..1000 quadrilaterals;
- estimate font_size_ratio as rendered font size divided by page width, plus the observed line_height;
- choose font_family (serif, sans, condensed, or display), font_weight, letter_spacing_em, and font_width_scale from the photographed type; use font_width_scale for historically ultra-condensed display faces, never to compensate for incorrect geometry;
- set visual_text only to reproduce photographed line breaks. It must contain exactly the grounded text with whitespace changed—never add, delete, or alter characters;
- choose each polygon and typography together so the rendered final line ends at the same visual boundary as the photographed final line; substantial blank tails are invalid;
- locate captions/bylines independently when visually distinct. A caption scattered around an illustration must become multiple tight text polygons; never use one large polygon covering the illustration or photograph;
- headings must use visibly larger type than their article body, but their polygon must end above the first body glyph. A heading may overlap the photographed heading by a few pixels only; it may never displace or cover body copy;
- body/caption polygons must hug the photographed glyph footprint. Do not add bottom padding, empty rows, or unused image area. If rendered text ends early, shrink the polygon to its real last line or correct the visual line breaks and typography after inspecting the render.

Never divide text by equal length. Read the actual first and last visible words of each physical region. Together, the anchored spans must partition the saved article text in reading order, without missing or duplicated non-whitespace. Decorative art and photographs are not text regions. Before returning, inspect the whole proposed puzzle again as a rendered page, not as isolated coordinates."""

TWO_POINT_PLACEMENT_PROMPT="""You are an archival multimodal placement agent. Use ONLY direct visual inspection of the untouched page. Never calculate coordinates from text length, column count, Markdown structure, neighboring boxes, or any layout formula.

For each real photographed text component:
1. find the exact first visible glyph and mark its TOP-LEFT image point;
2. find the exact last visible glyph and mark its BOTTOM-RIGHT image point;
3. assign the exact corresponding transcription span using start_quote/end_quote;
4. visually choose font size, line height, family, weight, spacing, and optional visual line breaks.

Return only these two image points for geometry. A component may be a headline, physical body column, caption, byline, or metadata line. Follow the actual photographed components and reading order. Do not split a region evenly, move text between columns, extend a box into blank space, or place a box based on another box. The two points must be observations from the photograph. Every transcription character must belong to exactly one photographed component. Before returning, re-inspect every top-left and bottom-right point against the untouched image."""

TWO_POINT_AUDIT_PROMPT="""Independently inspect an archival overlay. Image 1 is the untouched photograph, Image 2 is the actual browser render, and Image 3 labels block IDs. For every supplied block, compare the rendered text with the exact photographed component selected by its top-left and bottom-right points. Reject any block whose box begins before/after the first glyph, ends before/after the last glyph, contains the wrong transcription span, uses the wrong physical column, or materially mismatches font size/spacing. Rectangles are intentional in this experimental two-point mode; do not invent skew corrections. Include every block ID exactly once in checked_block_ids. Accept only when the whole photographed page is populated correctly."""

AUDIT_PROMPT="""You are the independent visual auditor for an archival text overlay. Image 1 is the untouched source. Image 2 is the ACTUAL browser-rendered overlay users will see. Image 3 is a labeled region map used only to identify block IDs. Compare Image 2 to Image 1 from top-left to bottom-right.

For EVERY proposed block ID, add it exactly once to checked_block_ids after checking: correct article identity; exact first and last source line; real photographed polygon including skew; heading/body boundary; font scale; line height; and whether the rendered last line lands on the photographed last line. Reject oversized headings that displace the apparent body start, body type not visibly smaller than its heading, any substantial blank tail or empty bottom rows, text extending beyond its source section, a caption polygon that covers illustration/photo area, rectangular geometry where the photographed shape is skewed/irregular, missing source regions, wrong column order, or a column handoff that does not match the visible text. Put every failure in issues with a concrete required_action. Do not reward text coverage or lack of browser overflow by itself. accepted may be true only when every supplied block ID was checked, every visual verdict is true, and issues is empty."""

REGION_AUDIT_PROMPT="""You are auditing one batch from an archival overlay. Image 1 is the full untouched page for context. Image 2 is the full actual browser render. Image 3 is a contact sheet: each labeled row shows the untouched source crop on the LEFT and the browser-rendered crop on the RIGHT for one block. Audit ONLY the requested IDs and include every requested ID exactly once in checked_block_ids. At crop scale, reject a polygon that misses/overlaps the photographed text region, wrong first/last words, incorrect physical column handoff, materially wrong font width/size/line spacing, overflow, or more than a small final-line tail. All verdict booleans and accepted may be true only if every requested crop is a near one-to-one match."""


def _minimum_rendered_fill(block:OverlayBlock)->float:
    """Hard release guardrail: text polygons end at their last rendered glyph."""
    if block.role=="heading":return .86
    if block.role in {"body","caption","byline","metadata"}:return .93
    return .90


def _articles(path:Path):
    payload=json.loads(path.read_text(encoding="utf-8"));articles=[]
    for article in payload.get("articles",[]):
        source=article.get("verbatim_text",article.get("text",""));heading=(article.get("heading") or "").strip()
        # Digitization commonly preserves the printed heading as the first line of
        # verbatim_text. It is placed by its own heading block and must not be
        # duplicated into the first body column.
        body=source;heading_inline=False
        if heading and source.lstrip().startswith(heading):
            leading=len(source)-len(source.lstrip());remainder=source[leading+len(heading):]
            heading_inline=bool(remainder and not remainder.startswith(("\r","\n")))
            if not heading_inline:body=remainder.lstrip("\r\n ")
        articles.append(dict(article,body_text=body,heading_inline=heading_inline,lines=body.splitlines()))
    return payload,articles


def _ground(draft:PlacementDraft,articles:list[dict]):
    by_id={article["article_id"]:article for article in articles};blocks=[]
    for item in draft.blocks:
        if item.article_id not in by_id:raise ValueError(f"unknown article_id: {item.article_id}")
        article=by_id[item.article_id];lines=article["lines"];source=article.get("body_text")
        if source is None:source="\n".join(lines)
        start=end=None
        if item.role=="heading":
            if article.get("heading_inline"):raise ValueError(f"{item.block_id}: inline caption label must not be duplicated as a heading block")
            text=(article.get("heading") or article.get("label") or "").strip()
        elif item.start_quote is not None:
            start=source.find(item.start_quote)
            if start<0:raise ValueError(f"{item.block_id}: exact start_quote not found")
            end_at=source.find(item.end_quote,start+len(item.start_quote)) if item.end_quote!=item.start_quote else start
            if end_at<0:raise ValueError(f"{item.block_id}: exact end_quote not found after start_quote")
            end=end_at+len(item.end_quote);text=source[start:end].strip()
        else:
            if item.end_line>=len(lines):raise ValueError(f"{item.block_id}: source line is outside the article")
            text="\n".join(lines[item.start_line:item.end_line+1]).strip()
        if not text:raise ValueError(f"{item.block_id}: grounded text is empty")
        visual=item.visual_text or "\n".join(line for line in text.splitlines() if line.strip())
        if "".join(visual.split())!="".join(text.split()):raise ValueError(f"{item.block_id}: visual_text may change whitespace only")
        data=item.model_dump(exclude={"start_quote","end_quote","visual_text"});blocks.append(OverlayBlock(**data,start_offset=start,end_offset=end,article_label=article.get("heading") or article.get("label") or item.article_id,text=text,overlay_text=visual))
    for article in articles:
        lines=article["lines"];claimed=[];spans=[];source=article.get("body_text")
        if source is None:source="\n".join(lines)
        for block in blocks:
            if block.article_id==article["article_id"] and block.role!="heading":
                if block.start_offset is not None:spans.append((block.start_offset,block.end_offset))
                else:claimed.extend(range(block.start_line,block.end_line+1))
        if spans:
            if claimed:raise ValueError(f'{article["article_id"]}: mixed text selector modes are invalid')
            spans.sort();cursor=0
            for start,end in spans:
                if source[cursor:start].strip():raise ValueError(f'{article["article_id"]}: anchored coverage has a gap or is reordered')
                if start<cursor:raise ValueError(f'{article["article_id"]}: anchored coverage overlaps')
                cursor=end
            if source[cursor:].strip():raise ValueError(f'{article["article_id"]}: anchored coverage is incomplete')
            required=[];actual=[]
        else:
            required=[index for index,line in enumerate(lines) if line.strip()]
            actual=[index for index in claimed if lines[index].strip()]
        if actual!=required:raise ValueError(f'{article["article_id"]}: body line coverage is incomplete, duplicated, or out of order')
        heading=(article.get("heading") or article.get("label") or "").strip()
        if heading and not article.get("heading_inline") and not any(block.article_id==article["article_id"] and block.role=="heading" for block in blocks):raise ValueError(f'{article["article_id"]}: heading placement is missing')
    return blocks


def _two_point_draft(draft:TwoPointPlacementDraft)->PlacementDraft:
    blocks=[]
    for item in draft.blocks:
        x1,y1=item.top_left;x2,y2=item.bottom_right
        data=item.model_dump(exclude={"top_left","bottom_right"})
        blocks.append(DraftBlock(**data,polygon=[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]))
    return PlacementDraft(inspected_by=draft.inspected_by,blocks=blocks,notes=draft.notes)


def _data(raw:bytes,mime="image/png"):return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def apply_typography(block:dict,page_aspect:float=1.4)->dict:
    """Legacy-only helper for non-agent demos; accepted agent plans must already contain typography."""
    visual_text=block.get("overlay_text") or "\n".join(line for line in block.get("text","").splitlines() if line.strip());block={**block,"overlay_text":visual_text}
    if block.get("font_size_ratio"):return block
    polygon=block["polygon"];text=visual_text;role=block.get("role","body")
    width=max(point[0] for point in polygon)-min(point[0] for point in polygon);height=max(point[1] for point in polygon)-min(point[1] for point in polygon);lines=text.splitlines() or [""];longest=max(map(len,lines),default=1)
    width_ratio=width/max(1,longest*.58)/1000;height_ratio=height*page_aspect/max(1,len(lines)*1.04)/1000
    return {**block,"font_size_ratio":round(max(.0005,min(.05,min(width_ratio,height_ratio)*(1.08 if role=="heading" else 1))),5),"line_height":block.get("line_height",1.04)}


def validate_accepted_plan(payload:dict)->None:
    """Reject anything that was not explicitly placed and accepted by the vision workflow."""
    audit=payload.get("audit") or {}
    required=("accepted","coverage_complete","article_identity_correct","line_ownership_exact",
              "geometry_matches_source","typography_matches_source","regions_filled_to_source_bounds")
    if not payload.get("coverage_complete") or not all(audit.get(field) is True for field in required):
        raise ValueError("overlay plan has not passed the complete independent vision audit")
    if audit.get("issues"):raise ValueError("overlay plan still contains visual audit issues")
    if payload.get("schema_version",1)>=3:
        expected={block.get("block_id") for block in payload.get("blocks",[])}
        checked=audit.get("checked_block_ids",[])
        if len(checked)!=len(expected) or set(checked)!=expected:
            raise ValueError("overlay plan has not been individually vision-audited for every region")
    for block in payload.get("blocks",[]):
        if not block.get("font_size_ratio") or not block.get("line_height"):
            raise ValueError(f'{block.get("block_id","overlay block")}: vision-authored typography is missing')


def without_inline_caption_duplicates(blocks:list[dict])->list[dict]:
    """Compatibility cleanup for v2 plans created before inline labels existed."""
    result=[]
    for block in blocks:
        duplicate=block.get("role")=="heading" and any(
            other.get("article_id")==block.get("article_id")
            and other.get("role")=="caption"
            and other.get("polygon")==block.get("polygon")
            and (other.get("text") or "").startswith(block.get("text") or "\0")
            for other in blocks
        )
        if not duplicate:result.append(block)
    return result


def _region_map(image_path:Path,blocks:list[OverlayBlock]):
    image=Image.open(image_path).convert("RGBA");layer=Image.new("RGBA",image.size,(0,0,0,0));draw=ImageDraw.Draw(layer)
    for index,block in enumerate(blocks,1):
        points=[(x*image.width/1000,y*image.height/1000) for x,y in block.polygon];draw.polygon(points,fill=(255,238,214,210),outline=(183,79,53,255),width=max(2,image.width//900))
        left,top=max(0,int(min(x for x,_ in points))),max(0,int(min(y for _,y in points)));right=max(left+1,int(max(x for x,_ in points)))
        size=max(5,round(block.font_size_ratio*image.width));font=ImageFont.truetype("arial.ttf",size)
        words=(block.overlay_text or block.text).replace("\n"," ").split();lines=[];line=""
        for word in words:
            candidate=(line+" "+word).strip()
            if line and draw.textlength(candidate,font=font)>right-left:lines.append(line);line=word
            else:line=candidate
        if line:lines.append(line)
        step=max(1,round(size*block.line_height));draw.multiline_text((left,top),"\n".join(lines),font=font,fill=(20,15,10,255),spacing=max(0,step-size))
        label=f"{index} {block.article_id} {block.role} {block.start_line if block.start_line is not None else 'H'}-{block.end_line if block.end_line is not None else 'H'}"
        draw.text(points[0],label,fill=(80,25,20,255),stroke_width=max(1,image.width//1600),stroke_fill=(255,250,242,255))
    out=io.BytesIO();Image.alpha_composite(image,layer).convert("RGB").save(out,"JPEG",quality=88);return out.getvalue()


def _rendered_preview(image_path:Path,blocks:list[OverlayBlock],render_width:int|None=None,capture:bool=True):
    """Render the exact overlay contract in Chromium so vision audits what users will see."""
    from playwright.sync_api import sync_playwright
    image=Image.open(image_path);width=min(render_width or 1600,image.width);height=round(width*image.height/image.width)
    source=_data(Path(image_path).read_bytes(),{".png":"image/png",".webp":"image/webp"}.get(Path(image_path).suffix.casefold(),"image/jpeg")) if capture else ""
    regions=[]
    for block in blocks:
        xs=[point[0]/10 for point in block.polygon];ys=[point[1]/10 for point in block.polygon]
        left,top=min(xs),min(ys);w,h=max(xs)-left,max(ys)-top
        polygon=",".join(f"{(point[0]/10-left)/w*100:.3f}% {(point[1]/10-top)/h*100:.3f}%" for point in block.polygon)
        text=html.escape(block.overlay_text or block.text).replace("\n","<br>")
        # This value is embedded in a double-quoted style attribute; keep family
        # names unquoted so the attribute cannot be terminated early.
        family={"serif":'Georgia,Times New Roman,serif',"sans":'Arial,sans-serif',"condensed":'Arial Narrow,Roboto Condensed,Arial,sans-serif',"display":'Impact,Arial Narrow,sans-serif'}[block.font_family]
        regions.append(f'<article data-block="{html.escape(block.block_id)}" style="left:{left}%;top:{top}%;width:{w}%;height:{h}%;--polygon:{polygon};--font-width:{block.font_width_scale};font-size:{block.font_size_ratio*width}px;line-height:{block.line_height};font-family:{family};font-weight:{block.font_weight};letter-spacing:{block.letter_spacing_em}em"><span>{text}</span></article>')
    document=f'''<!doctype html><style>*{{box-sizing:border-box}}html,body{{margin:0;background:#ddd}}#canvas{{position:relative;width:{width}px;height:{height}px;overflow:hidden}}img{{display:block;width:100%;height:100%}}#layer{{position:absolute;inset:0}}article{{position:absolute;padding:0;background:rgba(255,250,242,.86);color:#15110c;border:.5px solid rgba(232,160,124,.75);font-family:Georgia,"Times New Roman",serif;overflow:hidden;clip-path:polygon(var(--polygon))}}article>span{{display:block;width:calc(100% / var(--font-width));transform:scaleX(var(--font-width));transform-origin:top left}}</style><div id=canvas><img src="{source}"><div id=layer>{''.join(regions)}</div></div>'''
    with sync_playwright() as runtime:
        browser=runtime.chromium.launch(headless=True);page=browser.new_page(viewport={"width":width,"height":height})
        page.set_content(document,wait_until="load");raw=page.locator("#canvas").screenshot(type="png") if capture else b""
        metrics=page.locator("article").evaluate_all("""(elements,renderWidth) => elements.map(element => {
          const text=element.firstElementChild,range=document.createRange();range.selectNodeContents(text);const rects=[...range.getClientRects()];const box=element.getBoundingClientRect(),textBox=text.getBoundingClientRect();
          const top=rects.length?Math.min(...rects.map(rect=>rect.top)):box.top,bottom=rects.length?Math.max(...rects.map(rect=>rect.bottom)):box.top;
          const style=getComputedStyle(element);
          const bottomOverflow=Math.max(0,bottom-box.bottom),rightOverflow=Math.max(0,textBox.right-box.right);
          return {block_id:element.dataset.block,render_width:renderWidth,fill_ratio:(bottom-top)/box.height,overflow:bottomOverflow>1||rightOverflow>1,bottom_overflow_px:bottomOverflow,right_overflow_px:rightOverflow,box_height_px:box.height,box_width_px:box.width,font_family:style.fontFamily,font_weight:style.fontWeight,letter_spacing:style.letterSpacing};
        })""",width)
        browser.close();return raw,metrics


def _responsive_render_metrics(image_path:Path,blocks:list[OverlayBlock],widths:tuple[int,...]=(1600,1200,960,720,560))->list[dict]:
    """Measure the authored overlay at full-page and narrow inspection widths.

    Browser rounding is proportionally larger in the 50/50 inspector. A plan is
    publishable only when its text fits at every representative width, not just
    in the full-page audit render.
    """
    image_width=Image.open(image_path).width;tested=[];samples=[]
    for width in widths:
        width=min(width,image_width)
        if width in tested:continue
        tested.append(width);_,metrics=_rendered_preview(image_path,blocks,width,capture=False)
        samples.append(metrics)
    reference={item["block_id"]:dict(item) for item in samples[0]};by_width=[{item["block_id"]:item for item in sample} for sample in samples]
    for block_id,item in reference.items():
        measured=[sample[block_id] for sample in by_width]
        item["overflow"]=any(metric["overflow"] for metric in measured)
        item["bottom_overflow_px"]=max(metric["bottom_overflow_px"] for metric in measured)
        item["right_overflow_px"]=max(metric["right_overflow_px"] for metric in measured)
        item["bottom_overflow_ratio"]=max(metric["bottom_overflow_px"]/max(1,metric["box_height_px"]) for metric in measured)
        item["right_overflow_ratio"]=max(metric["right_overflow_px"]/max(1,metric["box_width_px"]) for metric in measured)
        item["minimum_fill_ratio"]=min(metric["fill_ratio"] for metric in measured)
        item["maximum_fill_ratio"]=max(metric["fill_ratio"] for metric in measured)
        item["minimum_box_height_px"]=min(metric["box_height_px"] for metric in measured)
        item["responsive_widths"]=tested
    return list(reference.values())


def _scale_polygon_bottom(block:OverlayBlock,factor:float)->None:
    top_left,top_right,bottom_right,bottom_left=block.polygon
    block.polygon=[top_left,top_right,
        [min(1000,max(0,top_right[0]+(bottom_right[0]-top_right[0])*factor)),min(1000,max(0,top_right[1]+(bottom_right[1]-top_right[1])*factor))],
        [min(1000,max(0,top_left[0]+(bottom_left[0]-top_left[0])*factor)),min(1000,max(0,top_left[1]+(bottom_left[1]-top_left[1])*factor))]]


def tighten_rendered_bottom_edges(image_path:str|Path,blocks:list[OverlayBlock],target_fill:float=.97,rounds:int=2)->tuple[list[OverlayBlock],list[dict]]:
    """Tighten non-heading polygons to the last browser-rendered glyph.

    This is a render correction, not placement inference: left/right/top edges,
    text ownership, font height, and reading order stay untouched. A tiny width
    compression is allowed only to neutralize narrow-viewport rounding wraps.
    The following multimodal audit still has to verify the corrected render
    against the photograph before a plan can be accepted.
    """
    adjusted=[block.model_copy(deep=True) for block in blocks];metrics=[]
    for _ in range(max(1,rounds)):
        _,metrics=_rendered_preview(Path(image_path),adjusted,capture=False);by_id={item["block_id"]:item for item in metrics};changed=False
        for block in adjusted:
            if block.role=="banner":continue
            metric=by_id[block.block_id];fill=metric["fill_ratio"];required=.90 if block.role=="heading" else target_fill
            if metric["overflow"] or fill>=required or fill<=0:continue
            factor=max(.08,min(1,fill/required));_scale_polygon_bottom(block,factor)
            changed=True
        if not changed:break
    # The live inspector renders the same plan at a much narrower width. First
    # correct browser-rounding line wraps with at most a very small horizontal
    # type compression. This preserves the visually authored glyph height and
    # bottom edge instead of adding an empty line of geometry.
    for _ in range(4):
        metrics=_responsive_render_metrics(Path(image_path),adjusted);by_id={item["block_id"]:item for item in metrics};changed=False
        for block in adjusted:
            metric=by_id[block.block_id]
            responsive_wrap=metric["maximum_fill_ratio"]>metric["fill_ratio"]+.04
            if metric["bottom_overflow_px"]<=1 or not responsive_wrap or block.font_width_scale<=.94:continue
            block.font_width_scale=max(.94,block.font_width_scale*.985);changed=True
        if not changed:break
    # A block that originally overflowed was intentionally excluded from the
    # first bottom-edge tightening. Once its responsive wrap is repaired, trim
    # that formerly oversized box to the last rendered line as well.
    for _ in range(max(1,rounds)):
        _,metrics=_rendered_preview(Path(image_path),adjusted,capture=False);by_id={item["block_id"]:item for item in metrics};changed=False
        for block in adjusted:
            metric=by_id[block.block_id];required=.90 if block.role=="heading" else target_fill
            if metric["overflow"] or metric["fill_ratio"]>=required or metric["fill_ratio"]<=0:continue
            _scale_polygon_bottom(block,max(.08,min(1,metric["fill_ratio"]/required)));changed=True
        if not changed:break
    # Any remaining sub-pixel vertical overflow gets transparent fit allowance.
    # The painted text span still ends at the final glyph, so this cannot
    # recreate a visible white tail.
    for _ in range(4):
        metrics=_responsive_render_metrics(Path(image_path),adjusted);by_id={item["block_id"]:item for item in metrics};changed=False
        for block in adjusted:
            metric=by_id[block.block_id]
            if metric["bottom_overflow_px"]<=1:continue
            safety_ratio=1.25/max(1,metric["minimum_box_height_px"])
            factor=min(1.12,1+metric["bottom_overflow_ratio"]+safety_ratio)
            _scale_polygon_bottom(block,factor);changed=True
        if not changed:break
    metrics=_responsive_render_metrics(Path(image_path),adjusted)
    return adjusted,metrics


def _comparison_sheet(image_path:Path,rendered:bytes,blocks:list[OverlayBlock])->bytes:
    """High-resolution paired crops let vision inspect narrow newspaper regions."""
    source=Image.open(image_path).convert("RGB");render=Image.open(io.BytesIO(rendered)).convert("RGB")
    canvas=Image.new("RGB",(1600,max(1,len(blocks))*300),(238,232,220));draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype("arial.ttf",24)
    for row,block in enumerate(blocks):
        y=row*300;draw.text((10,y+8),f"{block.block_id}  SOURCE",font=font,fill=(30,25,20));draw.text((810,y+8),"RENDER",font=font,fill=(30,25,20))
        xs=[p[0] for p in block.polygon];ys=[p[1] for p in block.polygon];pad=12
        def crop(page):
            box=(max(0,int(min(xs)*page.width/1000)-pad),max(0,int(min(ys)*page.height/1000)-pad),min(page.width,int(max(xs)*page.width/1000)+pad),min(page.height,int(max(ys)*page.height/1000)+pad))
            piece=page.crop(box);piece.thumbnail((770,245),Image.Resampling.LANCZOS);return piece
        left,right=crop(source),crop(render);canvas.paste(left,(10,y+48));canvas.paste(right,(810,y+48))
    output=io.BytesIO();canvas.save(output,"JPEG",quality=92);return output.getvalue()


def place_two_point_with_api(image_path:str|Path,articles_path:str|Path,client,model="gpt-5.6-sol",audit_model="gpt-5.6-sol",attempts=4)->OverlayPlan:
    """Experimental pure-vision placement: the model authors exactly two points.

    This path deliberately does not call render tightening, coordinate repair,
    column inference, or any other deterministic layout mutation. Failed renders
    go back to the multimodal agent for a new visual placement.
    """
    image=Path(image_path);records=Path(articles_path);raw=image.read_bytes();_,articles=_articles(records)
    numbered="\n\n".join(f"ARTICLE {a['article_id']}\nHEADING: {a.get('heading') or a.get('label') or ''}\nDIPLOMATIC TEXT:\n{a.get('body_text','')}" for a in articles)
    mime={".png":"image/png",".webp":"image/webp"}.get(image.suffix.casefold(),"image/jpeg");feedback="";review_rendered=None;review_map=None;passed_once=False
    # Reserve one extra call because an accepted first pass must still be
    # re-authored from the source-versus-browser visual comparison.
    for _ in range(max(2,attempts+1)):
        content=[{"type":"input_text","text":numbered+feedback},{"type":"input_image","image_url":_data(raw,mime),"detail":"original"}]
        if review_rendered is not None:
            content.extend([{"type":"input_image","image_url":_data(review_rendered,"image/png"),"detail":"original"},{"type":"input_image","image_url":_data(review_map,"image/jpeg"),"detail":"original"}])
        response=client.responses.parse(model=model,instructions=TWO_POINT_PLACEMENT_PROMPT,input=[{"role":"user","content":content}],text_format=TwoPointPlacementDraft)
        visual=_two_point_draft(response.output_parsed);blocks=_ground(visual,articles)
        rendered,metrics=_rendered_preview(image,blocks);region_map=_region_map(image,blocks);expected={block.block_id for block in blocks}
        summary="\n".join(f"{block.block_id}: {block.article_id} {block.role}; begins {block.text[:70]!r}; ends {block.text[-70:]!r}; browser_overflow={metrics[index]['overflow']}" for index,block in enumerate(blocks))
        audit=client.responses.parse(model=audit_model,instructions=TWO_POINT_AUDIT_PROMPT,input=[{"role":"user","content":[{"type":"input_text","text":summary},{"type":"input_image","image_url":_data(raw,mime),"detail":"original"},{"type":"input_image","image_url":_data(rendered,"image/png"),"detail":"original"},{"type":"input_image","image_url":_data(region_map,"image/jpeg"),"detail":"original"}]}],text_format=PlacementAudit).output_parsed
        checked=len(audit.checked_block_ids)==len(expected) and set(audit.checked_block_ids)==expected
        passed=audit.accepted and audit.coverage_complete and audit.article_identity_correct and audit.line_ownership_exact and audit.geometry_matches_source and audit.typography_matches_source and audit.regions_filled_to_source_bounds and not audit.issues and checked and not any(metric["overflow"] for metric in metrics)
        if passed and not passed_once:
            passed_once=True;review_rendered=rendered;review_map=region_map
            feedback="\n\nMANDATORY FINAL MULTIMODAL REFINEMENT. Image 2 is your ACTUAL browser render and Image 3 labels its blocks. Click-equivalent inspection must compare each rendered component with the same photographed text in Image 1. Re-identify both visual corner points and typography for every block. Preserve a value only when the images support it. Return the complete plan again; this pass cannot be skipped.\nLAST PLAN:\n"+response.output_parsed.model_dump_json()+"\nFIRST AUDIT:\n"+audit.model_dump_json()
            continue
        if passed and passed_once:
            return OverlayPlan(source_sha256=hashlib.sha256(raw).hexdigest(),transcript_sha256=hashlib.sha256(records.read_bytes()).hexdigest(),inspected_by=response.output_parsed.inspected_by,blocks=blocks,coverage_complete=True,audit=audit,notes=[*response.output_parsed.notes,"Experimental two-point vision placement; no deterministic geometry correction applied."])
        review_rendered=rendered;review_map=region_map
        feedback="\n\nTHE LAST TWO-POINT PLAN WAS REJECTED. Image 2 is the actual browser render and Image 3 labels its blocks. Re-inspect the photograph and move only points or typography you can justify visually. Do not calculate replacements.\nLAST PLAN:\n"+response.output_parsed.model_dump_json()+"\nAUDIT:\n"+audit.model_dump_json()+"\nBROWSER METRICS (evidence only, never geometry instructions):\n"+json.dumps(metrics)
    raise RuntimeError("two-point vision placement failed independent visual audit")


def place_with_api(image_path:str|Path,articles_path:str|Path,client,model="gpt-5.6-sol",audit_model="gpt-5.6-sol",attempts=5)->OverlayPlan:
    image=Path(image_path);records=Path(articles_path);raw=image.read_bytes();payload,articles=_articles(records)
    numbered="\n\n".join(f"ARTICLE {a['article_id']}\nHEADING ({'INLINE; DO NOT PLACE SEPARATELY' if a.get('heading_inline') else 'SEPARATE PRINTED BLOCK'}): {a.get('heading') or a.get('label') or ''}\n"+"\n".join(f"{i}: {line}" for i,line in enumerate(a["lines"])) for a in articles)
    mime={".png":"image/png",".webp":"image/webp"}.get(image.suffix.casefold(),"image/jpeg");feedback="";passed_once=False
    # Reserve one call for the mandatory post-pass refinement even when the
    # first accepted candidate arrives on the final normal repair attempt.
    for attempt in range(max(2,attempts+1)):
        response=client.responses.parse(model=model,instructions=PLACEMENT_PROMPT,input=[{"role":"user","content":[{"type":"input_text","text":numbered+feedback},{"type":"input_image","image_url":_data(raw,mime),"detail":"original"}]}],text_format=PlacementDraft)
        blocks=_ground(response.output_parsed,articles);blocks,metrics=tighten_rendered_bottom_edges(image,blocks);rendered,_=_rendered_preview(image,blocks);region_map=_region_map(image,blocks)
        metric_by_id={metric["block_id"]:metric for metric in metrics}
        summary="\n".join(f"{b.block_id}: {b.article_id} {b.role} lines {b.start_line}-{b.end_line}; rendered_fill={metric_by_id[b.block_id]['fill_ratio']:.3f}; empty_bottom={max(0,1-metric_by_id[b.block_id]['fill_ratio']):.3f}; required_fill={_minimum_rendered_fill(b):.3f}; overflow={metric_by_id[b.block_id]['overflow']}; begins {b.text[:60]!r}; ends {b.text[-60:]!r}" for b in blocks)
        overview=client.responses.parse(model=audit_model,instructions=AUDIT_PROMPT,input=[{"role":"user","content":[{"type":"input_text","text":summary},{"type":"input_image","image_url":_data(raw,mime),"detail":"original"},{"type":"input_image","image_url":_data(rendered,"image/png"),"detail":"original"},{"type":"input_image","image_url":_data(region_map,"image/jpeg"),"detail":"original"}]}],text_format=PlacementAudit).output_parsed
        audits=[overview]
        for offset in range(0,len(blocks),12):
            batch=blocks[offset:offset+12];ids=[block.block_id for block in batch];sheet=_comparison_sheet(image,rendered,batch)
            batch_summary="AUDIT ONLY THESE IDS: "+json.dumps(ids)+"\n"+"\n".join(line for line in summary.splitlines() if any(line.startswith(block_id+":") for block_id in ids))
            audits.append(client.responses.parse(model=audit_model,instructions=REGION_AUDIT_PROMPT,input=[{"role":"user","content":[{"type":"input_text","text":batch_summary},{"type":"input_image","image_url":_data(raw,mime),"detail":"original"},{"type":"input_image","image_url":_data(rendered,"image/png"),"detail":"original"},{"type":"input_image","image_url":_data(sheet,"image/jpeg"),"detail":"original"}]}],text_format=PlacementAudit).output_parsed)
        expected={block.block_id for block in blocks};checked_ids=[block_id for audit in audits[1:] for block_id in audit.checked_block_ids]
        all_checked=len(checked_ids)==len(expected) and set(checked_ids)==expected
        rendered_complete=all(not metric_by_id[block.block_id]["overflow"] and metric_by_id[block.block_id]["fill_ratio"]>=_minimum_rendered_fill(block) for block in blocks)
        audit_ok=all(a.accepted and a.coverage_complete and a.article_identity_correct and a.line_ownership_exact and a.geometry_matches_source and a.typography_matches_source and a.regions_filled_to_source_bounds and not a.issues for a in audits)
        if audit_ok and all_checked and rendered_complete:
            if not passed_once:
                passed_once=True
                feedback="\n\nMANDATORY FINAL VISUAL REFINEMENT PASS. The previous rendered plan passed its first audit, but it may not be published yet. Re-inspect the untouched page and the complete LAST PLAN below. For every block, visually re-identify the top-left first-glyph pixel and bottom-right last-glyph pixel, refine the remaining corners for skew, and compare heading size, body size, line spacing, final-line landing, and blank tail. Preserve a block exactly when the visual evidence supports it; adjust it when it does not. Return the entire plan again.\nLAST PLAN:\n"+response.output_parsed.model_dump_json()+"\nFIRST-PASS AUDITS:\n"+json.dumps([a.model_dump() for a in audits],ensure_ascii=False)
                continue
            overview.checked_block_ids=checked_ids
            return OverlayPlan(source_sha256=hashlib.sha256(raw).hexdigest(),transcript_sha256=hashlib.sha256(records.read_bytes()).hexdigest(),inspected_by=response.output_parsed.inspected_by,blocks=blocks,coverage_complete=True,audit=overview,notes=[*response.output_parsed.notes,"Mandatory final visual refinement and full re-audit completed."])
        missing=sorted(expected-set(checked_ids));render_failures=[metric for metric in metrics if metric["overflow"] or metric["fill_ratio"]<_minimum_rendered_fill(next(block for block in blocks if block.block_id==metric["block_id"]))]
        feedback="\n\nINDEPENDENT AUDIT REJECTED THE LAST PLAN. Preserve blocks without reported defects and repair every concrete issue. Re-inspect the untouched photograph; do not calculate replacement geometry.\nLAST PLAN:\n"+response.output_parsed.model_dump_json()+"\nAUDITS:\n"+json.dumps([a.model_dump() for a in audits],ensure_ascii=False)+"\nUNCHECKED BLOCK IDS:\n"+json.dumps(missing)+"\nBROWSER RENDER FAILURES (rejection evidence only; visually choose the repair):\n"+json.dumps(render_failures)
    raise RuntimeError("overlay placement failed independent visual audit after all attempts")


def save_plan(plan:OverlayPlan,path:str|Path):Path(path).write_text(json.dumps(plan.model_dump(),ensure_ascii=False,indent=2),encoding="utf-8")


def materialize_development_plan(image_path:str|Path,articles_path:str|Path,draft_path:str|Path,audit_path:str|Path,output_path:str|Path)->OverlayPlan:
    """Ground a Codex-produced visual draft, but publish it only after a separate audit accepts it."""
    image=Path(image_path);records=Path(articles_path)
    draft=PlacementDraft.model_validate_json(Path(draft_path).read_text(encoding="utf-8"))
    audit=PlacementAudit.model_validate_json(Path(audit_path).read_text(encoding="utf-8"))
    expected={block.block_id for block in draft.blocks}
    if not audit.accepted or not audit.coverage_complete or not audit.article_identity_correct or not audit.line_ownership_exact or not audit.geometry_matches_source or not audit.typography_matches_source or not audit.regions_filled_to_source_bounds or audit.issues or len(audit.checked_block_ids)!=len(expected) or set(audit.checked_block_ids)!=expected:
        raise ValueError("development placement has not passed independent visual audit")
    _,articles=_articles(records);blocks=_ground(draft,articles);blocks,_=tighten_rendered_bottom_edges(image,blocks)
    plan=OverlayPlan(source_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),transcript_sha256=hashlib.sha256(records.read_bytes()).hexdigest(),inspected_by=draft.inspected_by,blocks=blocks,coverage_complete=True,audit=audit,notes=draft.notes)
    save_plan(plan,output_path);return plan
