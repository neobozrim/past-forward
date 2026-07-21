import pytest
from pydantic import ValidationError
from PIL import Image
from sofia_harness.overlay_agent import DraftBlock,OverlayBlock,OverlayPlan,PlacementAudit,PlacementDraft,_articles,_ground,_rendered_preview,tighten_rendered_bottom_edges,validate_accepted_plan,without_inline_caption_duplicates


def test_overlay_plan_accepts_normalized_skewed_quadrilaterals():
    block=OverlayBlock(block_id="a1-c1",text="Verified text",polygon=[[100,100],[410,90],[420,800],[110,820]],rotation=-1,role="body",font_size_ratio=.01,confidence=.97)
    plan=OverlayPlan(source_sha256="a",transcript_sha256="b",inspected_by="codex",blocks=[block],coverage_complete=True,audit=PlacementAudit(accepted=True,coverage_complete=True,article_identity_correct=True,line_ownership_exact=True,geometry_matches_source=True,typography_matches_source=True,regions_filled_to_source_bounds=True))
    assert plan.blocks[0].polygon[1]==[410.0,90.0]


def test_overlay_plan_rejects_unbounded_or_non_quad_geometry():
    with pytest.raises(ValidationError):OverlayBlock(block_id="bad",text="x",polygon=[[0,0],[1001,0],[0,1]],confidence=.5)


def test_audit_requires_explicit_identity_and_visual_verdicts():
    with pytest.raises(ValidationError):PlacementAudit(accepted=True,coverage_complete=True)


def test_accepted_plan_requires_vision_audit_and_authored_typography():
    audit={"accepted":True,"coverage_complete":True,"article_identity_correct":True,"line_ownership_exact":True,"geometry_matches_source":True,"typography_matches_source":True,"regions_filled_to_source_bounds":True,"issues":[]}
    with pytest.raises(ValueError,match="typography"):
        validate_accepted_plan({"coverage_complete":True,"audit":audit,"blocks":[{"block_id":"b","line_height":1.0}]})
    with pytest.raises(ValueError,match="vision audit"):
        validate_accepted_plan({"coverage_complete":False,"audit":{},"blocks":[]})


def test_v3_plan_requires_each_region_to_be_individually_checked():
    audit={"accepted":True,"coverage_complete":True,"article_identity_correct":True,"line_ownership_exact":True,"geometry_matches_source":True,"typography_matches_source":True,"regions_filled_to_source_bounds":True,"checked_block_ids":[],"issues":[]}
    with pytest.raises(ValueError,match="every region"):
        validate_accepted_plan({"schema_version":3,"coverage_complete":True,"audit":audit,"blocks":[{"block_id":"b","font_size_ratio":.01,"line_height":1}]})


def test_visual_audit_preview_is_the_actual_browser_render(tmp_path):
    source=tmp_path/"source.png";Image.new("RGB",(600,800),"white").save(source)
    block=OverlayBlock(block_id="body",article_id="a",text="First line\nSecond line",polygon=[[100,100],[500,90],[510,500],[110,510]],role="body",confidence=.99,font_size_ratio=.02,line_height=1.0)
    raw,metrics=_rendered_preview(source,[block])
    assert raw.startswith(b"\x89PNG") and len(raw)>1000
    assert metrics[0]["block_id"]=="body" and metrics[0]["overflow"] is False
    assert "Georgia" in metrics[0]["font_family"] and metrics[0]["font_weight"]=="400"


def test_render_correction_tightens_text_bottoms_without_moving_top_or_side_edges(tmp_path):
    source=tmp_path/"source.png";Image.new("RGB",(600,800),"white").save(source)
    heading=OverlayBlock(block_id="heading",article_id="a",text="HEADING",polygon=[[100,60],[500,60],[500,160],[100,160]],role="heading",confidence=.99,font_size_ratio=.02,line_height=1)
    body=OverlayBlock(block_id="body",article_id="a",text="First line\nSecond line",polygon=[[100,180],[500,180],[500,780],[100,780]],role="body",confidence=.99,font_size_ratio=.02,line_height=1)
    tightened,metrics=tighten_rendered_bottom_edges(source,[heading,body])
    assert tightened[0].polygon[:2]==heading.polygon[:2]
    assert tightened[0].polygon[2][1]<heading.polygon[2][1]
    assert tightened[1].polygon[2][1]<400 and tightened[1].polygon[3][1]<400
    body_metric=next(item for item in metrics if item["block_id"]=="body")
    assert body_metric["overflow"] is False and body_metric["fill_ratio"]>=.95


def test_render_correction_prevents_narrow_inspector_rounding_wraps(tmp_path):
    source=tmp_path/"source.png";Image.new("RGB",(1600,2133),"white").save(source)
    text="\n".join([
        "Васил Коларов, подпредседа-","тел на Министерския съвет","От мое име и от името на прави-",
        "телството на Полската република","изпращам Вам и на българското пра-","вителство най-дълбоки съчувствия",
        "по случай смъртта на вожда на бъл-","гарския народ и м-р председателя на","Народната република България —",
        "Георги Димитров.","Величествената фигура на Георги","Димитров, непримиримия борец за",
        "свободата и прогреса, изгря с всич-","кия си блясък над Европа, когато","по време на Лайпцигския процес той",
        "даде безсмъртен пример за несло-","мима борба с хитлеровия фашизъм.","След войната, като създател на на-",
        "(Продължава на 4 стр.)",
    ])
    block=OverlayBlock(block_id="responsive",article_id="a",text=text,overlay_text=text,polygon=[[802,829],[928,828],[928,931.628],[809.837,932.628]],role="body",confidence=.99,font_size_ratio=.0071,line_height=1,font_width_scale=1)
    _,before=_rendered_preview(source,[block],560,capture=False)
    assert before[0]["overflow"] is True
    corrected,metrics=tighten_rendered_bottom_edges(source,[block])
    assert corrected[0].polygon==block.polygon
    assert corrected[0].font_width_scale==pytest.approx(.985)
    assert metrics[0]["overflow"] is False and metrics[0]["fill_ratio"]>=.93
    assert metrics[0]["responsive_widths"]==[1600,1200,960,720,560]


def test_grounding_uses_exact_saved_lines_in_column_order():
    articles=[{"article_id":"a1","heading":"HEAD","lines":["first-","second","third"]}]
    draft=PlacementDraft(inspected_by="vision",blocks=[
        DraftBlock(block_id="h",article_id="a1",role="heading",polygon=[[0,0],[100,0],[100,20],[0,20]],font_size_ratio=.01,confidence=.9),
        DraftBlock(block_id="c1",article_id="a1",role="body",start_line=0,end_line=1,polygon=[[0,20],[50,20],[50,100],[0,100]],font_size_ratio=.01,confidence=.9),
        DraftBlock(block_id="c2",article_id="a1",role="body",start_line=2,end_line=2,polygon=[[50,20],[100,20],[100,100],[50,100]],font_size_ratio=.01,confidence=.9),
    ])
    blocks=_ground(draft,articles)
    assert [block.text for block in blocks]==["HEAD","first-\nsecond","third"]


def test_exact_quote_anchors_can_split_a_markdown_paragraph_at_a_physical_column_handoff():
    articles=[{"article_id":"a1","heading":"HEAD","body_text":"one two three four","lines":["one two three four"]}]
    draft=PlacementDraft(inspected_by="vision",blocks=[
        DraftBlock(block_id="h",article_id="a1",role="heading",polygon=[[0,0],[100,0],[100,20],[0,20]],font_size_ratio=.01,confidence=.9),
        DraftBlock(block_id="c1",article_id="a1",role="body",start_quote="one",end_quote="two",polygon=[[0,20],[50,20],[50,100],[0,100]],font_size_ratio=.01,confidence=.9),
        DraftBlock(block_id="c2",article_id="a1",role="body",start_quote="three",end_quote="four",polygon=[[50,20],[100,20],[100,100],[50,100]],font_size_ratio=.01,confidence=.9),
    ])
    blocks=_ground(draft,articles)
    assert [block.text for block in blocks]==["HEAD","one two","three four"]


def test_visual_line_breaks_preserve_exact_transcribed_characters():
    articles=[{"article_id":"a","heading":"LONG HEAD","body_text":"body","lines":["body"]}]
    draft=PlacementDraft(inspected_by="vision",blocks=[
        DraftBlock(block_id="h",article_id="a",role="heading",visual_text="LONG\nHEAD",polygon=[[0,0],[1,0],[1,1],[0,1]],font_size_ratio=.01,confidence=1),
        DraftBlock(block_id="b",article_id="a",role="body",start_line=0,end_line=0,polygon=[[0,1],[1,1],[1,2],[0,2]],font_size_ratio=.01,confidence=1),
    ])
    assert _ground(draft,articles)[0].overlay_text=="LONG\nHEAD"
    draft.blocks[0].visual_text="WRONG"
    with pytest.raises(ValueError,match="whitespace only"):_ground(draft,articles)


def test_article_heading_is_not_duplicated_into_body_grounding(tmp_path):
    source=tmp_path/"page.articles.json"
    source.write_text('{"articles":[{"article_id":"a","heading":"HEAD","verbatim_text":"HEAD\\n\\nbody"}]}',encoding="utf-8")
    _,articles=_articles(source)
    assert articles[0]["body_text"]=="body"
    assert articles[0]["lines"]==["body"]


def test_inline_caption_label_stays_in_caption_and_cannot_be_duplicated(tmp_path):
    source=tmp_path/"page.articles.json"
    source.write_text('{"articles":[{"article_id":"c","heading":"Caption lead","verbatim_text":"Caption lead continues here"}]}',encoding="utf-8")
    _,articles=_articles(source)
    assert articles[0]["heading_inline"] is True
    assert articles[0]["body_text"]=="Caption lead continues here"
    with pytest.raises(ValueError,match="must not be duplicated"):
        _ground(PlacementDraft(inspected_by="vision",blocks=[DraftBlock(block_id="h",article_id="c",role="heading",polygon=[[0,0],[1,0],[1,1],[0,1]],font_size_ratio=.01,confidence=1)]),articles)


def test_legacy_inline_caption_duplicate_is_removed():
    polygon=[[0,0],[1,0],[1,1],[0,1]]
    blocks=[{"block_id":"h","article_id":"c","role":"heading","text":"Lead","polygon":polygon},{"block_id":"c","article_id":"c","role":"caption","text":"Lead continues","polygon":polygon}]
    assert [block["block_id"] for block in without_inline_caption_duplicates(blocks)]==["c"]


@pytest.mark.parametrize("ranges",[[(0,0),(2,2)],[(0,1),(1,2)],[(1,2),(0,0)]])
def test_grounding_rejects_gaps_duplicates_and_reordered_columns(ranges):
    articles=[{"article_id":"a1","heading":"HEAD","lines":["one","two","three"]}]
    blocks=[DraftBlock(block_id="h",article_id="a1",role="heading",polygon=[[0,0],[1,0],[1,1],[0,1]],font_size_ratio=.01,confidence=1)]
    blocks += [DraftBlock(block_id=f"b{i}",article_id="a1",role="body",start_line=start,end_line=end,polygon=[[0,0],[1,0],[1,1],[0,1]],font_size_ratio=.01,confidence=1) for i,(start,end) in enumerate(ranges)]
    with pytest.raises(ValueError,match="coverage"):_ground(PlacementDraft(inspected_by="vision",blocks=blocks),articles)
