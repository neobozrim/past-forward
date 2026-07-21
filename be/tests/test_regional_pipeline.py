import json
from pathlib import Path
from threading import Lock

from PIL import Image

from sofia_harness.digitization_agent import (DigitizationAgent, Inspection, LayoutInspection,
    PlannedLayout, SemanticArticle, SemanticHeading, SemanticPagePlan)
from sofia_harness.models import Article, LayoutRegion, PageLayout, Transcription


def response(value):
    return type("Response", (), {"output_parsed": value})()


class RegionalResponses:
    def __init__(self, disagree=False, fail_once=False, invalid_layout_once=False, terra_uncertain=False):
        self.disagree = disagree
        self.fail_once = fail_once
        self.failed = False
        self.lock = Lock()
        self.inspections = 0
        self.layout_prompt = ""
        self.invalid_layout_once = invalid_layout_once
        self.layout_calls = 0
        self.terra_uncertain = terra_uncertain

    @staticmethod
    def semantic_plan():
        return SemanticPagePlan(headings=[SemanticHeading(id="h1",verbatim_text="ЗАГЛАВИЕ",level="headline",
            article_id="a1",polygon=[[50,50],[950,50],[950,250],[50,250]],reading_order=0,confidence=.99)],
            articles=[SemanticArticle(id="a1",descriptive_label="test article",polygon=[[50,50],[950,50],[950,950],[50,950]],
                heading_ids=["h1"],body_column_count=1,reading_order=0,confidence=.98)])

    def page_layout(self, invalid=False):
        return PageLayout(page_type="newspaper",language_candidates=["bg"],orientation_degrees=0,
            layout_difficulty=.4,confidence=.98,regions=[
                LayoutRegion(id="heading",type="headline",polygon=[[50,50],[950,50],[950,250],[50,250]],
                    article_id="a1",reading_order=0,confidence=.99,semantic_heading_id="h1",detected_text="ЗАГЛАВИЕ"),
                LayoutRegion(id="body",type="article_body",polygon=[[50,300],[950,300],[950,950],[50,950]],
                    article_id="a1",reading_order=0 if invalid else 1,confidence=.95,column_index=1,column_count=1),
            ],articles=[Article(id="a1",region_ids=["heading","body"],reading_order=0)])

    def parse(self, model, input, text_format):
        prompt = input[0]["content"][0]["text"]
        if text_format is SemanticPagePlan:
            self.layout_prompt += prompt
            return response(self.semantic_plan())
        if text_format is LayoutInspection:
            return response(LayoutInspection(text_coverage=1,all_visible_text_covered=True,
                no_text_edges_clipped=True,headings_complete=True,subheadings_complete=True,
                article_grouping_coherent=True,semantic_plan_correct=True,column_structure_correct=True,
                reading_order_correct=True))
        if text_format is PageLayout:
            self.layout_prompt += "\n"+prompt
            self.layout_calls += 1
            return response(self.page_layout(self.invalid_layout_once and self.layout_calls==1))
        if text_format is PlannedLayout:
            self.layout_calls += 1
            return response(PlannedLayout(semantic_plan=self.semantic_plan(),page_layout=self.page_layout()))
        if text_format is Transcription:
            with self.lock:
                if self.fail_once and not self.failed:
                    self.failed = True
                    raise TimeoutError("temporary timeout")
            region = "heading" if "headline" in prompt else "body"
            reader_b = "reader B" in prompt
            text = "ЗАГЛАВИЕ" if region == "heading" else ("ТЕКСТ Б" if self.disagree and reader_b else "ТЕКСТ А")
            return response(Transcription(region_id=region, verbatim_text=text, confidence=.97))
        self.inspections += 1
        if self.terra_uncertain and model == "gpt-5.6-terra":
            return response(Inspection(final_text="[неясно: А | Б]",confidence=.70))
        if self.terra_uncertain and model == "gpt-5.6-sol":
            return response(Inspection(final_text="SOL ФИНАЛ",confidence=.97))
        return response(Inspection(final_text="ТЕКСТ ФИНАЛ", confidence=.96))


class RegionalClient:
    def __init__(self, **kwargs):
        self.responses = RegionalResponses(**kwargs)


def make_source(tmp_path):
    source = tmp_path / "page.jpg"
    Image.new("RGB", (300, 500), "white").save(source)
    return source


def test_regional_pipeline_checkpoints_and_reconstructs_in_layout_order(tmp_path):
    client = RegionalClient()
    result = DigitizationAgent(tmp_path / "digitized", client=client).process_regions(make_source(tmp_path), "complete page")
    assert result["text"] == "ЗАГЛАВИЕ\n\nТЕКСТ А"
    assert result["region_count"] == 2 and result["verified"]
    assert client.responses.inspections == 0
    checkpoint = next((tmp_path / "digitized").glob("*/audit/region-checkpoint.json"))
    assert '"status": "complete"' in checkpoint.read_text(encoding="utf-8")
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert len(saved["regions"]["heading"]["polygon_pixels"]) == 4
    assert len(saved["regions"]["heading"]["perspective_transform"]) == 3
    assert all(x >= 8 for x in saved["regions"]["heading"]["rectified_dimensions"])
    assert len(saved["calls"]) == 7  # semantic plan + synthesis + audit + two blinded reads per region
    assert all(call["status"] == "success" and call["latency_ms"] >= 0 for call in saved["calls"])
    assert [call["model"] for call in saved["calls"]].count("gpt-5.6-sol") == 3
    assert [call["model"] for call in saved["calls"]].count("gpt-5.6-luna") == 4
    assert len(list(checkpoint.parent.glob("regions/*/original.png"))) == 2
    overlay=checkpoint.parent/"layout-regions.png"
    assert overlay.is_file()
    with Image.open(overlay) as rendered:
        assert rendered.size == (300,500)
    prompt = client.responses.layout_prompt
    assert "larger, thicker, heavier" in prompt
    assert "article-level reading order" in prompt
    assert "fixed number or width" in prompt
    assert "right rail" not in prompt


def test_regional_pipeline_adjudicates_only_disagreement(tmp_path):
    client = RegionalClient(disagree=True)
    result = DigitizationAgent(tmp_path / "digitized", client=client).process_regions(make_source(tmp_path), "complete page")
    assert result["text"] == "ЗАГЛАВИЕ\n\nТЕКСТ ФИНАЛ"
    assert client.responses.inspections == 1


def test_regional_pipeline_retries_a_timed_out_region_read(tmp_path):
    client = RegionalClient(fail_once=True);events=[]
    result = DigitizationAgent(tmp_path / "digitized", client=client).process_regions(
        make_source(tmp_path), "complete page", lambda stage, detail: events.append((stage, detail)), retries=1)
    assert result["verified"]
    assert any(stage == "Retrying" and "attempt 2/2" in detail for stage, detail in events)


def test_trapezoid_is_padded_and_perspective_rectified():
    image=Image.new("RGB",(300,400),"white")
    crop,source,padded,matrix=DigitizationAgent._rectify_polygon(
        image,[[150,100],[850,150],[900,900],[100,850]],8)
    assert crop.width > 200 and crop.height > 280
    assert source.shape == padded.shape == (4,2)
    assert matrix.shape == (3,3)
    assert (source != padded).any()


def test_invalid_layout_is_repaired_before_ocr(tmp_path):
    client=RegionalClient(invalid_layout_once=True);events=[]
    result=DigitizationAgent(tmp_path/"digitized",client=client).process_regions(
        make_source(tmp_path),"complete page",lambda stage,detail:events.append((stage,detail)))
    assert result["region_count"] == 2
    assert client.responses.layout_calls == 2
    assert any(stage == "Repairing article hierarchy" and "duplicate reading_order" in detail for stage,detail in events)


def test_terra_uncertainty_escalates_only_that_region_to_sol(tmp_path):
    client=RegionalClient(disagree=True,terra_uncertain=True)
    result=DigitizationAgent(tmp_path/"digitized",client=client).process_regions(make_source(tmp_path),"complete page")
    assert result["text"] == "ЗАГЛАВИЕ\n\nSOL ФИНАЛ"
    checkpoint=next((tmp_path/"digitized").glob("*/audit/region-checkpoint.json"))
    saved=json.loads(checkpoint.read_text(encoding="utf-8"))
    body=saved["regions"]["body"]
    assert body["accepted"]["method"] == "sol_escalation"
    models=[call["model"] for call in saved["calls"]]
    assert models.count("gpt-5.6-terra") == 1
    assert models.count("gpt-5.6-sol") == 4  # semantic plan, synthesis, audit, and final escalation


def test_layout_only_stops_before_preprocessing_and_ocr(tmp_path):
    client=RegionalClient();events=[]
    result=DigitizationAgent(tmp_path/"digitized",client=client).process_regions(
        make_source(tmp_path),"analyse layout only",lambda stage,detail:events.append((stage,detail)),stop_after_layout=True)
    assert result["layout_only"] and result["markdown_path"] is None
    assert result["region_count"] == 2
    assert client.responses.layout_calls == 1 and client.responses.inspections == 0
    assert (result_path:=Path(result["layout_path"])).is_file()
    assert Path(result["overlay_path"]).is_file()
    checkpoint=json.loads((result_path.parent/"region-checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["status"] == "layout_complete"
    assert len(checkpoint["calls"]) == 3  # semantic plan, layout synthesis, and hierarchy audit
    assert not any((result_path.parent/"regions").iterdir())
    assert any(stage == "Layout complete" for stage,detail in events)
