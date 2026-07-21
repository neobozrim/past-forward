import json
from types import SimpleNamespace

import pytest
from PIL import Image

from sofia_harness.models import Transcription
from sofia_harness.pen_agent import OCRDecision,PageReview,PenAgent,RegionReview


def call(name,call_id,arguments):
    return SimpleNamespace(type="function_call",name=name,call_id=call_id,arguments=json.dumps(arguments))


class FakeResponses:
    def __init__(self):self.turn=0;self.parse_models=[]
    def create(self,**kwargs):
        self.turn+=1
        article={"article_id":"a1","label":"Article","column_count":0,"column_order":[],"notes":"test"}
        mark={"region_id":"r1","article_id":"a1","role":"heading","label":"ЗАГЛАВИЕ",
              "points":[[80,90],[920,110],[900,360],[70,340]],"local_order":0,"column_index":None,"notes":"skewed heading"}
        transcribe={"region_id":"r1"}
        outputs={
            1:[call("declare_article","c1",article),
               call("declare_article","c1b",{**article,"article_id":"false-article","label":"False article"}),
               call("remove_article","c1c",{"article_id":"false-article","reason":"visual inspection disproved it"}),
               call("mark_and_crop","c2",mark),call("transcribe_region","c3",transcribe)],
            2:[call("transcribe_region","c4",transcribe)],
            3:[call("render_overlay","c5",{})],
            4:[call("finish_page","c6",{"article_order":["a1"],"source_checked":True,"no_unmarked_text":True,
                "headings_grounded":True,"columns_verified":True,"final_notes":"done"})],
        }
        return SimpleNamespace(id=f"response-{self.turn}",output=outputs[self.turn],usage=None,output_text="")
    def parse(self,**kwargs):
        self.parse_models.append(kwargs["model"])
        if kwargs["text_format"] is Transcription:
            value=Transcription(region_id="r1",verbatim_text="ЗАГЛАВИЕ",confidence=.99)
        elif kwargs["text_format"] is OCRDecision:
            value=OCRDecision(final_text="ЗАГЛАВИЕ",confidence=.99,unresolved=False,crop_complete=True,role_matches_pixels=True)
        elif kwargs["text_format"] is RegionReview:
            value=RegionReview(crop_complete=True,role_matches_pixels=True,contains_unmarked_heading=False,
                mixed_articles=False,column_flow_complete=True,required_action="accept")
        else:
            value=PageReview(passed=True,no_unmarked_text=True,all_headings_have_heading_regions=True,
                articles_separated_correctly=True,column_structure_correct=True)
        return SimpleNamespace(id="review",output_parsed=value,usage=None)


class FakeClient:
    def __init__(self):self.responses=FakeResponses()


class FakeLayoutResponses(FakeResponses):
    def create(self,**kwargs):
        self.turn+=1
        article={"article_id":"a1","label":"Article","column_count":0,"column_order":[],"notes":"test"}
        mark={"region_id":"r1","article_id":"a1","role":"heading","label":"ЗАГЛАВИЕ",
              "points":[[80,90],[920,110],[900,360],[70,340]],"local_order":0,"column_index":None,"notes":"skewed heading"}
        outputs={
            1:[call("declare_article","c1",article),call("mark_and_crop","c2",mark)],
            2:[call("render_overlay","c3",{})],
            3:[call("finish_layout","c4",{"article_order":["a1"],"source_checked":True,"no_unmarked_text":True,
                "headings_grounded":True,"columns_verified":True,"final_notes":"layout done"})],
        }
        return SimpleNamespace(id=f"layout-{self.turn}",output=outputs[self.turn],usage=None,output_text="")


class FakeLayoutClient:
    def __init__(self):self.responses=FakeLayoutResponses()


class FailingReviewResponses(FakeResponses):
    def parse(self,**kwargs):
        if kwargs["text_format"] is Transcription:raise TimeoutError("review timed out")
        return super().parse(**kwargs)


class FailingReviewClient:
    def __init__(self):self.responses=FailingReviewResponses()


def test_pen_agent_requires_returned_crop_and_overlay_evidence(tmp_path):
    source=tmp_path/"page.png";Image.new("RGB",(300,400),"white").save(source)
    client=FakeClient();result=PenAgent(tmp_path/"digitized",client).run(source,"complete page",max_turns=6)
    assert result["workflow"]=="agentic_pen" and result["text"]=="ЗАГЛАВИЕ"
    state=json.loads((next((tmp_path/"digitized").glob("*/audit"))/"pen-agent-state.json").read_text(encoding="utf-8"))
    assert state["status"]=="complete" and state["overlay_turn"]==3
    rejected=[x for x in state["actions"] if x["tool"]=="transcribe_region" and not x["result"]["ok"]]
    assert rejected and "not been returned" in rejected[0]["result"]["error"]
    removed=[x for x in state["actions"] if x["tool"]=="remove_article" and x["result"]["ok"]]
    assert removed and "false-article" not in state["articles"]
    audit=next((tmp_path/"digitized").glob("*/audit"))
    assert (audit/"regions"/"r1"/"revision-001"/"exact.png").is_file()
    assert (audit/"regions"/"r1"/"revision-001"/"ocr-routing.json").is_file()
    assert state["routing"]=={"layout":"gpt-5.6-sol","transcription":"gpt-5.6-luna","adjudication":"gpt-5.6-terra","escalation":"gpt-5.6-sol"}
    assert client.responses.parse_models.count("gpt-5.6-luna")==2 and "gpt-5.6-terra" in client.responses.parse_models


def test_layout_only_finishes_without_transcription(tmp_path):
    source=tmp_path/"page.png";Image.new("RGB",(300,400),"white").save(source)
    result=PenAgent(tmp_path/"digitized",FakeLayoutClient()).run(source,"layout only",stop_after_layout=True,max_turns=5)
    assert result["layout_only"] is True and result["markdown_path"] is None and result["text"]==""
    state=json.loads((next((tmp_path/"digitized").glob("*/audit"))/"pen-agent-state.json").read_text(encoding="utf-8"))
    assert state["status"]=="complete_layout"
    assert not [x for x in state["actions"] if x["tool"] in {"save_transcription","transcribe_region"}]


def test_verifier_timeout_checkpoints_state_and_fresh_overlay(tmp_path):
    source=tmp_path/"page.png";Image.new("RGB",(300,400),"white").save(source)
    with pytest.raises(TimeoutError) as caught:
        PenAgent(tmp_path/"digitized",FailingReviewClient()).run(source,"complete page",max_turns=5)
    audit=next((tmp_path/"digitized").glob("*/audit"))
    state=json.loads((audit/"pen-agent-state.json").read_text(encoding="utf-8"))
    assert state["status"]=="interrupted" and "review timed out" in state["error"]
    assert state["overlay_version"]==state["version"] and (audit/"agent-overlay.png").is_file()
    assert getattr(caught.value,"run_folder","")==str(audit.parent)


def test_quadrilateral_ordering_handles_ties_and_rejects_degenerate_marks():
    image=Image.new("RGB",(400,400),"white")
    crop,matrix=PenAgent._crop(image,[[500,50],[950,500],[500,950],[50,500]])
    assert crop.width>100 and crop.height>100 and abs(float(matrix[2,2]))>0
    # A phone photo commonly puts the visual top-right slightly above the
    # top-left.  The rectified crop must remain landscape, not rotate 90 deg.
    landscape,_=PenAgent._crop(image,[[100,220],[900,200],[880,500],[120,520]])
    assert landscape.width>landscape.height*2
    with pytest.raises(ValueError,match="unique"):
        PenAgent._crop(image,[[100,100],[100,100],[900,900],[100,900]])
