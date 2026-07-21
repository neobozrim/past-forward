import pytest
import json
from PIL import Image
from sofia_harness.models import PageLayout,Transcription
from sofia_harness.runner import _atomic_json,run


def test_manifest_atomic_write_refuses_overwrite(tmp_path):
    path = tmp_path / "manifest.json"
    _atomic_json(path, {"status":"completed"})
    with pytest.raises(FileExistsError): _atomic_json(path, {"status":"different"})
    assert not path.with_suffix(".tmp").exists()


def test_full_page_ocr_happens_before_layout_and_survives_empty_layout(tmp_path):
    image=tmp_path/"page.png";Image.new("RGB",(120,160),"white").save(image)
    config_file=tmp_path/"config.yaml";config_file.write_text("test",encoding="utf-8")
    calls=[]
    class Adapter:
        def transcribe(self,path,region_id,model):
            calls.append(("ocr",region_id));return Transcription(region_id=region_id,verbatim_text="Цял текст",confidence=.9),{"input_tokens":1,"output_tokens":1}
        def analyze_layout(self,path,model):
            calls.append(("layout",None));return PageLayout(page_type="document",language_candidates=["bg"],orientation_degrees=0,layout_difficulty=.1,confidence=.9,regions=[]),{"input_tokens":1,"output_tokens":1}
    config={"_path":str(config_file),"dataset":{"images":str(tmp_path),"patterns":["*.png"],"output":str(tmp_path/"runs")},"models":{"layout":"l","layout_escalation":"l2","ocr_easy":"o","ocr_hard":"o2"},"routing":{"layout_confidence_threshold":.8,"difficult_page_threshold":.7,"disagreement_threshold":.9,"confidence_threshold":.8},"qc":{"min_short_edge_px":10}}
    output=run(config,adapter=Adapter(),scan_paths=[image])
    manifest=json.loads((output/"manifest.json").read_text(encoding="utf-8"))
    assert calls[0]==("ocr","full_page")
    assert manifest["pages"][0]["full_page_read"]["verbatim_text"]=="Цял текст"
    assert manifest["pages"][0]["status"]=="completed"
