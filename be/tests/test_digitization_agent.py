from PIL import Image
from sofia_harness.digitization_agent import DigitizationAgent,Inspection
from sofia_harness.models import Transcription


class FakeResponses:
    def __init__(self):self.calls=[]
    def parse(self,model,input,text_format):
        self.calls.append((model,text_format))
        parsed=(Transcription(region_id="full_page",verbatim_text="А ЗА СМЪРТЬТА",confidence=.8)
                if text_format is Transcription else Inspection(final_text="АКТЪ ЗА СМЪРТЬТА",omissions_found=["КТЪ"],corrections_made=["А → АКТЪ"],confidence=.98))
        return type("Response",(),{"output_parsed":parsed})()
class FakeClient:
    def __init__(self):self.responses=FakeResponses()


def test_agent_saves_paired_source_and_inspected_markdown(tmp_path):
    source=tmp_path/"boris.jpg";Image.new("RGB",(100,100),"white").save(source)
    client=FakeClient();result=DigitizationAgent(tmp_path/"digitized",client=client).process(source,"Digitize the complete document")
    assert "АКТЪ ЗА СМЪРТЬТА" in result["text"]
    assert result["source_path"].endswith("_boris.jpg")
    assert result["markdown_path"].endswith("_boris.md")
    markdown=open(result["markdown_path"],encoding="utf-8").read()
    assert "source: \"./" in markdown and "А → АКТЪ" in markdown
    formats=[call[1] for call in client.responses.calls]
    assert formats[:3]==[Transcription,Transcription,Inspection]
    assert formats.count(Transcription)>=6 and formats[-1] is Inspection
    assert 0 <= result["coverage"] <= 1
    assert any((tmp_path/"digitized").glob("*/audit/reads.json"))
