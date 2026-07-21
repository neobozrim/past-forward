from pathlib import Path
from types import SimpleNamespace
import pytest
from PIL import Image
from sofia_harness.openai_adapter import OpenAIAdapter, OCRCallError


class FailingResponses:
    def __init__(self): self.calls = 0
    def parse(self, **kwargs): self.calls += 1; raise TimeoutError("simulated timeout")


def test_retries_and_attempt_telemetry(tmp_path, monkeypatch):
    image = tmp_path / "crop.png"; Image.new("RGB", (5,5)).save(image)
    responses = FailingResponses()
    monkeypatch.setattr("sofia_harness.openai_adapter.time.sleep", lambda _: None)
    adapter = OpenAIAdapter(SimpleNamespace(responses=responses))
    with pytest.raises(OCRCallError) as caught:
        adapter.transcribe(Path(image), "r", "fake", max_attempts=3)
    assert responses.calls == 3
    assert [a["status"] for a in caught.value.attempts] == ["failure"] * 3
    assert all(a["error_type"] == "TimeoutError" for a in caught.value.attempts)
