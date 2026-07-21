from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path

from openai import OpenAI

from .models import PageLayout, Transcription

OCR_PROMPT = """Transcribe this Bulgarian historical print crop diplomatically.
Do not modernize spelling or invent missing text. Preserve capitalization, punctuation,
abbreviations, and paragraphs. Mark uncertainty with alternatives and distinguish printed
hyphens from line-wrap hyphenation. Return only data matching the supplied schema."""

LAYOUT_PROMPT = """Analyze this complete historical newspaper page. Identify a hierarchical
layout of masthead, article, headline, subtitle, body_column, photo, caption, advertisement,
table, footer, and other meaningful regions. Group regions into articles and assign a unique
global reading_order. Polygons use normalized 0..1000 page coordinates. Do not transcribe body
text. Do not infer regions not visibly supported by the scan."""


class OpenAIAdapter:
    def __init__(self, client=None):
        self.client = client or OpenAI()

    def transcribe(self, image_path: Path, region_id: str, model: str, max_attempts: int = 3) -> tuple[Transcription, dict]:
        mime = mimetypes.guess_type(image_path)[0] or "image/png"
        data = base64.b64encode(image_path.read_bytes()).decode()
        attempts, response = [], None
        for attempt in range(1, max_attempts + 1):
            started = time.perf_counter()
            try:
                # Every invocation is a fresh stateless request; no prior read is included.
                response = self.client.responses.parse(
                    model=model,
                    input=[{"role": "user", "content": [
                        {"type": "input_text", "text": f"{OCR_PROMPT}\nregion_id={region_id}"},
                        {"type": "input_image", "image_url": f"data:{mime};base64,{data}", "detail": "high"},
                    ]}], text_format=Transcription)
                attempts.append({"attempt": attempt, "status": "success", "latency_ms": round((time.perf_counter()-started)*1000),
                                 "request_id": getattr(response, "_request_id", None)})
                break
            except Exception as exc:
                attempts.append({"attempt": attempt, "status": "failure", "latency_ms": round((time.perf_counter()-started)*1000),
                                 "error_type": type(exc).__name__, "error": str(exc)[:500]})
                if attempt == max_attempts: raise OCRCallError(attempts) from exc
                time.sleep(min(2 ** (attempt - 1), 4))
        usage = response.usage
        parsed = response.output_parsed
        offset_errors = parsed.validate_offsets()
        if offset_errors: raise InvalidTranscriptionError(offset_errors, attempts)
        meta = {"model": model, "attempts": attempts,
                "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                "cached_input_tokens": getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0)}
        return parsed, meta

    def analyze_layout(self, image_path: Path, model: str) -> tuple[PageLayout, dict]:
        mime = mimetypes.guess_type(image_path)[0] or "image/png"
        data = base64.b64encode(image_path.read_bytes()).decode(); started = time.perf_counter()
        response = self.client.responses.parse(model=model, input=[{"role":"user", "content":[
            {"type":"input_text", "text":LAYOUT_PROMPT},
            {"type":"input_image", "image_url":f"data:{mime};base64,{data}", "detail":"high"}]}],
            text_format=PageLayout)
        parsed = response.output_parsed; errors = parsed.validate_layout()
        if errors: raise InvalidTranscriptionError(errors, [{"attempt":1, "status":"invalid_layout"}])
        usage = response.usage
        return parsed, {"model":model, "stage":"layout", "latency_ms":round((time.perf_counter()-started)*1000),
            "request_id":getattr(response,"_request_id",None), "input_tokens":usage.input_tokens,
            "output_tokens":usage.output_tokens}


class OCRCallError(RuntimeError):
    def __init__(self, attempts): super().__init__("OCR failed after retries"); self.attempts = attempts


class InvalidTranscriptionError(RuntimeError):
    def __init__(self, errors, attempts): super().__init__("; ".join(errors)); self.errors, self.attempts = errors, attempts
