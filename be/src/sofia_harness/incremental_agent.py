from __future__ import annotations

import argparse
import base64
import difflib
import hashlib
import json
import mimetypes
import re
import shutil
import time
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from .metadata import metadata_from_filename
from typing import Literal

from pydantic import BaseModel, Field


class ArticleInput(BaseModel):
    article_id: str
    article_order: int = Field(ge=1)
    heading: str
    verbatim_text: str
    confidence: float = Field(ge=0, le=1)
    uncertainties: list[str] = Field(default_factory=list)
    source_anchor: str


class InventoryArticle(BaseModel):
    inventory_id: str
    article_order: int = Field(ge=1)
    heading: str
    source_anchor: str
    ownership_notes: str


class IndependentInventory(BaseModel):
    articles: list[InventoryArticle]
    unresolved_inventory_questions: list[str] = Field(default_factory=list)


class InventoryFinding(BaseModel):
    kind: Literal["missing_article", "ownership", "reading_order", "suspicious_article"]
    related_article_id: str | None = None
    expected_order: int | None = Field(default=None, ge=1)
    heading: str
    pixel_evidence: str
    required_action: str


class VerifiedArticleLocator(BaseModel):
    article_id: str
    article_order: int = Field(ge=1)
    independently_read_heading: str
    visual_anchor: str
    ownership_notes: str


class InventoryAudit(BaseModel):
    passed: bool
    all_articles_found: bool
    ownership_correct: bool
    reading_order_correct: bool
    findings: list[InventoryFinding] = Field(default_factory=list)
    article_locators: list[VerifiedArticleLocator] = Field(default_factory=list)


class TextComparison(BaseModel):
    lexically_equal: bool
    similarity: float = Field(ge=0, le=1)
    difference_count: int = Field(ge=0)
    differences: list[dict] = Field(default_factory=list)


class ChangeDecision(BaseModel):
    change_id: str
    choice: Literal["reader_a", "adjudicated"]
    confidence: float = Field(ge=0, le=1)
    pixel_evidence: str


class ChangeVerification(BaseModel):
    all_changes_checked: bool
    decisions: list[ChangeDecision]


class TranscriptFile(BaseModel):
    source_sha256: str
    transcribed_by: str
    reviewed_by: str | None = None
    articles: list[ArticleInput]


class ArticleIntegrityError(ValueError):
    """A transcript cannot be checkpointed because it contains pipeline corruption."""


_TRUNCATION_MARKER = re.compile(
    r"(?:truncated\s+output|output\s+(?:was\s+)?truncated|"
    r"\b\d[\d,._ ]*\s+(?:tokens?|characters?|chars?|bytes?)\s+truncated\b|"
    r"[\[<({]\s*(?:output\s+)?truncated\s*[\]>)}])",
    re.IGNORECASE,
)


def transcript_text_integrity_errors(text: str) -> list[str]:
    """Detect transport/UI artefacts which can never be diplomatic source text."""
    errors: list[str] = []
    if _TRUNCATION_MARKER.search(text):
        errors.append("contains an internal output-truncation marker")
    if "�" in text:
        errors.append("contains a Unicode replacement character")
    return errors


NO_EXTERNAL_OCR = """You must read the supplied pixels with your own vision. You have no permission to
call, imitate, or rely on Tesseract, pytesseract, OCRmyPDF, system OCR, external OCR engines, hidden
transcription drafts, or article-body text produced by another worker. Do not use shell or filesystem tools as an OCR
shortcut. If pixels are genuinely undecidable, preserve that uncertainty instead of inventing text."""


PROMPT = f"""You are Past Forward's Sol archival newspaper digitisation agent. Work directly from the
untouched complete page. Identify one semantic article and transcribe it directly, then save it immediately
before continuing to the next article. Do not create a page-wide coordinate plan, polygons, a generic column
grid, or a layout approval gate. Discover articles incrementally.

{NO_EXTERNAL_OCR}

Use larger/heavier headings, subheadings, signatures, rules, whitespace and text continuity to decide
which text belongs to each article. A separately signed contribution can be its own article. A deck may
span fewer physical columns than the article. Neighboring text is never context and must not be copied
into the article.

For each article, preserve every pixel-supported glyph, historical spelling, capitalization, punctuation,
paragraph break and printed line-end hyphen. Do not normalize, summarize, improve Bulgarian, or complete
plausible sentences. Use [неясно: ...] only when the pixels genuinely cannot decide. Include captions,
signatures, metadata and notices when they are printed text. Do not transcribe photographs.

Before saving each article, make a second focused pass over that article's pixels. Actively challenge every
fluent or plausible-looking word and resolve confusable glyphs from their printed shapes, not language
probability. If the second pass cannot decide, record the uncertainty; never conceal it with high confidence.
In source_anchor, identify both the visible start and the visible end of the article (final line, signature,
rule, whitespace boundary or next heading). Never reuse another article's anchor. Confirm that the last saved
line is immediately before that visible end and that no text after the end belongs to a neighboring article.

Honor the operator's requested scope exactly. Call save_articles for each single completed article; never
hold several completed articles in an unsaved answer. Continue from the saved article list after each tool
result. Before finish_page, scan the requested scope for missed headings, headless signed items, column
hand-offs and text assigned to the wrong article.
You must use tools and must not return the transcription only as prose."""


TERRA_INVENTORY_PROMPT = f"""You are an independent Terra page-inventory checker. Inspect only the
untouched source page. Identify the semantic articles and their ownership/order using headings, subheadings,
heavier type, whitespace, rules, signatures and textual continuation. Return an inventory, not a page plan:
do not draw polygons, coordinates, rectangles, or a generic column grid. A deck can span fewer columns than
its article. A photograph caption belongs to the surrounding article unless visibly independent.

Do not transcribe article bodies and do not judge exact wording. You are checking only article inventory,
article boundaries/ownership and article order. You have not seen Sol's article list or transcription.

{NO_EXTERNAL_OCR}"""


TERRA_COMPARE_PROMPT = f"""You are Terra's independent inventory/ownership auditor. Compare the blind
source inventory that Terra produced from the untouched page with the saved Sol article summaries. The
summaries deliberately contain no article body text. Flag only concrete missing articles, wrong ownership,
wrong article order, or a saved article whose identity is suspicious. Do not evaluate or rewrite primary
text. Do not demand polygons or a page-wide layout plan. Every finding must cite visible pixel evidence and
give one precise action for a fresh Sol inspection.
Allow a visually unified compilation to be saved either as one composite article or as separately headed
or signed child articles. That one-to-many granularity difference is not an error when all children remain
inside the correct umbrella item, appear once, and no neighboring text crosses ownership boundaries.
For every non-missing finding, related_article_id must be the exact ID from the saved Sol summaries.
For every saved Sol article, also return an article_locator keyed by its exact article_id. Re-read the
heading and visual location independently from the page pixels; do not merely copy Sol's summary wording.
The locator must contain no body excerpt, confidence or uncertainty hint. Use signatures, typography and
page position for headless contributions. These locators let a blind text reader find the exact saved item
even when Terra treats several separately signed children as one umbrella inventory article.

{NO_EXTERNAL_OCR}"""


TARGETED_SOL_PROMPT = f"""You are a fresh Sol archival vision reader resolving one concrete finding.
Inspect the untouched full page and act only on the flagged article. Establish that article's complete
semantic extent, column ownership and reading order, then transcribe the complete article directly from the
pixels. Incidental neighboring text may be visible but never belongs to the article. Return one complete
ArticleInput. Preserve diplomatic spelling, punctuation, paragraph breaks and printed line-end hyphens; do
not normalize or complete plausible prose. Use [неясно: ...] only when pixels genuinely cannot decide.
Recheck the targeted words and ownership against the pixel shapes a second time before returning. Record
both the visible start and visible end (final line/byline/boundary) in source_anchor.

{NO_EXTERNAL_OCR}"""


BLIND_SOL_READ_PROMPT = f"""You are Sol reader B, an independent archival vision transcriber. Inspect the
untouched full page and transcribe only the identified semantic article. You have not seen reader A's body
text, confidence, uncertainties or wording. The supplied heading and visual anchor identify the article;
they are not a transcription draft. Incidental neighboring text may be visible but never belongs to it.

Preserve every pixel-supported glyph, historical spelling, capitalization, punctuation, paragraph break
and printed line-end hyphen. Do not normalize or complete plausible prose. Use [неясно: ...] only when the
pixels genuinely cannot decide. Re-read the article a second time from the glyph shapes before returning
one complete ArticleInput. In source_anchor, state the visible beginning and ending boundary, including the
final line or byline; do not reuse an anchor belonging to a neighboring article.

{NO_EXTERNAL_OCR}"""


SOL_TEXT_ADJUDICATION_PROMPT = """You are a fresh Sol archival vision adjudicator. Two genuinely blind Sol
readers independently transcribed the same article and their diplomatic readings differ. Inspect the
untouched full-page pixels yourself. Use the supplied article identity, both complete readings and the
machine-generated difference summary only to locate disputes; decide from glyph shapes, never fluency,
grammar, majority or plausibility.

Return one complete ArticleInput. Preserve reader A byte-for-byte wherever the pixels do not justify a
change. Correct every pixel-supported omission, added passage, wrong glyph, punctuation error, printed
line-end hyphen or article-ownership leak. Do not merge neighboring text. When the source cannot decide,
use [неясно: ...] and record the uncertainty instead of silently choosing a plausible word. Recheck every
changed span once more against the pixels before returning.

You must read the supplied pixels with your own vision. You have no permission to call, imitate, or rely on
Tesseract, pytesseract, OCRmyPDF, system OCR or any external OCR engine. Do not use shell or filesystem OCR
shortcuts."""


SOL_CHANGE_VERIFICATION_PROMPT = """You are a fresh Sol source-evidence verifier. A previous adjudicator
proposed a small set of exact changes to reader A's diplomatic transcription. Inspect the untouched
full-page pixels and decide each supplied change independently. You receive only the local before/after
candidate and short locating windows—not reader B's draft or its confidence.

For every change_id, choose reader_a unless the printed glyphs, punctuation, line break, or article boundary
visibly support the adjudicated candidate. Never choose by fluency, grammar, modern spelling, or confidence
in another model. A deletion is valid only when the deleted material is visibly duplicate or belongs to a
neighboring article. Mark all_changes_checked false if any candidate cannot be located or visually decided;
do not silently guess. Return exactly one decision for every supplied change_id and no extra decisions.

You must read the supplied pixels with your own vision. You have no permission to call, imitate, or rely on
Tesseract, pytesseract, OCRmyPDF, system OCR or any external OCR engine. Do not use shell or filesystem OCR
shortcuts."""

TOOLS = [
    {"type": "function", "name": "save_articles", "description": "Checkpoint exactly one newly completed article immediately.",
     "parameters": {"type": "object", "properties": {"articles": {"type": "array", "minItems": 1, "maxItems": 1,
         "items": {"type": "object", "properties": {
             "article_id": {"type": "string"}, "article_order": {"type": "integer", "minimum": 1},
             "heading": {"type": "string"}, "verbatim_text": {"type": "string"},
             "confidence": {"type": "number", "minimum": 0, "maximum": 1},
             "uncertainties": {"type": "array", "items": {"type": "string"}},
             "source_anchor": {"type": "string"}},
             "required": ["article_id", "article_order", "heading", "verbatim_text", "confidence", "uncertainties", "source_anchor"],
             "additionalProperties": False}}}, "required": ["articles"], "additionalProperties": False}, "strict": True},
    {"type": "function", "name": "revise_article", "description": "Replace one saved article after pixel review found an omission, hallucination, ownership error or wrong order.",
     "parameters": {"type": "object", "properties": {
         "article": {"type": "object", "properties": {
             "article_id": {"type": "string"}, "article_order": {"type": "integer", "minimum": 1},
             "heading": {"type": "string"}, "verbatim_text": {"type": "string"},
             "confidence": {"type": "number", "minimum": 0, "maximum": 1},
             "uncertainties": {"type": "array", "items": {"type": "string"}},
             "source_anchor": {"type": "string"}},
             "required": ["article_id", "article_order", "heading", "verbatim_text", "confidence", "uncertainties", "source_anchor"],
             "additionalProperties": False}, "reason": {"type": "string"}},
         "required": ["article", "reason"], "additionalProperties": False}, "strict": True},
    {"type": "function", "name": "finish_page", "description": "Request independent inventory inspection after every article in the operator's scope is saved.",
     "parameters": {"type": "object", "properties": {"final_scan_completed": {"type": "boolean"}, "notes": {"type": "string"}},
                    "required": ["final_scan_completed", "notes"], "additionalProperties": False}, "strict": True},
]


class IncrementalRun:
    def __init__(self, output_root: str | Path, source: str | Path, instruction: str,
                 resume_folder: str | Path | None = None):
        root = Path(output_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        if resume_folder is not None:
            folder = Path(resume_folder).resolve()
            if folder != root and root not in folder.parents:
                raise ValueError("resume folder is outside the output archive")
            self.folder = folder
            self.audit = folder / "audit"
            self.state_path = self.audit / "incremental-state.json"
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.source = Path(self.state["source"]).resolve()
            actual_source_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()
            if actual_source_hash.casefold() != self.state["source_sha256"].casefold():
                raise ValueError("archived source hash changed; refusing to reuse prior article verification")
            self.articles_path = next(folder.glob("*.articles.json"))
            self.markdown_path = next(folder.glob("*.md"))
            self.stamp = self.source.name.split("_", 1)[0]
            self.state.setdefault("text_verification", {})
            self.state["status"] = "transcribing"
            self.state["instruction"] = instruction
            self.state.pop("error", None)
            self.state["events"].append({
                "event": "resumed",
                "at": datetime.now().astimezone().isoformat(timespec="seconds"),
            })
            self.persist()
            return
        source = Path(source).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        folder = root / f"{stamp}_{source.stem}_incremental"
        if folder.exists():
            folder = root / f"{folder.name}_{uuid.uuid4().hex[:6]}"
        self.folder = folder
        self.audit = folder / "audit"
        self.audit.mkdir(parents=True)
        self.source = folder / f"{stamp}_{source.name}"
        shutil.copy2(source, self.source)
        self.stamp = stamp
        self.state_path = self.audit / "incremental-state.json"
        self.articles_path = folder / f"{stamp}_{source.stem}.articles.json"
        self.markdown_path = folder / f"{stamp}_{source.stem}.md"
        self.state = {
            "schema_version": 1,
            "workflow": "incremental-full-page",
            "status": "transcribing",
            "instruction": instruction,
            "source": str(self.source),
            "source_sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(),
            "metadata": metadata_from_filename(source.name),
            "articles": {},
            "events": [],
            "model_calls": [],
            "audits": [],
            "text_verification": {},
        }
        self.persist()

    @staticmethod
    def _atomic(path: Path, text: str):
        temporary = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
        temporary.write_text(text, encoding="utf-8")
        # Windows indexers/antivirus can briefly hold the destination between writes.
        # Keep atomic replacement, but tolerate that short external lock.
        for attempt in range(5):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(.02 * (2 ** attempt))

    @staticmethod
    def article_fingerprint(article: dict) -> str:
        value = {key: article[key] for key in (
            "article_id", "article_order", "heading", "verbatim_text", "uncertainties", "source_anchor"
        )}
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    @staticmethod
    def _anchor_key(value: str) -> str:
        return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip().casefold()

    @staticmethod
    def _ownership_tokens(value: str) -> list[str]:
        return re.findall(r"\w+", unicodedata.normalize("NFC", value).casefold(), flags=re.UNICODE)

    @classmethod
    def _substantial_boundary_overlap(cls, first: str, second: str) -> dict | None:
        """Find a long verbatim block leaked across an article boundary.

        Common short phrases are deliberately ignored.  A finding requires at least 60
        consecutive words, at least 35% of the shorter article, and proximity to an end
        of either article—the characteristic shape of a wrong column hand-off.
        """
        left, right = cls._ownership_tokens(first), cls._ownership_tokens(second)
        if min(len(left), len(right)) < 60:
            return None
        block = difflib.SequenceMatcher(a=left, b=right, autojunk=False).find_longest_match()
        shorter_fraction = block.size / min(len(left), len(right))
        touches_boundary = (
            block.a <= 8 or len(left) - (block.a + block.size) <= 8
            or block.b <= 8 or len(right) - (block.b + block.size) <= 8
        )
        if block.size < 60 or shorter_fraction < .35 or not touches_boundary:
            return None
        return {"words": block.size, "shorter_fraction": round(shorter_fraction, 3)}

    def article_integrity_errors(self, article: ArticleInput,
                                 ignore_article_id: str | None = None) -> list[str]:
        errors = transcript_text_integrity_errors(article.heading)
        errors.extend(transcript_text_integrity_errors(article.verbatim_text))
        errors.extend(transcript_text_integrity_errors(article.source_anchor))
        anchor = self._anchor_key(article.source_anchor)
        for existing_id, existing in self.state["articles"].items():
            if existing_id == ignore_article_id:
                continue
            if anchor and anchor == self._anchor_key(existing["source_anchor"]):
                errors.append(f"source anchor duplicates article {existing_id}")
            overlap = self._substantial_boundary_overlap(
                article.verbatim_text, existing["verbatim_text"])
            if overlap:
                errors.append(
                    f"substantial boundary text overlap with article {existing_id} "
                    f"({overlap['words']} words; {overlap['shorter_fraction']:.0%} of shorter article)"
                )
        return errors

    def page_integrity_errors(self) -> list[str]:
        errors: list[str] = []
        for article_id, raw in self.state["articles"].items():
            article = ArticleInput.model_validate(raw)
            for error in self.article_integrity_errors(article, ignore_article_id=article_id):
                # Each pair is otherwise reported twice. Keep a stable, useful diagnostic.
                if "article " in error:
                    related = error.rsplit("article ", 1)[-1].split(" ", 1)[0]
                    if related < article_id:
                        continue
                errors.append(f"{article_id}: {error}")
        return errors

    @staticmethod
    def _markdown_body(article: dict):
        """Hide a duplicated leading heading in Markdown without altering diplomatic JSON."""
        text = article["verbatim_text"].strip()
        first, separator, remainder = text.partition("\n\n")
        normalize = lambda value: re.sub(r"\s+", " ", value).strip().casefold()
        if separator and normalize(first) == normalize(article["heading"]):
            return remainder.strip()
        if not separator and normalize(first) == normalize(article["heading"]):
            return ""
        return text

    def persist(self):
        rows = sorted(self.state["articles"].values(), key=lambda value: (value["article_order"], value["article_id"]))
        self._atomic(self.state_path, json.dumps(self.state, ensure_ascii=False, indent=2))
        self._atomic(self.articles_path, json.dumps({
            "schema_version": 1, "workflow": self.state["workflow"], "source": self.source.name,
            "source_sha256": self.state["source_sha256"], "metadata": self.state.get("metadata", {}),
            "articles": rows,
        }, ensure_ascii=False, indent=2))
        sections = []
        for article in rows:
            note = ""
            if article["uncertainties"]:
                note = "\n\n> Неясно в източника: " + "; ".join(article["uncertainties"])
            body = self._markdown_body(article)
            sections.append(f"## {article['heading']}" + (f"\n\n{body}" if body else "") + note)
        self._atomic(self.markdown_path,
            f"---\nsource: {self.source.name}\nworkflow: incremental-full-page\narticles: {len(rows)}\nstatus: {self.state['status']}\n---\n\n" + "\n\n".join(sections) + "\n")

    def save_read_artifact(self, article_id: str, role: str, payload: dict) -> Path:
        """Atomically retain every independent read and adjudication beside the page."""
        safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", article_id).strip("._") or "article"
        identity = hashlib.sha256(article_id.encode("utf-8")).hexdigest()[:10]
        reads = self.audit / "reads"
        reads.mkdir(exist_ok=True)
        path = reads / f"{safe_id[:80]}-{identity}.{role}.{uuid.uuid4().hex[:10]}.json"
        self._atomic(path, json.dumps(payload, ensure_ascii=False, indent=2))
        return path

    def save(self, article: ArticleInput, revision_reason: str | None = None):
        if not article.verbatim_text.strip():
            raise ValueError(f"{article.article_id}: verbatim_text is empty")
        integrity_errors = self.article_integrity_errors(
            article, ignore_article_id=article.article_id if revision_reason else None)
        if integrity_errors:
            raise ArticleIntegrityError(f"{article.article_id}: " + "; ".join(integrity_errors))
        for existing_id, existing in self.state["articles"].items():
            if existing_id != article.article_id and existing["article_order"] == article.article_order:
                raise ValueError(f"article_order {article.article_order} already belongs to {existing_id}")
        value = article.model_dump()
        value["revision_reason"] = revision_reason
        value["saved_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        self.state["articles"][article.article_id] = value
        self.state["events"].append({"event": "revised" if revision_reason else "saved", "article_id": article.article_id,
                                     "at": value["saved_at"], "reason": revision_reason})
        self.persist()

    def insert(self, article: ArticleInput, revision_reason: str):
        """Insert a newly recovered article without losing the established page order."""
        if article.article_id in self.state["articles"]:
            self.save(article, revision_reason)
            return
        integrity_errors = self.article_integrity_errors(article)
        if integrity_errors:
            raise ArticleIntegrityError(f"{article.article_id}: " + "; ".join(integrity_errors))
        for existing in self.state["articles"].values():
            if existing["article_order"] >= article.article_order:
                existing["article_order"] += 1
        self.save(article, revision_reason)

    def revise(self, article: ArticleInput, revision_reason: str):
        """Replace an article and move it without creating duplicate order numbers."""
        existing = self.state["articles"].get(article.article_id)
        if existing is None:
            raise ValueError(f"unknown article {article.article_id}")
        integrity_errors = self.article_integrity_errors(article, ignore_article_id=article.article_id)
        if integrity_errors:
            raise ArticleIntegrityError(f"{article.article_id}: " + "; ".join(integrity_errors))
        old_order = existing["article_order"]
        new_order = article.article_order
        if new_order < old_order:
            for other_id, other in self.state["articles"].items():
                if other_id != article.article_id and new_order <= other["article_order"] < old_order:
                    other["article_order"] += 1
        elif new_order > old_order:
            for other_id, other in self.state["articles"].items():
                if other_id != article.article_id and old_order < other["article_order"] <= new_order:
                    other["article_order"] -= 1
        self.save(article, revision_reason)

    def completed_summary(self):
        """Return navigation anchors only; never seed a fresh reader with prior OCR text."""
        rows = sorted(self.state["articles"].values(), key=lambda value: (value["article_order"], value["article_id"]))
        return [{key: row[key] for key in ("article_id", "article_order", "heading", "source_anchor")} for row in rows]

    def result(self):
        rows = list(self.state["articles"].values())
        verification = self.state.get("text_verification", {})
        verified_text = [article_id for article_id, value in verification.items()
                         if article_id in self.state["articles"]
                         and value.get("status") in {"matched", "adjudicated"}
                         and value.get("final_sha256") == self.article_fingerprint(self.state["articles"][article_id])]
        pending_text = [article_id for article_id in self.state["articles"] if article_id not in verified_text]
        return {
            "source_name": self.source.name, "source_path": str(self.source), "folder": str(self.folder),
            "markdown_path": str(self.markdown_path), "articles_path": str(self.articles_path),
            "layout_path": str(self.state_path), "overlay_path": None, "semantic_path": None, "semantic_overlay_path": None,
            "article_count": len(rows), "region_count": 0, "confidence": min((x["confidence"] for x in rows), default=0),
            "uncertain_regions": [x["article_id"] for x in rows if x["uncertainties"]],
            "omissions_found": [finding.get("heading", "") for audit in self.state["audits"]
                                for finding in audit.get("findings", []) if finding.get("kind") == "missing_article"],
            "corrections_made": [x["article_id"] for x in rows if x.get("revision_reason")],
            "text_verified_articles": verified_text, "text_verification_pending": pending_text,
            "verified": self.state["status"] == "complete", "failed_regions": {}, "workflow": "incremental_full_page",
            "status": self.state["status"], "partial": self.state["status"] not in {"complete", "complete_with_review_findings"},
        }


class IncrementalArticleAgent:
    """Lightweight Sol article loop with a blind Terra inventory and fresh Sol repairs."""

    def __init__(self, output_root: str | Path, client, model: str = "gpt-5.6-sol",
                 inventory_model: str = "gpt-5.6-terra", inspector_model: str = "gpt-5.6-sol"):
        self.output_root = Path(output_root)
        self.client = client
        self.model = model
        self.inventory_model = inventory_model
        self.inspector_model = inspector_model

    @staticmethod
    def _image(path: Path):
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"

    @staticmethod
    def _record_call(run: IncrementalRun, kind: str, model: str, started: float,
                     response=None, error=None, **extra):
        record = {
            "kind": kind,
            "model": model,
            "request_id": getattr(response, "id", None),
            "latency_ms": round((time.perf_counter() - started) * 1000),
            **extra,
        }
        if error is not None:
            record["error"] = f"{type(error).__name__}: {error}"
        usage = getattr(response, "usage", None)
        if usage is not None:
            record["usage"] = {
                key: getattr(usage, key, None)
                for key in ("input_tokens", "output_tokens", "total_tokens")
                if getattr(usage, key, None) is not None
            }
        run.state["model_calls"].append(record)

    @staticmethod
    def _require_complete_response(response, stage: str):
        """Reject token-limited Responses before any partial text can be checkpointed."""
        status = getattr(response, "status", None)
        incomplete = getattr(response, "incomplete_details", None)
        if status == "incomplete" or incomplete:
            reason = getattr(incomplete, "reason", None)
            if reason is None and isinstance(incomplete, dict):
                reason = incomplete.get("reason")
            raise ArticleIntegrityError(
                f"{stage} returned an incomplete response" + (f": {reason}" if reason else "")
            )

    @staticmethod
    def _require_article_text_integrity(article: ArticleInput, stage: str):
        errors: list[str] = []
        for field in (article.heading, article.verbatim_text, article.source_anchor):
            errors.extend(transcript_text_integrity_errors(field))
        if errors:
            raise ArticleIntegrityError(f"{stage}: " + "; ".join(errors))

    def _fresh_sol_input(self, run: IncrementalRun, instruction: str, reason: str):
        summary = json.dumps(run.completed_summary(), ensure_ascii=False, indent=2)
        prompt = (
            f"OPERATOR REQUEST: {instruction}\n{reason}\n"
            "The following articles are safely checkpointed. Use only their identity and visible source "
            "anchors to avoid duplication; body text is deliberately withheld. Read the untouched page "
            "pixels and save the next single completed article.\n\n"
            f"COMPLETED ARTICLE ANCHORS:\n{summary}"
        )
        return [{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": self._image(run.source), "detail": "original"},
        ]}]

    @staticmethod
    def _article_hash(article: dict) -> str:
        return IncrementalRun.article_fingerprint(article)

    @staticmethod
    def _comparison(reader_a: str, reader_b: str) -> TextComparison:
        """Ignore horizontal spacing noise while preserving diplomatic line and paragraph boundaries."""
        def normalize(value: str) -> str:
            value = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
            lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.split("\n")]
            while lines and not lines[0]:
                lines.pop(0)
            while lines and not lines[-1]:
                lines.pop()
            return "\n".join(lines)
        text_a, text_b = normalize(reader_a), normalize(reader_b)
        tokens_a, tokens_b = text_a.split(), text_b.split()
        matcher = difflib.SequenceMatcher(a=tokens_a, b=tokens_b, autojunk=False)
        differences = []
        count = 0
        for tag, a1, a2, b1, b2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            count += 1
            if len(differences) >= 80:
                continue
            differences.append({
                "kind": tag,
                "reader_a": " ".join(tokens_a[a1:a2]),
                "reader_b": " ".join(tokens_b[b1:b2]),
                "reader_a_window": " ".join(tokens_a[max(0, a1 - 5):min(len(tokens_a), a2 + 5)]),
                "reader_b_window": " ".join(tokens_b[max(0, b1 - 5):min(len(tokens_b), b2 + 5)]),
            })
        if text_a != text_b and count == 0:
            count = 1
            differences.append({
                "kind": "line_or_paragraph_break",
                "reader_a": text_a,
                "reader_b": text_b,
            })
        return TextComparison(
            lexically_equal=text_a == text_b,
            similarity=difflib.SequenceMatcher(a=text_a, b=text_b, autojunk=False).ratio(),
            difference_count=count,
            differences=differences,
        )

    @staticmethod
    def _truncation_signals(reader_a: str, reader_b: str) -> list[str]:
        """Recognize likely cut-off text without adding another model call.

        The independent B read already exists.  We only classify high-signal cases:
        an abrupt terminal glyph/line-end word split, or a substantial suffix which B
        can see but A lacks.  The normal adjudication path then inspects the pixels.
        """
        signals: list[str] = []

        def abrupt_ending(value: str) -> bool:
            tail = value.rstrip()
            return bool(tail and (
                re.search(r"\w-$", tail, flags=re.UNICODE)
                or tail[-1] in {",", ";", ":", "—", "–", "(", "[", "«"}
            ))

        if abrupt_ending(reader_a):
            signals.append("reader A ends at a possible mid-word or mid-sentence boundary")
        if abrupt_ending(reader_b):
            signals.append("reader B ends at a possible mid-word or mid-sentence boundary")

        tokens_a = IncrementalRun._ownership_tokens(reader_a)
        tokens_b = IncrementalRun._ownership_tokens(reader_b)
        matcher = difflib.SequenceMatcher(a=tokens_a, b=tokens_b, autojunk=False)
        for tag, a1, a2, b1, b2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            touches_end = a2 >= len(tokens_a) - 3 and b2 >= len(tokens_b) - 3
            if not touches_end:
                continue
            missing_from_a = (b2 - b1) - (a2 - a1)
            if missing_from_a >= 8:
                signals.append(
                    f"independent reader B found {missing_from_a} additional ending words"
                )
            elif missing_from_a <= -8:
                signals.append(
                    f"reader A has {-missing_from_a} ending words absent from independent reader B"
                )
        return list(dict.fromkeys(signals))

    def _blind_sol_request(self, image_url: str, article: dict, terra_locator: dict):
        identity = {
            "opaque_article_id": article["article_id"],
            "article_order": article["article_order"],
            "independent_terra_locator": terra_locator,
        }
        response = self.client.responses.parse(
            model=self.inspector_model,
            instructions=BLIND_SOL_READ_PROMPT,
            input=[{"role": "user", "content": [
                {"type": "input_text", "text":
                    "Transcribe this article independently from the page pixels. The locator was produced "
                    "independently by Terra and contains no reader-A heading, body text, confidence, source "
                    "anchor or uncertainty hints.\n\nARTICLE IDENTITY:\n" +
                    json.dumps(identity, ensure_ascii=False, indent=2)},
                {"type": "input_image", "image_url": image_url, "detail": "original"},
            ]}],
            text_format=ArticleInput,
        )
        self._require_complete_response(response, "blind Sol read")
        self._require_article_text_integrity(response.output_parsed, "blind Sol read")
        return response

    def _adjudication_request(self, image_url: str, reader_a: dict, reader_b: ArticleInput,
                              comparison: TextComparison):
        payload = {
            "article_identity": {key: reader_a[key] for key in (
                "article_id", "article_order", "heading", "source_anchor"
            )},
            "reader_a": {"heading": reader_a["heading"], "verbatim_text": reader_a["verbatim_text"],
                         "uncertainties": reader_a["uncertainties"]},
            "reader_b": {"heading": reader_b.heading, "verbatim_text": reader_b.verbatim_text,
                         "uncertainties": reader_b.uncertainties},
            "comparison": comparison.model_dump(),
        }
        response = self.client.responses.parse(
            model=self.inspector_model,
            instructions=SOL_TEXT_ADJUDICATION_PROMPT,
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": json.dumps(payload, ensure_ascii=False, indent=2)},
                {"type": "input_image", "image_url": image_url, "detail": "original"},
            ]}],
            text_format=ArticleInput,
        )
        self._require_complete_response(response, "Sol text adjudication")
        self._require_article_text_integrity(response.output_parsed, "Sol text adjudication")
        return response

    @staticmethod
    def _proposed_changes(reader_a: dict, adjudicated: ArticleInput) -> list[dict]:
        """Produce exact, replayable character edits with enough local text to find them in the scan."""
        changes: list[dict] = []
        for field in ("heading", "verbatim_text"):
            before = reader_a[field]
            after = getattr(adjudicated, field)
            matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
            index = 0
            for tag, a1, a2, b1, b2 in matcher.get_opcodes():
                if tag == "equal":
                    continue
                change_id = f"{field}-{index}"
                index += 1
                changes.append({
                    "change_id": change_id,
                    "field": field,
                    "kind": tag,
                    "reader_a": before[a1:a2],
                    "adjudicated": after[b1:b2],
                    "reader_a_window": before[max(0, a1 - 80):min(len(before), a2 + 80)],
                    "adjudicated_window": after[max(0, b1 - 80):min(len(after), b2 + 80)],
                    "_a": [a1, a2], "_b": [b1, b2],
                })
        if reader_a["uncertainties"] != adjudicated.uncertainties:
            changes.append({
                "change_id": "uncertainties-0", "field": "uncertainties", "kind": "replace",
                "reader_a": reader_a["uncertainties"], "adjudicated": adjudicated.uncertainties,
                "reader_a_window": reader_a["uncertainties"],
                "adjudicated_window": adjudicated.uncertainties,
            })
        return changes

    def _change_verification_request(self, image_url: str, article: dict,
                                     adjudicated: ArticleInput, changes: list[dict]):
        public_changes = [{key: value for key, value in change.items() if not key.startswith("_")}
                          for change in changes]
        payload = {
            "article_identity": {
                "article_id": article["article_id"], "article_order": article["article_order"],
                "heading": article["heading"], "source_anchor": article["source_anchor"],
            },
            "proposed_changes": public_changes,
        }
        response = self.client.responses.parse(
            model=self.inspector_model,
            instructions=SOL_CHANGE_VERIFICATION_PROMPT,
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": json.dumps(payload, ensure_ascii=False, indent=2)},
                {"type": "input_image", "image_url": image_url, "detail": "original"},
            ]}],
            text_format=ChangeVerification,
        )
        self._require_complete_response(response, "Sol change verification")
        return response

    @staticmethod
    def _apply_verified_changes(reader_a: dict, adjudicated: ArticleInput, changes: list[dict],
                                verification: ChangeVerification) -> ArticleInput:
        decisions = {decision.change_id: decision for decision in verification.decisions}
        values = {"heading": reader_a["heading"], "verbatim_text": reader_a["verbatim_text"]}
        for field in ("heading", "verbatim_text"):
            before = reader_a[field]
            after = getattr(adjudicated, field)
            field_changes = [change for change in changes if change["field"] == field]
            if not field_changes:
                continue
            by_span = {(tuple(change["_a"]), tuple(change["_b"])): change
                       for change in field_changes}
            pieces: list[str] = []
            for tag, a1, a2, b1, b2 in difflib.SequenceMatcher(
                    a=before, b=after, autojunk=False).get_opcodes():
                if tag == "equal":
                    pieces.append(before[a1:a2])
                    continue
                change = by_span[((a1, a2), (b1, b2))]
                decision = decisions[change["change_id"]]
                pieces.append(after[b1:b2] if decision.choice == "adjudicated" else before[a1:a2])
            values[field] = "".join(pieces)
        uncertainty_change = next((change for change in changes
                                   if change["field"] == "uncertainties"), None)
        uncertainties = reader_a["uncertainties"]
        if uncertainty_change and decisions[uncertainty_change["change_id"]].choice == "adjudicated":
            uncertainties = adjudicated.uncertainties
        confidence = min(
            reader_a["confidence"], adjudicated.confidence,
            *(decision.confidence for decision in verification.decisions),
        )
        return ArticleInput(
            article_id=reader_a["article_id"], article_order=reader_a["article_order"],
            heading=values["heading"], verbatim_text=values["verbatim_text"],
            confidence=confidence, uncertainties=uncertainties,
            source_anchor=reader_a["source_anchor"],
        )

    def _finalize_adjudication(self, run: IncrementalRun, emit, image_url: str, article: dict,
                               adjudicated: ArticleInput, record: dict,
                               min_confidence: float) -> bool:
        verified_path = (Path(record["change_verification_path"])
                         if record.get("status") == "change_verification_ready"
                         and record.get("change_verification_path") else None)
        if verified_path and verified_path.is_file():
            try:
                artifact = json.loads(verified_path.read_text(encoding="utf-8"))
                recovered = ArticleInput.model_validate(artifact["article"]).model_copy(update={
                    "article_id": article["article_id"], "article_order": article["article_order"],
                    "source_anchor": article["source_anchor"],
                })
                if self._article_hash(article) == record.get("primary_sha256"):
                    run.revise(recovered, "recovered source-verified Sol adjudication after interruption")
                    final_row = run.state["articles"][article["article_id"]]
                    record.update(status="adjudicated", final_sha256=self._article_hash(final_row))
                    if record.get("truncation_review"):
                        record["truncation_review"].update(
                            status="resolved_by_source_adjudication",
                            final_sha256=self._article_hash(final_row),
                        )
                    record.pop("error", None)
                    run.persist()
                    emit("Text disagreement resolved", article["heading"])
                    return True
            except (OSError, KeyError, json.JSONDecodeError, ValueError):
                pass
        if adjudicated.confidence < min_confidence:
            record.update(status="adjudication_low_confidence",
                          error=f"final confidence {adjudicated.confidence:.3f} is below {min_confidence:.3f}")
            run.persist()
            emit("Text adjudication uncertain", article["heading"])
            return False

        changes = self._proposed_changes(article, adjudicated)
        final = adjudicated
        if changes:
            emit("Verifying proposed text edits", article["heading"])
            started = time.perf_counter()
            try:
                response = self._change_verification_request(image_url, article, adjudicated, changes)
                checked = response.output_parsed
                expected_ids = {change["change_id"] for change in changes}
                returned_ids = [decision.change_id for decision in checked.decisions]
                if (not checked.all_changes_checked or len(returned_ids) != len(set(returned_ids))
                        or set(returned_ids) != expected_ids):
                    raise ValueError("source verifier did not decide every proposed change exactly once")
                if any(decision.confidence < min_confidence for decision in checked.decisions):
                    raise ValueError("source verifier confidence is below the configured floor")
                final = self._apply_verified_changes(article, adjudicated, changes, checked)
            except Exception as exc:
                self._record_call(run, "sol_change_verification", self.inspector_model, started,
                                  error=exc, article_id=article["article_id"])
                record.update(status="change_verification_failed", error=f"{type(exc).__name__}: {exc}")
                run.persist()
                emit("Proposed edit verification failed", article["heading"])
                return False
            self._record_call(run, "sol_change_verification", self.inspector_model, started,
                              response=response, article_id=article["article_id"])
            path = run.save_read_artifact(article["article_id"], "sol-change-verification", {
                "role": "fresh_sol_source_change_verifier", "model": self.inspector_model,
                "request_id": getattr(response, "id", None),
                "proposed_changes": [
                    {key: value for key, value in change.items() if not key.startswith("_")}
                    for change in changes
                ],
                "verification": checked.model_dump(), "article": final.model_dump(),
            })
            record.update(status="change_verification_ready", change_verification_path=str(path))
            run.persist()

        changed = any(final.model_dump()[key] != article[key]
                      for key in ("heading", "verbatim_text", "confidence", "uncertainties"))
        if changed:
            run.revise(final, "fresh Sol A/B adjudication with source-only change verification")
        final_row = run.state["articles"][article["article_id"]]
        record.update(status="adjudicated", final_sha256=self._article_hash(final_row))
        if record.get("truncation_review"):
            record["truncation_review"].update(
                status="resolved_by_source_adjudication",
                final_sha256=self._article_hash(final_row),
            )
        record.pop("error", None)
        run.persist()
        emit("Text disagreement resolved", article["heading"])
        return True

    def _verify_article_texts(self, run: IncrementalRun, emit, audit: InventoryAudit,
                              max_workers: int = 3, min_confidence: float = .80) -> bool:
        """Blind-read every final article and adjudicate only genuine A/B lexical disagreements."""
        articles = sorted(run.state["articles"].values(),
                          key=lambda value: (value["article_order"], value["article_id"]))
        if not articles:
            run.state["status"] = "text_review_required"
            run.state["text_verification_error"] = "no articles were available for verification"
            run.persist()
            return False
        verification = run.state.setdefault("text_verification", {})
        terra_locators = {locator.article_id: locator.model_dump() for locator in audit.article_locators}
        missing_locators = [article["article_id"] for article in articles
                            if article["article_id"] not in terra_locators]
        if missing_locators:
            run.state["status"] = "text_review_required"
            run.state["text_verification_error"] = (
                f"Terra did not independently locate {len(missing_locators)} saved article(s)"
            )
            run.persist()
            emit("Saved — independent locators missing", f"{len(missing_locators)} article(s)")
            return False
        image_url = self._image(run.source)
        blind_results = []
        requests = []

        for article in articles:
            article_hash = self._article_hash(article)
            saved = verification.get(article["article_id"], {})
            if saved.get("status") in {"matched", "adjudicated"} and saved.get("final_sha256") == article_hash:
                continue
            adjudication_path = (Path(saved["adjudication_path"])
                                 if saved.get("status") in {
                                     "adjudication_ready", "adjudication_low_confidence",
                                     "change_verification_failed", "change_verification_ready",
                                 }
                                 and saved.get("adjudication_path") else None)
            if adjudication_path and adjudication_path.is_file():
                try:
                    artifact = json.loads(adjudication_path.read_text(encoding="utf-8"))
                    proposed = ArticleInput.model_validate(artifact["article"]).model_copy(update={
                        "article_id": article["article_id"], "article_order": article["article_order"],
                        "source_anchor": article["source_anchor"],
                    })
                    proposed_hash = self._article_hash(proposed.model_dump())
                    if article_hash == saved.get("primary_sha256"):
                        self._finalize_adjudication(
                            run, emit, image_url, article, proposed, saved, min_confidence)
                        continue
                    if article_hash == proposed_hash:
                        saved.update(status="adjudicated", final_sha256=article_hash)
                        if saved.get("truncation_review"):
                            saved["truncation_review"].update(
                                status="resolved_by_source_adjudication",
                                final_sha256=article_hash,
                            )
                        run.persist()
                        continue
                except (OSError, KeyError, json.JSONDecodeError, ValueError):
                    pass
            blind_path = Path(saved.get("blind_read_path", "")) if saved.get("blind_read_path") else None
            if saved.get("primary_sha256") == article_hash and blind_path and blind_path.is_file():
                try:
                    artifact = json.loads(blind_path.read_text(encoding="utf-8"))
                    blind_results.append((article, ArticleInput.model_validate(artifact["article"])))
                    continue
                except (OSError, KeyError, json.JSONDecodeError, ValueError):
                    pass
            requests.append(article)

        if requests:
            emit("Independent Sol verification", f"Blind reads for {len(requests)} article(s)")
            with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(requests)))) as pool:
                futures = {}
                for article in requests:
                    started = time.perf_counter()
                    terra_locator = terra_locators[article["article_id"]]
                    future = pool.submit(self._blind_sol_request, image_url, article, terra_locator)
                    futures[future] = (article, started)
                completed = 0
                for future in as_completed(futures):
                    article, started = futures[future]
                    completed += 1
                    try:
                        response = future.result()
                        blind = response.output_parsed.model_copy(update={
                            "article_id": article["article_id"], "article_order": article["article_order"],
                        })
                        if not blind.verbatim_text.strip():
                            raise ValueError("blind Sol reader returned empty article text")
                    except Exception as exc:
                        self._record_call(run, "sol_blind_article_read", self.inspector_model, started,
                                          error=exc, article_id=article["article_id"])
                        verification[article["article_id"]] = {
                            "status": "blind_read_failed", "primary_sha256": self._article_hash(article),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        run.persist()
                        emit("Independent read failed", article["heading"])
                        continue
                    self._record_call(run, "sol_blind_article_read", self.inspector_model, started,
                                      response=response, article_id=article["article_id"])
                    path = run.save_read_artifact(article["article_id"], "sol-b", {
                        "role": "blind_sol_b", "model": self.inspector_model,
                        "request_id": getattr(response, "id", None), "article": blind.model_dump(),
                    })
                    verification[article["article_id"]] = {
                        "status": "blind_read_complete", "primary_sha256": self._article_hash(article),
                        "blind_read_path": str(path), "blind_confidence": blind.confidence,
                    }
                    run.persist()
                    blind_results.append((article, blind))
                    emit("Independent article read", f"{completed}/{len(requests)} · {article['heading']}")

        disagreements = []
        for article, blind in blind_results:
            reader_a_text = article["heading"] + "\n\n" + article["verbatim_text"]
            reader_b_text = blind.heading + "\n\n" + blind.verbatim_text
            comparison = self._comparison(reader_a_text, reader_b_text)
            truncation_signals = self._truncation_signals(
                article["verbatim_text"], blind.verbatim_text)
            if truncation_signals:
                comparison = comparison.model_copy(update={
                    "lexically_equal": False,
                    "difference_count": comparison.difference_count + 1,
                    "differences": comparison.differences + [{
                        "kind": "possible_truncation",
                        "signals": truncation_signals,
                    }],
                })
            if article["uncertainties"] != blind.uncertainties and comparison.lexically_equal:
                comparison = comparison.model_copy(update={
                    "lexically_equal": False,
                    "difference_count": 1,
                    "differences": [{
                        "kind": "uncertainty_disagreement",
                        "reader_a": article["uncertainties"],
                        "reader_b": blind.uncertainties,
                    }],
                })
            if blind.confidence < min_confidence and comparison.lexically_equal:
                comparison = comparison.model_copy(update={
                    "lexically_equal": False,
                    "difference_count": 1,
                    "differences": [{
                        "kind": "blind_reader_low_confidence",
                        "reader_b_confidence": blind.confidence,
                    }],
                })
            record = verification.setdefault(article["article_id"], {})
            record["comparison"] = comparison.model_dump()
            if truncation_signals:
                record["truncation_review"] = {
                    "status": "suspected",
                    "signals": truncation_signals,
                }
                run.persist()
                emit("Possible truncation detected", article["heading"])
            if comparison.lexically_equal:
                record.update(status="matched", final_sha256=self._article_hash(article))
                run.persist()
                emit("Independent reads agree", article["heading"])
            else:
                disagreements.append((article, blind, comparison))

        if disagreements:
            emit("Sol resolving text disagreements", f"{len(disagreements)} article(s)")
            with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(disagreements)))) as pool:
                futures = {}
                for article, blind, comparison in disagreements:
                    started = time.perf_counter()
                    future = pool.submit(self._adjudication_request, image_url, article, blind, comparison)
                    futures[future] = (article, blind, comparison, started)
                for future in as_completed(futures):
                    article, blind, comparison, started = futures[future]
                    record = verification[article["article_id"]]
                    try:
                        response = future.result()
                        final = response.output_parsed.model_copy(update={
                            "article_id": article["article_id"], "article_order": article["article_order"],
                            "source_anchor": article["source_anchor"],
                        })
                        if not final.verbatim_text.strip():
                            raise ValueError("Sol adjudicator returned empty article text")
                    except Exception as exc:
                        self._record_call(run, "sol_text_adjudication", self.inspector_model, started,
                                          error=exc, article_id=article["article_id"])
                        record.update(status="adjudication_failed", error=f"{type(exc).__name__}: {exc}")
                        run.persist()
                        emit("Text adjudication failed", article["heading"])
                        continue
                    self._record_call(run, "sol_text_adjudication", self.inspector_model, started,
                                      response=response, article_id=article["article_id"])
                    path = run.save_read_artifact(article["article_id"], "sol-adjudication", {
                        "role": "sol_lexical_adjudicator", "model": self.inspector_model,
                        "request_id": getattr(response, "id", None),
                        "reader_a": ArticleInput.model_validate(article).model_dump(),
                        "reader_b": blind.model_dump(), "comparison": comparison.model_dump(),
                        "article": final.model_dump(),
                    })
                    record.update(status="adjudication_ready", adjudication_path=str(path),
                                  proposed_confidence=final.confidence,
                                  primary_sha256=self._article_hash(article))
                    run.persist()
                    self._finalize_adjudication(
                        run, emit, image_url, article, final, record, min_confidence)

        failures = [article_id for article_id in run.state["articles"]
                    if verification.get(article_id, {}).get("status") not in {"matched", "adjudicated"}]
        if failures:
            run.state["status"] = "text_review_required"
            run.state["text_verification_error"] = f"verification incomplete for {len(failures)} article(s)"
            run.persist()
            emit("Saved — text verification pending", f"{len(failures)} article(s)")
            return False
        run.state["text_verification_completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        run.state.pop("text_verification_error", None)
        run.persist()
        return True

    def _terra_inventory(self, run: IncrementalRun):
        started = time.perf_counter()
        try:
            response = self.client.responses.parse(
                model=self.inventory_model,
                instructions=TERRA_INVENTORY_PROMPT,
                input=[{"role": "user", "content": [
                    {"type": "input_text", "text":
                        "Inventory every semantic article required by this operator request; do not inventory "
                        "out-of-scope articles as missing.\n\nOPERATOR REQUEST: " + run.state["instruction"]},
                    {"type": "input_image", "image_url": self._image(run.source), "detail": "original"},
                ]}],
                text_format=IndependentInventory,
            )
            self._require_complete_response(response, "Terra inventory")
        except Exception as exc:
            self._record_call(run, "terra_inventory", self.inventory_model, started, error=exc)
            run.persist()
            raise
        inventory = response.output_parsed
        self._record_call(run, "terra_inventory", self.inventory_model, started, response=response)
        run.state["audits"].append({"kind": "terra_inventory", **inventory.model_dump()})
        run.persist()
        return inventory

    def _terra_compare(self, run: IncrementalRun, inventory: IndependentInventory):
        payload = {
            "operator_request": run.state["instruction"],
            "blind_terra_inventory": inventory.model_dump(),
            "saved_sol_article_summaries": run.completed_summary(),
        }
        started = time.perf_counter()
        try:
            response = self.client.responses.parse(
                model=self.inventory_model,
                instructions=TERRA_COMPARE_PROMPT,
                input=[{"role": "user", "content": [
                    {"type": "input_text", "text": json.dumps(payload, ensure_ascii=False, indent=2)},
                    {"type": "input_image", "image_url": self._image(run.source), "detail": "original"},
                ]}],
                text_format=InventoryAudit,
            )
            self._require_complete_response(response, "Terra inventory comparison")
        except Exception as exc:
            self._record_call(run, "terra_inventory_comparison", self.inventory_model, started, error=exc)
            run.persist()
            raise
        audit = response.output_parsed
        self._record_call(run, "terra_inventory_comparison", self.inventory_model, started, response=response)
        run.state["audits"].append({"kind": "terra_inventory_comparison", **audit.model_dump()})
        run.persist()
        return audit

    def _targeted_sol(self, run: IncrementalRun, finding: InventoryFinding):
        existing = run.state["articles"].get(finding.related_article_id or "")
        if finding.kind != "missing_article" and not existing:
            raise ValueError(f"Terra finding {finding.kind} must identify an existing article_id")
        payload = {"terra_finding": finding.model_dump(), "existing_article_summary": None}
        if existing:
            payload["existing_article_summary"] = {
                key: existing[key] for key in ("article_id", "article_order", "heading", "source_anchor")
            }
        started = time.perf_counter()
        try:
            response = self.client.responses.parse(
                model=self.inspector_model,
                instructions=TARGETED_SOL_PROMPT,
                input=[{"role": "user", "content": [
                    {"type": "input_text", "text": json.dumps(payload, ensure_ascii=False, indent=2)},
                    {"type": "input_image", "image_url": self._image(run.source), "detail": "original"},
                ]}],
                text_format=ArticleInput,
            )
            self._require_complete_response(response, "targeted Sol inspection")
            self._require_article_text_integrity(response.output_parsed, "targeted Sol inspection")
        except Exception as exc:
            self._record_call(run, "targeted_sol_inspection", self.inspector_model, started,
                              error=exc, finding=finding.model_dump())
            run.persist()
            raise
        article = response.output_parsed
        self._record_call(run, "targeted_sol_inspection", self.inspector_model, started,
                          response=response, finding=finding.model_dump())
        if existing:
            target_order = finding.expected_order or existing["article_order"]
            article = article.model_copy(update={
                "article_id": existing["article_id"],
                "article_order": target_order,
            })
            run.revise(article, f"targeted Sol inspection: {finding.required_action}")
        else:
            expected = finding.expected_order or (
                max((row["article_order"] for row in run.state["articles"].values()), default=0) + 1
            )
            article_id = article.article_id
            if article_id in run.state["articles"]:
                article_id = f"{article_id}-recovered-{uuid.uuid4().hex[:6]}"
            article = article.model_copy(update={"article_id": article_id, "article_order": expected})
            run.insert(article, f"recovered by targeted Sol inspection: {finding.required_action}")
        run.persist()
        return article

    def run(self, source: str | Path, instruction: str, progress=None, max_turns: int = 40,
            max_audits: int = 2, max_context_turns: int = 4, max_context_restarts: int = 3,
            resume_folder: str | Path | None = None, verify_text: bool = True,
            verification_workers: int = 3, min_text_confidence: float = .80):
        emit = progress or (lambda stage, detail="": None)
        if not 0 <= min_text_confidence <= 1:
            raise ValueError("min_text_confidence must be between 0 and 1")
        if verification_workers < 1:
            raise ValueError("verification_workers must be at least 1")
        run = IncrementalRun(self.output_root, source, instruction, resume_folder=resume_folder)
        emit("Resuming saved page" if resume_folder else "Reading complete page", Path(run.source).name)
        response = None
        pending = self._fresh_sol_input(
            run,
            instruction,
            "Resume at the next unsaved article." if resume_folder else "Begin at the top of the page.",
        )
        finish_requested = False
        context_turns = 0
        context_index = 0
        failure_restarts = 0

        for turn in range(1, max_turns + 1):
            started = time.perf_counter()
            kwargs = {
                "model": self.model,
                "instructions": PROMPT,
                "tools": TOOLS,
                "parallel_tool_calls": False,
            }
            if response is None:
                kwargs["input"] = pending
            else:
                kwargs.update(previous_response_id=response.id, input=pending)
            try:
                response = self.client.responses.create(**kwargs)
                self._require_complete_response(response, "Sol article transcription")
            except Exception as exc:
                self._record_call(run, "sol_article_transcription", self.model, started, error=exc,
                                  turn=turn, context_index=context_index)
                if failure_restarts < max_context_restarts:
                    failure_restarts += 1
                    context_index += 1
                    response = None
                    context_turns = 0
                    pending = self._fresh_sol_input(
                        run,
                        instruction,
                        "The previous Sol request failed. Resume at the next unsaved article; do not redo saved work.",
                    )
                    run.persist()
                    emit("Fresh Sol context", f"retry {failure_restarts} · {len(run.state['articles'])} articles retained")
                    continue
                run.state["status"] = "interrupted"
                run.state["error"] = f"{type(exc).__name__}: {exc}"
                run.persist()
                emit("Paused — saved progress", f"{len(run.state['articles'])} articles retained")
                return run.result()

            context_turns += 1
            failure_restarts = 0
            self._record_call(run, "sol_article_transcription", self.model, started, response=response,
                              turn=turn, context_index=context_index)
            calls = [value for value in response.output if getattr(value, "type", None) == "function_call"]
            pending = []
            if not calls:
                pending = [{"role": "user", "content": [{
                    "type": "input_text",
                    "text": "Continue with tools: save the next single article, or finish only after the final page scan.",
                }]}]
            for call in calls:
                try:
                    args = json.loads(call.arguments or "{}")
                    if call.name == "save_articles":
                        saved = []
                        for raw in args["articles"]:
                            article = ArticleInput.model_validate(raw)
                            if article.article_id in run.state["articles"]:
                                raise ValueError(
                                    f"save_articles cannot overwrite {article.article_id}; use revise_article")
                            run.save(article)
                            saved.append(article.article_id)
                            emit("Article saved", f"{article.article_order} · {article.heading}")
                        value = {"ok": True, "saved": saved, "article_count": len(run.state["articles"])}
                    elif call.name == "revise_article":
                        article = ArticleInput.model_validate(args["article"])
                        if article.article_id not in run.state["articles"]:
                            value = {"ok": False, "error": "unknown article; use save_articles"}
                        else:
                            run.revise(article, args["reason"])
                            emit("Article repaired", f"{article.article_order} · {article.heading}")
                            value = {"ok": True, "revised": article.article_id}
                    elif call.name == "finish_page":
                        finish_requested = bool(args["final_scan_completed"])
                        integrity_errors = run.page_integrity_errors() if finish_requested else []
                        if integrity_errors:
                            finish_requested = False
                        value = {
                            "ok": finish_requested and not integrity_errors,
                            "article_count": len(run.state["articles"]),
                            "instruction": (
                                "Terra's independent inventory starts now." if finish_requested
                                else "Repair transcript integrity before finishing."
                                if integrity_errors else "Complete the source scan."
                            ),
                            "integrity_errors": integrity_errors,
                        }
                    else:
                        value = {"ok": False, "error": "unknown tool"}
                except ArticleIntegrityError as exc:
                    value = {
                        "ok": False,
                        "error": str(exc),
                        "required_action": (
                            "Re-inspect the affected article boundary from source pixels. Retranscribe any "
                            "incomplete article in a fresh pass; revise wrong ownership, then retry the save."
                        ),
                    }
                    run.state["events"].append({
                        "event": "article_integrity_rejected",
                        "at": datetime.now().astimezone().isoformat(timespec="seconds"),
                        "error": str(exc),
                    })
                    emit("Article integrity check", str(exc))
                except Exception as exc:
                    run.state["status"] = "interrupted"
                    run.state["error"] = (
                        f"invalid {getattr(call, 'name', 'tool')} call: {type(exc).__name__}: {exc}"
                    )
                    run.persist()
                    emit("Paused — invalid model tool call", f"{len(run.state['articles'])} articles retained")
                    return run.result()
                pending.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(value, ensure_ascii=False),
                })
            run.persist()
            if finish_requested:
                break
            if context_turns >= max_context_turns:
                context_index += 1
                response = None
                context_turns = 0
                pending = self._fresh_sol_input(
                    run,
                    instruction,
                    "Start a fresh visual pass at the next unsaved article. Do not repeat saved articles.",
                )
                emit("Fresh Sol context", f"{len(run.state['articles'])} articles retained")

        if not finish_requested:
            run.state["status"] = "interrupted"
            run.persist()
            emit("Paused — saved progress", f"{len(run.state['articles'])} articles retained")
            return run.result()

        last_audit = None
        repairs_attempted = 0
        for audit_index in range(max_audits):
            emit("Terra checking article inventory", f"Pass {audit_index + 1}/{max_audits}")
            try:
                inventory = self._terra_inventory(run)
                audit = self._terra_compare(run, inventory)
            except Exception as exc:
                run.state["status"] = "transcribed_pending_review"
                run.state["review_error"] = f"{type(exc).__name__}: {exc}"
                run.persist()
                emit("Transcription saved — review pending", str(exc))
                return run.result()
            last_audit = audit
            if audit.passed and audit.all_articles_found and audit.ownership_correct and audit.reading_order_correct:
                if verify_text and not self._verify_article_texts(
                        run, emit, audit, max_workers=verification_workers,
                        min_confidence=min_text_confidence):
                    return run.result()
                run.state["status"] = "complete"
                run.persist()
                emit("Complete", f"{len(run.state['articles'])} articles")
                return run.result()
            if not audit.findings:
                break
            emit("Sol inspecting Terra findings", f"{len(audit.findings)} targeted checks")
            for finding in audit.findings:
                try:
                    article = self._targeted_sol(run, finding)
                except Exception as exc:
                    run.state.setdefault("targeted_errors", []).append({
                        "finding": finding.model_dump(),
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    run.persist()
                    emit("Targeted Sol check failed", finding.heading)
                    continue
                repairs_attempted += 1
                stage = "Missing article recovered" if finding.kind == "missing_article" else "Article repaired"
                emit(stage, article.heading)

        # Every concrete Terra finding is routed to Sol, including findings from the final
        # configured audit.  If that final audit required a repair, another independent
        # inventory pass is still needed before the run may truthfully be called complete.
        if run.state.get("targeted_errors"):
            run.state["status"] = "review_required"
        elif last_audit is not None and last_audit.findings and repairs_attempted:
            run.state["status"] = "repaired_pending_inventory_confirmation"
        else:
            run.state["status"] = "inventory_review_required"
        run.persist()
        emit("Saved — inventory confirmation pending", f"{len(run.state['articles'])} articles")
        return run.result()


def materialize_transcript(source: str | Path, transcript_file: str | Path, output_root: str | Path = "digitized"):
    """Materialize a Codex-produced real-page checkpoint through the production filesystem store."""
    transcript_path = Path(transcript_file).resolve()
    transcript = TranscriptFile.model_validate_json(transcript_path.read_text(encoding="utf-8"))
    run = IncrementalRun(output_root, source, "Digitise the complete page article by article.")
    if transcript.source_sha256.casefold() != run.state["source_sha256"].casefold():
        raise ValueError("transcript source hash does not match the saved scan")
    for article in transcript.articles:
        run.save(article)
    run.state["development_transcript"] = {
        "path": str(transcript_path), "sha256": hashlib.sha256(transcript_path.read_bytes()).hexdigest(),
        "transcribed_by": transcript.transcribed_by, "reviewed_by": transcript.reviewed_by,
    }
    run.state["status"] = "complete" if transcript.reviewed_by else "awaiting_independent_review"
    run.persist()
    return run.result()


def main():
    parser = argparse.ArgumentParser(description="Materialize an incremental real-page transcript without paid API calls.")
    parser.add_argument("source")
    parser.add_argument("transcript")
    parser.add_argument("--output", default="digitized")
    args = parser.parse_args()
    print(json.dumps(materialize_transcript(args.source, args.transcript, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
