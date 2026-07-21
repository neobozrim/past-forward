from __future__ import annotations

from pydantic import BaseModel, Field


class Region(BaseModel):
    id: str
    label: str
    polygon: list[list[float]]
    reading_order: int | None = None
    text: str | None = None


class Page(BaseModel):
    id: str
    image_path: str
    width: int
    height: int
    regions: list[Region]
    reading_edges: list[tuple[str, str]] = Field(default_factory=list)


class LayoutRegion(BaseModel):
    id: str
    type: str
    polygon: list[list[float]] = Field(description="Polygon in normalized 0..1000 page coordinates")
    article_id: str | None = None
    parent_id: str | None = None
    reading_order: int
    confidence: float = Field(ge=0, le=1)
    needs_enhancement: bool = False
    semantic_heading_id: str | None = None
    detected_text: str | None = None
    column_index: int | None = None
    column_count: int | None = None


class Article(BaseModel):
    id: str
    region_ids: list[str]
    reading_order: int


class PageLayout(BaseModel):
    page_type: str
    language_candidates: list[str]
    orientation_degrees: int
    layout_difficulty: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    regions: list[LayoutRegion]
    articles: list[Article] = Field(default_factory=list)

    def validate_layout(self) -> list[str]:
        errors, ids = [], set()
        for region in self.regions:
            if region.id in ids: errors.append(f"duplicate region id: {region.id}")
            ids.add(region.id)
            if len(region.polygon) != 4: errors.append(f"{region.id}: text polygon must have exactly 4 corners")
            if any(len(p) != 2 or any(v < 0 or v > 1000 for v in p) for p in region.polygon):
                errors.append(f"{region.id}: invalid normalized polygon")
            if region.parent_id is not None and region.parent_id == region.id:
                errors.append(f"{region.id}: region cannot parent itself")
        for region in self.regions:
            if region.parent_id is not None and region.parent_id not in ids:
                errors.append(f"{region.id}: unknown parent {region.parent_id}")
        article_ids, memberships = set(), {}
        for article in self.articles:
            if article.id in article_ids: errors.append(f"duplicate article id: {article.id}")
            article_ids.add(article.id)
            if not article.region_ids: errors.append(f"{article.id}: article has no regions")
            for region_id in article.region_ids:
                if region_id not in ids: errors.append(f"{article.id}: unknown region {region_id}")
                if region_id in memberships and memberships[region_id] != article.id:
                    errors.append(f"{region_id}: belongs to multiple articles")
                memberships[region_id] = article.id
        if len([a.reading_order for a in self.articles]) != len(set(a.reading_order for a in self.articles)):
            errors.append("duplicate article reading_order")
        for region in self.regions:
            if region.article_id is not None and region.article_id not in article_ids:
                errors.append(f"{region.id}: unknown article {region.article_id}")
            if region.article_id is not None and memberships.get(region.id) not in (None, region.article_id):
                errors.append(f"{region.id}: article_id disagrees with article membership")
        for article in self.articles:
            local_orders=[r.reading_order for r in self.regions if r.id in article.region_ids]
            if len(local_orders)!=len(set(local_orders)):
                errors.append(f"duplicate reading_order within article {article.id}")
        return errors


class UncertainSpan(BaseModel):
    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    alternatives: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    reason: str = ""


class PrintedHyphenation(BaseModel):
    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    is_line_wrap: bool


class Transcription(BaseModel):
    region_id: str
    verbatim_text: str
    confidence: float = Field(ge=0, le=1)
    uncertain_spans: list[UncertainSpan] = Field(default_factory=list)
    printed_hyphenations: list[PrintedHyphenation] = Field(default_factory=list)

    def validate_offsets(self) -> list[str]:
        errors, previous_end = [], 0
        for index, span in enumerate(sorted(self.uncertain_spans, key=lambda x: x.start)):
            if span.end < span.start or span.end > len(self.verbatim_text):
                errors.append(f"span[{index}] out of bounds")
            elif self.verbatim_text[span.start:span.end] != span.text:
                errors.append(f"span[{index}] text does not match verbatim_text")
            if span.start < previous_end:
                errors.append(f"span[{index}] overlaps previous span")
            previous_end = max(previous_end, span.end)
        return errors
