from sofia_harness.models import Article, LayoutRegion, PageLayout, Transcription, UncertainSpan


def tx(spans):
    return Transcription(region_id="r", verbatim_text="Добър ден", confidence=.8, uncertain_spans=spans)


def test_valid_unicode_offsets():
    assert tx([UncertainSpan(text="Добър", start=0, end=5, confidence=.5)]).validate_offsets() == []


def test_offset_text_mismatch():
    assert "does not match" in tx([UncertainSpan(text="ден", start=0, end=3, confidence=.5)]).validate_offsets()[0]


def test_offset_out_of_bounds():
    assert "out of bounds" in tx([UncertainSpan(text="x", start=20, end=21, confidence=.5)]).validate_offsets()[0]


def test_overlapping_offsets():
    spans = [UncertainSpan(text="Добър",start=0,end=5,confidence=.5), UncertainSpan(text="ър д",start=3,end=7,confidence=.5)]
    assert any("overlaps" in error for error in tx(spans).validate_offsets())


def test_layout_rejects_bad_coordinates_and_article_reference():
    layout=PageLayout(page_type="newspaper",language_candidates=["bg"],orientation_degrees=0,
        layout_difficulty=.5,confidence=.8,regions=[LayoutRegion(id="r",type="body_column",
        polygon=[[0,0],[1100,0],[0,10]],reading_order=0,confidence=.8)],
        articles=[Article(id="a",region_ids=["missing"],reading_order=0)])
    errors=layout.validate_layout()
    assert any("invalid normalized polygon" in e for e in errors)
    assert any("unknown region" in e for e in errors)


def test_transcription_schema_defines_every_required_property():
    schema=Transcription.model_json_schema()
    assert set(schema["required"]) <= set(schema["properties"])
    hyphen=schema["$defs"]["PrintedHyphenation"]
    assert set(hyphen["required"]) <= set(hyphen["properties"])


def test_layout_rejects_inconsistent_article_hierarchy():
    regions=[
        LayoutRegion(id="h",type="headline",polygon=[[0,0],[100,0],[100,50]],reading_order=0,confidence=.9,article_id="a"),
        LayoutRegion(id="b",type="body_column",polygon=[[0,60],[100,60],[100,200]],reading_order=1,confidence=.9,article_id="missing",parent_id="ghost"),
    ]
    layout=PageLayout(page_type="newspaper",language_candidates=["bg"],orientation_degrees=0,
        layout_difficulty=.5,confidence=.8,regions=regions,
        articles=[Article(id="a",region_ids=["h","b"],reading_order=0)])
    errors=layout.validate_layout()
    assert any("unknown parent" in error for error in errors)
    assert any("unknown article" in error for error in errors)
    assert any("article_id disagrees" in error for error in errors)
