import pytest
from sofia_harness.metrics import (cer, hungarian_match, iou, normalize_text,
    normalized_agreement, pairwise_precedence, score_text, wer)
from sofia_harness.models import Region


def region(identifier, label, x1, y1, x2, y2):
    return Region(id=identifier, label=label, polygon=[[x1,y1],[x2,y1],[x2,y2],[x1,y2]])


def test_iou_hand_calculated_half_overlap():
    assert iou([[0,0],[10,10]], [[5,0],[15,10]]) == pytest.approx(1/3)


def test_cer_hand_calculated():
    assert cer("котка", "кола") == pytest.approx(2/5)


def test_wer_hand_calculated():
    assert wer("един два три", "един три") == pytest.approx(1/3)


def test_unicode_nfc_policy():
    assert normalize_text("и\u0306", "verbatim") == "й"


def test_search_policy_joins_printed_line_wrap():
    assert normalize_text("комунисти-\nческата", "search") == "комунистическата"


def test_historical_letters_not_modernized():
    assert normalize_text("ѣ ъ", "search") == "ѣ ъ"


def test_unknown_policy_rejected():
    with pytest.raises(ValueError): normalize_text("текст", "modernize")


def test_score_reports_policy():
    assert score_text("Добър", "добър", "search") == {"policy":"search", "cer":0, "wer":0}


def test_one_to_one_matching_and_unmatched_penalties():
    gold = [region("g1","Text",0,0,10,10), region("g2","Text",20,0,30,10)]
    pred = [region("p1","Text",0,0,10,10), region("p2","Heading",20,0,30,10)]
    result = hungarian_match(gold, pred)
    assert (result["true_positive"], result["false_negative"], result["false_positive"]) == (1,1,1)


def test_pairwise_reading_order_handles_disconnected_nodes():
    assert pairwise_precedence(["x","b","a"], [("a","b"), ("missing","b")]) == 0


def test_agreement_normalizes_case_and_space():
    assert normalized_agreement("  Добър  ден ", "добър ден") == 1
