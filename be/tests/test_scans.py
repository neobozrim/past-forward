from PIL import Image
from sofia_harness.scans import discover_scans, inspect_scan, mark_duplicates, normalized_to_pixels
import numpy as np
import pytest
from sofia_harness.scans import _skew


def test_scan_discovery_ignores_non_images(tmp_path):
    Image.new("RGB",(10,20)).save(tmp_path/"B.PNG")
    Image.new("RGB",(10,20)).save(tmp_path/"a.png")
    (tmp_path/"notes.txt").write_text("not a scan")
    assert [p.name for p in discover_scans(tmp_path,["*.png","*.PNG"])] == ["a.png","B.PNG"]


def test_technical_inspection_has_hash_and_dimensions(tmp_path):
    image=tmp_path/"page.png"; Image.new("L",(30,40),128).save(image)
    qc=inspect_scan(image)
    assert qc["width"]==30 and qc["height"]==40 and len(qc["sha256"])==64
    assert qc["analysis"]["unassessed"]==["perspective_distortion"]
    assert "perspective_unassessed" not in qc["flags"]


def test_normalized_layout_coordinates_to_pixels():
    assert normalized_to_pixels([[0,0],[500,250],[1000,1000]],200,400)==[[0,0],[100,100],[200,400]]


def test_blank_and_low_resolution_flags(tmp_path):
    image=tmp_path/"blank.png"; Image.new("L",(100,120),255).save(image)
    qc=inspect_scan(image,{"min_short_edge_px":200})
    assert "blank_page" in qc["flags"] and "insufficient_resolution" in qc["flags"]


def test_duplicate_detection(tmp_path):
    for name in ["a.png","b.png"]: Image.new("L",(100,100),200).save(tmp_path/name)
    records=[inspect_scan(tmp_path/"a.png"),inspect_scan(tmp_path/"b.png")]
    mark_duplicates(records,0)
    assert "duplicate_page" in records[1]["flags"]


def test_blur_score_is_reported(tmp_path):
    image=tmp_path/"flat.png"; Image.new("L",(100,100),128).save(image)
    qc=inspect_scan(image)
    assert qc["scores"]["blur_laplacian_variance"] == 0
    assert "blur" in qc["flags"]


def test_skew_accepts_opencv5_flat_hough_shape(monkeypatch):
    monkeypatch.setattr("sofia_harness.scans.cv2.HoughLinesP",lambda *a,**k:np.array([[0,0,90,3]],dtype=np.int32))
    angle,confidence=_skew(np.zeros((100,100),dtype=np.uint8))
    assert angle == pytest.approx(1.909,abs=.001) and confidence > 0
