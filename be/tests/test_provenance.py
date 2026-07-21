from PIL import Image
from sofia_harness.image_ops import crop_region, sha256


def test_crop_provenance_and_hashes(tmp_path):
    source = tmp_path / "source.png"; Image.new("RGB", (100,80), "white").save(source)
    paths, provenance = crop_region(str(source), [[10,20],[40,20],[40,60],[10,60]], tmp_path/"out", "r1")
    assert Image.open(paths[0]).size == (31,41)
    assert provenance["source_dimensions"] == [100,80]
    assert provenance["crop_box"] == [10,20,41,61]
    assert provenance["transform_source_to_crop"] == [1,0,-10,0,1,-20,0,0,1]
    assert all(v["sha256"] == sha256(v["path"]) for v in provenance["variants"])

