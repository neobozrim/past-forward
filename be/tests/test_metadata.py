from sofia_harness.metadata import metadata_from_filename


def test_extracts_known_filename_hints():
    assert metadata_from_filename("IMG_0982_nm_07_04_1949_page1.png") == {
        "publication": "Народна младеж",
        "publication_code": "nm",
        "issue_date": "1949-07-04",
        "page_number": 1,
    }


def test_unknown_filename_is_valid_and_empty():
    assert metadata_from_filename("old-newspaper-photo.jpg") == {}


def test_publication_hints_are_extensible():
    result = metadata_from_filename("scan_custom_12_31_1950_page7.tif", {"custom": "Custom Daily"})
    assert result == {
        "publication": "Custom Daily",
        "publication_code": "custom",
        "issue_date": "1950-12-31",
        "page_number": 7,
    }
