from __future__ import annotations

import re
from datetime import date
from pathlib import Path


# These are filename hints, not an allow-list. Unknown publication codes remain
# usable inputs and simply produce no publication metadata.
PUBLICATION_HINTS = {
    "ot": "Отечествен фронт",
    "nm": "Народна младеж",
    "rd": "Работническо дело",
    "st": "Стършел",
}


def metadata_from_filename(filename: str | Path,
                           publication_hints: dict[str, str] | None = None) -> dict:
    """Best-effort metadata hints from a source filename.

    Recognizes tokenized names such as IMG_0982_nm_07_04_1949_page1.png.
    Nothing here is required: unrecognized names return an empty dictionary.
    Callers may supply or extend the publication hint mapping.
    """
    stem = Path(filename).stem
    tokens = [token for token in re.split(r"[^0-9A-Za-zА-Яа-я]+", stem) if token]
    lowered = [token.casefold() for token in tokens]
    hints = PUBLICATION_HINTS if publication_hints is None else publication_hints
    metadata: dict[str, str | int] = {}

    for token in lowered:
        if token in hints:
            metadata["publication"] = hints[token]
            metadata["publication_code"] = token
            break

    for index in range(len(tokens) - 2):
        if not all(tokens[index + offset].isdigit() for offset in range(3)):
            continue
        month, day, year = (int(tokens[index + offset]) for offset in range(3))
        if len(tokens[index + 2]) != 4:
            continue
        try:
            metadata["issue_date"] = date(year, month, day).isoformat()
            break
        except ValueError:
            continue

    for token in lowered:
        match = re.fullmatch(r"page[-_ ]?(\d+)", token)
        if match:
            metadata["page_number"] = int(match.group(1))
            break

    return metadata
