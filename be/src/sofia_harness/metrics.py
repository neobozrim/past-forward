from __future__ import annotations

from difflib import SequenceMatcher
import unicodedata


def bbox(polygon):
    xs, ys = zip(*polygon); return min(xs), min(ys), max(xs), max(ys)


def iou(left, right) -> float:
    ax1, ay1, ax2, ay2 = bbox(left); bx1, by1, bx2, by2 = bbox(right)
    area = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - area
    return area / union if union else 0.0


def normalized_agreement(a: str, b: str) -> float:
    norm = lambda s: " ".join(s.lower().split())
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def edit_distance(a: list, b: list) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j-1] + (x != y)))
        row = nxt
    return row[-1]


def cer(reference: str, prediction: str) -> float:
    return edit_distance(list(reference), list(prediction)) / max(1, len(reference))


def wer(reference: str, prediction: str) -> float:
    return edit_distance(reference.split(), prediction.split()) / max(1, len(reference.split()))


def normalize_text(text: str, policy: str = "verbatim") -> str:
    text = unicodedata.normalize("NFC", text)
    if policy == "verbatim": return text
    if policy != "search": raise ValueError(f"unknown scoring policy: {policy}")
    lines = text.splitlines()
    joined = "".join(line[:-1] if line.endswith("-") else line + " " for line in lines)
    return " ".join(joined.casefold().split())


def score_text(reference: str, prediction: str, policy="verbatim") -> dict:
    a, b = normalize_text(reference, policy), normalize_text(prediction, policy)
    return {"policy": policy, "cer": cer(a, b), "wer": wer(a, b)}


def hungarian_match(gold: list, predicted: list, min_iou=.5, class_compatible=True) -> dict:
    """Globally optimal one-to-one assignment, then threshold with unmatched penalties."""
    size = max(len(gold), len(predicted))
    if not size: return {"matches":[], "true_positive":0, "false_negative":0, "false_positive":0}
    weights = [[0.0] * size for _ in range(size)]
    for gi, g in enumerate(gold):
        for pi, p in enumerate(predicted):
            if not class_compatible or g.label == p.label: weights[gi][pi] = iou(g.polygon, p.polygon)
    # Hungarian algorithm for minimum cost; maximizing IoU is minimizing 1-IoU.
    n = size; u = [0.0]*(n+1); v = [0.0]*(n+1); assignment = [0]*(n+1); way = [0]*(n+1)
    for row in range(1, n+1):
        assignment[0] = row; col0 = 0; minimum = [float("inf")]* (n+1); used = [False]*(n+1)
        while True:
            used[col0] = True; row0 = assignment[col0]; delta = float("inf"); col1 = 0
            for col in range(1, n+1):
                if used[col]: continue
                current = 1.0 - weights[row0-1][col-1] - u[row0] - v[col]
                if current < minimum[col]: minimum[col] = current; way[col] = col0
                if minimum[col] < delta: delta, col1 = minimum[col], col
            for col in range(n+1):
                if used[col]: u[assignment[col]] += delta; v[col] -= delta
                else: minimum[col] -= delta
            col0 = col1
            if assignment[col0] == 0: break
        while True:
            col1 = way[col0]; assignment[col0] = assignment[col1]; col0 = col1
            if col0 == 0: break
    matches = []
    for col in range(1, n+1):
        gi, pi = assignment[col]-1, col-1
        if gi < len(gold) and pi < len(predicted) and weights[gi][pi] >= min_iou:
            matches.append((gi, pi, weights[gi][pi]))
    return {"matches": matches, "true_positive": len(matches), "false_negative": len(gold)-len(matches),
            "false_positive": len(predicted)-len(matches)}


def pairwise_precedence(order: list[str], gold_edges: list[tuple[str, str]]) -> float:
    position = {item: i for i, item in enumerate(order)}
    evaluable = [(a,b) for a,b in gold_edges if a in position and b in position]
    return sum(position[a] < position[b] for a,b in evaluable) / len(evaluable) if evaluable else 1.0
