from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .image_ops import sha256


DEFAULT_THRESHOLDS = {
    "min_short_edge_px": 1800, "blur_variance_min": 80.0, "max_abs_skew_degrees": 2.0,
    "max_illumination_range": 70.0, "blank_ink_coverage_max": .003,
    "edge_ink_ratio_max": .08, "duplicate_hash_distance": 3,
}


def discover_scans(image_dir: str | Path, patterns: list[str]) -> list[Path]:
    root = Path(image_dir)
    found = {p.resolve() for pattern in patterns for p in root.glob(pattern) if p.is_file()}
    return sorted(found, key=lambda p: p.name.casefold())


def _skew(gray: np.ndarray) -> tuple[float, float]:
    small = cv2.resize(gray, None, fx=min(1, 1400/gray.shape[1]), fy=min(1, 1400/gray.shape[1]))
    edges = cv2.Canny(small, 60, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=small.shape[1]//8, maxLineGap=20)
    if lines is None: return 0.0, 0.0
    angles = []
    for x1,y1,x2,y2 in np.asarray(lines).reshape(-1,4):
        angle = np.degrees(np.arctan2(y2-y1, x2-x1))
        normalized = ((angle + 45) % 90) - 45
        if abs(normalized) <= 15: angles.append(normalized)
    return (round(float(np.median(angles)),3), min(1.0,len(angles)/30)) if angles else (0.0,0.0)


def _illumination(gray: np.ndarray) -> tuple[float, list[float]]:
    background = cv2.GaussianBlur(gray, (0,0), sigmaX=max(gray.shape)/30)
    cells = [float(np.mean(cell)) for row in np.array_split(background,4,axis=0) for cell in np.array_split(row,4,axis=1)]
    return round(max(cells)-min(cells),3), [round(x,2) for x in cells]


def _perceptual_hash(gray: np.ndarray) -> str:
    resized = cv2.resize(gray,(9,8),interpolation=cv2.INTER_AREA)
    bits = resized[:,1:] > resized[:,:-1]
    return f"{int(''.join('1' if b else '0' for b in bits.flat),2):016x}"


def hash_distance(left: str, right: str) -> int:
    return (int(left,16)^int(right,16)).bit_count()


def inspect_scan(path: str | Path, thresholds: dict | None = None) -> dict:
    limits = DEFAULT_THRESHOLDS | (thresholds or {}); path = Path(path)
    raw = np.fromfile(path,dtype=np.uint8); image = cv2.imdecode(raw,cv2.IMREAD_COLOR)
    if image is None: raise ValueError(f"unreadable image: {path}")
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY); height,width = gray.shape
    analysis_scale=min(1.0,1600/max(width,height)); work=cv2.resize(gray,None,fx=analysis_scale,fy=analysis_scale,interpolation=cv2.INTER_AREA) if analysis_scale<1 else gray
    blur = float(cv2.Laplacian(work,cv2.CV_64F).var()); skew,skew_confidence = _skew(work)
    illum_range,illum_grid = _illumination(work)
    threshold = cv2.threshold(work,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    ink = threshold > 0; ink_coverage = float(np.mean(ink))
    margin=max(2,round(min(work.shape)*.015)); edge_mask=np.zeros_like(ink)
    edge_mask[:margin,:]=True; edge_mask[-margin:,:]=True; edge_mask[:,:margin]=True; edge_mask[:,-margin:]=True
    edge_ink_ratio=float(np.sum(ink & edge_mask)/max(1,np.sum(ink)))
    # A 90-degree error usually presents more strong vertical than horizontal text/rule energy.
    gx=float(np.mean(np.abs(cv2.Sobel(work,cv2.CV_32F,1,0)))); gy=float(np.mean(np.abs(cv2.Sobel(work,cv2.CV_32F,0,1))))
    orientation_suspect = gx < gy*.65
    flags=[]
    if blur < limits["blur_variance_min"]: flags.append("blur")
    if skew_confidence >= .2 and abs(skew)>limits["max_abs_skew_degrees"]: flags.append("skew")
    if orientation_suspect: flags.append("orientation_suspect")
    if illum_range>limits["max_illumination_range"]: flags.append("uneven_illumination")
    if ink_coverage<limits["blank_ink_coverage_max"]: flags.append("blank_page")
    if edge_ink_ratio>limits["edge_ink_ratio_max"]: flags.append("cropped_or_missing_edge")
    if min(width,height)<limits["min_short_edge_px"]: flags.append("insufficient_resolution")
    return {"path":str(path.resolve()),"sha256":sha256(path),"width":width,"height":height,
        "format":path.suffix.lstrip('.').upper(),"analysis":{"scale":round(analysis_scale,6),"width":work.shape[1],"height":work.shape[0],"unassessed":["perspective_distortion"]},"scores":{"blur_laplacian_variance":round(blur,3),
        "skew_degrees":skew,"skew_confidence":round(skew_confidence,3),"illumination_grid_range":illum_range,
        "illumination_grid_means":illum_grid,"ink_coverage":round(ink_coverage,6),"edge_ink_ratio":round(edge_ink_ratio,6),
        "orientation_gradient_ratio":round(gx/max(gy,.0001),3)},"perceptual_hash":_perceptual_hash(gray),
        "flags":flags,"qc_status":"review" if flags else "pass"}


def mark_duplicates(records: list[dict], max_distance: int = 3) -> None:
    for index, current in enumerate(records):
        for earlier in records[:index]:
            distance=hash_distance(current["perceptual_hash"],earlier["perceptual_hash"])
            if current["sha256"]==earlier["sha256"] or distance<=max_distance:
                current["flags"].append("duplicate_page")
                current["duplicate_of"]={"path":earlier["path"],"hash_distance":distance}
                current["qc_status"]="review"; break


def normalized_to_pixels(polygon: list[list[float]], width: int, height: int) -> list[list[float]]:
    return [[x*width/1000.0,y*height/1000.0] for x,y in polygon]
