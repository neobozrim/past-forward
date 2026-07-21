from __future__ import annotations

from pathlib import Path
import hashlib
import json
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def crop_region(image_path: str, polygon: list[list[float]], output_dir: Path, region_id: str) -> tuple[list[Path], dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    xs, ys = zip(*polygon)
    box = (max(0, int(min(xs))), max(0, int(min(ys))), min(image.width, int(max(xs) + 1)),
           min(image.height, int(max(ys) + 1)))
    original = image.crop(box)
    paths = [output_dir / f"{region_id}-original.png", output_dir / f"{region_id}-contrast.png",
             output_dir / f"{region_id}-threshold.png"]
    original.save(paths[0])
    enhanced = ImageOps.grayscale(original)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.7).resize((original.width * 2, original.height * 2))
    enhanced = enhanced.filter(ImageFilter.SHARPEN)
    enhanced.save(paths[1])
    threshold = ImageOps.grayscale(original).resize((original.width * 2, original.height * 2))
    threshold = ImageOps.autocontrast(threshold).point(lambda p: 255 if p > 145 else 0)
    threshold.save(paths[2])
    provenance = {"source_image": str(Path(image_path).resolve()), "source_sha256": sha256(image_path),
                  "source_dimensions": [image.width, image.height], "polygon_source_pixels": polygon,
                  "crop_box": list(box), "transform_source_to_crop": [1, 0, -box[0], 0, 1, -box[1], 0, 0, 1],
                  "variants": [
                      {"path": str(paths[0]), "recipe": ["crop_rgb"], "sha256": sha256(paths[0])},
                      {"path": str(paths[1]), "recipe": ["grayscale", "contrast:1.7", "resize:2x", "sharpen"], "sha256": sha256(paths[1])},
                      {"path": str(paths[2]), "recipe": ["grayscale", "resize:2x", "autocontrast", "threshold:145"], "sha256": sha256(paths[2])},
                  ]}
    (output_dir / f"{region_id}-provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return paths, provenance
