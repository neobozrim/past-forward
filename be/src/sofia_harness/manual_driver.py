from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from .image_ops import crop_region
from .scans import inspect_scan, normalized_to_pixels


def materialize(layout_file="manual_layouts.json", config_file="config.yaml"):
    import yaml
    layout_path=Path(layout_file); config=yaml.safe_load(Path(config_file).read_text(encoding="utf-8"))
    spec=json.loads(layout_path.read_text(encoding="utf-8")); root=Path(config["dataset"]["images"])
    run_id=datetime.now(timezone.utc).strftime("manual-%Y%m%dT%H%M%SZ"); out=Path(config["dataset"]["output"])/run_id; out.mkdir(parents=True)
    manifest={"schema_version":"manual-1.0","run_id":run_id,"driver":spec["driver"],"api_calls":0,
        "validation_scope":"agent_driven_visual_pilot","created_at":datetime.now(timezone.utc).isoformat(),"pages":[]}
    for page_spec in spec["pages"]:
        source=root/page_spec["file"]; qc=inspect_scan(source,config.get("qc",{})); page={**page_spec,"qc":qc,"regions":[]}
        for region in page_spec["regions"]:
            pixels=normalized_to_pixels(region["polygon"],qc["width"],qc["height"])
            paths,provenance=crop_region(str(source),pixels,out/page_spec["file"]/"regions",region["id"])
            status="transcribed" if region.get("text") else ("non_text" if region["type"]=="photo" else "needs_crop_transcription")
            page["regions"].append({**region,"status":status,"artifacts":[str(p) for p in paths],"provenance":provenance})
        page.pop("file",None); manifest["pages"].append(page)
    target=out/"manifest.json"; target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return target


if __name__=="__main__": print(materialize().resolve())
