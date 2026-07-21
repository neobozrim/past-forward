from __future__ import annotations

import json, os, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

from .image_ops import crop_region, sha256
from .metrics import normalized_agreement
from .openai_adapter import OpenAIAdapter, OCRCallError, InvalidTranscriptionError
from .scans import discover_scans, inspect_scan, mark_duplicates, normalized_to_pixels

SCHEMA_VERSION = "2.0.0"
TEXT_TYPES = {"headline", "subtitle", "body_column", "caption", "advertisement", "table", "footer", "masthead"}


def _atomic_json(path: Path, data: dict):
    if path.exists(): raise FileExistsError(f"refusing to overwrite immutable manifest: {path}")
    temporary = path.with_suffix(".tmp"); temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _git_commit():
    try: return subprocess.check_output(["git","rev-parse","HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None


def run(config: dict, dry_run: bool = False, adapter=None, scan_paths=None, progress=None) -> Path:
    ds, models, routing = config["dataset"], config["models"], config["routing"]
    scans = [Path(p).resolve() for p in scan_paths] if scan_paths is not None else discover_scans(ds["images"], ds.get("patterns", ["*.png"]))
    limit = config.get("run", {}).get("max_pages"); scans = scans[:limit] if limit and scan_paths is None else scans
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"); out = Path(ds.get("output","runs"))/run_id; out.mkdir(parents=True)
    manifest = {"schema_version":SCHEMA_VERSION,"run_id":run_id,"status":"running","dry_run":dry_run,
        "validation_scope":"discovery_and_qc_only" if dry_run else "live_unscored_pipeline_run",
        "created_at":datetime.now(timezone.utc).isoformat(),"source_scan_count":len(scans),"config":config,
        "config_sha256":sha256(config.get("_path","config.yaml")),"git_commit":_git_commit(),
        "environment":{"python":sys.version,"platform":platform.platform()},
        "prompt_versions":{"layout":"hierarchical-layout-v1","ocr":"diplomatic-bg-v1"},
        "pages":[],"calls":[],"failures":[],"review_queue":[]}
    def emit(scan, stage, status, **detail):
        if progress: progress(str(Path(scan).resolve()), stage, status, detail)
    qc_records=[]
    for scan in scans:
        emit(scan,"qc","running")
        qc_records.append(inspect_scan(scan,config.get("qc",{})))
        emit(scan,"qc","complete",flags=qc_records[-1].get("flags",[]))
    mark_duplicates(qc_records,config.get("qc",{}).get("duplicate_hash_distance",3))
    api = None if dry_run else (adapter or OpenAIAdapter())
    for index, (scan,qc) in enumerate(zip(scans,qc_records)):
        page_id = f"p{index+1:06d}"; page = {"page_id":page_id,"source":qc,"status":"qc_only" if dry_run else "processing"}
        manifest["pages"].append(page)
        if dry_run:
            emit(scan,"qc_only_complete","complete")
            continue
        # During active development QC is diagnostic, never a processing gate.
        # Even blank, duplicate, low-resolution, skewed, or damaged scans continue
        # through layout and OCR so operators can inspect the actual outcome.
        page["qc_flags"]=list(qc.get("flags",[]))
        try:
            # OCR the complete original first. Layout/QC may improve or structure this
            # baseline, but can never prevent the model from seeing the source page.
            emit(scan,"full_page_ocr","running")
            try:
                full_read, full_meta = api.transcribe(scan,"full_page",models["ocr_hard"])
                manifest["calls"].append({**full_meta,"stage":"full_page_ocr"})
                page["full_page_read"] = full_read.model_dump()
                emit(scan,"full_page_ocr","complete",confidence=full_read.confidence)
            except (OCRCallError,InvalidTranscriptionError) as exc:
                manifest["failures"].append({"page_id":page_id,"stage":"full_page_ocr","error":str(exc),"attempts":getattr(exc,"attempts",[])})
                emit(scan,"full_page_ocr","failed",error=str(exc))
            emit(scan,"layout","running")
            layout, telemetry = api.analyze_layout(scan, models["layout"]); manifest["calls"].append(telemetry)
            if layout.confidence < routing["layout_confidence_threshold"] or layout.layout_difficulty > routing["difficult_page_threshold"]:
                first_layout = layout.model_dump()
                layout, telemetry = api.analyze_layout(scan, models["layout_escalation"]); manifest["calls"].append(telemetry)
                page["layout_escalated_from"] = first_layout
            page["layout"] = layout.model_dump(); page["status"] = "layout_complete"; page["regions"] = []; emit(scan,"layout","complete",regions=len(layout.regions))
            emit(scan,"cropping_enhancement","running")
            for region in sorted(layout.regions, key=lambda r:r.reading_order):
                if region.type not in TEXT_TYPES: continue
                polygon = normalized_to_pixels(region.polygon, qc["width"], qc["height"])
                variants, provenance = crop_region(str(scan), polygon, out/page_id/"regions", region.id)
                model = models["ocr_hard"] if region.needs_enhancement or region.confidence < .85 else models["ocr_easy"]
                item = {"region":region.model_dump(),"provenance":provenance,"ocr_model":model}
                page["regions"].append(item)
                try:
                    # Blinded reads: separate stateless requests, different image variants, no shared output.
                    emit(scan,"ocr_read_a","running",region_id=region.id); a, ta = api.transcribe(variants[0], region.id, model); emit(scan,"ocr_read_a","complete",region_id=region.id)
                    emit(scan,"ocr_read_b","running",region_id=region.id); b, tb = api.transcribe(variants[1], region.id, model); emit(scan,"ocr_read_b","complete",region_id=region.id)
                    manifest["calls"].extend([ta,tb]); agreement = normalized_agreement(a.verbatim_text,b.verbatim_text)
                    item.update({"read_a":a.model_dump(),"read_b":b.model_dump(),"agreement":agreement})
                    reasons=[]
                    if agreement < routing["disagreement_threshold"]: reasons.append("reader_disagreement")
                    if min(a.confidence,b.confidence) < routing["confidence_threshold"]: reasons.append("low_confidence")
                    if a.uncertain_spans or b.uncertain_spans: reasons.append("reported_uncertainty")
                    emit(scan,"adjudication","complete",region_id=region.id,agreement=agreement)
                    if reasons: manifest["review_queue"].append({"page_id":page_id,"region_id":region.id,"reasons":reasons})
                except (OCRCallError,InvalidTranscriptionError) as exc:
                    manifest["failures"].append({"page_id":page_id,"region_id":region.id,"stage":"ocr","error":str(exc),"attempts":getattr(exc,"attempts",[])})
            emit(scan,"cropping_enhancement","complete")
            has_region_text=any((r.get("read_a") or {}).get("verbatim_text") for r in page["regions"])
            if not page.get("full_page_read") and not has_region_text:
                raise RuntimeError("no OCR text was produced for the page")
            page["status"] = "completed"; emit(scan,"reconstructed","complete")
        except Exception as exc:
            page["status"] = "failed"; manifest["failures"].append({"page_id":page_id,"stage":"layout","error_type":type(exc).__name__,"error":str(exc)})
    manifest["status"]="completed_with_failures" if manifest["failures"] else "completed"
    manifest["completed_at"]=datetime.now(timezone.utc).isoformat(); _atomic_json(out/"manifest.json",manifest); return out
