"""Inspect an explicit split across all discovered Hugging Face configurations."""

from datetime import datetime, timezone
import json
import sys
import time
import uuid

from .huggingface import inspect_huggingface_configuration, inspect_huggingface_selection
from .inspection_history import append_history
from .inspection_runs import build_inspection_report


def inspect_all_configurations(catalog, settings, *, token=None, listing=False):
    """Keep each configuration's sampling, votes and saved report independent."""
    split = settings["split"]
    selected, skipped = [], []
    # Resolve availability at one pinned revision before publishing any results.
    for name in sorted(set(catalog["configs"])):
        configuration = inspect_huggingface_configuration(catalog, name, token=token)
        if split not in configuration["splits"]:
            skipped.append({"configuration": name, "reason": f"Split {split!r} is absent",
                            "available_splits": list(configuration["splits"])})
            print(f"Skipping {name}: split {split!r} is absent", file=sys.stderr)
        else:
            selected.append(configuration)
    if not selected:
        raise ValueError(f"No configurations provide split {split!r}; no inspections were saved")
    selection = {"requested_config": "*", "dataset": catalog["dataset_id"],
                 "revision": catalog["revision"], "split": split,
                 "selected_configurations": [item["config"] for item in selected],
                 "skipped_configurations": skipped}
    if listing:
        print(json.dumps(selection, ensure_ascii=False, indent=2))
        return 0
    selection["batch_id"] = uuid.uuid4().hex
    reports, failures = [], []
    for index, configuration in enumerate(selected, 1):
        name = configuration["config"]
        print(f"Inspection subset {index}/{len(selected)}: {name} / {split}", file=sys.stderr)
        started_at = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()
        try:
            inventory = inspect_huggingface_selection(
                configuration, split, token=token, metadata_only=settings["formats_only"])
            subset_settings = {**settings, "config": name, "revision": catalog["revision"],
                               "shards": inventory["dataset_shards"]}
            report = build_inspection_report(inventory, subset_settings, token=token)
            report.update(started_at=started_at, processing_seconds=round(time.perf_counter() - start, 3),
                          created_at=datetime.now(timezone.utc).isoformat(), configuration_batch=selection)
            if settings["record_history"] or settings["output"]:
                report = append_history(report, settings["history_sheet"])
                print(f"Inspection history: {settings['history_sheet']} · subset {name} · "
                      f"run {report['history']['run_id']}", file=sys.stderr)
            reports.append(report)
        except (OSError, ValueError, ImportError) as error:
            failures.append({"configuration": name, "error": str(error)})
            print(f"Inspection failed for {name}: {error}", file=sys.stderr)
    print(json.dumps({"report_type": "dataset_inspection_batch", **selection,
                      "completed": len(reports), "failed_configurations": failures,
                      "reports": reports}, ensure_ascii=False, indent=2, default=str), flush=True)
    print(f"Subset inspections: {len(reports)} completed, {len(skipped)} without requested split, "
          f"{len(failures)} failed", file=sys.stderr)
    if failures:
        print("Failed subset details:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure['configuration']}: {failure['error']}", file=sys.stderr)
    return 1 if failures else 0
