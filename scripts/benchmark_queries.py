"""Measure cold startup and warm queries against the configured local corpus."""
from __future__ import annotations

import argparse
import json
from time import perf_counter

from aic2026.app.api import RuntimeConfig, TaskRequest, load_retrieval_pipeline, run_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", action="append", help="English query; repeat for more cases")
    parser.add_argument("--task", choices=("kis", "qa", "trake"), default="kis")
    args = parser.parse_args()
    cases = args.query or [
        "a man riding a motorbike on a street",
        "a red car near a building",
    ]
    for task, text in [("kis", "warmup"), *((args.task, q) for q in cases)]:
        started = perf_counter()
        result = run_task(task, TaskRequest(
            text=text, runtime=RuntimeConfig(temporal_nms_enabled=True),
        ))
        print(json.dumps({
            "task": task, "query": text, "seconds": round(perf_counter() - started, 3),
            "candidates": len(result.candidates),
            "timing": result.trace[-1].detail,
            "corpus_cache": load_retrieval_pipeline.cache_info()._asdict(),
        }, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
