#!/usr/bin/env python3
"""Evaluate local benchmark JSONL files and export JSON and CSV summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.metrics import (
    DATASET_GROUP_FIELDS,
    EvaluationError,
    GROUP_FIELDS,
    evaluate_dataset_records,
    evaluate_records,
    load_jsonl,
    macro_average_accuracy,
)


CSV_FIELDS = (
    "run_id",
    "dataset",
    "model",
    "prompt_variant",
    "method",
    "total_samples",
    "successful_requests",
    "api_failures",
    "invalid_inputs",
    "parse_failures",
    "parse_coverage",
    "parse_coverage_numerator",
    "parse_coverage_denominator",
    "accuracy_all_labeled",
    "accuracy_all_labeled_numerator",
    "accuracy_all_labeled_denominator",
    "accuracy_parsed_labeled",
    "accuracy_parsed_labeled_numerator",
    "accuracy_parsed_labeled_denominator",
    "yes_precision",
    "yes_recall",
    "yes_f1",
    "yes_support",
    "no_precision",
    "no_recall",
    "no_f1",
    "no_support",
    "macro_f1",
    "unresolved_api_failures",
    "eligible",
    "actual_yes_predicted_yes",
    "actual_yes_predicted_no",
    "actual_no_predicted_yes",
    "actual_no_predicted_no",
)


def _csv_row(summary: dict[str, Any]) -> dict[str, Any]:
    row = {
        field: summary[field]
        for field in (
            "run_id",
            "dataset",
            "model",
            "prompt_variant",
            "method",
            "total_samples",
            "successful_requests",
            "api_failures",
            "invalid_inputs",
            "parse_failures",
        )
    }
    for metric in (
        "parse_coverage",
        "accuracy_all_labeled",
        "accuracy_parsed_labeled",
    ):
        row[metric] = summary[metric]["value"]
        row[f"{metric}_numerator"] = summary[metric]["numerator"]
        row[f"{metric}_denominator"] = summary[metric]["denominator"]
    for label in ("yes", "no"):
        for metric in ("precision", "recall", "f1", "support"):
            row[f"{label}_{metric}"] = summary["per_class"][label][metric]
    row["macro_f1"] = summary["macro_f1"]
    row["unresolved_api_failures"] = summary["unresolved_api_failures"]
    row["eligible"] = summary["eligible"]
    for actual in ("yes", "no"):
        for predicted in ("yes", "no"):
            field = f"actual_{actual}_predicted_{predicted}"
            row[field] = summary["confusion_matrix"][f"actual_{actual}"][
                f"predicted_{predicted}"
            ]
    return row


def write_summary_json(
    path: str | Path,
    summaries: list[dict[str, Any]],
    *,
    dataset_summaries: list[dict[str, Any]] | None = None,
    macro_averages: list[dict[str, Any]] | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "grouping": list(GROUP_FIELDS),
        "groups": summaries,
        "dataset_grouping": list(DATASET_GROUP_FIELDS),
        "datasets": dataset_summaries if dataset_summaries is not None else [],
        "macro_average_accuracy": macro_averages if macro_averages is not None else [],
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_summary_csv(path: str | Path, summaries: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(_csv_row(summary))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate local JSONL benchmark result files without network access."
    )
    parser.add_argument(
        "results",
        nargs="+",
        help="one or more JSONL result files",
    )
    parser.add_argument(
        "--summary-json",
        "--output-json",
        dest="summary_json",
        required=True,
        help="path for the JSON summary",
    )
    parser.add_argument(
        "--csv",
        "--output-csv",
        dest="csv_path",
        required=True,
        help="path for the tabular CSV summary",
    )
    parser.add_argument(
        "--allow-unscored-gold-labels",
        action="store_true",
        help=(
            "preserve unsupported native gold labels but exclude them from "
            "binary accuracy metrics"
        ),
    )
    return parser.parse_args(argv)


def cli(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        records = [
            record
            for result_path in arguments.results
            for record in load_jsonl(result_path)
        ]
        summaries = evaluate_records(
            records,
            allow_unscored_gold_labels=arguments.allow_unscored_gold_labels,
        )
        dataset_summaries = evaluate_dataset_records(
            records,
            allow_unscored_gold_labels=arguments.allow_unscored_gold_labels,
        )
        write_summary_json(
            arguments.summary_json,
            summaries,
            dataset_summaries=dataset_summaries,
            macro_averages=macro_average_accuracy(dataset_summaries),
        )
        write_summary_csv(arguments.csv_path, summaries)
    except (EvaluationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
