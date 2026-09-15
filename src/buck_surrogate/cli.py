"""Command-line interface for the CSV-first workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .explore import explore_candidates
from .learning_curve import (
    evaluate_learning_curve,
    save_learning_curve_plot,
    summary_as_json,
)
from .model import (
    benchmark_model,
    load_bundle,
    predict_frame,
    save_bundle,
    train_model,
)
from .physics import synthetic_dataset
from .schema import load_csv, validate_dataframe


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_validation(report) -> None:
    print(f"Rows: {report.rows}")
    print(f"Status: {'VALID' if report.valid else 'INVALID'}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")


def command_generate(args: argparse.Namespace) -> int:
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = synthetic_dataset(samples=args.samples, seed=args.seed)
    frame.to_csv(destination, index=False)
    print(f"Created {len(frame)} synthetic reference rows at {destination}")
    print("Data source: simplified equations; not experimental ground truth.")
    return 0


def command_validate(args: argparse.Namespace) -> int:
    report = validate_dataframe(load_csv(args.input), require_targets=not args.features_only)
    _print_validation(report)
    return 0 if report.valid else 2


def command_train(args: argparse.Namespace) -> int:
    from .plots import save_parity_plots

    frame = load_csv(args.input)
    report = validate_dataframe(frame, require_targets=True)
    _print_validation(report)
    if not report.valid:
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle, metrics, predictions = train_model(
        frame,
        test_fraction=args.test_fraction,
        seed=args.seed,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
    )
    save_bundle(bundle, output_dir / "buck_surrogate.joblib")
    _write_json(metrics, output_dir / "metrics.json")
    predictions.to_csv(output_dir / "predictions_test.csv", index=False)
    save_parity_plots(predictions, output_dir / "parity_plots.png")
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts saved to {output_dir}")
    return 0


def command_learning_curve(args: argparse.Namespace) -> int:
    frame = load_csv(args.input)
    report = validate_dataframe(frame, require_targets=True)
    _print_validation(report)
    if not report.valid:
        return 2

    raw, summary, metadata = evaluate_learning_curve(
        frame,
        sizes=args.sizes,
        test_fraction=args.test_fraction,
        repeats=args.repeats,
        seed=args.seed,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_dir / "learning_curve_raw.csv", index=False)
    summary.to_csv(output_dir / "learning_curve.csv", index=False)
    _write_json(summary_as_json(summary, metadata), output_dir / "learning_curve.json")
    save_learning_curve_plot(summary, output_dir / "learning_curve.png")

    columns = ["training_rows", "target", "r2_mean", "mape_pct_mean", "p95_ape_pct_mean"]
    print(summary[columns].to_string(index=False))
    print(f"Learning-curve artifacts saved to {output_dir}")
    print("Interpret the trend; row count alone does not prove dataset sufficiency.")
    return 0


def command_predict(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.model)
    result = predict_frame(bundle, load_csv(args.input))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    outside = int((~result["within_training_domain"]).sum())
    print(f"Saved {len(result)} predictions to {destination}")
    if outside:
        print(f"WARNING: {outside} rows are outside at least one training bound.")
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.model)
    frame = load_csv(args.input) if args.input else synthetic_dataset(1000, args.seed)
    result = benchmark_model(
        bundle,
        frame,
        rows=args.rows,
        repeats=args.repeats,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
    if args.output:
        _write_json(result, Path(args.output))
    return 0


def command_explore(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.model)
    result, feasible_count = explore_candidates(
        bundle,
        candidates=args.candidates,
        max_ripple_ratio=args.max_ripple_ratio,
        seed=args.seed,
        top=args.top,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    print(f"Feasible candidates: {feasible_count} / {args.candidates}")
    print(f"Saved the top {len(result)} screening candidates to {destination}")
    print("Verify shortlisted designs in the original data source before use.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="buck-surrogate",
        description="Build and use a Buck-converter surrogate from a stable CSV schema.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate transparent synthetic data")
    generate.add_argument("--output", required=True)
    generate.add_argument("--samples", type=int, default=20000)
    generate.add_argument("--seed", type=int, default=42)
    generate.set_defaults(handler=command_generate)

    validate = subparsers.add_parser("validate", help="Validate a compatible CSV")
    validate.add_argument("--input", required=True)
    validate.add_argument("--features-only", action="store_true")
    validate.set_defaults(handler=command_validate)

    train = subparsers.add_parser("train", help="Train and evaluate the surrogate")
    train.add_argument("--input", required=True)
    train.add_argument("--output-dir", default="outputs")
    train.add_argument("--test-fraction", type=float, default=0.20)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--max-iter", type=int, default=250)
    train.add_argument("--learning-rate", type=float, default=0.08)
    train.set_defaults(handler=command_train)

    curve = subparsers.add_parser(
        "learning-curve",
        help="Measure held-out performance as training data increase",
    )
    curve.add_argument("--input", required=True)
    curve.add_argument("--output-dir", default="outputs/learning_curve")
    curve.add_argument("--sizes", type=int, nargs="+")
    curve.add_argument("--test-fraction", type=float, default=0.20)
    curve.add_argument("--repeats", type=int, default=3)
    curve.add_argument("--seed", type=int, default=42)
    curve.add_argument("--max-iter", type=int, default=250)
    curve.add_argument("--learning-rate", type=float, default=0.08)
    curve.set_defaults(handler=command_learning_curve)

    predict = subparsers.add_parser("predict", help="Predict compatible design rows")
    predict.add_argument("--model", required=True)
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", required=True)
    predict.set_defaults(handler=command_predict)

    benchmark = subparsers.add_parser("benchmark", help="Measure surrogate inference time")
    benchmark.add_argument("--model", required=True)
    benchmark.add_argument("--input")
    benchmark.add_argument("--rows", type=int, default=10000)
    benchmark.add_argument("--repeats", type=int, default=15)
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--output")
    benchmark.set_defaults(handler=command_benchmark)

    explore = subparsers.add_parser("explore", help="Screen designs inside training bounds")
    explore.add_argument("--model", required=True)
    explore.add_argument("--candidates", type=int, default=100000)
    explore.add_argument("--max-ripple-ratio", type=float, default=0.30)
    explore.add_argument("--seed", type=int, default=42)
    explore.add_argument("--top", type=int, default=10)
    explore.add_argument("--output", default="outputs/top_candidates.csv")
    explore.set_defaults(handler=command_explore)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
