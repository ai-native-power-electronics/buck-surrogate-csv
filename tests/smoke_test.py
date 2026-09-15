"""Dependency-light functional check; run directly without pytest."""

from pathlib import Path
from tempfile import TemporaryDirectory

from buck_surrogate.explore import explore_candidates
from buck_surrogate.learning_curve import evaluate_learning_curve
from buck_surrogate.model import load_bundle, predict_frame, save_bundle, train_model
from buck_surrogate.physics import synthetic_dataset
from buck_surrogate.schema import validate_dataframe


def main() -> None:
    frame = synthetic_dataset(800, seed=11)
    report = validate_dataframe(frame, require_targets=True)
    assert report.valid, report.errors

    bundle, metrics, predictions = train_model(frame, max_iter=80, seed=11)
    assert metrics["data"]["rows_total"] == 800
    assert len(predictions) == 160

    predicted = predict_frame(bundle, frame.head(5))
    assert predicted["within_training_domain"].all()
    assert (predicted["predicted_loss_w"] > 0.0).all()

    ranked, feasible_count = explore_candidates(
        bundle,
        candidates=500,
        max_ripple_ratio=0.50,
        seed=11,
        top=5,
    )
    assert feasible_count >= len(ranked)

    _, curve_summary, curve_metadata = evaluate_learning_curve(
        frame,
        sizes=[100, 300],
        repeats=1,
        max_iter=20,
        seed=11,
    )
    assert curve_metadata["training_sizes"] == [100, 300]
    assert len(curve_summary) == 4

    with TemporaryDirectory() as temporary_directory:
        model_path = Path(temporary_directory) / "model.joblib"
        save_bundle(bundle, model_path)
        reloaded = load_bundle(model_path)
        assert reloaded["features"] == bundle["features"]

    print("SMOKE TEST: PASS")
    for target, values in metrics["targets"].items():
        print(f"{target}: R2={values['r2']:.4f}, MAPE={values['mape_pct']:.2f}%")


if __name__ == "__main__":
    main()
