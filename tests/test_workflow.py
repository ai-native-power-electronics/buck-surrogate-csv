from buck_surrogate.explore import explore_candidates
from buck_surrogate.learning_curve import evaluate_learning_curve
from buck_surrogate.model import predict_frame, train_model
from buck_surrogate.physics import synthetic_dataset


def test_small_end_to_end_workflow():
    frame = synthetic_dataset(800, seed=11)
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
    assert len(ranked) <= 5


def test_learning_curve_uses_requested_sizes_and_fixed_test_set():
    frame = synthetic_dataset(600, seed=17)
    raw, summary, metadata = evaluate_learning_curve(
        frame,
        sizes=[100, 300],
        repeats=1,
        max_iter=20,
        seed=17,
    )

    assert metadata["rows_fixed_test"] == 120
    assert metadata["training_sizes"] == [100, 300]
    assert set(raw["training_rows"]) == {100, 300}
    assert set(summary["target"]) == {"loss_w", "ripple_current_a"}
    assert "index" not in summary.columns
