"""Run CTNet through MOABB motor-imagery evaluations."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


DATASETS = {
    "BNCI2014_001": ("moabb.datasets", "BNCI2014_001"),
}

PARADIGMS = {
    "LeftRightImagery": ("moabb.paradigms", "LeftRightImagery"),
    "MotorImagery": ("moabb.paradigms", "MotorImagery"),
}

EVALUATIONS = {
    "TrainTest": None,
    "CrossSession": ("moabb.evaluations", "CrossSessionEvaluation"),
    "WithinSession": ("moabb.evaluations", "WithinSessionEvaluation"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CTNet with MOABB.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="BNCI2014_001")
    parser.add_argument("--paradigm", choices=sorted(PARADIGMS), default="MotorImagery")
    parser.add_argument(
        "--evaluation",
        choices=sorted(EVALUATIONS),
        default="TrainTest",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="*",
        help="Optional subject ids, e.g. --subjects 1 2 3.",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        help="Use the first N subjects for quick smoke runs.",
    )
    parser.add_argument("--train-session", default="0train")
    parser.add_argument("--test-session", default="1test")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/benchmarks"))
    parser.add_argument("--architecture", choices=("paper", "compact"), default="paper")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=72)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.5)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--validation-ratio", type=float, default=0.3)
    parser.add_argument("--augmentation-factor", type=int, default=3)
    parser.add_argument("--augmentation-segments", type=int, default=8)
    parser.add_argument("--input-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--sampling-rate", type=int, default=250)
    parser.add_argument("--fmin", type=float, default=8.0)
    parser.add_argument("--fmax", type=float, default=35.0)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--classifier-dropout", type=float, default=0.5)
    parser.add_argument("--positional-dropout", type=float, default=0.1)
    parser.add_argument("--att-depth", type=int, default=6)
    parser.add_argument("--att-heads", type=int, default=2)
    parser.add_argument("--att-dim", type=int, default=16)
    parser.add_argument("--att-mlp-dim", type=int, default=64)
    parser.add_argument("--n-filters-time", type=int, default=8)
    parser.add_argument("--filter-time-length", type=int, default=64)
    parser.add_argument("--depth-multiplier", type=int, default=2)
    parser.add_argument("--pool-time-length", type=int, default=8)
    parser.add_argument("--pool-time-stride", type=int, default=8)
    parser.add_argument("--second-filter-time-length", type=int, default=16)
    parser.add_argument("--second-pool-time-length", type=int, default=8)
    parser.add_argument("--second-pool-time-stride", type=int, default=8)
    parser.add_argument("--max-position-embeddings", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    moabb = _import_moabb()
    moabb.set_log_level("info" if args.verbose else "warning")
    estimator_cls = _import_estimator()

    dataset_cls = _resolve(DATASETS[args.dataset])
    paradigm_cls = _resolve(PARADIGMS[args.paradigm])
    dataset = dataset_cls()
    if args.subjects:
        dataset.subject_list = args.subjects
    elif args.max_subjects is not None:
        dataset.subject_list = dataset.subject_list[: args.max_subjects]

    paradigm_kwargs = {"fmin": args.fmin, "fmax": args.fmax}
    if args.paradigm == "MotorImagery":
        paradigm_kwargs["n_classes"] = None
    if args.architecture == "paper":
        paradigm_cls = _without_software_filter(paradigm_cls)
        paradigm_kwargs["tmax"] = (args.input_samples - 1) / args.sampling_rate
    paradigm = paradigm_cls(**paradigm_kwargs)

    metric_name = _metric_name(paradigm)
    if args.evaluation == "TrainTest":
        results = _run_train_test(args, dataset, paradigm, estimator_cls, metric_name)
    else:
        evaluation_cls = _resolve(EVALUATIONS[args.evaluation])
        pipelines = {"CTNet": _build_estimator(args, estimator_cls)}
        evaluation = evaluation_cls(
            paradigm=paradigm,
            datasets=[dataset],
            random_state=args.seed,
            n_jobs=args.n_jobs,
            overwrite=True,
            error_score="raise",
        )
        results = evaluation.process(pipelines).rename(columns={"score": metric_name})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / (
        f"moabb_{args.dataset}_{args.paradigm}_{args.evaluation}_"
        f"{args.architecture}.csv"
    )
    summary_path = output_path.with_name(f"{output_path.stem}_summary.csv")
    metric_names = list(
        dict.fromkeys(
            name for name in (metric_name, "accuracy", "cohen_kappa") if name in results
        )
    )
    summary = _summarize_results(results, metric_names)
    results.to_csv(output_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(results)
    print(f"Saved MOABB results to {output_path.resolve()}")
    print(f"Saved summary to {summary_path.resolve()}")
    _print_summary(summary)


def _build_estimator(args: argparse.Namespace, estimator_cls):
    return estimator_cls(
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        validation_ratio=args.validation_ratio,
        augmentation_factor=args.augmentation_factor,
        augmentation_segments=args.augmentation_segments,
        input_samples=args.input_samples,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        sampling_rate=args.sampling_rate,
        dropout=args.dropout,
        classifier_dropout=args.classifier_dropout,
        positional_dropout=args.positional_dropout,
        att_depth=args.att_depth,
        att_heads=args.att_heads,
        att_dim=args.att_dim,
        att_mlp_dim=args.att_mlp_dim,
        n_filters_time=args.n_filters_time,
        filter_time_length=args.filter_time_length,
        depth_multiplier=args.depth_multiplier,
        pool_time_length=args.pool_time_length,
        pool_time_stride=args.pool_time_stride,
        second_filter_time_length=args.second_filter_time_length,
        second_pool_time_length=args.second_pool_time_length,
        second_pool_time_stride=args.second_pool_time_stride,
        max_position_embeddings=args.max_position_embeddings,
        verbose=args.verbose,
    )


def _run_train_test(args, dataset, paradigm, estimator_cls, metric_name):
    try:
        import pandas as pd
        from sklearn.metrics import accuracy_score, cohen_kappa_score, get_scorer
    except ImportError as exc:
        raise SystemExit(
            "pandas and scikit-learn are required for TrainTest benchmarks."
        ) from exc

    rows = []
    scorer = get_scorer(metric_name)
    for subject in dataset.subject_list:
        x, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])
        y = np.asarray(y)
        sessions = metadata["session"].astype(str)
        train_mask = (sessions == args.train_session).to_numpy()
        test_mask = (sessions == args.test_session).to_numpy()
        if not train_mask.any() or not test_mask.any():
            available = ", ".join(sorted(map(str, sessions.unique())))
            raise SystemExit(
                f"Subject {subject} is missing sessions "
                f"{args.train_session!r}->{args.test_session!r}. "
                f"Available sessions: {available}"
            )

        estimator = _build_estimator(args, estimator_cls)
        start = time.perf_counter()
        estimator.fit(x[train_mask], y[train_mask])
        score = scorer(estimator, x[test_mask], y[test_mask])
        predictions = estimator.predict(x[test_mask])
        elapsed = time.perf_counter() - start
        metrics = {
            metric_name: score,
            "cohen_kappa": cohen_kappa_score(y[test_mask], predictions),
        }
        if metric_name != "accuracy":
            metrics["accuracy"] = accuracy_score(y[test_mask], predictions)
        rows.append(
            {
                **metrics,
                "time": elapsed,
                "samples": int(train_mask.sum()),
                "samples_test": int(test_mask.sum()),
                "n_classes": int(np.unique(y).size),
                "subject": subject,
                "train_session": args.train_session,
                "test_session": args.test_session,
                "channels": int(x.shape[1]),
                "n_sessions": int(sessions.nunique()),
                "dataset": getattr(dataset, "code", args.dataset),
                "architecture": args.architecture,
                "best_epoch": estimator.best_epoch_,
                "pipeline": f"CTNet-{args.architecture}",
            }
        )
    return pd.DataFrame(rows)


def _metric_name(paradigm) -> str:
    scoring = paradigm.scoring
    if not isinstance(scoring, str):
        raise ValueError(f"Unsupported MOABB scorer: {scoring!r}")
    return scoring


def _summarize_results(results, metric_names):
    import pandas as pd

    rows = []
    for metric in metric_names:
        values = pd.to_numeric(results[metric], errors="coerce")
        if "subject" in results:
            values = values.groupby(results["subject"]).mean()
        values = values.dropna()
        rows.append(
            {
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "n_subjects": int(values.size),
            }
        )
    return pd.DataFrame(rows, columns=["metric", "mean", "std", "n_subjects"])


def _print_summary(summary) -> None:
    print("\nSummary across subjects (mean +/- sample std):")
    for row in summary.itertuples(index=False):
        print(
            f"{row.metric}: {row.mean:.4f} +/- {row.std:.4f} " f"(n={row.n_subjects})"
        )


def _without_software_filter(paradigm_cls):
    from moabb.datasets.preprocessing import make_fixed_pipeline

    class PaperPreprocessingParadigm(paradigm_cls):
        def _get_raw_pipelines(self):
            return [make_fixed_pipeline(None)]

    return PaperPreprocessingParadigm


def _import_moabb():
    try:
        import moabb
    except ImportError as exc:
        raise SystemExit(
            "MOABB is not installed in the active Python environment. "
            "Install MOABB with your preferred conda/pip workflow, or use "
            "`pip install -e .[benchmark]` as a convenience shortcut."
        ) from exc
    return moabb


def _import_estimator():
    try:
        if __package__:
            from .ctnet_estimator import CtnetSklearnClassifier
        else:
            from ctnet_estimator import CtnetSklearnClassifier
    except ImportError as exc:
        raise SystemExit(
            "Could not import the CTNet benchmark estimator. Original import "
            f"error:\n{exc}"
        ) from exc
    return CtnetSklearnClassifier


def _resolve(target: tuple[str, str]):
    module_name, attribute_name = target
    module = __import__(module_name, fromlist=[attribute_name])
    return getattr(module, attribute_name)


if __name__ == "__main__":
    main()
