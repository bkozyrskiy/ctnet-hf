"""Aggregate the nine-subject benchmark across training seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


METRICS = ("accuracy", "cohen_kappa")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    paths = [
        path
        for path in sorted(args.input_dir.rglob("*TrainTest_paper_seed-*.csv"))
        if not path.stem.endswith("_summary")
    ]
    if not paths:
        raise SystemExit(f"No per-seed result CSVs found in {args.input_dir}.")
    results = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    results = results[results["seed"].isin(args.seeds)].copy()
    missing = sorted(set(args.seeds) - set(results["seed"].unique()))
    if missing:
        raise SystemExit(f"Missing result files for seeds: {missing}")
    duplicates = results.duplicated(["seed", "subject"])
    if duplicates.any():
        pairs = results.loc[duplicates, ["seed", "subject"]].to_dict("records")
        raise SystemExit(f"Duplicate seed/subject rows: {pairs}")
    expected_subjects = set(range(1, 10))
    for seed in args.seeds:
        actual_subjects = set(results.loc[results["seed"] == seed, "subject"])
        if actual_subjects != expected_subjects:
            raise SystemExit(
                f"Seed {seed} subjects are {sorted(actual_subjects)}; expected 1-9."
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = results.sort_values(["seed", "subject"])
    if "bundle_path" in results:
        results["bundle_path"] = results.apply(
            lambda row: (
                f"bundles/{row['dataset']}/{row['architecture']}/"
                f"seed-{int(row['seed'])}/subject-{int(row['subject'])}"
            ),
            axis=1,
        )
    results_path = args.output_dir / "bnci2014_001_three_seed_results.csv"
    results.to_csv(results_path, index=False)

    seed_rows = []
    subject_rows = []
    overall_rows = []
    for metric in METRICS:
        if metric not in results:
            continue
        for seed, values in results.groupby("seed")[metric]:
            seed_rows.append(
                {
                    "seed": int(seed),
                    "metric": metric,
                    "mean_across_subjects": float(values.mean()),
                    "std_across_subjects": float(values.std(ddof=1)),
                    "n_subjects": int(values.size),
                }
            )
        for subject, values in results.groupby("subject")[metric]:
            subject_rows.append(
                {
                    "subject": int(subject),
                    "metric": metric,
                    "mean_across_seeds": float(values.mean()),
                    "std_across_seeds": float(values.std(ddof=1)),
                    "n_seeds": int(values.size),
                }
            )
        subject_by_seed = results.pivot(index="subject", columns="seed", values=metric)
        subject_means = subject_by_seed.mean(axis=1)
        subject_seed_stds = subject_by_seed.std(axis=1, ddof=1)
        seed_means = subject_by_seed.mean(axis=0)
        overall_rows.append(
            {
                "metric": metric,
                "grand_mean": float(subject_means.mean()),
                "std_across_subject_means": float(subject_means.std(ddof=1)),
                "mean_within_subject_seed_std": float(subject_seed_stds.mean()),
                "min_seed_mean": float(seed_means.min()),
                "max_seed_mean": float(seed_means.max()),
                "n_subjects": int(subject_by_seed.shape[0]),
                "n_seeds": int(subject_by_seed.shape[1]),
            }
        )

    seed_summary = pd.DataFrame(seed_rows)
    subject_summary = pd.DataFrame(subject_rows)
    overall_summary = pd.DataFrame(overall_rows)
    seed_path = args.output_dir / "summary_by_seed.csv"
    subject_path = args.output_dir / "summary_by_subject.csv"
    overall_path = args.output_dir / "summary_overall.csv"
    seed_summary.to_csv(seed_path, index=False)
    subject_summary.to_csv(subject_path, index=False)
    overall_summary.to_csv(overall_path, index=False)
    manifest = {
        "format_version": 1,
        "inputs": [
            {"path": str(path), "sha256": _sha256(path)} for path in paths
        ],
        "n_rows": int(len(results)),
        "seeds": [int(seed) for seed in args.seeds],
        "subjects": sorted(int(subject) for subject in results["subject"].unique()),
        "outputs": [
            {"path": path.name, "sha256": _sha256(path)}
            for path in (results_path, seed_path, subject_path, overall_path)
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(overall_summary.to_string(index=False))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
