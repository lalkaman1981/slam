#!/usr/bin/env python3
"""Evaluate tracked TUM camera positions using rigid, no-scale aligned ATE.

Requires NumPy. Associates timestamps one-to-one by smallest absolute difference
within --max-difference, then fits the estimated translations to ground truth by
SVD. This measures the matched tracked subset; it does not score lost frames or
orientation error. Input poses must be camera-to-world, as in TUM trajectories.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def read_trajectory(path: Path) -> np.ndarray:
    """Read strictly increasing finite TUM rows: timestamp tx ty tz qx qy qz qw."""
    rows = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            text = line.split("#", 1)[0].strip()
            if not text:
                continue
            try:
                row = [float(value) for value in text.split()]
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: expected numeric TUM pose") from error
            if len(row) != 8 or not np.all(np.isfinite(row)):
                raise ValueError(f"{path}:{line_number}: expected eight finite TUM values")
            if rows and row[0] <= rows[-1][0]:
                raise ValueError(f"{path}:{line_number}: timestamps must be strictly increasing")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no poses")
    return np.asarray(rows, dtype=np.float64)


def associate_timestamps(reference_times: np.ndarray, estimate_times: np.ndarray,
                         max_difference: float) -> tuple[np.ndarray, np.ndarray]:
    """Greedily choose closest unused pairs, with deterministic tie breaking."""
    if not np.isfinite(max_difference) or max_difference <= 0:
        raise ValueError("maximum timestamp difference must be finite and positive")
    candidates = []
    for estimate_index, timestamp in enumerate(estimate_times):
        start = int(np.searchsorted(reference_times, timestamp - max_difference, side="left"))
        stop = int(np.searchsorted(reference_times, timestamp + max_difference, side="right"))
        for reference_index in range(start, stop):
            difference = abs(float(reference_times[reference_index] - timestamp))
            if difference <= max_difference:
                candidates.append((difference, estimate_index, reference_index))
    used_reference, used_estimate, matches = set(), set(), []
    for _, estimate_index, reference_index in sorted(candidates):
        if reference_index in used_reference or estimate_index in used_estimate:
            continue
        used_reference.add(reference_index)
        used_estimate.add(estimate_index)
        matches.append((estimate_index, reference_index))
    matches.sort()
    return (np.asarray([pair[1] for pair in matches], dtype=int),
            np.asarray([pair[0] for pair in matches], dtype=int))


def evaluate(reference: np.ndarray, estimate: np.ndarray, max_difference: float = 0.02) -> dict:
    reference_indices, estimate_indices = associate_timestamps(reference[:, 0], estimate[:, 0], max_difference)
    if len(reference_indices) < 3:
        raise ValueError(f"need at least three timestamp pairs for rigid alignment; found {len(reference_indices)}")
    target = reference[reference_indices, 1:4]
    source = estimate[estimate_indices, 1:4]
    target_center = target.mean(axis=0)
    source_center = source.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_transpose = np.linalg.svd(covariance)
    reflection = np.eye(3)
    reflection[2, 2] = -1.0 if np.linalg.det(right_transpose.T @ left.T) < 0.0 else 1.0
    rotation = right_transpose.T @ reflection @ left.T
    translation = target_center - rotation @ source_center
    aligned = source @ rotation.T + translation
    errors = np.linalg.norm(aligned - target, axis=1)
    differences = np.abs(reference[reference_indices, 0] - estimate[estimate_indices, 0])
    return {
        "reference_frames": int(len(reference)),
        "estimated_frames": int(len(estimate)),
        "matched_frames": int(len(errors)),
        "timestamp_tolerance_seconds": max_difference,
        "largest_matched_time_difference_seconds": float(differences.max()),
        "alignment": "rigid SE(3); no scale fitting",
        "scope": "matched tracked frames only; lost frames and orientation error are not scored",
        "rmse_metres": float(np.sqrt(np.mean(errors * errors))),
        "mean_metres": float(errors.mean()),
        "median_metres": float(np.median(errors)),
        "max_metres": float(errors.max()),
        "rotation_estimate_to_reference": rotation.tolist(),
        "translation_estimate_to_reference_metres": translation.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("groundtruth", type=Path)
    parser.add_argument("estimate", type=Path)
    parser.add_argument("--max-difference", type=float, default=0.02,
                        help="maximum timestamp separation in seconds (default: 0.02)")
    parser.add_argument("--json", action="store_true", help="emit full evaluation as JSON")
    arguments = parser.parse_args()
    try:
        result = evaluate(read_trajectory(arguments.groundtruth), read_trajectory(arguments.estimate),
                          arguments.max_difference)
    except (OSError, ValueError, np.linalg.LinAlgError) as error:
        print(f"evaluate_trajectory: {error}", file=sys.stderr)
        return 1
    if arguments.json:
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        print(f"Matched {result['matched_frames']}/{result['estimated_frames']} estimated frames "
              f"to {result['reference_frames']} reference poses (within {arguments.max_difference:g} s).")
        print(f"ATE RMSE: {result['rmse_metres']:.6f} m")
        print(f"Mean: {result['mean_metres']:.6f} m; median: {result['median_metres']:.6f} m; "
              f"maximum: {result['max_metres']:.6f} m")
        print("Rigid SE(3) alignment, no scale fitting; matched tracked subset only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
