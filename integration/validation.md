# Validation: 27 November 2026

Environment: Apple arm64, Apple Clang 21, C++23, Release, OpenCV 4.11.0.
Imported remote revision: `ddd9bdd` (`pose_estimation_base`), including `master` at
`15659fe`. `git diff --exit-code origin/pose_estimation_base --` reports no changes
to any imported tracked file. All integration source files are additions.

## Automated checks

All four CTest suites pass in Release and in Debug with AddressSanitizer plus
UndefinedBehaviorSanitizer:

- Current upstream pose tests: rigid estimation/RANSAC, SE(3), refinement and tracking.
- Adapter tests: descriptor bits, coordinate/angle/depth conversion, twist order,
  loop transform direction, missing depth/outliers and inlier indexing.
- Mapping modes: numerical Jacobians, scalar/SIMD/threaded parity, BA and pose graph
  convergence, landmark refinement and map correction.
- Pipeline: actual generated images, nonzero known motion, default configuration,
  reciprocal associations, lost tracking/relocalization, loop candidate consistency,
  fusion/culling, exports and invalid input.

The default-configuration image test tracks all 25 frames and inserts two keyframes;
its final translation is approximately 0.161 m against a known 0.160 m translation.
The nine-frame command-line fixture also tracks every frame and writes PLY/TUM outputs.

## Synthetic mapping sequence

The preserved 360-frame room simulation uses noisy features and drifting odometry.
All modes produce one confirmed loop, 44 inserted / 25 retained keyframes, 5568
active landmarks, and identical exported trajectories at their output precision.
The loop reduces graph error from `0.0225912` to `0.000172875`. Final keyframe position
RMSE against the known world frame is `0.0260344 m`.

| Mapping mode | Total updateMap time | Speedup |
| --- | ---: | ---: |
| serial | 2.39764 s | 1.00x |
| serial_simd | 2.36141 s | 1.02x |
| parallel | 1.66847 s | 1.44x |

These are single runs with other build/test activity on the machine. The driver
supplies simulated tracking; this is a mapping check, not an image-tracking benchmark.

## Public TUM recording

Data: [TUM Freiburg1 XYZ](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download),
downloaded from the public dataset server. The run uses all 794 image pairs selected
by this CLI's nearest-unused-depth policy (20 ms tolerance), intrinsics
`fx=517.3 fy=516.5 cx=318.6 cy=255.3`, default 1000 requested features and 5000 depth
units/metre. It uses the pinhole approximation without undistortion.

Every mode tracked 596/794 paired frames (75.1%): one initialization, 570 ordinary
tracked frames and 25 relocalizations. The other 198 are explicitly LOST and omitted
from the exported trajectory. There were 305 inserted keyframes, 38140 created
landmarks including subsequently culled ones, and **no natural loop closures**.
Controlled loop tests above establish the loop path independently.

| Mapping mode | Extraction | Tracking | Mapping |
| --- | ---: | ---: | ---: |
| serial | 15.9847 s | 0.375968 s | 1.88208 s |
| serial_simd | 15.9822 s | 0.373083 s | 1.72632 s |
| parallel | 16.0100 s | 0.419126 s | 2.59702 s |

Parallel mapping is slower for these small jobs: thread creation and reduction
overhead outweigh the benefit. The three trajectories differ by at most `1e-7` in
their exported components. Extraction and pose estimation are upstream serial code.

Rigid SE(3) alignment **without scale fitting**, using unique nearest ground-truth
timestamp matches within 20 ms, gives translation ATE RMSE **0.027447 m**, mean
0.019975 m and maximum 0.101204 m for all three modes. All 596 exported poses match;
the largest actual timestamp gap is 5.025 ms. This measures **only tracked frames**;
it neither scores the 198 losses nor measures orientation error. It does not establish
uninterrupted tracking or paper-level performance.

Reproduce the evaluation (NumPy is needed only for this reporting script):

```sh
python3 tools/evaluate_trajectory.py \
  /private/tmp/rgbd_dataset_freiburg1_xyz/groundtruth.txt \
  build/integration/results/tum-serial/trajectory.txt
```

The evaluator was checked on an exactly rigidly transformed synthetic trajectory,
on a scale mismatch (which must remain nonzero), on unique timestamp associations,
and on missing associations (which must fail).

## Residual/Jacobian kernel

One separate run of `slam_mapping_bench`, 60000 observations and 50 repetitions:

| Mode | Milliseconds per batch | Speedup | Maximum absolute numeric difference |
| --- | ---: | ---: | ---: |
| serial | 1.91587 | 1.00x | 0 |
| serial_simd | 0.72310 | 2.65x | 1.14e-13 |
| parallel | 0.32263 | 5.94x | 0 |

These are isolated kernel results. They must not be reported as whole-pipeline speedups.

Local copies of the real/synthetic maps, trajectories and run logs are in
`build/integration/results/`. That directory is an ignored build artifact and is not
part of the source changes. The original pre-merge files are preserved separately at
`/private/tmp/slam-before-merge-20261127-174852`.
