# Validation: 27 November 2026

## Person 3 paper-alignment revision

Reference and scope: [paper_mapping.md](paper_mapping.md). Six suites pass in Release
and Debug with AddressSanitizer/UndefinedBehaviorSanitizer. New regressions cover
covisibility, culling, triangulation, fusion, permanent BA outlier removal, fixed-camera
BA, scaled/invalid linear systems, nonidentity loop refinement with false matches,
missing sensor depth, uncertain map depth, reciprocal fusion and the fixed world origin.
The linear solver, BA and loop translation units also compile for x86-64 SSE2;
runtime and timing checks were performed on Apple arm64 NEON. The x86 build was not run.

On the 360-frame synthetic mapping sequence, all three modes close one loop at frame
286, insert 48 keyframes and retain 24. Final keyframe ATE RMSE is **0.0177912 m**;
the loop reduces graph chi-squared from `0.00458575` to `0.000071369`. This driver
supplies synthetic tracking, so it establishes mapping behavior separately from the
image pipeline. Reducing graph cost does not guarantee lower ground-truth ATE at
every step.

| Mapping mode | Total updateMap time | Speedup |
| --- | ---: | ---: |
| serial | 2.92076 s | 1.00x |
| serial_simd | 2.63588 s | 1.11x |
| parallel | 1.79490 s | 1.63x |

The 180-variable Cholesky benchmark measures `0.33060 / 0.16049 ms` for scalar/SIMD
(2.06x), with maximum known-solution error `4.44e-16`. The 60000-observation
residual/Jacobian benchmark measures `1.75210 / 0.71476 / 0.31431 ms` for
scalar/SIMD/parallel; maximum numeric difference is `1.14e-13`. These are single-run
illustrations on this machine, not whole-SLAM speedups or statistical estimates.

Pioneer 360 still tracks **31/830 paired frames**, with **799 LOST**, in both serial
and SIMD. Their exported trajectories are identical. Rigid-aligned translation MSE
on those 31 tracked poses is **0.0000786932 m²**, RMSE **0.00887092 m**. This is
slightly worse than the preceding short-subset result (`0.0000507416 m²`); mapping
agreement with the paper does not itself establish better real-sequence accuracy.
Person 2's initial depth-RANSAC failure and recovery behavior remain unchanged.

Current logs, trajectories, maps, MSE and benchmark CSVs are in
`output/person3-paper/`. Reproduce:

```sh
ctest --test-dir build/integration --output-on-failure
build/integration/slam_mapping_sequence output/person3-paper/synthetic --mode all
build/integration/slam_rgbd data/rgbd_dataset_freiburg2_pioneer_360 output/person3-paper/pioneer-serial --mode serial
build/integration/slam_rgbd data/rgbd_dataset_freiburg2_pioneer_360 output/person3-paper/pioneer-simd --mode serial_simd
build/integration/slam_mapping_bench --solver
build/integration/slam_mapping_bench
```

## Earlier integration baseline

The measurements below predate the mapping revision above and are retained for comparison.

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
