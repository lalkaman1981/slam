# Pipeline benchmarks: our implementation against OpenCV

This document explains how the benchmarks in `bench/` work, how to run them, and how the compared methods differ. The timing method is the one from lab3 (`modern-cpp26-27-lab3-optional-group-11`).

## Overview

The setup has two parts.

1. **C++ binaries.** Each operation has one source file, `bench/<operation>.cpp`. CMake builds it once per variant into `bench/compiled/`:
   - `<operation>_serial` runs our code in serial mode.
   - `<operation>_simd` runs our code in SIMD mode (mapping only).
   - `<operation>_opencv` runs the closest OpenCV counterpart.

   A binary reads an input file and prepares the data. It then times one call between two `get_current_time_fenced()` calls and writes the result to `sink`. It prints the elapsed time in nanoseconds to stdout and nothing else.

2. **Python driver.** `bench/run_benchmarks.py` generates synthetic inputs and runs every binary several times. It stores the times in `bench/results/times.csv` and draws one chart per operation.

The binaries and all of our code they measure are compiled with `-O2`. Persons 1 and 2 have serial implementations only, so their charts show our serial code against OpenCV. Person 3's mapping chart adds the SIMD mode.

## The four charts

| Chart | Ours | OpenCV | Input size |
|---|---|---|---|
| `results/features.png` | `extractFeatures` on two frames + `match_frames` | `cv::ORB::create(1000)->detectAndCompute` on two frames + `cv::BFMatcher(NORM_HAMMING, crossCheck=true)` | image side, 500 to 3900 px |
| `results/pose.png` | `estimatePoseRansac` + `refinePose` | `cv::estimateAffine3D` (RANSAC) + rotation by SVD + `cv::solvePnPRefineLM` | correspondences, 500 to 10,000 |
| `results/mapping.png` | `solveCholesky`, serial and SIMD | `cv::solve(..., DECOMP_CHOLESKY)` | unknowns, 60 to 1200 (6 per keyframe) |
| `results/pipeline.png` | per frame: `extractFeatures`, `match_frames`, `trackFrame`; then one pose-graph solve with `solveCholesky` | per frame: ORB, BFMatcher, `estimateAffine3D`, `solvePnPRefineLM`; then the same pose-graph system solved with `cv::solve` | image side, 500 to 3700 px in steps of 400, 8 frames per sequence |

## How the methods differ

These differences matter when reading the charts. Accuracy figures come from a check run on the generated inputs on November 28, 2026.

| Step | Ours | OpenCV | Effect |
|---|---|---|---|
| Feature extraction | FAST-9 with a Harris score, grid selection, 8 levels at scale 1.2, our BRIEF pattern (person 1) | FAST with Harris ranking, its own pyramid and orientation, hand-written NEON kernels | OpenCV keeps close to the 1000 requested features (946 to 1000 per frame); ours returns 350 to 520 after the grid filter. OpenCV does more work per frame and is still faster. |
| Matching | mutual nearest neighbour, ratio test, orientation-histogram filter | brute-force Hamming with cross-check only | OpenCV returns about 500 matches per pair, ours about 120 to 155. |
| 3D-3D pose | rigid 6-DOF fit (Horn), at most 500 RANSAC iterations, refit on inliers (person 2) | 12-DOF affine fit with its own iteration rule; the 3x3 part is projected to a rotation | An affine fit needs points that are not coplanar, so the synthetic scene is a three-plane roof. |
| 3D-2D refinement | Levenberg-Marquardt with Huber weights, at most 10 iterations | Levenberg-Marquardt without a robust kernel, at most 10 iterations | With 10% outlier observations, ours ends within 1 mm of the true pose; OpenCV ends 0.17 m off. |
| Cholesky | our row-oriented solver (`linalg.cpp`), scalar or two-lane NEON dot products | `cv::solve` hands the factorization to LAPACK; on macOS that is Apple's Accelerate | All three solutions agree to about 1e-11. Accelerate is a separately optimized library, so `-O2` does not apply to it. At 1200 unknowns its process user time did not exceed wall time, so it does not appear to use extra threads. |
| Pipeline mapping | `linearizePoseGraph` (ours) + `solveCholesky` | `linearizePoseGraph` (ours) + `cv::solve` | OpenCV has no SLAM mapping; only the linear solver differs. Both chains track all 7 frame pairs within a few centimetres. |

In the pipeline chain, ours passes the previous frame's matched points to `trackFrame` as the local map, so refinement runs. The OpenCV chain refines on the RANSAC inliers, the usual practice when there is no robust kernel.

## Synthetic data

`run_benchmarks.py` writes everything to `bench/inputs/` once per (kind, size, seed) and reuses it. It needs only numpy; PNG files are written with the standard library's `zlib`.

- **Images:** a camera views a textured roof of three planes. The camera moves 4 cm sideways and 1.5 cm down and turns 0.6 degrees per frame. The texture is evaluated per pixel, so every resolution shows the same scene. Intrinsics scale with the image (`fx = fy = 0.8 · side`), so the field of view is the same at every size. Depth is stored as 16-bit PNG at 5000 units per metre, as person 2's code expects.
- **Pose:** random points in a 640x480 view under a known rigid motion. Pairs have 5 mm noise and 30% outliers; observations have 1 px noise and 10% outliers.
- **Cholesky:** the normal matrix of a pose graph with chain and loop edges and a prior on the first keyframe.

## Files

| path | purpose |
|---|---|
| `src/utils/time_measurements.cpp` | timing helpers, copied unchanged from lab3 |
| `bench/options_parser/` | command line parsing (`-i` input file), copied from lab3; only the `-t` help text differs |
| `bench/bench.hpp` | variant to execution mode, `sink`, input readers, OpenCV helpers |
| `bench/<operation>.cpp` | one benchmark per file: `features`, `pose`, `mapping`, `pipeline` |
| `bench/CMakeLists.txt` | builds each operation per variant at `-O2`; included from `integration/CMakeLists.txt` |
| `bench/run_benchmarks.py` | generates inputs, runs binaries, writes CSV and charts |
| `bench/inputs/` | generated inputs (ignored by git) |
| `bench/compiled/` | built binaries (ignored by git) |
| `bench/results/` | `times.csv` and the four `.png` charts |

The other benchmarks in `bench/` (`mapping_bench.cpp`, `pose_estimation_bench.cpp`) are separate and unchanged.

## Requirements

- CMake 3.20 or newer and a compiler with C++23 support.
- Boost.Program_options. On macOS: `brew install boost`.
- OpenCV 4.11 with `core`, `imgproc`, `imgcodecs`, `flann`, `features2d` and `calib3d`. Without `features2d` and `calib3d` the integration still builds and skips the benchmarks.
- Python 3 with `numpy` and `matplotlib`.

The local OpenCV under `/private/tmp` was rebuilt for these benchmarks. macOS can clear `/private/tmp`, so rebuild it from source if it disappears:

```
cmake -S opencv-4.11.0 -B slam-opencv-build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_LIST=core,imgproc,imgcodecs,flann,features2d,calib3d \
  "-DCMAKE_CXX_FLAGS_RELEASE=-O2 -DNDEBUG" "-DCMAKE_C_FLAGS_RELEASE=-O2 -DNDEBUG" \
  -DCMAKE_INSTALL_PREFIX=/private/tmp/slam-opencv
cmake --build slam-opencv-build -j && cmake --install slam-opencv-build
```

## Building

```
cmake -S integration -B build/integration -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/private/tmp/slam-opencv/lib/cmake/opencv4
cmake --build build/integration -j
```

The binaries are written to `bench/compiled/`, independent of the build directory. Their compile commands end with `-O2`, which overrides the `-O3` that Release adds before it.

## Running

```
cd bench
python3 run_benchmarks.py --runs 5
```

- `--runs` sets the number of measured runs per case. One extra warm-up run is made first and not counted.
- `--operations` limits the run, for example `--operations mapping pose`.
- `--sizes` replaces the sizes of an operation, for example `--sizes features=500,1000 pipeline=500`.
- `--seed` sets the random seed (default 42).
- `--results` sets the output folder inside `bench/` (default `results`).

The script prints the median time of every case and writes `results/times.csv` with the columns `operation, variant, size, run, time_ns`, plus the four charts.

A single binary can also be run by hand:

```
bench/compiled/mapping_simd -i bench/inputs/system_600_42.bin
```

## Adding an operation

1. Write `bench/<operation>.cpp`. Read the input and prepare everything before `get_current_time_fenced()`. Time only the measured call, add its result to `sink` afterwards, and print `to_ns(total_time)`. Put the OpenCV counterpart under `#ifdef BENCH_OPENCV`.
2. Add the operation and its variants to `bench/CMakeLists.txt`.
3. Add its input generator, sizes, variants and chart labels to `run_benchmarks.py`.
4. Add a row to the method-differences table above.
