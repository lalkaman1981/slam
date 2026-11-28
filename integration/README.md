# Integrated RGB-D SLAM

The integration incorporates `origin/pose_estimation_base` at `ddd9bdd`, which already
contains `origin/master` at `15659fe`. Every file from that remote tree is retained
unchanged, including its root CMake project and placeholder `main.cpp`. Build the
working pipeline using this separate CMake entry point.

Person 3's original `headers/BA` implementation now lives in `src/our/mapping` with
`.hpp` headers, the existing four-space source style and camelCase APIs consistent
with the pose-estimation module. The original synthetic driver is preserved and
adapted as `examples/mapping_sequence.cpp`.

## Build and test

Requires C++23, CMake 3.20+, threads, and OpenCV 4 with `core`, `imgproc`, `imgcodecs`.
Boost is unnecessary for this executable. OpenCV supplies image operations and the
upstream matrix containers; mapping uses its own Schur-complement bundle adjustment
and Cholesky pose-graph solver. There is no Ceres/g2o/ready-made BA or graph optimizer.

From the repository root:

```sh
cmake -S integration -B build/integration -DCMAKE_BUILD_TYPE=Release
cmake --build build/integration -j 8
ctest --test-dir build/integration --output-on-failure
```

If OpenCV is outside the normal search path, add
`-DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv4`. In this workspace a minimal OpenCV
4.11 build was installed in `/private/tmp/slam-opencv`, so the actual configure command is:

```sh
cmake -S integration -B build/integration \
  -DOpenCV_DIR=/private/tmp/slam-opencv/lib/cmake/opencv4 \
  -DCMAKE_BUILD_TYPE=Release
```

That temporary dependency can be removed by the OS; use a permanent OpenCV installation
for ongoing development. To check memory and undefined behavior:

```sh
cmake -S integration -B build/integration-asan \
  -DOpenCV_DIR=/private/tmp/slam-opencv/lib/cmake/opencv4 \
  -DCMAKE_BUILD_TYPE=Debug -DSLAM_SANITIZERS=ON
cmake --build build/integration-asan -j 8
ctest --test-dir build/integration-asan --output-on-failure
```

The test suites cover the current upstream pose implementation, conversion conventions,
finite-difference BA Jacobians, SIMD/parallel parity and convergence, real image extraction
through tracking and mapping, default settings, lost tracking and relocalization,
loop candidate consistency and correction, landmark fusion/culling, and exports.
The remote's old `pose_estimation_core_tests.cpp` references headers absent from that
branch and is retained unchanged; the current `pose_estimation_tests.cpp` is built.

## Run on RGB-D images

Input is a TUM-style directory with `rgb.txt`, `depth.txt`, and their image files.
Each list contains `timestamp relative/image/path` rows, optionally preceded by `#`
comments. Lists must be strictly increasing. RGB frames are greedily paired with the
nearest unused depth frame within 20 ms. Skipped/unpaired images do not enter SLAM.
Provide the sequence's pinhole intrinsics and registered 16-bit depth images.

```sh
build/integration/slam_rgbd /path/to/dataset /path/to/output \
  --fx 517.3 --fy 516.5 --cx 318.6 --cy 255.3 \
  --depth-scale 5000 --mode serial
```

Image dimensions come from the first RGB frame. Depth units default to 5000 per metre;
another `--depth-scale` is converted at the input boundary, with invalid depths kept zero.
Additional options are `--features 1000`, `--max-frames 100` and
`--min-initial-points 60`. The upstream grid filter may return substantially fewer
features than requested. The integration defaults require 60 valid depth points to
initialize and more than 15 associated map points for keyframe insertion.

Outputs:

- `map.ply`: active 3D landmarks in the first camera's coordinate system.
- `trajectory.txt`: successfully tracked frames in TUM camera-to-world format.
- `keyframe_trajectory.txt`: active keyframes in the same format.
- `frames.csv`: initialization, tracking/relocalization/loss status, counts and timings.

Lost frames are omitted from the trajectory; they are explicitly marked in `frames.csv`.
All tracked frame poses follow later map corrections through their reference keyframes.
The CLI exits nonzero if input/export fails or fewer than two frames are tracked.

A reproducible generated-image smoke test needs no downloaded dataset:

```sh
python3 tests/generate_rgbd_fixture.py /tmp/slam-rgbd-fixture
build/integration/slam_rgbd /tmp/slam-rgbd-fixture /tmp/slam-fixture-output \
  --fx 300 --fy 300 --cx 191.5 --cy 127.5
```

## Module boundaries and execution modes

`RgbdPipeline` calls the unchanged upstream extractor, frame matcher, depth RANSAC,
pose refinement and relocalization. It carries verified landmark associations forward,
searches the local map by projection, geometrically filters observations, and invokes
local mapping and loop closing. Keyframe images support relocalization; culled
keyframes release their cached images.

`integration_adapters` converts the existing types without imposing new shared types
on teammates. Upstream features use pyramid-level integer coordinates, radian angles
and four 64-bit descriptor words. Mapping stores original-image coordinates, degree
angles, depth per feature and 32 descriptor bytes. Both use world-to-camera poses.
Mapping's rotation-first twists are explicitly permuted to the upstream translation-first
convention. Shared SE(3) operations delegate to person 2's code.

Loop matching/scoring reuse person 1; loop hypotheses call person 2's RANSAC on
existing map points expressed in the candidate/current cameras. Person 3 then performs
reciprocal guided matching and robust bidirectional reprojection refinement. These
steps support triangulated landmarks without stored sensor-depth samples.

The CLI's `--mode` selects **person 3's mapping implementation**. The remote only
provides serial extraction and tracking, and those remain serial in every run.

| Mapping mode | Implementation |
| --- | --- |
| `serial` | Scalar observations, optimization and map operations |
| `serial_simd` | Two-lane FP64 vectors for projection, residuals/Jacobians, loop refinement, Schur products and Cholesky dot products; scalar tails |
| `parallel` | Independent observation/point/edge/candidate jobs; private BA accumulators and deterministic ordered reductions; shared-map writes after workers join |

`serial_simd` runs on every platform. The backend is chosen at compile time:

| Target | Backend (`simdBackendName()`) |
| --- | --- |
| AArch64 (GCC, Clang, MSVC) | `neon`: explicit NEON intrinsics |
| x86-64, or x86 with SSE2 (GCC, Clang, MSVC) | `sse2`: explicit SSE2 intrinsics |
| Other GCC/Clang targets (32-bit ARM, RISC-V, POWER, WebAssembly, ...) | `vector-extensions`: portable GCC/Clang vectors, lowered to the target's SIMD unit or to scalar code |
| Other compilers | `scalar`: same paired algorithm on two scalars |

`slam_rgbd`, `slam_mapping_sequence` and `slam_mapping_bench` print the backend on
stderr. Configure with `-DSLAM_SIMD_BACKEND=portable` or `scalar` to force the
portable or scalar backend; `auto` (the default) uses the table above. Forcing
`scalar` separates the effect of the paired data layout from the effect of SIMD lanes.
Checked so far: all three backends on Apple arm64 and on x86-64 (Rosetta), and a
compile of the portable backend for 32-bit ARM. MSVC, RISC-V, POWER and WebAssembly
builds have not been run.

Parallel execution uses at most eight workers and runs small jobs serially to limit
thread overhead. Output parity is checked numerically; tiny floating-point differences
are expected from grouping sums differently.

```sh
# End-to-end mapping on recorded synthetic features/poses with known ground truth:
build/integration/slam_mapping_sequence /tmp/slam-map-benchmark --mode all

# Isolated residual/Jacobian throughput with numerical checks:
build/integration/slam_mapping_bench

# Native scalar/SIMD Cholesky solve with a known solution:
build/integration/slam_mapping_bench --solver
```

`slam_mapping_sequence` preserves the original 360-frame room trajectory. It supplies
simulated tracking, so use `slam_rgbd` to exercise actual extraction and pose estimation.
The benchmark reports kernel speedups, not whole-SLAM speedups. Use the same input,
configuration, compiler and machine when comparing modes.

## Paper scope and limits

The design follows local mapping, covisibility, keyframe/point culling and loop correction
from [ORB-SLAM (2015)](https://arxiv.org/abs/1502.00956). This project follows the team's
RGB-D adaptation: metric depth initialization and fixed-scale SE(3), with virtual stereo
depth residuals. It is not a reproduction of the paper's monocular initializer, learned
BoW vocabulary, or Sim(3) scale correction. Place recognition uses descriptor similarity
over old keyframes, and the dense graph solver is intended for small experiments.

Person 3's implementation has been checked against the authors' original C++ source.
See [paper_mapping.md](paper_mapping.md) for the routine-by-routine correspondence,
the corrected mapping behavior and the explicit RGB-D/team-interface differences.
This mapping revision does not replace person 2's depth-dependent tracking or recovery.

Input must already be registered to a pinhole camera; the integration does not estimate
calibration or remove lens distortion. Real sequence tracking can be lost; successful
synthetic tests do not establish paper-level accuracy on arbitrary recordings. See
`validation.md` for the measured results and their limits.
