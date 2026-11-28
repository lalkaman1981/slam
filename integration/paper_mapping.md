# Person 3: correspondence with ORB-SLAM

Reference: [Mur-Artal, Montiel and Tardós, ORB-SLAM (2015)](https://arxiv.org/abs/1502.00956).
The [official implementation](https://github.com/raulmur/ORB_SLAM) is already C++.
The comparison uses commit `ce199650a25653808f96b83557333bce3461d29f`, not ORB-SLAM2/3.
Our module implements the algorithms with native C++ containers and our own numerical
solvers; it does not link the reference's ROS, DBoW2, Eigen or g2o libraries.

## Mapping stages

| Paper / reference routine | Our implementation |
| --- | --- |
| V-E, `Tracking::NeedNewKeyFrame` | `shouldInsertKeyframe`: overlap, tracked points, frame intervals and mapper availability; culled keyframes do not count as active map entries |
| VI-A, `LocalMapping::ProcessNewKeyFrame` | `insertKeyframe`, `associateObservations`, `updateCovisibility`: reciprocal observations, representative descriptors, all connection weights and a separate selection of strong neighbors |
| VI-B, `MapPointCulling` | `cullLandmarks`: found/visible ratio and observation support for recent points |
| VI-C, `CreateNewMapPoints`, `SearchInNeighbors` | `triangulateNewLandmarks`, `fuseLandmarksInNeighbors`: epipolar, parallax, positive-depth, reprojection and scale checks, then bidirectional neighbor fusion |
| VI-D, `Optimizer::LocalBundleAdjustment` | `buildLocalBAProblem`, `localBundleAdjustment`: local cameras/points free, external observers fixed, origin anchored, five robust iterations, permanent outlier removal, ten further robust iterations |
| VI-E, `KeyFrameCulling` | `cullKeyframes`: redundant observations at comparable or finer scale; spanning tree and trajectory references survive culling |
| VII-A, `LoopClosing::DetectLoop` | `findLoopCandidates`: connected-keyframe exclusion, covisibility score accumulation and temporal group consistency |
| VII-B, `ComputeSim3`, `OptimizeSim3` | `computeLoopGeometry`, `refineLoopTransform`: map-point geometry, bidirectional guided matching, two-stage robust reprojection refinement and neighborhood projection |
| VII-C/D, `CorrectLoop`, `OptimizeEssentialGraph` | `correctLoop`, `buildEssentialGraph`, `optimizePoseGraph`, `applyMapCorrection`: fuse duplicates, add the verified loop constraint, optimize graph and move points through their reference cameras |

The corresponding source is in
[LocalMapping.cc](https://github.com/raulmur/ORB_SLAM/blob/ce199650a25653808f96b83557333bce3461d29f/src/LocalMapping.cc),
[KeyFrame.cc](https://github.com/raulmur/ORB_SLAM/blob/ce199650a25653808f96b83557333bce3461d29f/src/KeyFrame.cc),
[MapPoint.cc](https://github.com/raulmur/ORB_SLAM/blob/ce199650a25653808f96b83557333bce3461d29f/src/MapPoint.cc),
[LoopClosing.cc](https://github.com/raulmur/ORB_SLAM/blob/ce199650a25653808f96b83557333bce3461d29f/src/LoopClosing.cc),
[ORBmatcher.cc](https://github.com/raulmur/ORB_SLAM/blob/ce199650a25653808f96b83557333bce3461d29f/src/ORBmatcher.cc)
and [Optimizer.cc](https://github.com/raulmur/ORB_SLAM/blob/ce199650a25653808f96b83557333bce3461d29f/src/Optimizer.cc).

## Native C++ and execution modes

The module uses value types, fixed-size `std::array` matrices, contiguous `std::vector`
storage and `std::span` views. Local BA uses our damped Schur-complement solver;
the essential graph uses our damped Cholesky solver. No ready-made BA/graph optimizer
is called. Numerical graph Jacobians remain suitable for the project's small maps.

`serial_simd` uses explicit AArch64 NEON or x86 SSE2 double-precision intrinsics for
paired projections/Jacobians, matrix block products, Schur assembly and contiguous
dot products in Cholesky. Other GCC/Clang targets use portable vector extensions, and
other compilers use a scalar two-lane fallback, so the mode runs on every platform.
Loads accept unaligned storage and handle scalar tails.
There are no approximate reciprocals or fast-math flags. Scalar/SIMD sums can differ
by rounding; tests compare known solutions and geometric results with tolerances.
`parallel` uses bounded `std::jthread` partitions, private accumulators and ordered
merges; shared map changes occur after workers finish.

## Deliberate differences from the monocular system

This reproduces person 3's mapping stages within the team's RGB-D interfaces, not
the entire monocular implementation or its numerical output bit for bit:

- Depth initializes the metric map and supplies optional virtual-stereo BA residuals.
  Monocular initialization belongs outside this module. RGB-D scale conventions and
  the close-depth keyframe rule are retained.
- Loop correction fixes scale and optimizes SE(3), whereas the monocular paper uses
  Sim(3). The original world origin remains the graph anchor.
- Candidate scoring reuses person 1's descriptor similarity, not the learned BoW
  vocabulary. Loop hypotheses reuse person 2's rigid RANSAC; the new mapping adapter
  supplies mapped camera-coordinate points rather than requiring fresh depth values.
- Team-specific loop spatial-distribution checks and the pipeline's lower
  initialization/tracked-point thresholds remain explicit configuration choices.
- Mapping executes synchronously between frames. Parallel numerical work does not
  introduce the original system's asynchronous tracking/mapping/loop-closing threads.

Person 1's feature extraction and person 2's frame tracking/relocalization sources
are unchanged. Mapping improvements therefore do not remove the Pioneer sequence's
initial depth-RANSAC tracking failure. Coverage must accompany trajectory MSE.
