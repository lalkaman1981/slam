// Person 2 module tests on synthetic data. Plain checks (not assert: Release defines NDEBUG).
// Run: ctest --test-dir build   or   ./build/slam_pose_estimation_core_tests

#include "camera.hpp"
#include "geometry.hpp"
#include "pose_refine.hpp"
#include "ransac.hpp"
#include "tracking.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <random>
#include <vector>

using namespace slam;

static int gFailures = 0;
static int gChecks   = 0;

#define CHECK(cond) do { ++gChecks; if (!(cond)) { ++gFailures; \
    std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } } while (0)

#define CHECK_NEAR(a, b, tol) do { ++gChecks; const double _a = (a), _b = (b); \
    if (!(std::abs(_a - _b) <= (tol))) { ++gFailures; \
    std::printf("FAIL %s:%d: %s = %.17g vs %s = %.17g (tol %g)\n", \
                __FILE__, __LINE__, #a, _a, #b, _b, double(tol)); } } while (0)

// |a - b| <= tol * max(1, |b|)
#define CHECK_REL(a, b, tol) CHECK_NEAR(a, b, (tol) * std::max(1.0, std::abs(double(b))))

// ---- helpers --------------------------------------------------------------------

static const CameraIntrinsics kCam{525.0, 525.0, 319.5, 239.5, 640, 480};   // TUM fr1-like

static double poseDistance(const Pose& A, const Pose& B) {
    double d = 0;
    for (int i = 0; i < 9; ++i) d = std::max(d, std::abs(A.R.m[i] - B.R.m[i]));
    d = std::max({d, std::abs(A.t.x - B.t.x), std::abs(A.t.y - B.t.y), std::abs(A.t.z - B.t.z)});
    return d;
}

static Pose randomPose(std::mt19937_64& rng, double maxAngle, double maxTrans) {
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    Vec3 axis{U(rng), U(rng), U(rng)};
    const double n = std::sqrt(dot(axis, axis));
    const double angle = maxAngle * std::abs(U(rng));
    const Vec6 xi{maxTrans * U(rng), maxTrans * U(rng), maxTrans * U(rng),
                  angle * axis.x / n, angle * axis.y / n, angle * axis.z / n};
    return expSE3(xi);
}

// Points in front of a camera at the origin, inside the image.
static std::vector<Vec3> randomPointsInView(std::mt19937_64& rng, int n) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<Vec3> pts;
    for (int i = 0; i < n; ++i) {
        const double z = 1.0 + 3.0 * U(rng);
        const Pixel px{20 + U(rng) * (kCam.width - 40), 20 + U(rng) * (kCam.height - 40)};
        pts.push_back(backProject(px, z, kCam));
    }
    return pts;
}

// pairs with b = T a for inliers; outliers get a random b. Returns the outlier flags.
static std::vector<PointPair3D> makePairs(std::mt19937_64& rng, const Pose& T, int n,
                                          double outlierRatio, double noise,
                                          std::vector<uint8_t>& isInlier) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double> G(0.0, noise);
    const auto pts = randomPointsInView(rng, n);
    std::vector<PointPair3D> pairs;
    isInlier.assign(n, 0);
    for (int i = 0; i < n; ++i) {
        Vec3 b = T * pts[i];
        if (U(rng) < outlierRatio) {
            b = Vec3{4 * U(rng) - 2, 4 * U(rng) - 2, 1 + 3 * U(rng)};
        } else {
            isInlier[i] = 1;
            b = b + Vec3{G(rng), G(rng), G(rng)};
        }
        pairs.push_back({pts[i], b, size_t(i), size_t(i)});
    }
    return pairs;
}

static std::vector<MapObservation> makeObservations(std::mt19937_64& rng, const Pose& T_cw, int n,
                                                    double pixelNoise) {
    std::normal_distribution<double> G(0.0, pixelNoise);
    const Pose T_wc = inversePose(T_cw);
    std::vector<MapObservation> obs;
    for (const Vec3& pc : randomPointsInView(rng, n)) {
        const Pixel px = *project(pc, kCam);
        obs.push_back({T_wc * pc, {px.u + G(rng), px.v + G(rng)}, 1.0});
    }
    return obs;
}

static double reprojectionRms(std::span<const MapObservation> obs, const Pose& T_cw) {
    double s = 0;
    for (const auto& o : obs) {
        const Pixel p = *project(T_cw * o.pointWorld, kCam);
        s += (p.u - o.pixel.u) * (p.u - o.pixel.u) + (p.v - o.pixel.v) * (p.v - o.pixel.v);
    }
    return std::sqrt(s / double(obs.size()));
}

// ---- camera -----------------------------------------------------------------------

static void testProjectBackProjectRoundTrip() {
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (int i = 0; i < 1000; ++i) {
        const Pixel px{U(rng) * 640, U(rng) * 480};
        const double z = 0.2 + 7 * U(rng);
        const Vec3 p = backProject(px, z, kCam);
        CHECK_NEAR(p.z, z, 1e-15);
        const auto back = project(p, kCam);
        CHECK(back.has_value());
        CHECK_NEAR(back->u, px.u, 1e-9);
        CHECK_NEAR(back->v, px.v, 1e-9);
    }
    // principal point <- optical axis; behind the camera -> nullopt
    const auto c = project({0, 0, 2}, kCam);
    CHECK(c && c->u == kCam.cx && c->v == kCam.cy);
    CHECK(!project({0, 0, -1}, kCam).has_value());
    CHECK(!project({0, 0, 0}, kCam).has_value());
}

static void testReadValidDepth() {
    std::vector<uint16_t> raw(4 * 3, 0);
    raw[1 * 4 + 2] = 10000;        // (u=2, v=1) -> 2 m
    raw[2 * 4 + 3] = 100;          // 0.02 m -> below minDepth
    const DepthImage d{raw, 4, 3};
    const DepthConfig cfg;
    const auto z = readValidDepth(d, {2.3, 0.6}, cfg);
    CHECK(z && std::abs(*z - 2.0) < 1e-15);
    CHECK(!readValidDepth(d, {0, 0}, cfg));            // no reading
    CHECK(!readValidDepth(d, {3, 2}, cfg));            // too close
    CHECK(!readValidDepth(d, {-0.6, 0}, cfg));         // outside (rounds to -1)
    CHECK(!readValidDepth(d, {4.0, 0}, cfg));          // outside
    CHECK(!readValidDepth(d, {NAN, 0}, cfg));
}

template <Mode M>
static void testTransformPointsMatchesPose() {
    std::mt19937_64 rng(2);
    const Pose T = randomPose(rng, 1.0, 1.0);
    const auto pts = randomPointsInView(rng, 1003);       // not a multiple of 4
    const PointsSoA in = toSoA(pts);
    PointsSoA out;
    transformPoints<M>(in, T, out);
    CHECK(out.size() == pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        const Vec3 e = T * pts[i];
        CHECK_REL(out.x[i], e.x, 1e-12);
        CHECK_REL(out.y[i], e.y, 1e-12);
        CHECK_REL(out.z[i], e.z, 1e-12);
    }
}

// ---- SE(3) ----------------------------------------------------------------------

static void testExpLogSE3() {
    std::mt19937_64 rng(3);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (int i = 0; i < 2000; ++i) {
        // angles from ~0 up to just below pi, including tiny ones
        const double scale = (i % 4 == 0) ? 1e-9 : (i % 4 == 1) ? 1e-4 : (i % 4 == 2) ? 1.0 : 1.8;
        Vec6 xi{U(rng), U(rng), U(rng), scale * U(rng), scale * U(rng), scale * U(rng)};
        const double th = std::sqrt(xi[3] * xi[3] + xi[4] * xi[4] + xi[5] * xi[5]);
        if (th >= std::numbers::pi - 1e-6) continue;
        const Vec6 back = logSE3(expSE3(xi));
        for (int k = 0; k < 6; ++k) CHECK_NEAR(back[k], xi[k], 1e-9);
        // and exp(log(T)) == T
        const Pose T = expSE3(xi);
        CHECK(poseDistance(expSE3(logSE3(T)), T) < 1e-12);
    }
    // angle very close to pi
    const Vec6 nearPi{0.1, -0.2, 0.3, 0, (std::numbers::pi - 1e-7), 0};
    const Vec6 back = logSE3(expSE3(nearPi));
    for (int k = 0; k < 6; ++k) CHECK_NEAR(back[k], nearPi[k], 1e-6);
    // R is orthonormal with det 1
    const Pose T = expSE3({0, 0, 0, 0.3, -1.2, 2.0});
    const Mat3 RtR = multiply(transpose(T.R), T.R);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) CHECK_NEAR(RtR.m[r * 3 + c], r == c ? 1.0 : 0.0, 1e-14);
}

static void testComposeInverse() {
    std::mt19937_64 rng(4);
    for (int i = 0; i < 100; ++i) {
        const Pose A = randomPose(rng, 3.0, 2.0), B = randomPose(rng, 3.0, 2.0);
        CHECK(poseDistance(composePose(A, inversePose(A)), identityPose()) < 1e-13);
        const Vec3 p{0.3, -0.7, 1.9};
        const Vec3 viaCompose = composePose(A, B) * p, viaSteps = A * (B * p);
        CHECK_NEAR(viaCompose.x, viaSteps.x, 1e-13);
        CHECK_NEAR(viaCompose.y, viaSteps.y, 1e-13);
        CHECK_NEAR(viaCompose.z, viaSteps.z, 1e-13);
    }
}

// ---- Horn, sampling, degeneracy ------------------------------------------------------

static void testHornRecoversTransform() {
    std::mt19937_64 rng(5);
    for (int trial = 0; trial < 50; ++trial) {
        const Pose T = randomPose(rng, 3.1, 2.0);
        std::vector<uint8_t> inl;
        const auto pairs = makePairs(rng, T, 3 + trial, 0.0, 0.0, inl);
        std::vector<int> idx(pairs.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = int(i);
        const auto est = estimateRigidTransform(pairs, idx);
        CHECK(est.has_value());
        if (est) CHECK(poseDistance(*est, T) < 1e-9);
    }
    // fewer than 3 points -> nullopt
    std::vector<uint8_t> inl;
    const auto pairs = makePairs(rng, identityPose(), 5, 0.0, 0.0, inl);
    const std::vector<int> two{0, 1};
    CHECK(!estimateRigidTransform(pairs, two).has_value());
}

static void testHornRejectsCollinear() {
    std::vector<PointPair3D> pairs;
    for (int i = 0; i < 6; ++i) {
        const Vec3 p{0.1 * i, 0.2 * i, 1.0 + 0.3 * i};
        pairs.push_back({p, p + Vec3{0.5, 0, 0}, size_t(i), size_t(i)});
    }
    const std::vector<int> idx{0, 1, 2, 3, 4, 5};
    CHECK(!estimateRigidTransform(pairs, idx).has_value());
}

static void testDegenerateSample() {
    const RansacConfig cfg;
    std::vector<PointPair3D> pairs = {
        {{0, 0, 1}, {0, 0, 1}, 0, 0},
        {{0.5, 0, 1}, {0.5, 0, 1}, 1, 1},
        {{1.0, 0.0001, 1}, {1.0, 0.0001, 1}, 2, 2},    // almost on the line of 0, 1
        {{0, 0.5, 1}, {0, 0.5, 1}, 3, 3},
        {{0.01, 0, 1}, {0.01, 0, 1}, 4, 4},            // 1 cm from point 0
    };
    CHECK(isDegenerateSample(pairs, {0, 1, 2}, cfg));   // collinear
    CHECK(isDegenerateSample(pairs, {0, 1, 4}, cfg));   // too close
    CHECK(!isDegenerateSample(pairs, {0, 1, 3}, cfg));  // good triangle
    // degenerate in B only
    pairs[3].b = {0.25, 0, 1};
    CHECK(isDegenerateSample(pairs, {0, 1, 3}, cfg));
}

static void testSamplingDeterministic() {
    for (int it = 0; it < 1000; ++it) {
        const Sample s = sampleRansacHypothesis(7, it, 10);
        CHECK(s == sampleRansacHypothesis(7, it, 10));
        CHECK(s[0] != s[1] && s[0] != s[2] && s[1] != s[2]);
        for (int k : s) CHECK(k >= 0 && k < 10);
    }
    CHECK(sampleRansacHypothesis(7, 0, 1000) != sampleRansacHypothesis(8, 0, 1000));
    CHECK(sampleRansacHypothesis(7, 0, 1000) != sampleRansacHypothesis(7, 1, 1000));
    // exact case n = 3
    Sample s = sampleRansacHypothesis(1, 5, 3);
    std::sort(s.begin(), s.end());
    CHECK((s == Sample{0, 1, 2}));
}

static void testRequiredIterations() {
    CHECK(requiredIterations(1.0, 0.99, 3, 500) <= 1);
    CHECK(requiredIterations(0.0, 0.99, 3, 500) == 500);
    // w = 0.5: log(0.01) / log(1 - 0.125) = 34.5 -> 35
    CHECK(requiredIterations(0.5, 0.99, 3, 500) == 35);
}

// ---- RANSAC -----------------------------------------------------------------------

template <Mode M>
static void testRansacWithOutliers() {
    std::mt19937_64 rng(6);
    for (int trial = 0; trial < 5; ++trial) {
        const Pose T = randomPose(rng, 0.5, 0.5);
        std::vector<uint8_t> isInlier;
        const auto pairs = makePairs(rng, T, 500, 0.4, 0.003, isInlier);
        RansacConfig cfg;
        cfg.seed = 100 + trial;
        const RansacResult r = estimatePoseRansac<M>(pairs, cfg);
        CHECK(r.success);
        CHECK(poseDistance(r.pose, T) < 5e-3);
        int agree = 0;
        for (size_t i = 0; i < pairs.size(); ++i) agree += (r.inlierMask[i] == isInlier[i]);
        CHECK(agree >= int(pairs.size()) - 5);    // random outliers rarely land within 3 cm
        CHECK(r.rmse < 0.01);
    }
}

template <Mode M>
static void testRansacFailsOnPureOutliers() {
    std::mt19937_64 rng(7);
    std::vector<uint8_t> isInlier;
    const auto pairs = makePairs(rng, identityPose(), 200, 1.0, 0.0, isInlier);
    const RansacResult r = estimatePoseRansac<M>(pairs, RansacConfig{});
    CHECK(!r.success);
}

// ---- refinement -----------------------------------------------------------------

static void testJacobianNumerically() {
    std::mt19937_64 rng(8);
    const Pose T = randomPose(rng, 0.3, 0.3);
    const auto obs = makeObservations(rng, T, 7, 0.0);
    const ObservationsSoA soa = toSoA(obs);
    std::vector<Jacobian26> J(obs.size());
    computePoseJacobians<Mode::SEQUENTIAL>(soa, T, kCam, J);
    const double h = 1e-6;
    for (size_t i = 0; i < obs.size(); ++i) {
        for (int k = 0; k < 6; ++k) {
            Vec6 d{};
            d[k] = h;
            const Pixel p1 = *project(composePose(expSE3(d), T) * obs[i].pointWorld, kCam);
            d[k] = -h;
            const Pixel p0 = *project(composePose(expSE3(d), T) * obs[i].pointWorld, kCam);
            CHECK_REL(J[i][k], (p1.u - p0.u) / (2 * h), 1e-5);
            CHECK_REL(J[i][6 + k], (p1.v - p0.v) / (2 * h), 1e-5);
        }
    }
}

static void testSolvePoseIncrement() {
    // H = A^T A + I (SPD), b = H x  ->  x
    std::mt19937_64 rng(9);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    double A[36];
    for (double& a : A) a = U(rng);
    Mat6 H{};
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c) {
            for (int k = 0; k < 6; ++k) H[r * 6 + c] += A[k * 6 + r] * A[k * 6 + c];
            if (r == c) H[r * 6 + c] += 1.0;
        }
    const Vec6 x{0.1, -0.2, 0.3, -0.4, 0.5, -0.6};
    Vec6 b{};
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c) b[r] += H[r * 6 + c] * x[c];
    const auto sol = solvePoseIncrement(H, b, 0.0);
    CHECK(sol.has_value());
    if (sol) for (int k = 0; k < 6; ++k) CHECK_NEAR((*sol)[k], x[k], 1e-12);
    // singular: rank 1
    Mat6 S{};
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c) S[r * 6 + c] = (r + 1.0) * (c + 1.0);
    CHECK(!solvePoseIncrement(S, b, 0.0).has_value());
    CHECK(!solvePoseIncrement(Mat6{}, b, 1e-3).has_value());
}

template <Mode M>
static void testRefineLowersRms() {
    std::mt19937_64 rng(10);
    const Pose T = randomPose(rng, 0.3, 0.3);
    const auto obs = makeObservations(rng, T, 500, 0.5);
    const Pose start = composePose(expSE3({0.03, -0.02, 0.04, 0.02, -0.015, 0.01}), T);
    const RefineResult r = refinePose<M>(obs, start, kCam, RefineConfig{});
    CHECK(r.success);
    CHECK_NEAR(r.initialRms, reprojectionRms(obs, start), 1e-9);
    CHECK_NEAR(r.finalRms, reprojectionRms(obs, r.pose), 1e-9);
    CHECK(r.finalRms < 0.1 * r.initialRms);
    CHECK(r.finalRms < 0.8);                     // 2D noise RMS = 0.5 * sqrt(2) = 0.71
    CHECK(poseDistance(r.pose, T) < 2e-3);
}

template <Mode M>
static void testRefineRobustToOutliers() {
    std::mt19937_64 rng(11);
    const Pose T = randomPose(rng, 0.3, 0.3);
    auto obs = makeObservations(rng, T, 400, 0.5);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (size_t i = 0; i < obs.size(); i += 5) obs[i].pixel = {U(rng) * 640, U(rng) * 480};   // 20 %
    const Pose start = composePose(expSE3({0.01, -0.01, 0.01, 0.005, -0.005, 0.005}), T);
    const RefineResult r = refinePose<M>(obs, start, kCam, RefineConfig{});
    CHECK(r.success);
    CHECK(poseDistance(r.pose, T) < 5e-3);
    CHECK(r.numInliers >= 300 && r.numInliers <= 330);
}

// ---- tracking ---------------------------------------------------------------------

// Two synthetic RGB-D frames that see the same world points.
struct SyntheticScene {
    std::vector<uint16_t> depthRef, depthCur;
    std::vector<Pixel>    kpRef, kpCur;
    Matches               matches;
    Pose                  T_rw, T_cw;
    FrameView ref() const { return {kpRef, {depthRef, kCam.width, kCam.height}}; }
    FrameView cur() const { return {kpCur, {depthCur, kCam.width, kCam.height}}; }
};

// Keypoints are integer pixels so the depth lookup is exact up to the 1/5000 m quantisation.
static SyntheticScene makeScene(std::mt19937_64& rng, int n, double wrongMatchRatio) {
    SyntheticScene s;
    s.T_rw = randomPose(rng, 0.2, 0.5);
    s.T_cw = composePose(expSE3({0.05, -0.03, 0.04, 0.03, 0.02, -0.04}), s.T_rw);
    s.depthRef.assign(size_t(kCam.width) * kCam.height, 0);
    s.depthCur = s.depthRef;
    const Pose T_wr = inversePose(s.T_rw);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    auto put = [](std::vector<uint16_t>& d, const Pixel& p, double z) {
        d[size_t(p.v) * kCam.width + size_t(p.u)] = uint16_t(std::lround(z * 5000));
    };
    for (const Vec3& pr : randomPointsInView(rng, 3 * n)) {
        if (int(s.kpRef.size()) == n) break;
        const Vec3 pc = s.T_cw * (T_wr * pr);
        const auto ur = project(pr, kCam);
        const auto uc = project(pc, kCam);
        if (!uc || uc->u < 1 || uc->v < 1 || uc->u > kCam.width - 2 || uc->v > kCam.height - 2) continue;
        const Pixel a{std::round(ur->u), std::round(ur->v)}, b{std::round(uc->u), std::round(uc->v)};
        // depth of the rounded pixel's ray, so back-projection hits the point's depth
        put(s.depthRef, a, pr.z);
        put(s.depthCur, b, pc.z);
        s.matches[s.kpRef.size()] = s.kpCur.size();
        s.kpRef.push_back(a);
        s.kpCur.push_back(b);
    }
    // wrong matches: shuffle the targets of a fraction of the matches
    const size_t m = s.kpRef.size();
    for (size_t i = 0; i < m; ++i)
        if (U(rng) < wrongMatchRatio) std::swap(s.matches[i], s.matches[size_t(U(rng) * double(m))]);
    return s;
}

template <Mode M>
static void testTrackFrameOk() {
    std::mt19937_64 rng(12);
    const SyntheticScene s = makeScene(rng, 300, 0.3);
    const TrackResult r = trackFrame<M>(s.ref(), s.T_rw, s.cur(), s.matches, {}, kCam, TrackingConfig{});
    CHECK(r.status == TrackingStatus::OK);
    CHECK(r.reason == LostReason::None);
    CHECK(poseDistance(r.T_cw, s.T_cw) < 5e-3);
    CHECK(r.inlierMatches.size() >= 100);         // swaps break ~half of the 300 matches
    int wrong = 0;                                 // correct matches are the identity
    for (const auto& [a, b] : r.inlierMatches) wrong += (a != b);
    CHECK(wrong <= 3);                             // a wrong match may land within 3 cm by chance
}

template <Mode M>
static void testTrackFrameLost() {
    std::mt19937_64 rng(13);
    const TrackingConfig cfg;
    // too few matches with depth
    const SyntheticScene few = makeScene(rng, 10, 0.0);
    const TrackResult a = trackFrame<M>(few.ref(), few.T_rw, few.cur(), few.matches, {}, kCam, cfg);
    CHECK(a.status == TrackingStatus::LOST);
    CHECK(a.reason == LostReason::TooFewCorrespondences);
    // enough matches, but almost all wrong -> too few inliers
    const SyntheticScene bad = makeScene(rng, 200, 1.0);
    const TrackResult b = trackFrame<M>(bad.ref(), bad.T_rw, bad.cur(), bad.matches, {}, kCam, cfg);
    CHECK(b.status == TrackingStatus::LOST);
    CHECK(b.reason == LostReason::RansacFailed || b.reason == LostReason::TooFewInliers);
}

// ---- equivalence: SIMD / PARALLEL vs serial ------------------------------------------

template <Mode M>
static void testEquivalence() {
    std::mt19937_64 rng(14);
    const Pose T = randomPose(rng, 0.4, 0.4);
    std::vector<uint8_t> isInlier;
    for (int n : {3, 5, 1003, 2048}) {
        const auto pairs = makePairs(rng, T, n, 0.3, 0.004, isInlier);
        const PairsSoA soa = toSoA(pairs);

        PointsSoA o0, o1;
        transformPoints<Mode::SEQUENTIAL>(soa.a, T, o0);
        transformPoints<M>(soa.a, T, o1);
        for (int i = 0; i < n; ++i) {
            CHECK_REL(o1.x[i], o0.x[i], 1e-12);
            CHECK_REL(o1.y[i], o0.y[i], 1e-12);
            CHECK_REL(o1.z[i], o0.z[i], 1e-12);
        }

        std::vector<double> e0(n), e1(n);
        computeAlignmentErrors<Mode::SEQUENTIAL>(soa, T, e0);
        computeAlignmentErrors<M>(soa, T, e1);
        for (int i = 0; i < n; ++i) CHECK_REL(e1[i], e0[i], 1e-9);

        std::vector<uint8_t> m0(n), m1(n);
        const int c0 = selectInliers<Mode::SEQUENTIAL>(e0, 0.01, m0);
        const int c1 = selectInliers<M>(e0, 0.01, m1);
        CHECK(c0 == c1);
        CHECK(m0 == m1);
        CHECK(countInliers<M>(soa, T, 0.01) == c0);

        const RansacResult r0 = estimatePoseRansac<Mode::SEQUENTIAL>(pairs, RansacConfig{});
        const RansacResult r1 = estimatePoseRansac<M>(pairs, RansacConfig{});
        CHECK(r0.success == r1.success);
        CHECK(r0.numInliers == r1.numInliers);
        CHECK(r0.iterationsUsed == r1.iterationsUsed);
        CHECK(r0.inlierMask == r1.inlierMask);
        CHECK(poseDistance(r0.pose, r1.pose) < 1e-9);
        CHECK_REL(r1.rmse, r0.rmse, 1e-6);
    }

    for (int n : {1, 7, 501, 2000}) {
        auto obs = makeObservations(rng, T, n, 1.0);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        for (int i = 0; i < n; i += 7) obs[i].pixel = {U(rng) * 640, U(rng) * 480};
        for (int i = 0; i < n; i += 3) obs[i].information = 1.0 / 1.44;
        if (n > 10) obs[3].pointWorld = inversePose(T) * Vec3{0.1, 0.1, -1.0};  // behind camera
        const ObservationsSoA soa = toSoA(obs);
        const Pose P = composePose(expSE3({0.01, 0.0, -0.01, 0.0, 0.01, 0.0}), T);

        std::vector<double> ru0(n), rv0(n), ru1(n), rv1(n);
        computeReprojectionResiduals<Mode::SEQUENTIAL>(soa, P, kCam, ru0, rv0);
        computeReprojectionResiduals<M>(soa, P, kCam, ru1, rv1);
        for (int i = 0; i < n; ++i) {
            if (n > 10 && i == 3) continue;
            CHECK_REL(ru1[i], ru0[i], 1e-9);
            CHECK_REL(rv1[i], rv0[i], 1e-9);
        }

        std::vector<Jacobian26> J0(n), J1(n);
        computePoseJacobians<Mode::SEQUENTIAL>(soa, P, kCam, J0);
        computePoseJacobians<M>(soa, P, kCam, J1);
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < 12; ++k) CHECK_REL(J1[i][k], J0[i][k], 1e-9);

        const PoseSystem s0 = accumulatePoseSystem<Mode::SEQUENTIAL>(soa, P, kCam, 2.4477);
        const PoseSystem s1 = accumulatePoseSystem<M>(soa, P, kCam, 2.4477);
        CHECK(s0.numValid == s1.numValid);
        CHECK(s0.numInliers == s1.numInliers);
        CHECK_NEAR(s1.cost, s0.cost, 1e-6 * std::max(1.0, s0.cost));
        CHECK_NEAR(s1.sumSqError, s0.sumSqError, 1e-6 * std::max(1.0, s0.sumSqError));
        double hmax = 1.0, bmax = 1.0;
        for (double h : s0.H) hmax = std::max(hmax, std::abs(h));
        for (double b : s0.b) bmax = std::max(bmax, std::abs(b));
        for (int k = 0; k < 36; ++k) CHECK_NEAR(s1.H[k], s0.H[k], 1e-6 * hmax);
        for (int k = 0; k < 6; ++k) CHECK_NEAR(s1.b[k], s0.b[k], 1e-6 * bmax);

        if (n >= 7) {
            const RefineResult f0 = refinePose<Mode::SEQUENTIAL>(obs, P, kCam, RefineConfig{});
            const RefineResult f1 = refinePose<M>(obs, P, kCam, RefineConfig{});
            CHECK(f0.success == f1.success);
            CHECK(f0.iterations == f1.iterations);
            CHECK(f0.numInliers == f1.numInliers);
            CHECK(poseDistance(f0.pose, f1.pose) < 1e-6);
            CHECK_REL(f1.finalRms, f0.finalRms, 1e-6);
        }
    }

    const SyntheticScene s = makeScene(rng, 300, 0.3);
    const TrackResult t0 = trackFrame<Mode::SEQUENTIAL>(s.ref(), s.T_rw, s.cur(), s.matches, {}, kCam, TrackingConfig{});
    const TrackResult t1 = trackFrame<M>(s.ref(), s.T_rw, s.cur(), s.matches, {}, kCam, TrackingConfig{});
    CHECK(t0.status == t1.status);
    CHECK(t0.inlierMatches == t1.inlierMatches);
    CHECK(poseDistance(t0.T_cw, t1.T_cw) < 1e-6);
}

template <Mode M>
static void runModeTests() {
    const int before = gFailures;
    testTransformPointsMatchesPose<M>();
    testRansacWithOutliers<M>();
    testRansacFailsOnPureOutliers<M>();
    testRefineLowersRms<M>();
    testRefineRobustToOutliers<M>();
    testTrackFrameOk<M>();
    testTrackFrameLost<M>();
    if (M != Mode::SEQUENTIAL) testEquivalence<M>();
    std::printf("[%s] %s\n", modeName(M).data(), gFailures == before ? "ok" : "FAILED");
}

int main() {
    testProjectBackProjectRoundTrip();
    testReadValidDepth();
    testExpLogSE3();
    testComposeInverse();
    testHornRecoversTransform();
    testHornRejectsCollinear();
    testDegenerateSample();
    testSamplingDeterministic();
    testRequiredIterations();
    testJacobianNumerically();
    testSolvePoseIncrement();
    std::printf("[mode-independent] %s\n", gFailures == 0 ? "ok" : "FAILED");

    runModeTests<Mode::SEQUENTIAL>();
    runModeTests<Mode::SIMD>();
    runModeTests<Mode::PARALLEL>();

    std::printf("%d checks, %d failures (default mode: %s)\n", gChecks, gFailures, modeName(kMode).data());
    return gFailures == 0 ? 0 : 1;
}
