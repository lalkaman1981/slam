// Person 2 benchmark: every mode on the same synthetic data, median of N samples.
// Run: ./build/slam_pose_estimation_bench [samples]      (env SLAM_THREADS sets the thread count)

#include "camera.hpp"
#include "kernels.hpp"
#include "pose_refine.hpp"
#include "ransac.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace slam;

namespace {

const CameraIntrinsics kCam{525.0, 525.0, 319.5, 239.5, 640, 480};

std::vector<Vec3> pointsInView(std::mt19937_64& rng, int n) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<Vec3> pts;
    for (int i = 0; i < n; ++i)
        pts.push_back(backProject({20 + U(rng) * 600, 20 + U(rng) * 440}, 1.0 + 3.0 * U(rng), kCam));
    return pts;
}

std::vector<PointPair3D> makePairs(std::mt19937_64& rng, const Pose& T, int n, double outlierRatio) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double> G(0.0, 0.005);
    std::vector<PointPair3D> pairs;
    for (const Vec3& a : pointsInView(rng, n)) {
        Vec3 b = (U(rng) < outlierRatio) ? Vec3{4 * U(rng) - 2, 4 * U(rng) - 2, 1 + 3 * U(rng)}
                                         : T * a + Vec3{G(rng), G(rng), G(rng)};
        pairs.push_back({a, b, pairs.size(), pairs.size()});
    }
    return pairs;
}

std::vector<MapObservation> makeObservations(std::mt19937_64& rng, const Pose& T_cw, int n, double outlierRatio) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double> G(0.0, 1.0);
    const Pose T_wc = inversePose(T_cw);
    std::vector<MapObservation> obs;
    for (const Vec3& pc : pointsInView(rng, n)) {
        Pixel px = *project(pc, kCam);
        px = (U(rng) < outlierRatio) ? Pixel{U(rng) * 640, U(rng) * 480} : Pixel{px.u + G(rng), px.v + G(rng)};
        obs.push_back({T_wc * pc, px, 1.0});
    }
    return obs;
}

int gSamples = 31;

using clock_type = std::chrono::steady_clock;

// Repetitions so one sample of `f` lasts >= 2 ms.
template <class F>
int calibrate(F& f) {
    int reps = 1;
    for (;;) {
        const auto t0 = clock_type::now();
        for (int r = 0; r < reps; ++r) f();
        if (clock_type::now() - t0 > std::chrono::milliseconds(2) || reps > (1 << 20)) return reps;
        reps *= 2;
    }
}

template <class F>
double sampleMicros(F& f, int reps) {
    const auto t0 = clock_type::now();
    for (int r = 0; r < reps; ++r) f();
    return std::chrono::duration<double, std::micro>(clock_type::now() - t0).count() / reps;
}

double median(std::vector<double> t) {
    std::nth_element(t.begin(), t.begin() + t.size() / 2, t.end());
    return t[t.size() / 2];
}

volatile double gSink = 0;   // keeps results alive

// Median time of one call per mode. The three modes are interleaved in every sample round,
// so CPU frequency drift (laptop, powersave governor) affects all of them alike.
template <class Bench>
void row(const char* name, Bench&& bench) {
    auto fs = bench.template operator()<Mode::SEQUENTIAL>();
    auto fv = bench.template operator()<Mode::SIMD>();
    auto fp = bench.template operator()<Mode::PARALLEL>();
    const int rs = calibrate(fs), rv = calibrate(fv), rp = calibrate(fp);
    std::vector<double> ts, tv, tp;
    for (int k = 0; k < gSamples; ++k) {
        ts.push_back(sampleMicros(fs, rs));
        tv.push_back(sampleMicros(fv, rv));
        tp.push_back(sampleMicros(fp, rp));
    }
    const double s = median(ts), v = median(tv), p = median(tp);
    std::printf("| %-30s | %9.1f | %9.1f | %5.2fx | %9.1f | %5.2fx |\n", name, s, v, s / v, p, s / p);
}

void ransacRows(const char* label, int n, double outliers) {
    std::mt19937_64 rng(1);
    const Pose T = expSE3({0.05, -0.02, 0.03, 0.02, -0.01, 0.03});
    const auto pairs = makePairs(rng, T, n, outliers);
    const PairsSoA soa = toSoA(pairs);
    std::vector<double> errors(pairs.size());
    std::vector<uint8_t> mask(pairs.size());
    PointsSoA out;
    const RansacConfig cfg;
    const auto r = estimatePoseRansac<Mode::SEQUENTIAL>(pairs, cfg);
    std::printf("\n**%s**: %d pairs, %.0f%% outliers -> %d inliers, %d iterations\n\n", label, n,
                100 * outliers, r.numInliers, r.iterationsUsed);
    std::printf("| function | serial us | simd us | simd speedup | parallel us | parallel speedup |\n");
    std::printf("|---|---|---|---|---|---|\n");
    row("transformPoints", [&]<Mode M>() { return ([&] { transformPoints<M>(soa.a, T, out); gSink = out.x[0]; }); });
    row("computeAlignmentErrors", [&]<Mode M>() { return ([&] { computeAlignmentErrors<M>(soa, T, errors); gSink = errors[0]; }); });
    row("selectInliers", [&]<Mode M>() { return ([&] { gSink = selectInliers<M>(errors, 0.03, mask); }); });
    row("countInliers (fused)", [&]<Mode M>() { return ([&] { gSink = countInliers<M>(soa, T, 0.03); }); });
    row("estimatePoseRansac", [&]<Mode M>() { return ([&] { gSink = estimatePoseRansac<M>(pairs, cfg).numInliers; }); });
}

void refineRows(const char* label, int n) {
    std::mt19937_64 rng(2);
    const Pose T = expSE3({0.05, -0.02, 0.03, 0.02, -0.01, 0.03});
    const auto obs = makeObservations(rng, T, n, 0.1);
    const ObservationsSoA soa = toSoA(obs);
    std::vector<double> ru(obs.size()), rv(obs.size());
    std::vector<Jacobian26> J(obs.size());
    const Pose start = composePose(expSE3({0.02, 0.01, -0.02, 0.01, 0.005, -0.01}), T);
    const RefineConfig cfg;
    const auto r = refinePose<Mode::SEQUENTIAL>(obs, start, kCam, cfg);
    std::printf("\n**%s**: %d observations, 10%% outliers, 1 px noise -> %d inliers, %d LM iterations\n\n",
                label, n, r.numInliers, r.iterations);
    std::printf("| function | serial us | simd us | simd speedup | parallel us | parallel speedup |\n");
    std::printf("|---|---|---|---|---|---|\n");
    row("computeReprojectionResiduals", [&]<Mode M>() { return ([&] { computeReprojectionResiduals<M>(soa, T, kCam, ru, rv); gSink = ru[0]; }); });
    row("computePoseJacobians", [&]<Mode M>() { return ([&] { computePoseJacobians<M>(soa, T, kCam, J); gSink = J[0][0]; }); });
    row("accumulatePoseSystem", [&]<Mode M>() { return ([&] { gSink = accumulatePoseSystem<M>(soa, T, kCam, 2.4477).cost; }); });
    row("refinePose", [&]<Mode M>() { return ([&] { gSink = refinePose<M>(obs, start, kCam, cfg).finalRms; }); });
}

// The brief's workload end to end: RANSAC on 2000 pairs (30% outliers) + refinePose on 500 observations.
void pipelineRow() {
    std::mt19937_64 rng(3);
    const Pose T = expSE3({0.05, -0.02, 0.03, 0.02, -0.01, 0.03});
    const auto pairs = makePairs(rng, T, 2000, 0.3);
    const auto obs = makeObservations(rng, T, 500, 0.1);
    const RansacConfig rc;
    const RefineConfig fc;
    std::printf("\n**Brief workload**: estimatePoseRansac(2000 pairs, 30%% outliers) + refinePose(500 obs from the RANSAC pose)\n\n");
    std::printf("| function | serial us | simd us | simd speedup | parallel us | parallel speedup |\n");
    std::printf("|---|---|---|---|---|---|\n");
    row("RANSAC + refinePose", [&]<Mode M>() {
        return ([&] {
            const RansacResult r = estimatePoseRansac<M>(pairs, rc);
            gSink = refinePose<M>(obs, r.pose, kCam, fc).finalRms;
        });
    });
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1) gSamples = std::max(1, std::atoi(argv[1]));
    std::printf("threads: %d, AVX2+FMA: %s, samples: %d (median)\n", detail::numThreads(),
                detail::hasAvx2Fma() ? "yes" : "no", gSamples);
    pipelineRow();
    ransacRows("RANSAC, brief size", 2000, 0.3);
    ransacRows("RANSAC, hard", 2000, 0.6);
    ransacRows("RANSAC, large", 20000, 0.5);
    refineRows("Refinement, brief size", 500);
    refineRows("Refinement, large", 5000);
    return 0;
}
