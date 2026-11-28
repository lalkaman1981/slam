// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include "bundle_adjustment.hpp"
#include "execution.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

namespace slam {

namespace {

constexpr double kInitialLambdaScale = 1e-5; // g2o LM tau
constexpr int kMaxLambdaTries = 10;
constexpr int kPointIterations = 10;

double chi2Threshold(const BAProblem &p, const BAObservation &obs) {
    return obs.hasDepth ? p.chi2Stereo : p.chi2Mono;
}

// Huber kernel applied to the squared error, as g2o::RobustKernelHuber with delta^2 = threshold.
double robustCost(double chi2, double delta2, bool robust) {
    if (!robust || chi2 <= delta2) return chi2;
    return 2.0 * std::sqrt(chi2 * delta2) - delta2;
}

double robustWeight(double chi2, double delta2, bool robust) {
    if (!robust || chi2 <= delta2) return 1.0;
    return std::sqrt(delta2 / chi2);
}

struct NormalEquations {
    std::vector<Mat6> A; // pose-pose blocks (diagonal only)
    std::vector<Vec6> gp;
    std::vector<Mat3> C; // point-point blocks
    std::vector<Vec3> gx;
    std::vector<Mat63> B; // pose-point block per observation
    std::vector<std::vector<std::size_t>> pointObservations;
};

double evaluateCost(const BAProblem &problem, bool robust, ExecutionMode mode) {
    std::vector<ObservationResidual> residuals;
    computeObservationResiduals(problem, residuals, false, mode);
    double cost = 0.0;
    for (std::size_t k = 0; k < residuals.size(); ++k) {
        const BAObservation &obs = problem.observations[k];
        if (obs.active) cost += robustCost(residuals[k].chi2, chi2Threshold(problem, obs), robust);
    }
    return cost;
}

void buildNormalEquations(const BAProblem &problem, const std::vector<ObservationResidual> &residuals,
                          bool robust, NormalEquations &ne, ExecutionMode mode) {
    ne.A.assign(problem.poses.size(), Mat6{});
    ne.gp.assign(problem.poses.size(), Vec6{});
    ne.C.assign(problem.points.size(), Mat3{});
    ne.gx.assign(problem.points.size(), Vec3{});
    ne.B.assign(problem.observations.size(), Mat63{});
    ne.pointObservations.assign(problem.points.size(), {});
    const std::size_t workers = detail::workerCount(problem.observations.size(), mode);
    struct Accumulator {
        std::vector<Mat6> A;
        std::vector<Vec6> gp;
        std::vector<Mat3> C;
        std::vector<Vec3> gx;
    };
    std::vector<Accumulator> local(workers);
    for (auto &acc : local) {
        acc.A.resize(problem.poses.size());
        acc.gp.resize(problem.poses.size());
        acc.C.resize(problem.points.size());
        acc.gx.resize(problem.points.size());
    }
    detail::runPartitions(
        problem.observations.size(), workers, [&](std::size_t begin, std::size_t end, std::size_t worker) {
            auto &acc = local[worker];
            for (std::size_t k = begin; k < end; ++k) {
                const BAObservation &obs = problem.observations[k];
                if (!obs.active) continue;
                const ObservationResidual &r = residuals[k];
                const double w = obs.information * robustWeight(r.chi2, chi2Threshold(problem, obs), robust);
                const auto JpT = r.jacobianPose.transpose();
                const auto JxT = r.jacobianPoint.transpose();
                if (!problem.fixedPose[obs.poseIndex]) {
                    acc.A[obs.poseIndex] += w * detail::multiply(JpT, r.jacobianPose, mode);
                    acc.gp[obs.poseIndex] += w * (JpT * r.error);
                    ne.B[k] = w * detail::multiply(JpT, r.jacobianPoint, mode);
                }
                acc.C[obs.pointIndex] += w * detail::multiply(JxT, r.jacobianPoint, mode);
                acc.gx[obs.pointIndex] += w * (JxT * r.error);
            }
        });
    // Merge private normal equations in worker order; only B has disjoint direct writes.
    for (const auto &acc : local) {
        for (std::size_t i = 0; i < ne.A.size(); ++i) {
            ne.A[i] += acc.A[i];
            ne.gp[i] += acc.gp[i];
        }
        for (std::size_t i = 0; i < ne.C.size(); ++i) {
            ne.C[i] += acc.C[i];
            ne.gx[i] += acc.gx[i];
        }
    }
    for (std::size_t k = 0; k < problem.observations.size(); ++k)
        if (problem.observations[k].active)
            ne.pointObservations[problem.observations[k].pointIndex].push_back(k);
}

// Solves the damped system with the points eliminated (Schur complement):
// (A - B C^-1 B^T) dp = -gp + B C^-1 gx,  dx = C^-1 (-gx - B^T dp).
bool solveSchur(const BAProblem &problem, const NormalEquations &ne, double lambda, std::vector<Vec6> &dp,
                std::vector<Vec3> &dx, ExecutionMode mode) {
    std::vector<int> freeIndex(problem.poses.size(), -1);
    std::size_t nFree = 0;
    for (std::size_t i = 0; i < problem.poses.size(); ++i)
        if (!problem.fixedPose[i]) freeIndex[i] = static_cast<int>(nFree++);

    const Mat3 damp3 = lambda * Mat3::identity();
    const Mat6 damp6 = lambda * Mat6::identity();
    DenseMatrix S(6 * nFree, 6 * nFree);
    std::vector<double> rhs(6 * nFree, 0.0);
    for (std::size_t i = 0; i < problem.poses.size(); ++i) {
        if (freeIndex[i] < 0) continue;
        const std::size_t o = 6 * static_cast<std::size_t>(freeIndex[i]);
        S.addBlock(o, o, ne.A[i] + damp6);
        for (std::size_t r = 0; r < 6; ++r)
            rhs[o + r] = -ne.gp[i][r];
    }

    std::vector<Mat3> Cinv(problem.points.size());
    for (std::size_t j = 0; j < problem.points.size(); ++j) {
        if (!invert3(ne.C[j] + damp3, Cinv[j])) return false;
        std::vector<std::size_t> coupled;
        for (std::size_t k : ne.pointObservations[j])
            if (freeIndex[problem.observations[k].poseIndex] >= 0) coupled.push_back(k);
        for (std::size_t a : coupled) {
            const Mat63 BCinv = detail::multiply(ne.B[a], Cinv[j], mode);
            const std::size_t oa = 6 * static_cast<std::size_t>(freeIndex[problem.observations[a].poseIndex]);
            const Vec6 r = BCinv * ne.gx[j];
            for (std::size_t q = 0; q < 6; ++q)
                rhs[oa + q] += r[q];
            for (std::size_t b : coupled) {
                const std::size_t ob =
                    6 * static_cast<std::size_t>(freeIndex[problem.observations[b].poseIndex]);
                S.addBlock(oa, ob, -detail::multiply(BCinv, ne.B[b].transpose(), mode));
            }
        }
    }

    std::vector<double> x;
    if (nFree > 0 && !solveCholesky(S, rhs, x, mode)) return false;
    dp.assign(problem.poses.size(), Vec6{});
    for (std::size_t i = 0; i < problem.poses.size(); ++i) {
        if (freeIndex[i] < 0) continue;
        const std::size_t o = 6 * static_cast<std::size_t>(freeIndex[i]);
        for (std::size_t q = 0; q < 6; ++q)
            dp[i][q] = x[o + q];
    }
    dx.assign(problem.points.size(), Vec3{});
    for (std::size_t j = 0; j < problem.points.size(); ++j) {
        Vec3 r = -ne.gx[j];
        for (std::size_t k : ne.pointObservations[j]) {
            const std::size_t pi = problem.observations[k].poseIndex;
            if (freeIndex[pi] >= 0) r -= ne.B[k].transpose() * dp[pi];
        }
        dx[j] = Cinv[j] * r;
    }
    return true;
}

} // namespace

ObservationResidual evaluateObservation(const Pose &T_cw, const Vec3 &point, const BAObservation &obs,
                                        const CameraIntrinsics &camera, bool withJacobians) {
    ObservationResidual r;
    const Vec3 Xc = transformPoint(T_cw, point);
    r.inFront = Xc[2] > 0.0;
    const double z = std::abs(Xc[2]) < 1e-9 ? 1e-9 : Xc[2];
    const double invZ = 1.0 / z;
    const double u = camera.fx * Xc[0] * invZ + camera.cx;
    const double v = camera.fy * Xc[1] * invZ + camera.cy;
    r.error[0] = obs.measured[0] - u;
    r.error[1] = obs.measured[1] - v;
    r.error[2] = obs.hasDepth ? obs.measured[2] - (u - camera.bf * invZ) : 0.0;
    r.chi2 = obs.information * squaredNorm(r.error);
    if (!withJacobians) return r;

    // d projection / d Xc.
    Mat3 P{};
    P(0, 0) = camera.fx * invZ;
    P(0, 2) = -camera.fx * Xc[0] * invZ * invZ;
    P(1, 1) = camera.fy * invZ;
    P(1, 2) = -camera.fy * Xc[1] * invZ * invZ;
    if (obs.hasDepth) {
        P(2, 0) = P(0, 0);
        P(2, 2) = P(0, 2) + camera.bf * invZ * invZ;
    }
    // d Xc / d [omega, upsilon] for T <- exp(delta) * T.
    Mat36 dXc{};
    setBlock(dXc, 0, 0, -skew(Xc));
    setBlock(dXc, 0, 3, Mat3::identity());
    r.jacobianPose = -(P * dXc);
    r.jacobianPoint = -(P * T_cw.R);
    return r;
}

namespace {

void evaluateObservationPair(const Pose &poseA, const Pose &poseB, const Vec3 &pointA, const Vec3 &pointB,
                             const BAObservation &obsA, const BAObservation &obsB,
                             const CameraIntrinsics &camera, bool withJacobians, ObservationResidual &outA,
                             ObservationResidual &outB) {
    using detail::Double2;
    auto xyz = detail::transformPair(poseA, poseB, pointA, pointB);
    double depths[2];
    xyz[2].store(depths);
    outA = {};
    outB = {};
    outA.inFront = depths[0] > 0.0;
    outB.inFront = depths[1] > 0.0;
    for (double &z : depths)
        if (std::abs(z) < 1e-9) z = 1e-9;
    const Double2 iz = Double2(1.0) / Double2(depths[0], depths[1]);
    const Double2 u = Double2(camera.fx) * xyz[0] * iz + Double2(camera.cx);
    const Double2 v = Double2(camera.fy) * xyz[1] * iz + Double2(camera.cy);
    auto scatter = [](Double2 v2, double &a, double &b) {
        double lanes[2];
        v2.store(lanes);
        a = lanes[0];
        b = lanes[1];
    };
    scatter(Double2(obsA.measured[0], obsB.measured[0]) - u, outA.error[0], outB.error[0]);
    scatter(Double2(obsA.measured[1], obsB.measured[1]) - v, outA.error[1], outB.error[1]);
    scatter(Double2(obsA.measured[2], obsB.measured[2]) - (u - Double2(camera.bf) * iz), outA.error[2],
            outB.error[2]);
    if (!obsA.hasDepth) outA.error[2] = 0.0;
    if (!obsB.hasDepth) outB.error[2] = 0.0;
    outA.chi2 = obsA.information * squaredNorm(outA.error);
    outB.chi2 = obsB.information * squaredNorm(outB.error);
    if (!withJacobians) return;
    std::array<std::array<Double2, 3>, 3> P{};
    P[0][0] = Double2(camera.fx) * iz;
    P[0][2] = -Double2(camera.fx) * xyz[0] * iz * iz;
    P[1][1] = Double2(camera.fy) * iz;
    P[1][2] = -Double2(camera.fy) * xyz[1] * iz * iz;
    const Double2 depthMask(obsA.hasDepth ? 1.0 : 0.0, obsB.hasDepth ? 1.0 : 0.0);
    P[2][0] = P[0][0] * depthMask;
    P[2][2] = (P[0][2] + Double2(camera.bf) * iz * iz) * depthMask;
    std::array<std::array<Double2, 6>, 3> dX{};
    dX[0][1] = xyz[2];
    dX[0][2] = -xyz[1];
    dX[0][3] = Double2(1.0);
    dX[1][0] = -xyz[2];
    dX[1][2] = xyz[0];
    dX[1][4] = Double2(1.0);
    dX[2][0] = xyz[1];
    dX[2][1] = -xyz[0];
    dX[2][5] = Double2(1.0);
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 6; ++c) {
            Double2 j(0.0);
            for (std::size_t k = 0; k < 3; ++k)
                j = j + P[r][k] * dX[k][c];
            scatter(-j, outA.jacobianPose(r, c), outB.jacobianPose(r, c));
        }
        for (std::size_t c = 0; c < 3; ++c) {
            Double2 j(0.0);
            for (std::size_t k = 0; k < 3; ++k)
                j = j + P[r][k] * Double2(poseA.R(k, c), poseB.R(k, c));
            scatter(-j, outA.jacobianPoint(r, c), outB.jacobianPoint(r, c));
        }
    }
}

} // namespace

void computeObservationResiduals(const BAProblem &problem, std::vector<ObservationResidual> &out,
                                 bool withJacobians, ExecutionMode mode) {
    out.resize(problem.observations.size());
    const auto workers = detail::workerCount(out.size(), mode);
    detail::runPartitions(out.size(), workers, [&](std::size_t begin, std::size_t end, std::size_t) {
        std::size_t k = begin;
        if (mode == ExecutionMode::SerialSimd) {
            for (; k + 1 < end; k += 2) {
                const auto &a = problem.observations[k];
                const auto &b = problem.observations[k + 1];
                evaluateObservationPair(problem.poses[a.poseIndex], problem.poses[b.poseIndex],
                                        problem.points[a.pointIndex], problem.points[b.pointIndex], a, b,
                                        problem.camera, withJacobians, out[k], out[k + 1]);
            }
        }
        for (; k < end; ++k) {
            const BAObservation &obs = problem.observations[k];
            out[k] = evaluateObservation(problem.poses[obs.poseIndex], problem.points[obs.pointIndex], obs,
                                         problem.camera, withJacobians);
        }
    });
}

double optimizeBAProblem(BAProblem &problem, int iterations, bool robust, ExecutionMode mode) {
    double lambda = -1.0;
    double nu = 2.0;
    std::vector<ObservationResidual> residuals;
    NormalEquations ne;
    double cost = evaluateCost(problem, robust, mode);

    for (int it = 0; it < iterations; ++it) {
        computeObservationResiduals(problem, residuals, true, mode);
        buildNormalEquations(problem, residuals, robust, ne, mode);
        if (lambda < 0.0) {
            double maxDiag = 0.0;
            for (const Mat6 &a : ne.A)
                for (std::size_t q = 0; q < 6; ++q)
                    maxDiag = std::max(maxDiag, a(q, q));
            for (const Mat3 &c : ne.C)
                for (std::size_t q = 0; q < 3; ++q)
                    maxDiag = std::max(maxDiag, c(q, q));
            lambda = kInitialLambdaScale * std::max(maxDiag, 1e-12);
        }

        bool accepted = false;
        double maxStep = 0.0;
        for (int attempt = 0; attempt < kMaxLambdaTries && !accepted; ++attempt) {
            std::vector<Vec6> dp;
            std::vector<Vec3> dx;
            if (!solveSchur(problem, ne, lambda, dp, dx, mode)) {
                lambda *= nu;
                nu *= 2.0;
                continue;
            }
            BAProblem candidate = problem;
            double predicted = 0.0;
            maxStep = 0.0;
            for (std::size_t i = 0; i < problem.poses.size(); ++i) {
                if (problem.fixedPose[i]) continue;
                candidate.poses[i] = applyLeftIncrement(problem.poses[i], dp[i]);
                predicted += dot(dp[i], lambda * dp[i] - ne.gp[i]);
                for (std::size_t q = 0; q < 6; ++q)
                    maxStep = std::max(maxStep, std::abs(dp[i][q]));
            }
            for (std::size_t j = 0; j < problem.points.size(); ++j) {
                candidate.points[j] = problem.points[j] + dx[j];
                predicted += dot(dx[j], lambda * dx[j] - ne.gx[j]);
                for (std::size_t q = 0; q < 3; ++q)
                    maxStep = std::max(maxStep, std::abs(dx[j][q]));
            }
            const double newCost = evaluateCost(candidate, robust, mode);
            const double rho = predicted > 0.0 ? (cost - newCost) / predicted : -1.0;
            if (std::isfinite(newCost) && rho > 0.0) {
                problem.poses = std::move(candidate.poses);
                problem.points = std::move(candidate.points);
                cost = newCost;
                const double t = 2.0 * rho - 1.0;
                lambda *= std::max(1.0 / 3.0, 1.0 - t * t * t);
                nu = 2.0;
                accepted = true;
            } else {
                lambda *= nu;
                nu *= 2.0;
            }
        }
        if (!accepted || maxStep < 1e-10) break;
    }
    return cost;
}

BAProblem buildLocalBAProblem(const Map &map, int keyframeId) {
    BAProblem problem;
    problem.camera = map.camera;
    problem.chi2Mono = map.config.chi2Mono;
    problem.chi2Stereo = map.config.chi2Stereo;

    std::vector<int> localKeyframes{keyframeId};
    for (int id : covisibleKeyframes(map, keyframeId))
        localKeyframes.push_back(id);
    const std::set<int> localSet(localKeyframes.begin(), localKeyframes.end());

    std::set<int> pointSet;
    for (int kfId : localKeyframes) {
        for (int slot : map.keyframes.at(kfId).landmarkIds) {
            const int id = resolveLandmarkId(map, slot);
            if (id >= 0 && pointSet.insert(id).second) problem.landmarkIds.push_back(id);
        }
    }
    std::vector<int> fixedKeyframes;
    std::set<int> fixedSet;
    for (int id : problem.landmarkIds) {
        for (const auto &[kfId, idx] : map.landmarks.at(id).observations) {
            if (localSet.count(kfId) == 0 && !map.keyframes.at(kfId).bad && fixedSet.insert(kfId).second)
                fixedKeyframes.push_back(kfId);
        }
    }

    std::map<int, std::size_t> poseIndex;
    auto addPose = [&](int kfId, bool fixed) {
        poseIndex[kfId] = problem.poses.size();
        problem.keyframeIds.push_back(kfId);
        problem.poses.push_back(map.keyframes.at(kfId).T_cw);
        problem.fixedPose.push_back(fixed);
    };
    for (int kfId : localKeyframes)
        addPose(kfId, kfId == map.originKeyframeId);
    for (int kfId : fixedKeyframes)
        addPose(kfId, true);

    for (std::size_t j = 0; j < problem.landmarkIds.size(); ++j) {
        const Landmark &lm = map.landmarks.at(problem.landmarkIds[j]);
        problem.points.push_back(lm.position);
        for (const auto &[kfId, idx] : lm.observations) {
            const auto it = poseIndex.find(kfId);
            if (it == poseIndex.end()) continue;
            const Keyframe &kf = map.keyframes.at(kfId);
            const Keypoint &kp = kf.frame.keypoints[static_cast<std::size_t>(idx)];
            const float depth = kf.frame.depths[static_cast<std::size_t>(idx)];
            BAObservation obs;
            obs.poseIndex = it->second;
            obs.pointIndex = j;
            obs.hasDepth = map.config.useDepthInBundleAdjustment && depth > 0.0f;
            obs.measured = makeVec3(kp.u, kp.v, obs.hasDepth ? kp.u - map.camera.bf / depth : 0.0);
            obs.information = map.invLevelSigma2[static_cast<std::size_t>(kp.octave)];
            obs.keyframeId = kfId;
            obs.landmarkId = lm.id;
            problem.observations.push_back(obs);
        }
    }
    return problem;
}

BAResult localBundleAdjustment(Map &map, int keyframeId, ExecutionMode mode) {
    BAResult result;
    BAProblem problem = buildLocalBAProblem(map, keyframeId);
    for (bool fixed : problem.fixedPose) {
        if (fixed)
            ++result.fixedKeyframes;
        else
            ++result.optimizedKeyframes;
    }
    result.optimizedPoints = static_cast<int>(problem.points.size());
    result.observations = static_cast<int>(problem.observations.size());
    if (problem.observations.empty()) return result;

    result.initialCost = evaluateCost(problem, true, mode);
    optimizeBAProblem(problem, map.config.baFirstIterations, true, mode);

    // ORB-SLAM 2015 removes first-pass outlier edges permanently, then runs ten more
    // iterations with the Huber kernels still attached.
    std::vector<ObservationResidual> residuals;
    computeObservationResiduals(problem, residuals, false, mode);
    for (std::size_t k = 0; k < residuals.size(); ++k) {
        BAObservation &obs = problem.observations[k];
        if (!std::isfinite(residuals[k].chi2) || residuals[k].chi2 > chi2Threshold(problem, obs) ||
            !residuals[k].inFront)
            obs.active = false;
    }
    result.finalCost = optimizeBAProblem(problem, map.config.baSecondIterations, true, mode);

    // ...and at the end.
    computeObservationResiduals(problem, residuals, false, mode);
    std::vector<std::pair<int, int>> outliers;
    for (std::size_t k = 0; k < residuals.size(); ++k) {
        const BAObservation &obs = problem.observations[k];
        if (!obs.active || !std::isfinite(residuals[k].chi2) ||
            residuals[k].chi2 > chi2Threshold(problem, obs) || !residuals[k].inFront)
            outliers.emplace_back(obs.landmarkId, obs.keyframeId);
    }
    for (const auto &[landmarkId, kfId] : outliers) {
        if (map.landmarks.at(landmarkId).bad) continue;
        eraseObservation(map, landmarkId, kfId);
        ++result.outliersRemoved;
    }

    for (std::size_t i = 0; i < problem.poses.size(); ++i)
        if (!problem.fixedPose[i]) map.keyframes.at(problem.keyframeIds[i]).T_cw = problem.poses[i];
    for (std::size_t j = 0; j < problem.points.size(); ++j) {
        const int id = problem.landmarkIds[j];
        if (map.landmarks.at(id).bad) continue;
        map.landmarks.at(id).position = problem.points[j];
        updateNormalAndDepth(map, id);
    }
    return result;
}

LandmarkRefinement refineLandmark(const Map &map, int landmarkId, ExecutionMode mode) {
    LandmarkRefinement result;
    result.landmarkId = landmarkId;
    const Landmark &lm = map.landmarks.at(landmarkId);
    result.position = lm.position;

    std::vector<std::pair<Pose, BAObservation>> observations;
    for (const auto &[kfId, idx] : lm.observations) {
        const Keyframe &kf = map.keyframes.at(kfId);
        const Keypoint &kp = kf.frame.keypoints[static_cast<std::size_t>(idx)];
        const float depth = kf.frame.depths[static_cast<std::size_t>(idx)];
        BAObservation obs;
        obs.hasDepth = map.config.useDepthInBundleAdjustment && depth > 0.0f;
        obs.measured = makeVec3(kp.u, kp.v, obs.hasDepth ? kp.u - map.camera.bf / depth : 0.0);
        obs.information = map.invLevelSigma2[static_cast<std::size_t>(kp.octave)];
        observations.emplace_back(kf.T_cw, obs);
    }
    if (observations.empty()) return result;

    auto evaluate = [&](const Vec3 &X, bool jacobians) {
        std::vector<ObservationResidual> values(observations.size());
        std::size_t i = 0;
        if (mode == ExecutionMode::SerialSimd) {
            for (; i + 1 < observations.size(); i += 2)
                evaluateObservationPair(observations[i].first, observations[i + 1].first, X, X,
                                        observations[i].second, observations[i + 1].second, map.camera,
                                        jacobians, values[i], values[i + 1]);
        }
        for (; i < observations.size(); ++i)
            values[i] =
                evaluateObservation(observations[i].first, X, observations[i].second, map.camera, jacobians);
        return values;
    };
    auto cost = [&](const Vec3 &X) {
        double c = 0.0;
        const auto values = evaluate(X, false);
        for (std::size_t i = 0; i < observations.size(); ++i) {
            const auto &obs = observations[i].second;
            const double threshold = obs.hasDepth ? map.config.chi2Stereo : map.config.chi2Mono;
            c += robustCost(values[i].chi2, threshold, true);
        }
        return c;
    };

    Vec3 X = lm.position;
    double current = cost(X);
    result.initialChi2 = current;
    for (int it = 0; it < kPointIterations; ++it) {
        Mat3 H{};
        Vec3 g{};
        const auto values = evaluate(X, true);
        for (std::size_t i = 0; i < observations.size(); ++i) {
            const auto &obs = observations[i].second;
            const ObservationResidual &r = values[i];
            if (!r.inFront) continue;
            const double threshold = obs.hasDepth ? map.config.chi2Stereo : map.config.chi2Mono;
            const double w = obs.information * robustWeight(r.chi2, threshold, true);
            const Mat3 JT = r.jacobianPoint.transpose();
            H += w * detail::multiply(JT, r.jacobianPoint, mode);
            g += w * (JT * r.error);
        }
        Mat3 Hinv;
        if (!invert3(H, Hinv)) break;
        const Vec3 step = -(Hinv * g);
        const Vec3 candidate = X + step;
        const double next = cost(candidate);
        if (!(next < current)) break;
        X = candidate;
        current = next;
        if (norm(step) < 1e-8) break;
    }
    result.position = X;
    result.finalChi2 = current;
    result.improved = current < result.initialChi2;
    return result;
}

int refineLandmarks(Map &map, const LocalMap &localMap, ExecutionMode mode) {
    std::vector<int> ids;
    std::set<int> seen;
    for (int slot : localMap.landmarkIds) {
        const int id = resolveLandmarkId(map, slot);
        if (id >= 0 && seen.insert(id).second) ids.push_back(id);
    }
    std::vector<LandmarkRefinement> results(ids.size());
    detail::runPartitions(ids.size(), detail::workerCount(ids.size(), mode, 16),
                          [&](std::size_t begin, std::size_t end, std::size_t) {
                              for (std::size_t i = begin; i < end; ++i)
                                  results[i] = refineLandmark(map, ids[i], mode);
                          });
    // Shared map is written only after all points are computed.
    int improved = 0;
    for (const LandmarkRefinement &r : results) {
        if (!r.improved) continue;
        map.landmarks.at(r.landmarkId).position = r.position;
        updateNormalAndDepth(map, r.landmarkId);
        ++improved;
    }
    return improved;
}

double meanReprojectionError(const Map &map, const std::vector<int> &landmarkIds) {
    double sum = 0.0;
    std::size_t count = 0;
    for (int slot : landmarkIds) {
        const int id = resolveLandmarkId(map, slot);
        if (id < 0) continue;
        const Landmark &lm = map.landmarks.at(id);
        for (const auto &[kfId, idx] : lm.observations) {
            const Keyframe &kf = map.keyframes.at(kfId);
            const Keypoint &kp = kf.frame.keypoints[static_cast<std::size_t>(idx)];
            BAObservation obs;
            obs.measured = makeVec3(kp.u, kp.v, 0.0);
            obs.information = 1.0;
            const ObservationResidual r = evaluateObservation(kf.T_cw, lm.position, obs, map.camera, false);
            sum += std::sqrt(r.chi2);
            ++count;
        }
    }
    return count > 0 ? sum / static_cast<double>(count) : 0.0;
}

} // namespace slam
