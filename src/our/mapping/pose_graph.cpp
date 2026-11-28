// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include "pose_graph.hpp"
#include "execution.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace slam {

namespace {

constexpr double kNumericStep = 1e-6;
constexpr int kMaxLambdaTries = 10;

void addEdge(PoseGraph &graph, int a, int b, const Pose &relativePose, GraphEdgeType type) {
    GraphEdge edge;
    edge.from = graph.nodeIndex.at(a);
    edge.to = graph.nodeIndex.at(b);
    edge.measurement = relativePose;
    edge.type = type;
    graph.edges.push_back(edge);
}

double edgeChi2(const GraphEdge &edge, const Vec6 &e) { return dot(e, edge.information * e); }

} // namespace

std::size_t addGraphNode(PoseGraph &graph, int keyframeId, const Pose &T_iw, bool fixed) {
    const std::size_t index = graph.poses.size();
    graph.nodeIndex[keyframeId] = index;
    graph.keyframeIds.push_back(keyframeId);
    graph.poses.push_back(T_iw);
    graph.fixed.push_back(fixed);
    return index;
}

void addOdometryEdge(PoseGraph &graph, int a, int b, const Pose &relativePose) {
    addEdge(graph, a, b, relativePose, GraphEdgeType::Odometry);
}

void addCovisibilityEdge(PoseGraph &graph, int a, int b, const Pose &relativePose) {
    addEdge(graph, a, b, relativePose, GraphEdgeType::Covisibility);
}

void addLoopEdge(PoseGraph &graph, int a, int b, const Pose &relativePose) {
    addEdge(graph, a, b, relativePose, GraphEdgeType::Loop);
}

Vec6 computeGraphResidual(const GraphEdge &edge, const std::vector<Pose> &poses) {
    const Pose &Tiw = poses[edge.from];
    const Pose &Tjw = poses[edge.to];
    return logSE3(composePose(composePose(edge.measurement, Tjw), inversePose(Tiw)));
}

EdgeLinearization linearizeEdge(const GraphEdge &edge, const std::vector<Pose> &poses) {
    EdgeLinearization lin;
    lin.residual = computeGraphResidual(edge, poses);
    std::vector<Pose> local{poses[edge.from], poses[edge.to]};
    GraphEdge localEdge = edge;
    localEdge.from = 0;
    localEdge.to = 1;
    for (std::size_t node = 0; node < 2; ++node) {
        Mat6 &J = node == 0 ? lin.jacobianFrom : lin.jacobianTo;
        const Pose original = local[node];
        for (std::size_t k = 0; k < 6; ++k) {
            Vec6 d{};
            d[k] = kNumericStep;
            local[node] = applyLeftIncrement(original, d);
            const Vec6 plus = computeGraphResidual(localEdge, local);
            d[k] = -kNumericStep;
            local[node] = applyLeftIncrement(original, d);
            const Vec6 minus = computeGraphResidual(localEdge, local);
            for (std::size_t r = 0; r < 6; ++r)
                J(r, k) = (plus[r] - minus[r]) / (2.0 * kNumericStep);
        }
        local[node] = original;
    }
    return lin;
}

GraphSystem linearizePoseGraph(const PoseGraph &graph, ExecutionMode mode) {
    GraphSystem system;
    system.variableOfNode.assign(graph.poses.size(), -1);
    std::size_t nVars = 0;
    for (std::size_t i = 0; i < graph.poses.size(); ++i)
        if (!graph.fixed[i]) system.variableOfNode[i] = static_cast<int>(nVars++);
    system.H = DenseMatrix(6 * nVars, 6 * nVars);
    system.b.assign(6 * nVars, 0.0);

    struct Contribution {
        Mat6 Hii{}, Hjj{}, Hij{};
        Vec6 gi{}, gj{};
        double chi2 = 0.0;
    };
    // Edge blocks are private to each worker. The dense graph is assembled only after joining.
    std::vector<Contribution> contributions(graph.edges.size());
    detail::runPartitions(graph.edges.size(), detail::workerCount(graph.edges.size(), mode, 4),
                          [&](std::size_t begin, std::size_t end, std::size_t) {
                              for (std::size_t e = begin; e < end; ++e) {
                                  const auto &edge = graph.edges[e];
                                  const auto lin = linearizeEdge(edge, graph.poses);
                                  auto &c = contributions[e];
                                  c.chi2 = edgeChi2(edge, lin.residual);
                                  const auto JiT =
                                      detail::multiply(lin.jacobianFrom.transpose(), edge.information, mode);
                                  const auto JjT =
                                      detail::multiply(lin.jacobianTo.transpose(), edge.information, mode);
                                  c.Hii = detail::multiply(JiT, lin.jacobianFrom, mode);
                                  c.Hjj = detail::multiply(JjT, lin.jacobianTo, mode);
                                  c.Hij = detail::multiply(JiT, lin.jacobianTo, mode);
                                  c.gi = JiT * lin.residual;
                                  c.gj = JjT * lin.residual;
                              }
                          });
    for (std::size_t e = 0; e < graph.edges.size(); ++e) {
        const auto &edge = graph.edges[e];
        const auto &c = contributions[e];
        system.chi2 += c.chi2;
        const int vi = system.variableOfNode[edge.from], vj = system.variableOfNode[edge.to];
        if (vi >= 0) {
            const std::size_t oi = 6 * static_cast<std::size_t>(vi);
            system.H.addBlock(oi, oi, c.Hii);
            for (std::size_t q = 0; q < 6; ++q)
                system.b[oi + q] += c.gi[q];
        }
        if (vj >= 0) {
            const std::size_t oj = 6 * static_cast<std::size_t>(vj);
            system.H.addBlock(oj, oj, c.Hjj);
            for (std::size_t q = 0; q < 6; ++q)
                system.b[oj + q] += c.gj[q];
        }
        if (vi >= 0 && vj >= 0) {
            const std::size_t oi = 6 * static_cast<std::size_t>(vi), oj = 6 * static_cast<std::size_t>(vj);
            system.H.addBlock(oi, oj, c.Hij);
            system.H.addBlock(oj, oi, c.Hij.transpose());
        }
    }

    return system;
}

bool solveGraphIncrement(const GraphSystem &system, double lambda, std::vector<double> &delta,
                         ExecutionMode mode) {
    DenseMatrix damped = system.H;
    std::vector<double> rhs(system.b.size());
    for (std::size_t i = 0; i < rhs.size(); ++i) {
        damped(i, i) += lambda;
        rhs[i] = -system.b[i];
    }
    return solveCholesky(damped, rhs, delta, mode);
}

double poseGraphChi2(const PoseGraph &graph) {
    double chi2 = 0.0;
    for (const GraphEdge &edge : graph.edges)
        chi2 += edgeChi2(edge, computeGraphResidual(edge, graph.poses));
    return chi2;
}

PoseGraphResult optimizePoseGraph(PoseGraph &graph, int iterations, ExecutionMode mode) {
    PoseGraphResult result;
    result.initialChi2 = poseGraphChi2(graph);
    double cost = result.initialChi2;
    double lambda = -1.0;
    double nu = 2.0;

    for (int it = 0; it < iterations; ++it) {
        const GraphSystem system = linearizePoseGraph(graph, mode);
        if (system.b.empty()) break;
        if (lambda < 0.0) {
            double maxDiag = 0.0;
            for (std::size_t i = 0; i < system.b.size(); ++i)
                maxDiag = std::max(maxDiag, system.H(i, i));
            lambda = 1e-5 * std::max(maxDiag, 1e-12);
        }
        bool accepted = false;
        double maxStep = 0.0;
        for (int attempt = 0; attempt < kMaxLambdaTries && !accepted; ++attempt) {
            std::vector<double> delta;
            if (!solveGraphIncrement(system, lambda, delta, mode)) {
                lambda *= nu;
                nu *= 2.0;
                continue;
            }
            std::vector<Pose> candidate = graph.poses;
            double predicted = 0.0;
            maxStep = 0.0;
            for (std::size_t i = 0; i < graph.poses.size(); ++i) {
                const int v = system.variableOfNode[i];
                if (v < 0) continue;
                Vec6 d;
                for (std::size_t q = 0; q < 6; ++q) {
                    d[q] = delta[6 * static_cast<std::size_t>(v) + q];
                    maxStep = std::max(maxStep, std::abs(d[q]));
                }
                candidate[i] = applyLeftIncrement(graph.poses[i], d);
            }
            for (std::size_t q = 0; q < delta.size(); ++q)
                predicted += delta[q] * (lambda * delta[q] - system.b[q]);
            std::swap(candidate, graph.poses);
            const double newCost = poseGraphChi2(graph);
            const double rho = predicted > 0.0 ? (cost - newCost) / predicted : -1.0;
            if (std::isfinite(newCost) && rho > 0.0) {
                cost = newCost;
                const double t = 2.0 * rho - 1.0;
                lambda *= std::max(1.0 / 3.0, 1.0 - t * t * t);
                nu = 2.0;
                accepted = true;
            } else {
                std::swap(candidate, graph.poses);
                lambda *= nu;
                nu *= 2.0;
            }
        }
        ++result.iterations;
        if (!accepted || maxStep < 1e-12) break;
    }
    result.finalChi2 = cost;
    return result;
}

} // namespace slam
