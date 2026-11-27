#pragma once

// Pose graph optimization over the Essential Graph (ORB-SLAM Sec. VII-D and appendix).
// ORB-SLAM optimizes Sim(3) for monocular input; with RGB-D the scale is observable and fixed
// to one, so nodes and edges are SE(3) (as ORB-SLAM2 does for stereo / RGB-D).
//
// Edge (i, j) stores the relative pose T_ij = T_iw * T_wj measured when the graph is built.
// Residual (paper appendix): e_ij = log_SE3(T_ij * T_jw * T_iw^-1), information = identity.

#include <map>
#include <vector>

#include "linalg.hpp"
#include "slam_types.hpp"

namespace slam {

enum class GraphEdgeType { Odometry, Covisibility, Loop };

struct GraphEdge {
    std::size_t from = 0; // node index i
    std::size_t to = 0;   // node index j
    Pose measurement;     // T_ij
    GraphEdgeType type = GraphEdgeType::Odometry;
    Mat6 information = Mat6::identity();
};

struct PoseGraph {
    std::vector<int> keyframeIds;
    std::map<int, std::size_t> nodeIndex;
    std::vector<Pose> poses; // T_iw
    std::vector<bool> fixed;
    std::vector<GraphEdge> edges;
};

struct EdgeLinearization {
    Vec6 residual{};
    Mat6 jacobianFrom{};
    Mat6 jacobianTo{};
};

struct GraphSystem {
    DenseMatrix H;
    std::vector<double> b;
    std::vector<int> variableOfNode; // -1 for fixed nodes
    double chi2 = 0.0;
};

struct PoseGraphResult {
    double initialChi2 = 0.0;
    double finalChi2 = 0.0;
    int iterations = 0;
};

std::size_t addGraphNode(PoseGraph &graph, int keyframeId, const Pose &T_iw, bool fixed);
void addOdometryEdge(PoseGraph &graph, int a, int b, const Pose &relativePose);
void addCovisibilityEdge(PoseGraph &graph, int a, int b, const Pose &relativePose);
void addLoopEdge(PoseGraph &graph, int a, int b, const Pose &relativePose);

Vec6 computeGraphResidual(const GraphEdge &edge, const std::vector<Pose> &poses);
// Numerical central differences w.r.t. left increments of both poses.
EdgeLinearization linearizeEdge(const GraphEdge &edge, const std::vector<Pose> &poses);
GraphSystem linearizePoseGraph(const PoseGraph &graph, ExecutionMode mode);
bool solveGraphIncrement(const GraphSystem &system, double lambda, std::vector<double> &delta);
double poseGraphChi2(const PoseGraph &graph);
// Levenberg-Marquardt; fixed nodes (at least the first keyframe) are not moved.
PoseGraphResult optimizePoseGraph(PoseGraph &graph, int iterations, ExecutionMode mode);

} // namespace slam
