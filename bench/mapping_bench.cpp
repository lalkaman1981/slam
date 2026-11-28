#include "bundle_adjustment.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {
using namespace slam;
BAProblem makeProblem() {
    BAProblem problem;
    problem.poses.resize(5);
    problem.fixedPose.assign(5, false);
    problem.fixedPose[0] = true;
    for (std::size_t i = 0; i < 5; ++i)
        problem.poses[i].t[0] = -0.1 * static_cast<double>(i);
    for (int i = 0; i < 12000; ++i) {
        const Vec3 point = makeVec3((i % 100 - 50) * 0.02, ((i / 100) % 100 - 50) * 0.02, 2.0 + 0.001 * i);
        problem.points.push_back(point);
        for (std::size_t j = 0; j < 5; ++j) {
            const auto x = transformPoint(problem.poses[j], point);
            const double u = problem.camera.fx * x[0] / x[2] + problem.camera.cx;
            BAObservation observation;
            observation.poseIndex = j;
            observation.pointIndex = static_cast<std::size_t>(i);
            observation.hasDepth = (i % 3) != 0;
            observation.measured =
                makeVec3(u + 0.1, problem.camera.fy * x[1] / x[2] + problem.camera.cy - 0.2,
                         u - problem.camera.bf / x[2] + 0.05);
            problem.observations.push_back(observation);
        }
    }
    return problem;
}

// The dense reduced system of a 30-camera window, with a known solution.
void benchmarkSolver() {
    constexpr std::size_t size = 180;
    constexpr int repetitions = 30;
    DenseMatrix matrix(size, size);
    std::vector<double> truth(size), rhs(size), solution;
    for (std::size_t i = 0; i < size; ++i) {
        truth[i] = 0.1 * (static_cast<int>(i % 7) - 3);
        for (std::size_t j = 0; j < size; ++j)
            matrix(i, j) = i == j ? 2.0 : 0.05 / (1.0 + std::abs(static_cast<double>(i) - j));
    }
    for (std::size_t i = 0; i < size; ++i)
        for (std::size_t j = 0; j < size; ++j) rhs[i] += matrix(i, j) * truth[j];
    double serialTime = 0.0;
    std::cout << "mode,dimension,repetitions,ms_per_solve,speedup,max_solution_error\n";
    for (auto mode : {ExecutionMode::Serial, ExecutionMode::SerialSimd}) {
        auto solve = [&] {
            if (!solveCholesky(matrix, rhs, solution, mode)) throw std::runtime_error("SPD solve failed");
        };
        for (int i = 0; i < 3; ++i) solve();
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repetitions; ++i) solve();
        const double elapsed = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - start).count() / repetitions;
        double error = 0.0;
        for (std::size_t i = 0; i < size; ++i) error = std::max(error, std::abs(solution[i] - truth[i]));
        if (error > 1e-10) throw std::runtime_error("solver returned an inaccurate solution");
        if (mode == ExecutionMode::Serial) serialTime = elapsed;
        std::cout << (mode == ExecutionMode::Serial ? "serial" : "serial_simd") << ',' << size << ','
                  << repetitions << ',' << std::setprecision(8) << elapsed << ',' << serialTime / elapsed
                  << ',' << error << '\n';
    }
}
} // namespace

int main(int argc, char **argv) {
    try {
        // stderr keeps the CSV on stdout unchanged.
        std::cerr << "serial_simd backend: " << slam::simdBackendName() << '\n';
        if (argc == 2 && std::string_view(argv[1]) == "--solver") {
            benchmarkSolver();
            return 0;
        }
        if (argc != 1) throw std::invalid_argument("Usage: slam_mapping_bench [--solver]");
        const auto problem = makeProblem();
        constexpr int repetitions = 50;
        std::vector<slam::ObservationResidual> reference, results;
        slam::computeObservationResiduals(problem, reference, true, slam::ExecutionMode::Serial);
        double serialTime = 0.0;
        std::cout << "mode,observations,repetitions,ms_per_batch,speedup,max_absolute_difference\n";
        for (const auto mode :
             {slam::ExecutionMode::Serial, slam::ExecutionMode::SerialSimd, slam::ExecutionMode::Parallel}) {
            for (int warmup = 0; warmup < 3; ++warmup)
                slam::computeObservationResiduals(problem, results, true, mode);
            const auto start = std::chrono::steady_clock::now();
            for (int iteration = 0; iteration < repetitions; ++iteration)
                slam::computeObservationResiduals(problem, results, true, mode);
            const double elapsed =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() /
                repetitions;
            double difference = 0.0;
            for (std::size_t i = 0; i < results.size(); ++i) {
                for (std::size_t q = 0; q < 3; ++q)
                    difference = std::max(difference, std::abs(results[i].error[q] - reference[i].error[q]));
                for (std::size_t q = 0; q < 18; ++q)
                    difference = std::max(
                        difference, std::abs(results[i].jacobianPose[q] - reference[i].jacobianPose[q]));
                for (std::size_t q = 0; q < 9; ++q)
                    difference = std::max(
                        difference, std::abs(results[i].jacobianPoint[q] - reference[i].jacobianPoint[q]));
            }
            if (difference > 1e-8) throw std::runtime_error("mode changed residuals or Jacobians");
            const char *name = mode == slam::ExecutionMode::Serial       ? "serial"
                               : mode == slam::ExecutionMode::SerialSimd ? "serial_simd"
                                                                         : "parallel";
            if (mode == slam::ExecutionMode::Serial) serialTime = elapsed;
            std::cout << name << ',' << results.size() << ',' << repetitions << ',' << std::setprecision(8)
                      << elapsed << ',' << serialTime / elapsed << ',' << difference << '\n';
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
