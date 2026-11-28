#include "bench.hpp"
#include "linalg.hpp"

// an SPD system shaped like a pose-graph normal matrix (6 unknowns per keyframe);
// measures one Cholesky solve, the linear solver of local BA and pose-graph optimization
long long run(const std::string& path) {
    const auto system = read_system(path);

#ifdef BENCH_OPENCV
    const cv::Mat A(static_cast<int>(system.n), static_cast<int>(system.n), CV_64F, const_cast<double*>(system.A.data()));
    const cv::Mat b(static_cast<int>(system.n), 1, CV_64F, const_cast<double*>(system.b.data()));
    cv::Mat x;

    auto start_time = get_current_time_fenced();
    const bool solved = cv::solve(A, b, x, cv::DECOMP_CHOLESKY);
    auto finish_time = get_current_time_fenced();
    auto total_time = finish_time - start_time;

    sink = sink + (solved ? micro(x.at<double>(0)) : -1);
#else
    slam::DenseMatrix A(system.n, system.n);
    for (size_t r = 0; r < system.n; ++r)
        for (size_t c = 0; c < system.n; ++c) A(r, c) = system.A[r * system.n + c];
    std::vector<double> x;

    auto start_time = get_current_time_fenced();
    const bool solved = slam::solveCholesky(A, system.b, x, bench_mode);
    auto finish_time = get_current_time_fenced();
    auto total_time = finish_time - start_time;

    sink = sink + (solved ? micro(x[0]) : -1);
#endif
    return to_ns(total_time);
}

int main(int argc, char** argv) {
    const command_line_options_t options(argc, argv);
    std::cout << run(options.get_input_file()) << std::endl;
}
