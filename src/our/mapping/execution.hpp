#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <thread>
#include <vector>

#include "slam_types.hpp"

#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace slam::detail {

// Fixed partitions and ordered reductions keep results independent of scheduling.
inline std::size_t workerCount(std::size_t count, ExecutionMode mode, std::size_t grain = 64) {
    if (mode != ExecutionMode::Parallel || count < grain * 2) return 1;
    return std::min({std::size_t{8},
                     std::max(std::size_t{1}, static_cast<std::size_t>(std::thread::hardware_concurrency())),
                     count / grain});
}

template <typename Function> void runPartitions(std::size_t count, std::size_t workers, Function function) {
    if (workers <= 1) {
        function(0, count, 0);
        return;
    }
    std::vector<std::exception_ptr> errors(workers);
    std::vector<std::jthread> threads;
    threads.reserve(workers);
    for (std::size_t w = 0; w < workers; ++w) {
        threads.emplace_back([&, w] {
            try {
                function(count * w / workers, count * (w + 1) / workers, w);
            } catch (...) {
                errors[w] = std::current_exception();
            }
        });
    }
    for (auto &thread : threads)
        thread.join();
    for (const auto &error : errors)
        if (error) std::rethrow_exception(error);
}

// Two independent observations per vector. No approximate reciprocal or fast-math.
struct Double2 {
#if defined(__aarch64__)
    float64x2_t value;
    Double2(double a, double b) {
        const double v[2]{a, b};
        value = vld1q_f64(v);
    }
    explicit Double2(float64x2_t v) : value(v) {}
    void store(double *v) const { vst1q_f64(v, value); }
#elif defined(__SSE2__)
    __m128d value;
    Double2(double a, double b) : value(_mm_set_pd(b, a)) {}
    explicit Double2(__m128d v) : value(v) {}
    void store(double *v) const { _mm_storeu_pd(v, value); }
#else
    std::array<double, 2> value;
    Double2(double a, double b) : value{a, b} {}
    void store(double *v) const {
        v[0] = value[0];
        v[1] = value[1];
    }
#endif
    Double2() : Double2(0.0, 0.0) {}
    explicit Double2(double v) : Double2(v, v) {}
};

#define SLAM_BINARY_OP(OP, NEON, SSE)                                                                        \
    inline Double2 operator OP(Double2 a, Double2 b) { return Double2(NEON(a.value, b.value)); }
#if defined(__aarch64__)
SLAM_BINARY_OP(+, vaddq_f64, _mm_add_pd)
SLAM_BINARY_OP(-, vsubq_f64, _mm_sub_pd)
SLAM_BINARY_OP(*, vmulq_f64, _mm_mul_pd)
SLAM_BINARY_OP(/, vdivq_f64, _mm_div_pd)
#elif defined(__SSE2__)
#undef SLAM_BINARY_OP
#define SLAM_BINARY_OP(OP, NEON, SSE)                                                                        \
    inline Double2 operator OP(Double2 a, Double2 b) { return Double2(SSE(a.value, b.value)); }
SLAM_BINARY_OP(+, vaddq_f64, _mm_add_pd)
SLAM_BINARY_OP(-, vsubq_f64, _mm_sub_pd)
SLAM_BINARY_OP(*, vmulq_f64, _mm_mul_pd)
SLAM_BINARY_OP(/, vdivq_f64, _mm_div_pd)
#else
#undef SLAM_BINARY_OP
#define SLAM_BINARY_OP(OP)                                                                                   \
    inline Double2 operator OP(Double2 a, Double2 b) {                                                       \
        return Double2(a.value[0] OP b.value[0], a.value[1] OP b.value[1]);                                  \
    }
SLAM_BINARY_OP(+)
SLAM_BINARY_OP(-)
SLAM_BINARY_OP(*)
SLAM_BINARY_OP(/)
#endif
#undef SLAM_BINARY_OP
inline Double2 operator-(Double2 a) { return Double2(0.0) - a; }

inline void requireSimdSupport(ExecutionMode mode) {
#if !defined(__aarch64__) && !defined(__SSE2__)
    if (mode == ExecutionMode::SerialSimd)
        throw std::runtime_error("serial_simd requires AArch64 NEON or SSE2");
#else
    (void)mode;
#endif
}

inline std::array<Double2, 3> transformPair(const Pose &a, const Pose &b, const Vec3 &x, const Vec3 &y) {
    std::array<Double2, 3> out;
    for (std::size_t r = 0; r < 3; ++r) {
        Double2 v(0.0);
        for (std::size_t c = 0; c < 3; ++c)
            v = v + Double2(a.R(r, c), b.R(r, c)) * Double2(x[c], y[c]);
        out[r] = v + Double2(a.t[r], b.t[r]);
    }
    return out;
}

template <std::size_t R, std::size_t K, std::size_t C>
Matrix<R, C> multiply(const Matrix<R, K> &a, const Matrix<K, C> &b, ExecutionMode mode) {
    if (mode != ExecutionMode::SerialSimd) return a * b;
    Matrix<R, C> out;
    for (std::size_t r = 0; r < R; ++r) {
        std::size_t c = 0;
        for (; c + 1 < C; c += 2) {
            Double2 v(0.0);
            for (std::size_t k = 0; k < K; ++k)
                v = v + Double2(a(r, k)) * Double2(b(k, c), b(k, c + 1));
            v.store(&out.data[r * C + c]);
        }
        for (; c < C; ++c)
            for (std::size_t k = 0; k < K; ++k)
                out(r, c) += a(r, k) * b(k, c);
    }
    return out;
}

} // namespace slam::detail
