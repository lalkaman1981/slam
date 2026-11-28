#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <span>
#include <thread>
#include <vector>

#include "slam_types.hpp"

// serial_simd backend, chosen at compile time. Explicit intrinsics are used where they
// exist; other GCC/Clang targets use portable vector extensions, which the compiler
// lowers to the target's SIMD unit (RVV, VSX, wasm simd128, ...) or to scalar code.
// SLAM_SIMD_FORCE_PORTABLE / SLAM_SIMD_FORCE_SCALAR override detection so every
// backend can be built and tested on one machine.
#if defined(SLAM_SIMD_FORCE_SCALAR)
#define SLAM_SIMD_SCALAR 1
#elif defined(SLAM_SIMD_FORCE_PORTABLE)
#if !defined(__GNUC__) && !defined(__clang__)
#error "SLAM_SIMD_FORCE_PORTABLE requires GCC or Clang vector extensions"
#endif
#define SLAM_SIMD_VECTOR_EXT 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#define SLAM_SIMD_NEON 1
#elif defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define SLAM_SIMD_SSE2 1
#elif defined(__GNUC__) || defined(__clang__)
#define SLAM_SIMD_VECTOR_EXT 1
#else
#define SLAM_SIMD_SCALAR 1
#endif

#if defined(SLAM_SIMD_NEON)
#include <arm_neon.h>
#elif defined(SLAM_SIMD_SSE2)
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
#if defined(SLAM_SIMD_NEON)
    float64x2_t value;
    Double2(double a, double b) {
        const double v[2]{a, b};
        value = vld1q_f64(v);
    }
    explicit Double2(float64x2_t v) : value(v) {}
    static Double2 load(const double *v) { return Double2(vld1q_f64(v)); }
    void store(double *v) const { vst1q_f64(v, value); }
    double sum() const { return vaddvq_f64(value); }
#elif defined(SLAM_SIMD_SSE2)
    __m128d value;
    Double2(double a, double b) : value(_mm_set_pd(b, a)) {}
    explicit Double2(__m128d v) : value(v) {}
    static Double2 load(const double *v) { return Double2(_mm_loadu_pd(v)); }
    void store(double *v) const { _mm_storeu_pd(v, value); }
    double sum() const { return _mm_cvtsd_f64(_mm_add_sd(value, _mm_unpackhi_pd(value, value))); }
#elif defined(SLAM_SIMD_VECTOR_EXT)
    typedef double Native __attribute__((vector_size(2 * sizeof(double))));
    Native value;
    Double2(double a, double b) : value(Native{a, b}) {}
    explicit Double2(Native v) : value(v) {}
    // memcpy keeps unaligned loads/stores well defined; compilers emit one vector move.
    static Double2 load(const double *v) {
        Native r;
        std::memcpy(&r, v, sizeof r);
        return Double2(r);
    }
    void store(double *v) const { std::memcpy(v, &value, sizeof value); }
    double sum() const { return value[0] + value[1]; }
#else
    std::array<double, 2> value;
    Double2(double a, double b) : value{a, b} {}
    static Double2 load(const double *v) { return Double2(v[0], v[1]); }
    double sum() const { return value[0] + value[1]; }
    void store(double *v) const {
        v[0] = value[0];
        v[1] = value[1];
    }
#endif
    Double2() : Double2(0.0, 0.0) {}
    explicit Double2(double v) : Double2(v, v) {}
};

#if defined(SLAM_SIMD_NEON)
inline Double2 operator+(Double2 a, Double2 b) { return Double2(vaddq_f64(a.value, b.value)); }
inline Double2 operator-(Double2 a, Double2 b) { return Double2(vsubq_f64(a.value, b.value)); }
inline Double2 operator*(Double2 a, Double2 b) { return Double2(vmulq_f64(a.value, b.value)); }
inline Double2 operator/(Double2 a, Double2 b) { return Double2(vdivq_f64(a.value, b.value)); }
#elif defined(SLAM_SIMD_SSE2)
inline Double2 operator+(Double2 a, Double2 b) { return Double2(_mm_add_pd(a.value, b.value)); }
inline Double2 operator-(Double2 a, Double2 b) { return Double2(_mm_sub_pd(a.value, b.value)); }
inline Double2 operator*(Double2 a, Double2 b) { return Double2(_mm_mul_pd(a.value, b.value)); }
inline Double2 operator/(Double2 a, Double2 b) { return Double2(_mm_div_pd(a.value, b.value)); }
#elif defined(SLAM_SIMD_VECTOR_EXT)
inline Double2 operator+(Double2 a, Double2 b) { return Double2(a.value + b.value); }
inline Double2 operator-(Double2 a, Double2 b) { return Double2(a.value - b.value); }
inline Double2 operator*(Double2 a, Double2 b) { return Double2(a.value * b.value); }
inline Double2 operator/(Double2 a, Double2 b) { return Double2(a.value / b.value); }
#else
inline Double2 operator+(Double2 a, Double2 b) { return Double2(a.value[0] + b.value[0], a.value[1] + b.value[1]); }
inline Double2 operator-(Double2 a, Double2 b) { return Double2(a.value[0] - b.value[0], a.value[1] - b.value[1]); }
inline Double2 operator*(Double2 a, Double2 b) { return Double2(a.value[0] * b.value[0], a.value[1] * b.value[1]); }
inline Double2 operator/(Double2 a, Double2 b) { return Double2(a.value[0] / b.value[0], a.value[1] / b.value[1]); }
#endif
inline Double2 operator-(Double2 a) { return Double2(0.0) - a; }

inline constexpr const char *simdBackend() {
#if defined(SLAM_SIMD_NEON)
    return "neon";
#elif defined(SLAM_SIMD_SSE2)
    return "sse2";
#elif defined(SLAM_SIMD_VECTOR_EXT)
    return "vector-extensions";
#else
    return "scalar";
#endif
}

// Contiguous, unaligned loads work with std::vector storage on every backend. Keep
// separate multiply/add operations and handle the tail without reading past it.
inline double innerProduct(std::span<const double> a, std::span<const double> b, ExecutionMode mode) {
    assert(a.size() == b.size());
    double sum = 0.0;
    std::size_t i = 0;
    if (mode == ExecutionMode::SerialSimd) {
        Double2 first, second;
        for (; i + 3 < a.size(); i += 4) {
            first = first + Double2::load(a.data() + i) * Double2::load(b.data() + i);
            second = second + Double2::load(a.data() + i + 2) * Double2::load(b.data() + i + 2);
        }
        if (i + 1 < a.size()) {
            first = first + Double2::load(a.data() + i) * Double2::load(b.data() + i);
            i += 2;
        }
        sum = (first + second).sum();
    }
    for (; i < a.size(); ++i) sum += a[i] * b[i];
    return sum;
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
                v = v + Double2(a(r, k)) * Double2::load(b.data.data() + k * C + c);
            v.store(&out.data[r * C + c]);
        }
        for (; c < C; ++c)
            for (std::size_t k = 0; k < K; ++k)
                out(r, c) += a(r, k) * b(k, c);
    }
    return out;
}

} // namespace slam::detail
