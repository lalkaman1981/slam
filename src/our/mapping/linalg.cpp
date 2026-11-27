// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include "linalg.hpp"

#include <algorithm>
#include <numeric>

namespace slam {

bool invert3(const Mat3 &a, Mat3 &inverse) {
    const double c00 = a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1);
    const double c01 = a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2);
    const double c02 = a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0);
    const double det = a(0, 0) * c00 + a(0, 1) * c01 + a(0, 2) * c02;
    double scale = 0.0;
    for (double v : a.data)
        scale = std::max(scale, std::abs(v));
    if (scale == 0.0 || std::abs(det) <= 1e-14 * scale * scale * scale) return false;
    const double inv = 1.0 / det;
    inverse(0, 0) = c00 * inv;
    inverse(0, 1) = (a(0, 2) * a(2, 1) - a(0, 1) * a(2, 2)) * inv;
    inverse(0, 2) = (a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1)) * inv;
    inverse(1, 0) = c01 * inv;
    inverse(1, 1) = (a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0)) * inv;
    inverse(1, 2) = (a(0, 2) * a(1, 0) - a(0, 0) * a(1, 2)) * inv;
    inverse(2, 0) = c02 * inv;
    inverse(2, 1) = (a(0, 1) * a(2, 0) - a(0, 0) * a(2, 1)) * inv;
    inverse(2, 2) = (a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0)) * inv;
    return true;
}

bool solveCholesky(const DenseMatrix &a, const std::vector<double> &b, std::vector<double> &x) {
    const std::size_t n = a.rows();
    if (a.cols() != n || b.size() != n) return false;
    DenseMatrix l(n, n);
    for (std::size_t j = 0; j < n; ++j) {
        double diag = a(j, j);
        for (std::size_t k = 0; k < j; ++k)
            diag -= l(j, k) * l(j, k);
        if (!(diag > 1e-12)) return false;
        const double ljj = std::sqrt(diag);
        l(j, j) = ljj;
        for (std::size_t i = j + 1; i < n; ++i) {
            double s = a(i, j);
            for (std::size_t k = 0; k < j; ++k)
                s -= l(i, k) * l(j, k);
            l(i, j) = s / ljj;
        }
    }
    // Forward substitution L y = b, then backward L^T x = y.
    x.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double s = b[i];
        for (std::size_t k = 0; k < i; ++k)
            s -= l(i, k) * x[k];
        x[i] = s / l(i, i);
    }
    for (std::size_t ii = n; ii-- > 0;) {
        double s = x[ii];
        for (std::size_t k = ii + 1; k < n; ++k)
            s -= l(k, ii) * x[k];
        x[ii] = s / l(ii, ii);
    }
    return true;
}

void symmetricEigen(const DenseMatrix &a, std::vector<double> &values, DenseMatrix &vectors) {
    const std::size_t n = a.rows();
    DenseMatrix m = a;
    DenseMatrix v(n, n);
    for (std::size_t i = 0; i < n; ++i)
        v(i, i) = 1.0;

    constexpr int kMaxSweeps = 100;
    for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
        double offDiagonal = 0.0;
        for (std::size_t p = 0; p < n; ++p)
            for (std::size_t q = p + 1; q < n; ++q)
                offDiagonal += m(p, q) * m(p, q);
        if (offDiagonal < 1e-30) break;

        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                if (std::abs(m(p, q)) < 1e-300) continue;
                const double theta = (m(q, q) - m(p, p)) / (2.0 * m(p, q));
                const double t =
                    (theta >= 0.0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                for (std::size_t k = 0; k < n; ++k) {
                    const double mkp = m(k, p);
                    const double mkq = m(k, q);
                    m(k, p) = c * mkp - s * mkq;
                    m(k, q) = s * mkp + c * mkq;
                }
                for (std::size_t k = 0; k < n; ++k) {
                    const double mpk = m(p, k);
                    const double mqk = m(q, k);
                    m(p, k) = c * mpk - s * mqk;
                    m(q, k) = s * mpk + c * mqk;
                }
                for (std::size_t k = 0; k < n; ++k) {
                    const double vkp = v(k, p);
                    const double vkq = v(k, q);
                    v(k, p) = c * vkp - s * vkq;
                    v(k, q) = s * vkp + c * vkq;
                }
            }
        }
    }

    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&m](std::size_t i, std::size_t j) { return m(i, i) < m(j, j); });
    values.resize(n);
    vectors = DenseMatrix(n, n);
    for (std::size_t k = 0; k < n; ++k) {
        values[k] = m(order[k], order[k]);
        for (std::size_t r = 0; r < n; ++r)
            vectors(r, k) = v(r, order[k]);
    }
}

} // namespace slam
