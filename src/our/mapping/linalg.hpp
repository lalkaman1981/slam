#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace slam {

// Small fixed-size row-major matrix. Column vectors are Matrix<N, 1>.
template <std::size_t R, std::size_t C> struct Matrix {
    std::array<double, R * C> data{};

    static Matrix identity() {
        static_assert(R == C, "identity() needs a square matrix");
        Matrix m{};
        for (std::size_t i = 0; i < R; ++i)
            m(i, i) = 1.0;
        return m;
    }

    double &operator()(std::size_t r, std::size_t c) { return data[r * C + c]; }
    double operator()(std::size_t r, std::size_t c) const { return data[r * C + c]; }
    double &operator[](std::size_t i) { return data[i]; }
    double operator[](std::size_t i) const { return data[i]; }

    Matrix &operator+=(const Matrix &o) {
        for (std::size_t i = 0; i < R * C; ++i)
            data[i] += o.data[i];
        return *this;
    }
    Matrix &operator-=(const Matrix &o) {
        for (std::size_t i = 0; i < R * C; ++i)
            data[i] -= o.data[i];
        return *this;
    }
    Matrix &operator*=(double s) {
        for (double &v : data)
            v *= s;
        return *this;
    }

    [[nodiscard]] Matrix<C, R> transpose() const {
        Matrix<C, R> out{};
        for (std::size_t r = 0; r < R; ++r)
            for (std::size_t c = 0; c < C; ++c)
                out(c, r) = (*this)(r, c);
        return out;
    }
};

template <std::size_t R, std::size_t C> Matrix<R, C> operator+(Matrix<R, C> a, const Matrix<R, C> &b) {
    a += b;
    return a;
}

template <std::size_t R, std::size_t C> Matrix<R, C> operator-(Matrix<R, C> a, const Matrix<R, C> &b) {
    a -= b;
    return a;
}

template <std::size_t R, std::size_t C> Matrix<R, C> operator-(Matrix<R, C> a) {
    a *= -1.0;
    return a;
}

template <std::size_t R, std::size_t C> Matrix<R, C> operator*(Matrix<R, C> a, double s) {
    a *= s;
    return a;
}

template <std::size_t R, std::size_t C> Matrix<R, C> operator*(double s, Matrix<R, C> a) {
    a *= s;
    return a;
}

template <std::size_t R, std::size_t K, std::size_t C>
Matrix<R, C> operator*(const Matrix<R, K> &a, const Matrix<K, C> &b) {
    Matrix<R, C> out{};
    for (std::size_t r = 0; r < R; ++r)
        for (std::size_t k = 0; k < K; ++k) {
            const double v = a(r, k);
            if (v == 0.0) continue;
            for (std::size_t c = 0; c < C; ++c)
                out(r, c) += v * b(k, c);
        }
    return out;
}

using Vec2 = Matrix<2, 1>;
using Vec3 = Matrix<3, 1>;
using Vec4 = Matrix<4, 1>;
using Vec6 = Matrix<6, 1>;
using Mat3 = Matrix<3, 3>;
using Mat6 = Matrix<6, 6>;
using Mat36 = Matrix<3, 6>;
using Mat63 = Matrix<6, 3>;

inline Vec3 makeVec3(double x, double y, double z) {
    Vec3 v;
    v[0] = x;
    v[1] = y;
    v[2] = z;
    return v;
}

template <std::size_t N> double dot(const Matrix<N, 1> &a, const Matrix<N, 1> &b) {
    double s = 0.0;
    for (std::size_t i = 0; i < N; ++i)
        s += a[i] * b[i];
    return s;
}

template <std::size_t N> double squaredNorm(const Matrix<N, 1> &a) { return dot(a, a); }

template <std::size_t N> double norm(const Matrix<N, 1> &a) { return std::sqrt(dot(a, a)); }

inline Vec3 cross(const Vec3 &a, const Vec3 &b) {
    return makeVec3(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

inline Mat3 skew(const Vec3 &v) {
    Mat3 m{};
    m(0, 1) = -v[2];
    m(0, 2) = v[1];
    m(1, 0) = v[2];
    m(1, 2) = -v[0];
    m(2, 0) = -v[1];
    m(2, 1) = v[0];
    return m;
}

// Writes block b into m starting at (r0, c0).
template <std::size_t BR, std::size_t BC, std::size_t R, std::size_t C>
void setBlock(Matrix<R, C> &m, std::size_t r0, std::size_t c0, const Matrix<BR, BC> &b) {
    for (std::size_t r = 0; r < BR; ++r)
        for (std::size_t c = 0; c < BC; ++c)
            m(r0 + r, c0 + c) = b(r, c);
}

// Inverse of a general 3x3 matrix. Returns false when it is (numerically) singular.
bool invert3(const Mat3 &a, Mat3 &inverse);

// Row-major dynamic matrix used for the reduced camera system and the pose graph.
class DenseMatrix {
  public:
    DenseMatrix() = default;
    DenseMatrix(std::size_t rows, std::size_t cols, double value = 0.0)
        : nRows(rows), nCols(cols), values(rows * cols, value) {}

    [[nodiscard]] std::size_t rows() const { return nRows; }
    [[nodiscard]] std::size_t cols() const { return nCols; }

    double &operator()(std::size_t r, std::size_t c) { return values[r * nCols + c]; }
    double operator()(std::size_t r, std::size_t c) const { return values[r * nCols + c]; }

    template <std::size_t BR, std::size_t BC>
    void addBlock(std::size_t r0, std::size_t c0, const Matrix<BR, BC> &b) {
        for (std::size_t r = 0; r < BR; ++r)
            for (std::size_t c = 0; c < BC; ++c)
                (*this)(r0 + r, c0 + c) += b(r, c);
    }

  private:
    std::size_t nRows = 0;
    std::size_t nCols = 0;
    std::vector<double> values;
};

// Solves a x = b for symmetric positive definite a (Cholesky). Returns false if a is not SPD.
bool solveCholesky(const DenseMatrix &a, const std::vector<double> &b, std::vector<double> &x);

// Cyclic Jacobi eigen decomposition of a symmetric matrix.
// Eigenvalues are sorted ascending; column k of `vectors` is the eigenvector of values[k].
void symmetricEigen(const DenseMatrix &a, std::vector<double> &values, DenseMatrix &vectors);

} // namespace slam
