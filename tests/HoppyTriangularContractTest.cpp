// SPDX-License-Identifier: MIT
// Triangular-compressed specification: decisions D1-D20; sections 14-20.

#include "HoppyTestSupport.hpp"
#include <SecUtility/Hoppy/Hoppy.hpp>
#include <Eigen/Core>
#include <complex>
#include <type_traits>
#include <utility>

namespace
{
	template <typename T, typename = void> struct has_sqrt : std::false_type {};
	template <typename T> struct has_sqrt<T, std::void_t<decltype(std::declval<const T&>().sqrt())>> : std::true_type {};
	template <typename T, typename = void> struct has_inverse_sqrt : std::false_type {};
	template <typename T> struct has_inverse_sqrt<T, std::void_t<decltype(std::declval<const T&>().inverseSqrt())>> : std::true_type {};
	template <typename T, typename = void> struct has_inner_stride : std::false_type {};
	template <typename T> struct has_inner_stride<T, std::void_t<decltype(std::declval<const T&>().innerStride())>> : std::true_type {};
	template <typename T, typename = void> struct has_transformed_by : std::false_type {};
	template <typename T> struct has_transformed_by<T, std::void_t<decltype(std::declval<const T&>().transformedBy(std::declval<const Eigen::MatrixXd&>()))>> : std::true_type {};

	using Matrix = Hoppy::HermitianMatrix<std::complex<double>, 3, Hoppy::TrianglePacking::Upper, Eigen::DontAlign>;
	using Traits = Eigen::internal::traits<Matrix>;
	static_assert(std::is_same_v<typename Matrix::Scalar, std::complex<double>>);
	static_assert(std::is_same_v<typename Matrix::RealScalar, double>);
	static_assert(std::is_same_v<typename Matrix::StorageIndex, Eigen::Index>);
	static_assert(Matrix::RowsAtCompileTime == 3 && Matrix::ColsAtCompileTime == 3);
	static_assert(Matrix::SizeAtCompileTime == 9);
	static_assert(Matrix::PackingValue == Hoppy::TrianglePacking::Upper);
	static_assert((Traits::Flags & Eigen::DirectAccessBit) == 0);
	static_assert((Traits::Flags & Eigen::PacketAccessBit) == 0);
	static_assert(!has_inner_stride<Matrix>::value);
	static_assert(!has_sqrt<Matrix>::value);
	static_assert(!has_inverse_sqrt<Matrix>::value);
	static_assert(!has_transformed_by<Matrix>::value);
	static_assert(std::is_same_v<Hoppy::HermitianMatrix<double>, Hoppy::SymmetricMatrix<double>>);
	static_assert(std::is_same_v<Hoppy::AntiHermitianMatrix<double>, Hoppy::AntiSymmetricMatrix<double>>);
	static_assert(noexcept(std::declval<const Matrix&>().rows()));
	static_assert(noexcept(std::declval<const Matrix&>().cols()));
	static_assert(noexcept(std::declval<const Matrix&>().dimension()));
	static_assert(noexcept(std::declval<const Matrix&>().storedSize()));
}

TEST_CASE("umbrella header exposes owner map expression and packed operations")
{
	double storage[6]{};
	Eigen::Map<Hoppy::SymmetricMatrix<double, 3>> mapped(storage);
	mapped.setIdentity();
	const auto expression = (mapped + mapped).transpose();
	Hoppy::SymmetricMatrix<double, 3, Hoppy::TrianglePacking::Upper> result(expression);
	Hoppy::Test::requireApprox(result.toDense(), 2.0 * Eigen::Matrix3d::Identity());
}

TEST_CASE("empty expressions retain specified neutral reductions")
{
	const Hoppy::SymmetricMatrixXd empty(0);
	REQUIRE(empty.rows() == 0);
	REQUIRE(empty.cols() == 0);
	REQUIRE(empty.size() == 0);
	REQUIRE(empty.storedSize() == 0);
	REQUIRE(empty.sum() == 0.0);
	REQUIRE(empty.squaredNorm() == 0.0);
	REQUIRE(empty.allFinite());
	REQUIRE_FALSE(empty.hasNaN());
}
