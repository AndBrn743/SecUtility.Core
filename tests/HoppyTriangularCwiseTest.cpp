// SPDX-License-Identifier: MIT

// Triangular-compressed specification: sections 12.1 and 14.

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/TriangularCompressedMatrix.hpp>

#include <Eigen/Core>

#include <complex>
#include <type_traits>
#include <utility>

namespace
{
	template <typename T, typename Factor, typename = void> struct has_multiply_assign : std::false_type {};
	template <typename T, typename Factor>
	struct has_multiply_assign<T, Factor, std::void_t<decltype(std::declval<T&>() *= std::declval<Factor>())>>
	    : std::true_type {};
	using Complex = std::complex<double>;
	using Plain = Hoppy::SymmetricMatrix<double, 2>;
	static_assert(std::is_same_v<decltype(+std::declval<Plain&>()), const Plain&>);
	static_assert(std::is_same_v<decltype(+std::declval<const Plain&>()), const Plain&>);
	static_assert(std::is_same_v<decltype(+std::declval<Plain&&>()), Plain&&>);
	static_assert(has_multiply_assign<Hoppy::HermitianMatrixXcd, double>::value);
	static_assert(!has_multiply_assign<Hoppy::HermitianMatrixXcd, Complex>::value);
	static_assert(has_multiply_assign<Hoppy::SymmetricMatrixXcd, Complex>::value);

	using SymSum = decltype(std::declval<const Hoppy::SymmetricMatrixXd&>()
	                        + std::declval<const Hoppy::SymmetricMatrixXcd&>());
	using MixedSum = decltype(std::declval<const Hoppy::SymmetricMatrixXd&>()
	                          + std::declval<const Hoppy::AntiSymmetricMatrixXd&>());
	using HermitianRealScale = decltype(std::declval<const Hoppy::HermitianMatrixXcd&>() * 2.0);
	using HermitianComplexScale = decltype(std::declval<const Hoppy::HermitianMatrixXcd&>() * Complex(2, 1));
	static_assert(SymSum::IsTriangularCompressed);
	static_assert(!MixedSum::IsTriangularCompressed);
	static_assert(HermitianRealScale::IsTriangularCompressed);
	static_assert(!HermitianComplexScale::IsTriangularCompressed);
	static_assert((Eigen::internal::traits<SymSum>::Flags & Eigen::DirectAccessBit) == 0);
}

TEST_CASE("unary and same-family binary expressions preserve structure")
{
	Hoppy::SymmetricMatrix<double, 2> left;
	Hoppy::SymmetricMatrix<double, 2, Hoppy::TrianglePacking::Upper> right;
	left.setZero(); right.setZero();
	left(0, 0) = 1; left(0, 1) = 2; left(1, 1) = 3;
	right(0, 0) = 4; right(0, 1) = 5; right(1, 1) = 6;
	const auto leftDense = left.toDense();
	const auto rightDense = right.toDense();
	const auto& identity = +left;
	REQUIRE(&identity == &left);
	Hoppy::Test::requireApprox((+left).toDense(), leftDense);
	Hoppy::Test::requireApprox((+Hoppy::SymmetricMatrix<double, 2>::Ones()).transpose().toDense(),
	                           Eigen::Matrix2d::Ones());
	Hoppy::Test::requireApprox((-left).toDense(), -leftDense);
	Hoppy::Test::requireApprox((left + right).toDense(), leftDense + rightDense);
	Hoppy::Test::requireApprox((left - right).toDense(), leftDense - rightDense);
	Hoppy::Test::requireApprox(((left + right) - left).toDense(), rightDense);
	left += right;
	Hoppy::Test::requireApprox(left.toDense(), leftDense + rightDense);
	left -= right;
	Hoppy::Test::requireApprox(left.toDense(), leftDense);
}

TEST_CASE("mixed families and complex Hermitian scaling promote to dense results")
{
	Hoppy::SymmetricMatrix<double, 2> symmetric;
	Hoppy::AntiSymmetricMatrix<double, 2> anti;
	symmetric.setZero(); anti.setZero();
	symmetric(0, 1) = 3;
	anti(0, 1) = 4;
	auto promoted = symmetric + anti;
	static_assert(!decltype(promoted)::IsTriangularCompressed);
	Hoppy::Test::requireApprox(promoted.toDense(), symmetric.toDense() + anti.toDense());

	Hoppy::HermitianMatrix<Complex, 2> hermitian;
	hermitian.setZero();
	hermitian(0, 1) = Complex(2, 3);
	Hoppy::Test::requireApprox((hermitian * Complex(1, 2)).toDense(),
	                           hermitian.toDense() * Complex(1, 2));
	Hoppy::Test::requireApprox((hermitian * 2.0).toDense(), hermitian.toDense() * 2.0);
	Hoppy::Test::requireApprox((2.0 * hermitian).toDense(), 2.0 * hermitian.toDense());
}

TEST_CASE("scalar compounds casts and rvalue chains preserve valid invariants")
{
	Hoppy::AntiHermitianMatrix<Complex, 2> matrix;
	matrix.setZero();
	matrix(0, 0) = Complex(0, 2);
	matrix(0, 1) = Complex(3, 4);
	auto dense = matrix.toDense();
	matrix *= 2.0;
	matrix /= 4.0;
	dense *= 2.0;
	dense /= 4.0;
	Hoppy::Test::requireApprox(matrix.toDense(), dense);
	Hoppy::Test::requireApprox(((-matrix) * 3.0 / 2.0).toDense(), -dense * 1.5);

	Hoppy::SymmetricMatrix<double, 2> real;
	real.setOnes();
	auto cast = real.cast<Complex>();
	static_assert(std::is_same_v<typename decltype(cast)::Scalar, Complex>);
	Hoppy::Test::requireApprox(cast.toDense(), real.toDense().cast<Complex>());
}

TEST_CASE("mapped compound operations are alias safe")
{
	double storage[3] = {1, 2, 3};
	Eigen::Map<Hoppy::SymmetricMatrix<double, 2>> map(storage);
	map += map;
	REQUIRE(map.coeff(0, 0) == 2);
	REQUIRE(map.coeff(1, 0) == 4);
	REQUIRE(map.coeff(1, 1) == 6);
	map *= 0.5;
	REQUIRE(map.coeff(1, 0) == 2);
}

#ifndef EIGEN_NO_DEBUG
TEST_CASE("compound dimension mismatch asserts before mutation")
{
	auto destination = Hoppy::SymmetricMatrixXd::Ones(2);
	auto source = Hoppy::SymmetricMatrixXd::Ones(3);
	REQUIRE_THROWS_AS(destination += source, Hoppy::Test::EigenAssertionFailure);
	REQUIRE(destination.isApprox(Hoppy::SymmetricMatrixXd::Ones(2)));
}
#else
TEST_CASE("no-debug compound mismatch returns without mutation")
{
	auto destination = Hoppy::SymmetricMatrixXd::Ones(2);
	auto source = Hoppy::SymmetricMatrixXd::Ones(3);
	destination += source;
	REQUIRE(destination.isApprox(Hoppy::SymmetricMatrixXd::Ones(2)));
}
#endif
