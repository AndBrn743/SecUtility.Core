// SPDX-License-Identifier: MIT

// Triangular-compressed specification: D16; sections 13-14 and 16.

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/TriangularCompressedMatrix.hpp>

#include <Eigen/Core>
#include <Eigen/LU>

#include <complex>
#include <type_traits>
#include <utility>

namespace
{
	template <typename T, typename = void> struct has_inverse : std::false_type {};
	template <typename T>
	struct has_inverse<T, std::void_t<decltype(std::declval<const T&>().inverse())>> : std::true_type {};
	template <typename T, typename = void> struct has_sqrt : std::false_type {};
	template <typename T>
	struct has_sqrt<T, std::void_t<decltype(std::declval<const T&>().sqrt())>> : std::true_type {};
	static_assert(has_inverse<Hoppy::UpperTriangularMatrixXd>::value);
	static_assert(has_inverse<Hoppy::LowerTriangularMatrixXcd>::value);
	static_assert(has_inverse<Hoppy::SymmetricMatrixXd>::value);
	static_assert(has_inverse<Hoppy::AntiSymmetricMatrixXd>::value);
	static_assert(has_inverse<Hoppy::HermitianMatrixXcd>::value);
	static_assert(has_inverse<Hoppy::AntiHermitianMatrixXcd>::value);
	static_assert(!has_inverse<Hoppy::SymmetricMatrixXi>::value);
	static_assert(!has_sqrt<Hoppy::SymmetricMatrixXd>::value);

	using Fixed = Hoppy::UpperTriangularMatrix<double, 3, Hoppy::TrianglePacking::Upper>;
	using Inverse = decltype(std::declval<const Fixed&>().inverse());
	static_assert(std::is_same_v<decltype(std::declval<const Inverse&>().toDense()), Eigen::Matrix3d>);
	static_assert(Inverse::RowsAtCompileTime == 3);
	static_assert(Inverse::PackingValue == Hoppy::TrianglePacking::Upper);
	static_assert(std::is_same_v<typename Inverse::StructureTag, Hoppy::Detail::UpperTriangularTag>);
	static_assert(std::is_same_v<typename Inverse::PlainObject, Fixed>);
}

TEST_CASE("inverse expressions re-evaluate referenced lvalue operands")
{
	Hoppy::UpperTriangularMatrix<double, 2> matrix;
	matrix.setIdentity();
	const auto inverse = matrix.inverse();
	Hoppy::Test::requireApprox(inverse.toDense(), Eigen::Matrix2d::Identity());

	matrix(0, 0) = 2.0;
	matrix(0, 1) = 3.0;
	matrix(1, 1) = 4.0;
	const Eigen::Matrix2d expected = matrix.toDense().inverse();
	Hoppy::Test::requireApprox(inverse.toDense(), expected);

	Eigen::Matrix2d assigned = inverse;
	Hoppy::Test::requireApprox(assigned, expected);
	Hoppy::Test::requireApprox(inverse.eval().toDense(), expected);
}

TEST_CASE("triangular and symmetric inverses retain their packed family")
{
	Hoppy::UpperTriangularMatrix<double, 3> upper;
	upper.setZero();
	upper(0, 0) = 2; upper(0, 1) = 1; upper(0, 2) = -1;
	upper(1, 1) = 3; upper(1, 2) = 2; upper(2, 2) = 4;
	const auto expected = upper.toDense().inverse().eval();
	const auto inverse = upper.inverse();
	Hoppy::Test::requireApprox(inverse.toDense(), expected);
	Hoppy::Test::requireApprox(inverse.eval().toDense(), expected);
	Eigen::Matrix3d dense = inverse;
	Hoppy::Test::requireApprox(dense, expected);
	upper = upper.inverse();
	Hoppy::Test::requireApprox(upper.toDense(), expected);

	Hoppy::SymmetricMatrix<double, 2> symmetric;
	symmetric.setZero(); symmetric(0, 0) = 4; symmetric(0, 1) = 1; symmetric(1, 1) = 3;
	Hoppy::Test::requireApprox(symmetric.inverse().toDense(), symmetric.toDense().inverse());
}

TEST_CASE("complex Hermitian and anti-Hermitian inverses match Eigen")
{
	using Complex = std::complex<double>;
	Hoppy::HermitianMatrix<Complex, 2> hermitian;
	hermitian.setZero(); hermitian(0, 0) = 4; hermitian(1, 1) = 5; hermitian(0, 1) = Complex(1, 2);
	Hoppy::Test::requireApprox(hermitian.inverse().toDense(), hermitian.toDense().inverse());

	Hoppy::AntiHermitianMatrix<Complex, 2> anti;
	anti.setZero(); anti(0, 0) = Complex(0, 2); anti(1, 1) = Complex(0, 3); anti(0, 1) = Complex(1, 1);
	Hoppy::Test::requireApprox(anti.inverse().toDense(), anti.toDense().inverse());

	Hoppy::AntiSymmetricMatrix<double, 2> antiSymmetric;
	antiSymmetric.setZero(); antiSymmetric(0, 1) = 2;
	Hoppy::Test::requireApprox(antiSymmetric.inverse().toDense(), antiSymmetric.toDense().inverse());
}

TEST_CASE("inverse supports maps products and nested rvalue chains")
{
	double storage[3] = {3, 1, 2};
	Eigen::Map<const Hoppy::SymmetricMatrix<double, 2>> map(storage);
	const auto expected = map.toDense().inverse().eval();
	Hoppy::Test::requireApprox(map.inverse().toDense(), expected);
	Hoppy::Test::requireApprox((map * map.inverse()).toDense(), Eigen::Matrix2d::Identity());
	Hoppy::Test::requireApprox(map.inverse().transpose().inverse().toDense(), map.toDense().transpose());
}
