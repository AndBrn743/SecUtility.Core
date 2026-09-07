// SPDX-License-Identifier: MIT

// Triangular-compressed specification: sections 10-11, 14, and 17.

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/TriangularCompressedMatrix.hpp>

#include <Eigen/Core>

#include <array>
#include <complex>
#include <type_traits>
#include <utility>

namespace
{
	template <typename T, typename = void> struct has_rvalue_diagonal : std::false_type {};
	template <typename T>
	struct has_rvalue_diagonal<T, std::void_t<decltype(std::declval<T&&>().diagonal())>> : std::true_type {};

	using Upper = Hoppy::UpperTriangularMatrix<std::complex<double>, 3, Hoppy::TrianglePacking::Lower>;
	using Transpose = decltype(std::declval<const Upper&>().transpose());
	using Conjugate = decltype(std::declval<const Upper&>().conjugate());
	using Adjoint = decltype(std::declval<const Upper&>().adjoint());
	static_assert(std::is_same_v<typename Transpose::StructureTag, Hoppy::Detail::LowerTriangularTag>);
	static_assert(std::is_same_v<typename Conjugate::StructureTag, Hoppy::Detail::UpperTriangularTag>);
	static_assert(std::is_same_v<typename Adjoint::StructureTag, Hoppy::Detail::LowerTriangularTag>);
	static_assert(Transpose::PackingValue == Hoppy::TrianglePacking::Upper);
	static_assert(Conjugate::PackingValue == Hoppy::TrianglePacking::Lower);
	static_assert(Adjoint::PackingValue == Hoppy::TrianglePacking::Upper);
	static_assert(!has_rvalue_diagonal<Upper>::value);
	static_assert((Eigen::internal::traits<Transpose>::Flags & Eigen::DirectAccessBit) == 0);
	static_assert(Eigen::internal::evaluator<Transpose>::Flags == 0);
	using LowerTranspose = decltype(std::declval<const Hoppy::LowerTriangularMatrixXd&>().transpose());
	using SymmetricTranspose = decltype(std::declval<const Hoppy::SymmetricMatrixXd&>().transpose());
	using AntiTranspose = decltype(std::declval<const Hoppy::AntiSymmetricMatrixXd&>().transpose());
	using HermitianTranspose = decltype(std::declval<const Hoppy::HermitianMatrixXcd&>().transpose());
	using AntiHermitianAdjoint = decltype(std::declval<const Hoppy::AntiHermitianMatrixXcd&>().adjoint());
	static_assert(std::is_same_v<typename LowerTranspose::StructureTag, Hoppy::Detail::UpperTriangularTag>);
	static_assert(std::is_same_v<typename SymmetricTranspose::StructureTag, Hoppy::Detail::SymmetricTag>);
	static_assert(std::is_same_v<typename AntiTranspose::StructureTag, Hoppy::Detail::AntiSymmetricTag>);
	static_assert(std::is_same_v<typename HermitianTranspose::StructureTag, Hoppy::Detail::HermitianTag>);
	static_assert(std::is_same_v<typename AntiHermitianAdjoint::StructureTag, Hoppy::Detail::AntiHermitianTag>);
}

TEST_CASE("transpose conjugate and adjoint are lazy structured expressions")
{
	using Complex = std::complex<double>;
	Upper matrix;
	matrix.setZero();
	matrix(0, 1) = Complex(2, 3);
	matrix(1, 2) = Complex(4, -5);
	const auto dense = matrix.toDense();
	Hoppy::Test::requireApprox(matrix.transpose().toDense(), dense.transpose());
	Hoppy::Test::requireApprox(matrix.conjugate().toDense(), dense.conjugate());
	Hoppy::Test::requireApprox(matrix.adjoint().toDense(), dense.adjoint());
	Hoppy::Test::requireApprox(matrix.transpose().transpose().toDense(), dense);
	Hoppy::Test::requireApprox(matrix.transpose().conjugate().toDense(), dense.transpose().conjugate());

	const auto evaluated = matrix.adjoint().eval();
	static_assert(std::is_same_v<typename std::decay_t<decltype(evaluated)>::StructureTag,
	                             Hoppy::Detail::LowerTriangularTag>);
	REQUIRE(evaluated.coeff(1, 0) == Complex(2, -3));
	const auto transposed = matrix.transpose();
	Eigen::internal::evaluator<decltype(transposed)> evaluator(transposed);
	REQUIRE(evaluator.coeff(1, 0) == Complex(2, 3));
	Eigen::MatrixXcd implicit = matrix.adjoint();
	REQUIRE(implicit(1, 0) == Complex(2, -3));
}

TEST_CASE("diagonal and logical subviews route writes through checked proxies")
{
	Hoppy::SymmetricMatrix<double, 4> matrix;
	matrix.setZero();
	auto diagonal = matrix.diagonal();
	diagonal[2] = 7;
	REQUIRE(matrix.coeff(2, 2) == 7);

	auto block = matrix.block(1, 0, 2, 3);
	block(0, 2) = 5;
	REQUIRE(matrix.coeff(1, 2) == 5);
	REQUIRE(matrix.coeff(2, 1) == 5);
	auto corner = matrix.topLeftCorner(2, 2);
	corner(0, 1) = 3;
	REQUIRE(matrix.coeff(1, 0) == 3);
	matrix.row(3)(0, 1) = 9;
	REQUIRE(matrix.coeff(3, 1) == 9);
	matrix.col(0)(2, 0) = 8;
	REQUIRE(matrix.coeff(0, 2) == 8);
}

TEST_CASE("triangular logical views expose implicit zeros and checked writes")
{
	Hoppy::SymmetricMatrix<double, 3> matrix;
	matrix.setZero();
	auto upper = matrix.triangularView<Eigen::Upper>();
	upper(0, 2) = 6;
	REQUIRE(upper.coeff(0, 2) == 6);
	REQUIRE(upper.coeff(2, 0) == 0);
	REQUIRE(matrix.coeff(2, 0) == 6);

	const auto& constant = matrix;
	auto strictLower = constant.triangularView<Eigen::StrictlyLower>();
	REQUIRE(strictLower.coeff(2, 0) == 6);
	REQUIRE(strictLower.coeff(0, 2) == 0);
	REQUIRE(strictLower.coeff(1, 1) == 0);
}

TEST_CASE("maps expose the same transform and logical view surface")
{
	using Complex = std::complex<double>;
	std::array<Complex, 6> storage{};
	Eigen::Map<Hoppy::HermitianMatrix<Complex, 3>> map(storage.data());
	map.setZero();
	map(0, 2) = Complex(2, 5);
	map.diagonal()[1] = Complex(7, 0);
	REQUIRE(map.block(0, 1, 2, 2).coeff(0, 1) == Complex(2, 5));
	REQUIRE(map.adjoint().coeff(2, 0) == Complex(2, -5));

	Eigen::Map<const Hoppy::HermitianMatrix<Complex, 3>> constant(storage.data());
	REQUIRE(constant.col(2).coeff(0, 0) == Complex(2, 5));
	Eigen::internal::evaluator<decltype(constant)> evaluator(constant);
	REQUIRE(evaluator.coeff(2, 0) == Complex(2, -5));
}

#ifndef EIGEN_NO_DEBUG
TEST_CASE("view bounds and structural constraints assert")
{
	Hoppy::AntiSymmetricMatrix<double, 3> anti;
	anti.setZero();
	REQUIRE_THROWS_AS(anti.diagonal()[1] = 2, Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(anti.block(2, 2, 2, 1), Hoppy::Test::EigenAssertionFailure);
	auto upper = anti.triangularView<Eigen::Upper>();
	REQUIRE_THROWS_AS(upper(2, 0) = 4, Hoppy::Test::EigenAssertionFailure);
}
#endif
