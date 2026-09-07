// SPDX-License-Identifier: MIT

// Triangular-compressed specification: sections 12.3, 14, and 16.

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/TriangularCompressedMatrix.hpp>

#include <Eigen/Core>

#include <type_traits>

namespace
{
	using UpperProduct = decltype(std::declval<const Hoppy::UpperTriangularMatrixXd&>()
	                              * std::declval<const Hoppy::UpperTriangularMatrixXd&>());
	using LowerProduct = decltype(std::declval<const Hoppy::LowerTriangularMatrixXd&>()
	                              * std::declval<const Hoppy::LowerTriangularMatrixXd&>());
	using SymmetricProduct = decltype(std::declval<const Hoppy::SymmetricMatrixXd&>()
	                                  * std::declval<const Hoppy::SymmetricMatrixXd&>());
	static_assert(UpperProduct::IsTriangularCompressed);
	static_assert(LowerProduct::IsTriangularCompressed);
	static_assert(!SymmetricProduct::IsTriangularCompressed);
}

TEST_CASE("upper and lower products preserve triangular structure")
{
	Hoppy::UpperTriangularMatrix<double, 3> left;
	Hoppy::UpperTriangularMatrix<double, 3, Hoppy::TrianglePacking::Upper> right;
	left.setZero(); right.setZero();
	left(0, 0) = 1; left(0, 1) = 2; left(0, 2) = 3; left(1, 1) = 4; left(1, 2) = 5; left(2, 2) = 6;
	right.setConstant(2);
	const auto product = left * right;
	static_assert(std::is_same_v<typename decltype(product)::StructureTag,
	                             Hoppy::Detail::UpperTriangularTag>);
	Hoppy::Test::requireApprox(product.toDense(), left.toDense() * right.toDense());
	Hoppy::Test::requireApprox(product.eval().toDense(), left.toDense() * right.toDense());
	Hoppy::Test::requireApprox((2.0 * left * right + left).toDense(),
	                           2.0 * left.toDense() * right.toDense() + left.toDense());
	const auto expectedAlias = (left.toDense() * right.toDense()).eval();
	left = left * right;
	Hoppy::Test::requireApprox(left.toDense(), expectedAlias);
}

TEST_CASE("general packed products promote to logical dense results")
{
	Hoppy::SymmetricMatrix<double, 2> left;
	Hoppy::AntiSymmetricMatrix<double, 2> right;
	left.setOnes(); right.setZero(); right(0, 1) = 3;
	const auto product = left * right;
	static_assert(!decltype(product)::IsTriangularCompressed);
	Hoppy::Test::requireApprox(product.toDense(), left.toDense() * right.toDense());
	Eigen::MatrixXd assigned = product;
	Hoppy::Test::requireApprox(assigned, left.toDense() * right.toDense());
}

TEST_CASE("packed dense and dense packed products support rectangular operands")
{
	Hoppy::SymmetricMatrix<double, 3> matrix;
	matrix.setOnes();
	Eigen::Matrix<double, 3, 2> right;
	right << 1, 2, 3, 4, 5, 6;
	Eigen::Matrix<double, 2, 3> left;
	left << 1, 2, 3, 4, 5, 6;
	Hoppy::Test::requireApprox((matrix * right).toDense(), matrix.toDense() * right);
	Hoppy::Test::requireApprox((left * matrix).toDense(), left * matrix.toDense());
	Eigen::Vector3d vector(1, 2, 3);
	Hoppy::Test::requireApprox((matrix * vector).toDense(), matrix.toDense() * vector);
}

TEST_CASE("nested product and transform chains retain temporary operands")
{
	auto left = Hoppy::LowerTriangularMatrix<double, 2>::Constant(2);
	auto right = Hoppy::LowerTriangularMatrix<double, 2>::Identity();
	const auto expected = (left.toDense() * right.toDense()).eval();
	Hoppy::Test::requireApprox(((left + left) / 2.0 * right).transpose().toDense(),
	                           expected.transpose());
}

#ifndef EIGEN_NO_DEBUG
TEST_CASE("product dimension mismatch asserts")
{
	auto matrix = Hoppy::SymmetricMatrixXd::Ones(2);
	Eigen::MatrixXd wrong = Eigen::MatrixXd::Ones(3, 1);
	REQUIRE_THROWS_AS((void) (matrix * wrong), Hoppy::Test::EigenAssertionFailure);
}
#endif
