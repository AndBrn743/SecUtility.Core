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
	static_assert(std::is_base_of_v<Eigen::MatrixBase<SymmetricProduct>, SymmetricProduct>);

	template <typename Tag>
	struct CountingExpression : Hoppy::TriangularCompressedMatrixExpr<CountingExpression<Tag>>
	{
		using Scalar = double;
		using StructureTag = Tag;
		static constexpr int RowsAtCompileTime = 2;
		static constexpr int ColsAtCompileTime = 2;
		static constexpr Hoppy::TrianglePacking PackingValue = Hoppy::TrianglePacking::Lower;
		static constexpr bool IsTriangularCompressed = true;
		static constexpr bool IsWritable = false;
		Eigen::Index dimension() const { return 2; }
		Eigen::Index rows() const { return 2; }
		Eigen::Index cols() const { return 2; }
		double coeff(Eigen::Index row, Eigen::Index column) const
		{
			++Calls;
			return row == column ? 2.0 : 1.0;
		}
		mutable int Calls = 0;
	};
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
	REQUIRE(product.norm() == Catch::Approx((left.toDense() * right.toDense()).norm()));
	Hoppy::Test::requireApprox(product.transpose().eval(),
	                           (left.toDense() * right.toDense()).transpose());
	Hoppy::Test::requireApprox((product * Eigen::Matrix2d::Constant(2.0)).eval(),
	                           left.toDense() * right.toDense() * Eigen::Matrix2d::Constant(2.0));
	Hoppy::Test::requireApprox((left * (left * right)).eval(),
	                           left.toDense() * left.toDense() * right.toDense());
}

TEST_CASE("packed dense and dense packed products support rectangular operands")
{
	Hoppy::SymmetricMatrix<double, 3> matrix;
	matrix.setOnes();
	Eigen::Matrix<double, 3, 2> right;
	right << 1, 2, 3, 4, 5, 6;
	Eigen::Matrix<double, 2, 3> left;
	left << 1, 2, 3, 4, 5, 6;
	using RectangularProduct = decltype(matrix * right);
	using RectangularReturn = typename Eigen::internal::traits<RectangularProduct>::ReturnType;
	static_assert(RectangularReturn::RowsAtCompileTime == 3);
	static_assert(RectangularReturn::ColsAtCompileTime == 2);
	Hoppy::Test::requireApprox((matrix * right).toDense(), matrix.toDense() * right);
	Hoppy::Test::requireApprox((left * matrix).toDense(), left * matrix.toDense());
	Eigen::Vector3d vector(1, 2, 3);
	Hoppy::Test::requireApprox((matrix * vector).toDense(), matrix.toDense() * vector);
	Hoppy::Test::requireApprox((matrix * right).transpose().eval(),
	                           (matrix.toDense() * right).transpose());
	Hoppy::Test::requireApprox(((matrix * right) * Eigen::Matrix2d::Constant(2.0)).eval(),
	                           matrix.toDense() * right * Eigen::Matrix2d::Constant(2.0));
	Eigen::Matrix<double, 3, 2, Eigen::RowMajor> rowMajor = matrix * right;
	Hoppy::Test::requireApprox(rowMajor, matrix.toDense() * right);
}

TEST_CASE("dense promoted products are alias safe")
{
	Hoppy::SymmetricMatrix<double, 2> packed;
	packed.setZero(); packed(0, 0) = 2; packed(0, 1) = 1; packed(1, 1) = 3;
	Eigen::Matrix2d dense;
	dense << 1, 2, 3, 4;
	const Eigen::Matrix2d original = dense;
	dense = packed * dense;
	Hoppy::Test::requireApprox(dense, packed.toDense() * original);
	dense = dense * packed;
	Hoppy::Test::requireApprox(dense, packed.toDense() * original * packed.toDense());
}

TEST_CASE("dense promoted product construction is lazy and nested evaluation materializes once")
{
	CountingExpression<Hoppy::Detail::SymmetricTag> left;
	CountingExpression<Hoppy::Detail::AntiSymmetricTag> right;
	const auto expression = (left * right) * Eigen::Matrix2d::Identity();
	REQUIRE(left.Calls == 0);
	REQUIRE(right.Calls == 0);
	const Eigen::Matrix2d evaluated = expression;
	REQUIRE(left.Calls == 8);
	REQUIRE(right.Calls == 8);
	REQUIRE(evaluated.allFinite());
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
