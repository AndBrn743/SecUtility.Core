// SPDX-License-Identifier: MIT

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/Hoppy.hpp>

#include <Eigen/Core>

#include <type_traits>
#include <utility>

namespace
{
	template <typename T, typename = void>
	struct has_as_diagonal : std::false_type {};
	template <typename T>
	struct has_as_diagonal<T, std::void_t<decltype(std::declval<T>().asDiagonal())>> : std::true_type {};

	template <typename T, typename = void>
	struct has_diagonal : std::false_type {};
	template <typename T>
	struct has_diagonal<T, std::void_t<decltype(std::declval<T>().diagonal())>> : std::true_type {};
}

TEST_CASE("matrix diagonal is a typed column-vector view")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2};
	matrix[0](0, 0) = 1.0;
	matrix[1] << 2.0, 3.0, 4.0, 5.0;

	auto diagonal = matrix.diagonal();
	using View = decltype(diagonal);
	static_assert(std::is_same_v<typename Eigen::internal::traits<View>::StorageKind,
	                             Hoppy::Detail::BlockVectorStorage>);
	static_assert(std::is_same_v<typename Eigen::internal::traits<View>::Orientation, Hoppy::Column>);
	static_assert(std::is_assignable_v<decltype(diagonal[1](0)), double>);
	REQUIRE(diagonal.blockingInfo() == std::vector<Eigen::Index>{1, 2});
	Hoppy::Test::requireApprox(diagonal.toDense(), matrix.toDense().diagonal());

	diagonal[1](0) = 7.0;
	diagonal[1](1) = 8.0;
	REQUIRE(matrix[1](0, 0) == 7.0);
	REQUIRE(matrix[1](1, 1) == 8.0);
	REQUIRE(matrix[1](0, 1) == 3.0);
	REQUIRE(matrix[1](1, 0) == 4.0);

	const auto& constMatrix = matrix;
	const auto constDiagonal = constMatrix.diagonal();
	static_assert(!std::is_assignable_v<decltype(constDiagonal[0](0)), double>);
}

TEST_CASE("diagonal assignment uses block-local temporaries for aliases")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{2};
	matrix[0] << 1.0, 11.0, 12.0, 2.0;
	Hoppy::BlockVector<double> increment{2};
	increment.asDense() << 3.0, 4.0;

	auto diagonal = matrix.diagonal();
	diagonal = diagonal + increment;
	Hoppy::Test::requireApprox(diagonal.toDense(), (Eigen::Vector2d() << 4.0, 6.0).finished());
	REQUIRE(matrix[0](0, 1) == 11.0);
	REQUIRE(matrix[0](1, 0) == 12.0);

	auto sameSource = matrix.diagonal();
	diagonal = sameSource;
	Hoppy::Test::requireApprox(diagonal.toDense(), sameSource.toDense());
}

TEST_CASE("column vectors form lazy block-diagonal wrappers")
{
	Hoppy::BlockVector<double> vector{1, 2};
	vector.asDense() << 1.0, 2.0, 3.0;
	const auto diagonal = vector.asDiagonal();
	using View = decltype(diagonal);
	static_assert(std::is_same_v<typename Eigen::internal::traits<View>::StorageKind,
	                             Hoppy::Detail::BlockDiagonalStorage>);
	Hoppy::Test::requireApprox(diagonal.toDense(), vector.asDense().asDiagonal().toDenseMatrix());
}

TEST_CASE("view participation separates owning rvalues, orientation, and expression prvalues")
{
	using Matrix = Hoppy::BlockDiagonalMatrix<double>;
	using ColumnVector = Hoppy::BlockVector<double>;
	using RowVector = Hoppy::BlockVector<double, Hoppy::Row>;
	using MatrixExpression = decltype(std::declval<const Matrix&>().transpose());
	using VectorExpression = decltype(std::declval<const ColumnVector&>().conjugate());
	using TwiceTransposed = decltype(std::declval<const ColumnVector&>().transpose().transpose());

	static_assert(has_diagonal<Matrix&>::value);
	static_assert(has_diagonal<const Matrix&>::value);
	static_assert(!has_diagonal<Matrix&&>::value);
	static_assert(!has_diagonal<const Matrix&&>::value);
	static_assert(has_diagonal<MatrixExpression&&>::value);
	static_assert(has_as_diagonal<const ColumnVector&>::value);
	static_assert(!has_as_diagonal<ColumnVector&&>::value);
	static_assert(!has_as_diagonal<const ColumnVector&&>::value);
	static_assert(!has_as_diagonal<const RowVector&>::value);
	static_assert(has_as_diagonal<VectorExpression&&>::value);
	static_assert(has_as_diagonal<TwiceTransposed&&>::value);
}

TEST_CASE("prvalue view chains retain their expression parents")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2};
	matrix[0](0, 0) = 1.0;
	matrix[1] << 2.0, 3.0, 4.0, 5.0;
	const auto transposedDiagonal = matrix.transpose().diagonal();
	Hoppy::Test::requireApprox(transposedDiagonal.toDense(), matrix.toDense().transpose().diagonal());

	Hoppy::BlockVector<double> vector{1, 2};
	vector.asDense() << 1.0, 2.0, 3.0;
	const auto conjugatedDiagonal = vector.conjugate().asDiagonal();
	const auto twiceTransposedDiagonal = vector.transpose().transpose().asDiagonal();
	Hoppy::Test::requireApprox(conjugatedDiagonal.toDense(),
	                           vector.asDense().conjugate().asDiagonal().toDenseMatrix());
	Hoppy::Test::requireApprox(twiceTransposedDiagonal.toDense(),
	                           vector.asDense().asDiagonal().toDenseMatrix());
}

#ifndef EIGEN_NO_DEBUG
TEST_CASE("diagonal assignment requires equal blocking")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2};
	Hoppy::BlockVector<double> vector{3};
	auto diagonal = matrix.diagonal();
	REQUIRE_THROWS_AS(diagonal = vector, Hoppy::Test::EigenAssertionFailure);
}
#endif
