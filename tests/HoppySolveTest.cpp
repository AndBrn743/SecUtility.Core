// SPDX-License-Identifier: MIT

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/Hoppy.hpp>

#include <Eigen/Core>

#include <complex>
#include <limits>
#include <type_traits>
#include <utility>

namespace
{
	template <typename Matrix, typename = void>
	struct has_inverse : std::false_type
	{};

	template <typename Matrix>
	struct has_inverse<Matrix, std::void_t<decltype(std::declval<const Matrix&>().inverse())>>
	    : std::true_type
	{};

	template <typename Matrix, typename Rhs, typename = void>
	struct has_solve : std::false_type
	{};

	template <typename Matrix, typename Rhs>
	struct has_solve<Matrix, Rhs,
	                 std::void_t<decltype(std::declval<const Matrix&>().solve(
	                         std::declval<const Rhs&>()))>> : std::true_type
	{};

	template <typename Matrix>
	void makeWellConditioned(Matrix& matrix)
	{
		for (Eigen::Index index = 0; index < matrix.blockCount(); ++index)
		{
			matrix[index].setRandom();
			matrix[index].diagonal().array() += typename Matrix::Scalar{4};
		}
	}
}  // namespace

TEST_CASE("inverse is lazy blockwise and supports expression chains")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2};
	makeWellConditioned(matrix);
	const auto inverse = matrix.inverse();
	static_assert(!std::is_same_v<std::remove_cv_t<decltype(inverse)>,
	                              Hoppy::BlockDiagonalMatrix<double>>);
	Hoppy::Test::requireApprox(inverse.toDense(), matrix.toDense().inverse());
	Hoppy::Test::requireApprox((matrix.inverse().transpose() / 2.0).toDense(),
	                           matrix.toDense().inverse().transpose() / 2.0);
	Hoppy::Test::requireApprox((matrix.inverse() * matrix).toDense(),
	                           matrix.toDense().inverse() * matrix.toDense());

	const Eigen::Matrix3d expected = matrix.toDense().inverse();
	matrix = matrix.inverse();
	Hoppy::Test::requireApprox(matrix.toDense(), expected);
}

TEST_CASE("complex inverse matches dense Eigen")
{
	using Complex = std::complex<double>;
	Hoppy::BlockDiagonalMatrix<Complex> matrix{1, 2};
	makeWellConditioned(matrix);
	Hoppy::Test::requireApprox(matrix.inverse().toDense(), matrix.toDense().inverse());
}

TEST_CASE("solve with a column block vector is eager and blockwise")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2};
	makeWellConditioned(matrix);
	Hoppy::BlockVector<double> rhs{1, 2};
	rhs.asDense() << 1.0, 2.0, -3.0;
	const auto result = matrix.solve(rhs);
	static_assert(std::is_same_v<std::remove_cv_t<decltype(result)>, Hoppy::BlockVector<double>>);
	REQUIRE(result.blockingInfo() == matrix.blockingInfo());
	Hoppy::Test::requireApprox(result.asDense(), matrix.toDense().partialPivLu().solve(rhs.asDense()));

	const auto expressionResult = matrix.solve(rhs + rhs);
	Hoppy::Test::requireApprox(expressionResult.asDense(),
	                           matrix.toDense().partialPivLu().solve(2.0 * rhs.asDense()));
}

TEST_CASE("solve with a dense matrix supports multiple and zero rhs columns")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2};
	makeWellConditioned(matrix);
	Eigen::Matrix<double, 3, 4> rhs;
	rhs.setRandom();
	const auto result = matrix.solve(rhs);
	static_assert(std::is_same_v<std::remove_cv_t<decltype(result)>, Eigen::MatrixXd>);
	Hoppy::Test::requireApprox(result, matrix.toDense().partialPivLu().solve(rhs));

	Eigen::MatrixXd oneColumn(3, 1);
	oneColumn.setRandom();
	Hoppy::Test::requireApprox(matrix.solve(oneColumn),
	                           matrix.toDense().partialPivLu().solve(oneColumn));
	Eigen::MatrixXd noColumns(3, 0);
	REQUIRE(matrix.solve(noColumns).rows() == 3);
	REQUIRE(matrix.solve(noColumns).cols() == 0);
}

TEST_CASE("complex solves and independently scaled blocks match dense Eigen")
{
	using Complex = std::complex<double>;
	Hoppy::BlockDiagonalMatrix<Complex> matrix{1, 2};
	matrix[0](0, 0) = Complex{2.0, 1.0};
	matrix[1] << Complex{10000.0, 2.0}, Complex{3.0, -1.0},
	             Complex{-2.0, 4.0}, Complex{12000.0, -3.0};
	Hoppy::BlockVector<Complex> vector{1, 2};
	vector.asDense() << Complex{1.0, 2.0}, Complex{-3.0, 1.0}, Complex{2.0, -4.0};
	Hoppy::Test::requireApprox(matrix.solve(vector).asDense(),
	                           matrix.toDense().partialPivLu().solve(vector.asDense()));

	Eigen::MatrixXcd dense(3, 2);
	dense.setRandom();
	Hoppy::Test::requireApprox(matrix.solve(dense),
	                           matrix.toDense().partialPivLu().solve(dense));
}

TEST_CASE("empty and one-dimensional dense solves are supported")
{
	const Hoppy::BlockDiagonalMatrix<double> empty;
	const Hoppy::BlockVector<double> emptyVector;
	REQUIRE(empty.solve(emptyVector).size() == 0);
	Eigen::MatrixXd emptyRhs(0, 3);
	REQUIRE(empty.solve(emptyRhs).rows() == 0);
	REQUIRE(empty.solve(emptyRhs).cols() == 3);

	Hoppy::BlockDiagonalMatrix<double> one{1};
	one[0](0, 0) = 2.0;
	Eigen::RowVector3d row;
	row << 2.0, 4.0, 6.0;
	Hoppy::Test::requireApprox(one.solve(row), row / 2.0);
}

TEST_CASE("inverse and solve availability follows the scalar and rhs contracts")
{
	using RealMatrix = Hoppy::BlockDiagonalMatrix<double>;
	using IntegerMatrix = Hoppy::BlockDiagonalMatrix<int>;
	using Column = Hoppy::BlockVector<double, Hoppy::BlockVectorOrientation::Column>;
	using Row = Hoppy::BlockVector<double, Hoppy::BlockVectorOrientation::Row>;
	using FloatColumn = Hoppy::BlockVector<float, Hoppy::BlockVectorOrientation::Column>;
	static_assert(has_inverse<RealMatrix>::value);
	static_assert(!has_inverse<IntegerMatrix>::value);
	static_assert(has_solve<RealMatrix, Column>::value);
	static_assert(!has_solve<RealMatrix, Row>::value);
	static_assert(!has_solve<RealMatrix, FloatColumn>::value);
	static_assert(!has_solve<IntegerMatrix, Hoppy::BlockVector<int>>::value);
	static_assert(!has_solve<RealMatrix, Eigen::Vector3d>::value);
	static_assert(!has_solve<RealMatrix, Eigen::MatrixXf>::value);
}

TEST_CASE("singular and nonfinite blocks add no library rank checks")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{2};
	matrix[0] << 1.0, 2.0, 2.0, 4.0;
	Eigen::MatrixXd rhs = Eigen::MatrixXd::Ones(2, 2);
	REQUIRE_NOTHROW((void)matrix.solve(rhs));
	matrix[0](0, 0) = std::numeric_limits<double>::quiet_NaN();
	REQUIRE_NOTHROW((void)matrix.solve(rhs));
}

#ifdef HOPPY_TEST_EIGEN_ASSERT_THROWS
TEST_CASE("solve enforces rhs dimensions and blockings")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2};
	makeWellConditioned(matrix);
	const Hoppy::BlockVector<double> wrongBlocking{3};
	REQUIRE_THROWS_AS((void)matrix.solve(wrongBlocking), Hoppy::Test::EigenAssertionFailure);
	const Eigen::MatrixXd wrongRows(2, 3);
	REQUIRE_THROWS_AS((void)matrix.solve(wrongRows), Hoppy::Test::EigenAssertionFailure);
}
#endif
