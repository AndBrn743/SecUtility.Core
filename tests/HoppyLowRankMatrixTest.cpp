// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Andy Brown

#include <SecUtility/Hoppy/LowRankMatrix.hpp>

#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <type_traits>
#include <utility>


namespace
{
	template <typename Matrix, typename = void>
	struct HasMutableCoefficients : std::false_type
	{
	};

	template <typename Matrix>
	struct HasMutableCoefficients<Matrix,
	                              std::void_t<decltype(std::declval<Matrix&>().coefficients()[Eigen::Index{}]
	                                                   = std::declval<typename Matrix::Scalar>())>> : std::true_type
	{
	};

	template <typename Matrix, typename = void>
	struct HasMutableLeftVectors : std::false_type
	{
	};

	template <typename Matrix>
	struct HasMutableLeftVectors<Matrix,
	                             std::void_t<decltype(std::declval<Matrix&>().leftVectors()(Eigen::Index{}, Eigen::Index{})
	                                                  = std::declval<typename Matrix::Scalar>())>> : std::true_type
	{
	};
}


using DynamicMatrix = Hoppy::LowRankMatrixX<double>;
using FixedMatrix = Hoppy::LowRankMatrix<double, 2, 3>;
using DynamicRowsMatrix = Hoppy::LowRankMatrix<double, Eigen::Dynamic, 3>;
using DynamicColsMatrix = Hoppy::LowRankMatrix<double, 2, Eigen::Dynamic>;
using DynamicMatrixTerm = decltype(std::declval<const DynamicMatrix&>().term(0));

static_assert(std::is_base_of_v<Eigen::EigenBase<DynamicMatrix>, DynamicMatrix>);
static_assert(std::is_base_of_v<Hoppy::LowRankMatrixBase<DynamicMatrix>, DynamicMatrix>);
static_assert(std::is_same_v<Eigen::internal::traits<DynamicMatrix>::Scalar, double>);
static_assert(std::is_same_v<Hoppy::LowRankMatrixBase<DynamicMatrix>::Scalar, double>);
static_assert(Hoppy::LowRankMatrixBase<FixedMatrix>::RowsAtCompileTime == 2);
static_assert(Hoppy::LowRankMatrixBase<FixedMatrix>::ColsAtCompileTime == 3);
static_assert(std::is_same_v<Eigen::internal::traits<DynamicMatrix>::StorageKind, Hoppy::LowRankStorage>);
static_assert(Eigen::internal::traits<FixedMatrix>::RowsAtCompileTime == 2);
static_assert(Eigen::internal::traits<FixedMatrix>::ColsAtCompileTime == 3);
static_assert(!HasMutableCoefficients<DynamicMatrix>::value);
static_assert(!HasMutableLeftVectors<DynamicMatrix>::value);
static_assert(std::is_const_v<std::remove_reference_t<decltype(std::declval<DynamicMatrixTerm>().coefficient)>>);
static_assert(std::is_lvalue_reference_v<decltype(std::declval<DynamicMatrixTerm>().coefficient)>);


namespace
{
	template <typename Derived>
	Eigen::Index TermCountThroughBase(const Hoppy::LowRankMatrixBase<Derived>& matrix)
	{
		return matrix.termCount();
	}
}


TEST_CASE("General low-rank matrices preserve dimensions and rank-zero state", "[Hoppy][LowRankMatrix]")
{
	SECTION("fully dynamic")
	{
		const DynamicMatrix defaultMatrix;
		CHECK(defaultMatrix.rows() == 0);
		CHECK(defaultMatrix.cols() == 0);
		CHECK(defaultMatrix.termCount() == 0);

		const DynamicMatrix matrix(2, 3);
		CHECK(matrix.rows() == 2);
		CHECK(matrix.cols() == 3);
		CHECK(matrix.termCount() == 0);
		CHECK(matrix.leftVectors().rows() == 2);
		CHECK(matrix.leftVectors().cols() == 0);
		CHECK(matrix.rightVectors().rows() == 3);
		CHECK(matrix.rightVectors().cols() == 0);
	}

	SECTION("fixed and partially dynamic")
	{
		const FixedMatrix fixed;
		CHECK(fixed.rows() == 2);
		CHECK(fixed.cols() == 3);

		const DynamicRowsMatrix dynamicRows(4, 3);
		CHECK(dynamicRows.rows() == 4);
		CHECK(dynamicRows.cols() == 3);

		const DynamicColsMatrix dynamicCols(2, 5);
		CHECK(dynamicCols.rows() == 2);
		CHECK(dynamicCols.cols() == 5);

	}
}


TEST_CASE("General low-rank matrices add and inspect individual terms", "[Hoppy][LowRankMatrix]")
{
	DynamicMatrix matrix(2, 3);
	const Eigen::Vector2d left{1, 2};
	const Eigen::Vector3d right{3, 4, 5};

	matrix.addTerm(2.5, left, right);
	REQUIRE(matrix.termCount() == 1);
	CHECK(TermCountThroughBase(matrix) == 1);
	const Hoppy::LowRankMatrixBase<DynamicMatrix>& base = matrix;
	CHECK(base.rows() == 2);
	CHECK(base.cols() == 3);
	CHECK(base.termCount() == 1);
	CHECK(base.coefficients() == matrix.coefficients());
	CHECK(base.leftVectors() == matrix.leftVectors());
	CHECK(base.rightVectors() == matrix.rightVectors());
	CHECK(base.coefficientOfTerm(0) == 2.5);
	CHECK(base.leftVectorOfTerm(0) == left);
	CHECK(base.rightVectorOfTerm(0) == right);
	CHECK(matrix.coefficientOfTerm(0) == 2.5);
	CHECK(matrix.leftVectorOfTerm(0) == left);
	CHECK(matrix.rightVectorOfTerm(0) == right);
	CHECK(matrix.coefficients() == Eigen::VectorXd::Constant(1, 2.5));
	CHECK(matrix.leftVectors() == left);
	CHECK(matrix.rightVectors() == right);

	const auto namedTerm = matrix.term(0);
	CHECK(namedTerm.leftVector == left);
	CHECK(namedTerm.coefficient == 2.5);
	CHECK(namedTerm.rightVector == right);
	CHECK(&namedTerm.coefficient == &matrix.coefficientOfTerm(0));

	const auto [termLeft, coefficient, termRight] = matrix.term(0);
	CHECK(termLeft == left);
	CHECK(coefficient == 2.5);
	CHECK(termRight == right);

}


TEST_CASE("General low-rank matrix constructors infer dimensions from terms", "[Hoppy][LowRankMatrix]")
{
	const Eigen::Vector2d left{1, 2};
	const Eigen::Vector3d right{3, 4, 5};
	const DynamicMatrix single(2.5, left, right);
	CHECK(single.rows() == 2);
	CHECK(single.cols() == 3);
	CHECK(single.termCount() == 1);

	Eigen::Matrix<double, 2, 3> leftVectors;
	leftVectors << 1, 2, 3, 4, 5, 6;
	Eigen::Matrix<double, 3, 3> rightVectors;
	rightVectors << 7, 8, 9, 10, 11, 12, 13, 14, 15;
	const Eigen::Vector3d coefficients{2, 0, -4};
	const DynamicMatrix batch(coefficients, leftVectors, rightVectors);
	CHECK(batch.rows() == 2);
	CHECK(batch.cols() == 3);
	REQUIRE(batch.termCount() == 2);
	CHECK((batch.coefficients() == Eigen::Vector2d{2, -4}));
	CHECK(batch.leftVectorOfTerm(0) == leftVectors.col(0));
	CHECK(batch.leftVectorOfTerm(1) == leftVectors.col(2));
	CHECK(batch.rightVectorOfTerm(0) == rightVectors.col(0));
	CHECK(batch.rightVectorOfTerm(1) == rightVectors.col(2));
}


TEST_CASE("General low-rank batch insertion filters only exact zeros and preserves order",
          "[Hoppy][LowRankMatrix]")
{
	using Complex = std::complex<double>;
	Hoppy::LowRankMatrixX<Complex> matrix(2, 3);
	const Eigen::VectorXcd coefficients{{Complex{0, 0}, Complex{1, 2}, Complex{0, 1}, Complex{-3, 0}}};
	const Eigen::MatrixXcd left = Eigen::MatrixXcd::Random(2, 4);
	const Eigen::MatrixXcd right = Eigen::MatrixXcd::Random(3, 4);

	matrix.addTerms(coefficients, left, right);
	REQUIRE(matrix.termCount() == 3);
	CHECK(matrix.coefficientOfTerm(0) == coefficients[1]);
	CHECK(matrix.coefficientOfTerm(1) == coefficients[2]);
	CHECK(matrix.coefficientOfTerm(2) == coefficients[3]);
	CHECK(matrix.leftVectorOfTerm(0) == left.col(1));
	CHECK(matrix.leftVectorOfTerm(1) == left.col(2));
	CHECK(matrix.leftVectorOfTerm(2) == left.col(3));
	CHECK(matrix.rightVectorOfTerm(0) == right.col(1));
	CHECK(matrix.rightVectorOfTerm(1) == right.col(2));
	CHECK(matrix.rightVectorOfTerm(2) == right.col(3));

	const auto oldTermCount = matrix.termCount();
	matrix.addTerm(Complex{}, left.col(0), right.col(0));
	CHECK(matrix.termCount() == oldTermCount);
	matrix.addTerm(Complex{}, Eigen::VectorXcd::Zero(7), Eigen::VectorXcd::Zero(9));
	CHECK(matrix.termCount() == oldTermCount);

	SECTION("an all-zero batch leaves an existing expansion unchanged")
	{
		const Eigen::VectorXcd oldCoefficients = matrix.coefficients();
		const Eigen::MatrixXcd oldLeft = matrix.leftVectors();
		const Eigen::MatrixXcd oldRight = matrix.rightVectors();
		const auto oldCapacity = matrix.capacity();

		matrix.addTerms(Eigen::VectorXcd::Zero(3), Eigen::MatrixXcd::Random(2, 3),
		                Eigen::MatrixXcd::Random(3, 3));

		CHECK(matrix.termCount() == oldTermCount);
		CHECK(matrix.capacity() == oldCapacity);
		CHECK(matrix.coefficients() == oldCoefficients);
		CHECK(matrix.leftVectors() == oldLeft);
		CHECK(matrix.rightVectors() == oldRight);
	}

	SECTION("an empty batch leaves an existing expansion unchanged")
	{
		const Eigen::VectorXcd oldCoefficients = matrix.coefficients();
		const auto oldCapacity = matrix.capacity();

		matrix.addTerms(Eigen::VectorXcd{}, Eigen::MatrixXcd(2, 0), Eigen::MatrixXcd(3, 0));

		CHECK(matrix.termCount() == oldTermCount);
		CHECK(matrix.capacity() == oldCapacity);
		CHECK(matrix.coefficients() == oldCoefficients);
	}
}


TEST_CASE("An all-zero batch constructor preserves inferred rectangular dimensions", "[Hoppy][LowRankMatrix]")
{
	const DynamicMatrix matrix(Eigen::Vector3d::Zero(), Eigen::MatrixXd::Random(2, 3),
	                           Eigen::MatrixXd::Random(4, 3));

	CHECK(matrix.rows() == 2);
	CHECK(matrix.cols() == 4);
	CHECK(matrix.termCount() == 0);
	CHECK(matrix.coefficients().size() == 0);
	CHECK(matrix.leftVectors().rows() == 2);
	CHECK(matrix.leftVectors().cols() == 0);
	CHECK(matrix.rightVectors().rows() == 4);
	CHECK(matrix.rightVectors().cols() == 0);
}


TEST_CASE("General low-rank insertion evaluates aliased views before reallocating", "[Hoppy][LowRankMatrix]")
{
	DynamicMatrix matrix(2, 3);
	Eigen::Matrix<double, 2, 2> left;
	left << 1, 3, 2, 4;
	Eigen::Matrix<double, 3, 2> right;
	right << 5, 8, 6, 9, 7, 10;
	matrix.addTerms(Eigen::Vector2d{2, 3}, left, right);

	const Eigen::MatrixXd expectedLeft = matrix.leftVectors();
	const Eigen::MatrixXd expectedRight = matrix.rightVectors();
	const Eigen::VectorXd expectedCoefficients = matrix.coefficients();
	matrix.addTerms(matrix.coefficients(), matrix.leftVectors(), matrix.rightVectors());

	REQUIRE(matrix.termCount() == 4);
	CHECK(matrix.leftVectors().leftCols(2) == expectedLeft);
	CHECK(matrix.leftVectors().rightCols(2) == expectedLeft);
	CHECK(matrix.rightVectors().leftCols(2) == expectedRight);
	CHECK(matrix.rightVectors().rightCols(2) == expectedRight);
	CHECK(matrix.coefficients().head(2) == expectedCoefficients);
	CHECK(matrix.coefficients().tail(2) == expectedCoefficients);

	const Eigen::Vector2d aliasedLeft = matrix.leftVectorOfTerm(0);
	const Eigen::Vector3d aliasedRight = matrix.rightVectorOfTerm(0);
	matrix.addTerm(matrix.coefficientOfTerm(0), matrix.leftVectorOfTerm(0), matrix.rightVectorOfTerm(0));
	REQUIRE(matrix.termCount() == 5);
	CHECK(matrix.leftVectorOfTerm(4) == aliasedLeft);
	CHECK(matrix.rightVectorOfTerm(4) == aliasedRight);
}


TEST_CASE("General low-rank values support reserve, clear, copy, and move", "[Hoppy][LowRankMatrix]")
{
	DynamicMatrix matrix(2, 3);
	matrix.reserve(8);
	CHECK(matrix.capacity() >= 8);

	matrix.addTerm(2, Eigen::Vector2d{1, 2}, Eigen::Vector3d{3, 4, 5});
	DynamicMatrix copy = matrix;
	CHECK(copy.rows() == 2);
	CHECK(copy.cols() == 3);
	CHECK(copy.coefficients() == matrix.coefficients());
	CHECK(copy.leftVectors() == matrix.leftVectors());
	CHECK(copy.rightVectors() == matrix.rightVectors());

	DynamicMatrix moved = std::move(copy);
	CHECK(moved.termCount() == 1);
	CHECK((moved.leftVectorOfTerm(0) == Eigen::Vector2d{1, 2}));
	DynamicMatrix moveAssigned;
	moveAssigned = std::move(moved);
	CHECK(moveAssigned.rows() == 2);
	CHECK(moveAssigned.cols() == 3);
	CHECK(moveAssigned.termCount() == 1);

	matrix.clear();
	CHECK(matrix.rows() == 2);
	CHECK(matrix.cols() == 3);
	CHECK(matrix.termCount() == 0);
	CHECK(matrix.capacity() >= 8);
}
