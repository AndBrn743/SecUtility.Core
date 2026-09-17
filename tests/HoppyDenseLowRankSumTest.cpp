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
	struct HasFactors : std::false_type
	{
	};

	template <typename Matrix>
	struct HasFactors<Matrix,
	                  std::void_t<decltype(std::declval<const Matrix&>().coefficients()),
	                              decltype(std::declval<const Matrix&>().leftVectors()),
	                              decltype(std::declval<const Matrix&>().rightVectors())>> : std::true_type
	{
	};

	Hoppy::LowRankMatrixX<double> Correction(const Eigen::Index rows, const Eigen::Index cols)
	{
		Hoppy::LowRankMatrixX<double> correction(rows, cols);
		correction.addTerm(2, Eigen::VectorXd::LinSpaced(rows, 1, static_cast<double>(rows)),
		                   Eigen::VectorXd::LinSpaced(cols, -1, static_cast<double>(cols - 2)));
		return correction;
	}
}


using FixedDense = Eigen::Matrix<double, 3, 2>;
using FixedCorrection = Hoppy::LowRankMatrix<double, 3, 2>;
using FixedSum = decltype(std::declval<const FixedDense&>() + std::declval<const FixedCorrection&>());
static_assert(FixedSum::RowsAtCompileTime == 3);
static_assert(FixedSum::ColsAtCompileTime == 2);
static_assert(std::is_base_of_v<Eigen::EigenBase<FixedSum>, FixedSum>);
static_assert(!std::is_base_of_v<Hoppy::LowRankMatrixBase<FixedSum>, FixedSum>);
static_assert(!HasFactors<FixedSum>::value);


TEST_CASE("Dense and low-rank sums support every sign and operand order", "[Hoppy][LowRankMatrix]")
{
	const Eigen::MatrixXd dense = Eigen::MatrixXd::Random(3, 2);
	const auto correction = Correction(3, 2);
	const Eigen::MatrixXd lowRankDense = correction.toDense();

	const auto densePlus = dense + correction;
	const auto denseMinus = dense - correction;
	const auto lowRankPlus = correction + dense;
	const auto lowRankMinus = correction - dense;
	CHECK(densePlus.toDense().isApprox(dense + lowRankDense));
	CHECK(denseMinus.toDense().isApprox(dense - lowRankDense));
	CHECK(lowRankPlus.toDense().isApprox(lowRankDense + dense));
	CHECK(lowRankMinus.toDense().isApprox(lowRankDense - dense));

	const Eigen::MatrixXd materialized = densePlus;
	CHECK(materialized.isApprox(dense + lowRankDense));
	Eigen::MatrixXd assigned(3, 2);
	assigned = lowRankMinus;
	CHECK(assigned.isApprox(lowRankDense - dense));
	assigned += denseMinus;
	CHECK(assigned.isApprox(lowRankDense - dense + dense - lowRankDense));
	assigned -= denseMinus;
	CHECK(assigned.isApprox(lowRankDense - dense));
}


TEST_CASE("Dense and low-rank sums multiply vectors, matrices, and blocks", "[Hoppy][LowRankMatrix]")
{
	const Eigen::MatrixXd dense = Eigen::MatrixXd::Random(3, 4);
	const auto correction = Correction(3, 4);
	const auto expression = dense - correction;
	const Eigen::MatrixXd reference = dense - correction.toDense();
	const Eigen::Vector4d vector = Eigen::Vector4d::Random();
	const Eigen::MatrixXd rhs = Eigen::MatrixXd::Random(4, 3);

	CHECK(Eigen::VectorXd(expression * vector).isApprox(reference * vector));
	CHECK(Eigen::MatrixXd(expression * rhs).isApprox(reference * rhs));
	CHECK(Eigen::MatrixXd(expression * rhs.leftCols(2)).isApprox(reference * rhs.leftCols(2)));

	Eigen::MatrixXd accumulated = Eigen::MatrixXd::Random(3, 3);
	const Eigen::MatrixXd initial = accumulated;
	using Expression = std::decay_t<decltype(expression)>;
	Eigen::internal::generic_product_impl<Expression, Eigen::MatrixXd>::scaleAndAddTo(
	        accumulated, expression, rhs, 2.5);
	CHECK(accumulated.isApprox(initial + 2.5 * reference * rhs));
}


TEST_CASE("Dense and low-rank sum products handle destination aliasing", "[Hoppy][LowRankMatrix]")
{
	const Eigen::Matrix3d dense = Eigen::Matrix3d::Random();
	Hoppy::LowRankMatrix<double, 3, 3> correction;
	correction.addTerm(0.5, Eigen::Vector3d{1, 2, -1}, Eigen::Vector3d{0.25, -2, 3});
	const auto expression = dense + correction;
	const Eigen::Matrix3d reference = expression.toDense();
	Eigen::Vector3d vector{1, -2, 0.5};
	const Eigen::Vector3d expected = reference * vector;
	vector = expression * vector;
	CHECK(vector.isApprox(expected));
}


TEST_CASE("Dense and low-rank sum diagonals avoid full materialization", "[Hoppy][LowRankMatrix]")
{
	const Eigen::MatrixXcd dense = Eigen::MatrixXcd::Random(4, 4);
	const Eigen::MatrixXcd vectors = Eigen::MatrixXcd::Random(4, 2);
	const Hoppy::LowRankSelfAdjointMatrixX<std::complex<double>> correction(Eigen::Vector2d{2, -1}, vectors);
	const auto expression = dense + correction;

	CHECK(expression.diagonal().isApprox((dense + correction.toDense()).diagonal()));
	CHECK(expression.toDense().isApprox(dense + correction.toDense()));
}


TEST_CASE("Dense and low-rank sums promote compatible scalar types", "[Hoppy][LowRankMatrix]")
{
	const Eigen::MatrixXd dense = Eigen::MatrixXd::Random(3, 3);
	const Eigen::MatrixXcd vectors = Eigen::MatrixXcd::Random(3, 1);
	const Hoppy::LowRankSelfAdjointMatrixX<std::complex<double>> correction(2.0, vectors);
	const auto expression = dense + correction;
	static_assert(std::is_same_v<typename std::decay_t<decltype(expression)>::Scalar, std::complex<double>>);
	CHECK(expression.toDense().isApprox(dense.cast<std::complex<double>>() + correction.toDense()));
	const Eigen::Vector3d rhs{1, -2, 0.5};
	CHECK(Eigen::VectorXcd(expression * rhs).isApprox(expression.toDense() * rhs));
}


TEST_CASE("Dense and rank-zero low-rank sums preserve rectangular dimensions", "[Hoppy][LowRankMatrix]")
{
	const Eigen::MatrixXd dense = Eigen::MatrixXd::Random(2, 4);
	const Hoppy::LowRankMatrixX<double> empty(2, 4);
	const auto expression = dense + empty;
	CHECK(expression.rows() == 2);
	CHECK(expression.cols() == 4);
	CHECK(expression.toDense().isApprox(dense));
	const Eigen::MatrixXd emptyRhs(4, 0);
	const Eigen::MatrixXd result = expression * emptyRhs;
	CHECK(result.rows() == 2);
	CHECK(result.cols() == 0);
}


TEST_CASE("Dense and low-rank sum expressions safely own temporaries", "[Hoppy][LowRankMatrix]")
{
	const Eigen::MatrixXd dense = Eigen::MatrixXd::Random(3, 3);
	const auto correction = Correction(3, 3);
	const Eigen::MatrixXd reference = 2.0 * dense - correction.toDense();
	const auto expression = (dense + dense) - Correction(3, 3);
	CHECK(expression.toDense().isApprox(reference));

	const auto reversed = Correction(3, 3) + Eigen::MatrixXd(dense);
	CHECK(reversed.toDense().isApprox(correction.toDense() + dense));

	const auto lowRankTemporaryDifference = Correction(3, 3) - Eigen::MatrixXd(dense);
	CHECK(lowRankTemporaryDifference.toDense().isApprox(correction.toDense() - dense));

	const auto densePlusLowRankTemporary = Eigen::MatrixXd(dense) + Correction(3, 3);
	CHECK(densePlusLowRankTemporary.toDense().isApprox(dense + correction.toDense()));
}


TEST_CASE("Low-rank diagonal preconditioner supports direct construction and staged initialization",
          "[Hoppy][LowRankMatrix]")
{
	Eigen::Matrix3d dense = Eigen::Matrix3d::Zero();
	dense.diagonal() << 2, 0, -4;
	const Hoppy::LowRankMatrix<double, 3, 3> empty;
	const auto expression = dense + empty;
	const Eigen::Vector3d rhs{4, 3, 8};
	const Eigen::Vector3d expected{2, 3, -2};

	const Hoppy::LowRankDiagonalPreconditioner<double> constructed(expression);
	CHECK(Eigen::Vector3d(constructed.solve(rhs)).isApprox(expected));

	Hoppy::LowRankDiagonalPreconditioner<double> staged;
	CHECK(&staged.analyzePattern(expression) == &staged);
	CHECK(&staged.factorize(expression) == &staged);
	CHECK(Eigen::Vector3d(staged.solve(rhs)).isApprox(expected));
}


TEST_CASE("Eigen iterative solvers consume dense and low-rank sums without sparse iteration",
          "[Hoppy][LowRankMatrix]")
{
	using Scalar = std::complex<double>;
	Eigen::MatrixXcd initial = Eigen::MatrixXcd::Zero(4, 4);
	initial.diagonal() << Scalar{5}, Scalar{6}, Scalar{7}, Scalar{8};
	const Eigen::MatrixXcd vectors = Eigen::MatrixXcd::Random(4, 2);
	const Hoppy::LowRankSelfAdjointMatrixX<Scalar> correction(Eigen::Vector2d{0.25, 0.5}, vectors);
	const auto expression = initial + correction;
	const Eigen::MatrixXcd reference = expression.toDense();
	const Eigen::VectorXcd expected = Eigen::VectorXcd::Random(4);
	const Eigen::VectorXcd rhs = reference * expected;

	using Expression = std::decay_t<decltype(expression)>;
	using Preconditioner = Hoppy::LowRankDiagonalPreconditioner<Scalar>;
	Eigen::ConjugateGradient<Expression, Eigen::Lower | Eigen::Upper, Preconditioner> solver(expression);
	solver.setTolerance(1e-12);
	solver.setMaxIterations(40);
	const Eigen::VectorXcd solution = solver.solve(rhs);

	CHECK(solver.info() == Eigen::Success);
	CHECK(solution.isApprox(expected, 1e-10));
	CHECK((reference * solution - rhs).norm() <= 1e-10 * rhs.norm());
}
