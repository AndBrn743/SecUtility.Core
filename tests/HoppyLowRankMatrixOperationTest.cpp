// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Andy Brown

#include <SecUtility/Hoppy/LowRankMatrix.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <type_traits>


namespace
{
	template <typename Scalar>
	Hoppy::LowRankMatrixX<Scalar> GeneralMatrix()
	{
		Eigen::MatrixX<Scalar> left(3, 3);
		Eigen::MatrixX<Scalar> right(4, 3);
		left << Scalar{1}, Scalar{2}, Scalar{-1}, Scalar{0.5}, Scalar{-2}, Scalar{3}, Scalar{4}, Scalar{1},
		        Scalar{0.25};
		right << Scalar{-1}, Scalar{2}, Scalar{0.5}, Scalar{3}, Scalar{1}, Scalar{-2}, Scalar{0.25},
		        Scalar{-4}, Scalar{2}, Scalar{1.5}, Scalar{0.75}, Scalar{-3};
		Eigen::VectorX<Scalar> coefficients(3);
		coefficients << Scalar{2}, Scalar{-0.5}, Scalar{3};
		return {coefficients, left, right};
	}

	template <typename Matrix>
	Eigen::MatrixX<typename Matrix::Scalar> DenseReference(const Matrix& matrix)
	{
		using Scalar = typename Matrix::Scalar;
		return matrix.leftVectors() * matrix.coefficients().template cast<Scalar>().asDiagonal()
		       * matrix.rightVectors().adjoint();
	}
}


using FixedProduct = decltype(std::declval<const Hoppy::LowRankMatrix<double, 2, 3>&>()
                              * std::declval<const Eigen::Vector3d&>());
static_assert(FixedProduct::RowsAtCompileTime == 2);
static_assert(FixedProduct::ColsAtCompileTime == 1);
static_assert(std::is_same_v<decltype(std::declval<const Hoppy::LowRankMatrix<double, 2, 3>&>().row(0)),
                             Eigen::Matrix<double, 1, 3>>);
static_assert(std::is_same_v<decltype(std::declval<const Hoppy::LowRankMatrix<double, 2, 3>&>().col(0)),
                             Eigen::Vector2d>);
static_assert(std::is_same_v<decltype(std::declval<const Hoppy::LowRankMatrix<double, 2, 3>&>().diagonal()),
                             Eigen::Vector2d>);


TEMPLATE_TEST_CASE("Low-rank products match dense products", "[Hoppy][LowRankMatrix]", double,
                   (std::complex<double>))
{
	const auto matrix = GeneralMatrix<TestType>();
	const Eigen::MatrixX<TestType> dense = DenseReference(matrix);
	const Eigen::VectorX<TestType> vector = Eigen::VectorX<TestType>::Random(4);
	const Eigen::MatrixX<TestType> block = Eigen::MatrixX<TestType>::Random(4, 3);

	const Eigen::VectorX<TestType> vectorResult = matrix * vector;
	const Eigen::MatrixX<TestType> blockResult = matrix * block;
	CHECK(vectorResult.isApprox(dense * vector));
	CHECK(blockResult.isApprox(dense * block));
	CHECK(Eigen::MatrixX<TestType>(matrix * block.leftCols(2)).isApprox(dense * block.leftCols(2)));

	Eigen::MatrixX<TestType> accumulated = Eigen::MatrixX<TestType>::Random(3, 3);
	const Eigen::MatrixX<TestType> initial = accumulated;
	using Matrix = std::decay_t<decltype(matrix)>;
	using Rhs = Eigen::MatrixX<TestType>;
	Eigen::internal::generic_product_impl<Hoppy::LowRankMatrixBase<Matrix>, Rhs>::scaleAndAddTo(
	        accumulated, matrix, block, TestType{2});
	CHECK(accumulated.isApprox(initial + TestType{2} * dense * block));

	const Eigen::MatrixX<TestType> emptyResult = matrix * block.leftCols(0);
	CHECK(emptyResult.rows() == 3);
	CHECK(emptyResult.cols() == 0);

	const Hoppy::LowRankMatrixX<TestType> rankZero(3, 4);
	CHECK(Eigen::MatrixX<TestType>(rankZero * block).isZero());
}


TEST_CASE("Low-rank products support compatible mixed scalars and destination aliasing", "[Hoppy][LowRankMatrix]")
{
	const auto matrix = GeneralMatrix<std::complex<double>>();
	const Eigen::MatrixXcd dense = DenseReference(matrix);
	const Eigen::Vector4d realVector{1, -2, 0.5, 3};
	const Eigen::VectorXcd mixedResult = matrix * realVector;
	CHECK(mixedResult.isApprox(dense * realVector));

	Hoppy::LowRankMatrixX<double> square(3, 3);
	square.addTerms(Eigen::Vector2d{2, -1}, Eigen::MatrixXd::Random(3, 2), Eigen::MatrixXd::Random(3, 2));
	const Eigen::MatrixXd squareDense = DenseReference(square);
	Eigen::Vector3d aliased{1, 2, 3};
	const Eigen::Vector3d expected = squareDense * aliased;
	aliased = square * aliased;
	CHECK(aliased.isApprox(expected));
}


TEST_CASE("Fixed low-rank products preserve compile-time dimensions", "[Hoppy][LowRankMatrix]")
{
	Hoppy::LowRankMatrix<double, 2, 3> matrix;
	matrix.addTerm(2, Eigen::Vector2d{1, -1}, Eigen::Vector3d{2, 0.5, -3});
	const Eigen::Vector3d rhs{1, 2, -1};
	const Eigen::Vector2d result = matrix * rhs;
	CHECK(result.isApprox(matrix.toDense() * rhs));
}


TEMPLATE_TEST_CASE("Low-rank inspection matches dense references", "[Hoppy][LowRankMatrix]", float, double,
                   (std::complex<float>), (std::complex<double>))
{
	const auto matrix = GeneralMatrix<TestType>();
	const Eigen::MatrixX<TestType> dense = DenseReference(matrix);

	CHECK(matrix.toDense().isApprox(dense));
	CHECK(matrix.row(1).isApprox(dense.row(1)));
	CHECK(matrix.col(2).isApprox(dense.col(2)));
	CHECK(matrix.diagonal().isApprox(dense.diagonal()));
	CHECK(matrix.squaredNorm() == Catch::Approx(dense.squaredNorm()).epsilon(1e-5));
	CHECK(matrix.norm() == Catch::Approx(dense.norm()).epsilon(1e-5));

	const Hoppy::LowRankMatrixX<TestType> empty(3, 4);
	CHECK(empty.toDense().isZero());
	CHECK(empty.row(2).isZero());
	CHECK(empty.col(3).isZero());
	CHECK(empty.diagonal().isZero());
	CHECK(empty.squaredNorm() == typename Hoppy::LowRankMatrixX<TestType>::RealScalar{});
	CHECK(empty.norm() == typename Hoppy::LowRankMatrixX<TestType>::RealScalar{});
}


TEST_CASE("Low-rank norms include cross terms and cancellation", "[Hoppy][LowRankMatrix]")
{
	const Eigen::Vector3cd left{std::complex<double>{1, 2}, {-0.5, 1}, {3, -2}};
	const Eigen::Vector2cd right{std::complex<double>{2, -1}, std::complex<double>{0.25, 3}};
	Hoppy::LowRankMatrixX<std::complex<double>> matrix(3, 2);
	matrix.addTerm({2, -1}, left, right);
	matrix.addTerm({-2, 1}, left, right);

	CHECK(matrix.termCount() == 2);
	CHECK(matrix.toDense().isZero(1e-12));
	CHECK(matrix.squaredNorm() == Catch::Approx(0).margin(1e-12));
	CHECK(matrix.norm() == Catch::Approx(0).margin(1e-12));
}


TEST_CASE("Structured low-rank products and norms honor their policies", "[Hoppy][LowRankMatrix]")
{
	const Eigen::MatrixXcd vectors = Eigen::MatrixXcd::Random(4, 3);
	Eigen::Vector3cd symmetricCoefficients;
	symmetricCoefficients << std::complex<double>{2, 1}, std::complex<double>{-1, 0.5},
	        std::complex<double>{3, -2};
	const Hoppy::LowRankSymmetricMatrixX<std::complex<double>> symmetric(symmetricCoefficients, vectors);
	const Eigen::MatrixXcd symmetricDense = DenseReference(symmetric);

	const Eigen::Vector3d selfAdjointCoefficients{2, -1, 3};
	const Hoppy::LowRankSelfAdjointMatrixX<std::complex<double>> selfAdjoint(selfAdjointCoefficients, vectors);
	const Eigen::MatrixXcd selfAdjointDense = DenseReference(selfAdjoint);
	const Eigen::MatrixXcd rhs = Eigen::MatrixXcd::Random(4, 2);

	CHECK(Eigen::MatrixXcd(symmetric * rhs).isApprox(symmetricDense * rhs));
	CHECK(Eigen::MatrixXcd(selfAdjoint * rhs).isApprox(selfAdjointDense * rhs));
	CHECK(symmetric.diagonal().isApprox(symmetricDense.diagonal()));
	CHECK(selfAdjoint.diagonal().isApprox(selfAdjointDense.diagonal()));
	CHECK(symmetric.squaredNorm() == Catch::Approx(symmetricDense.squaredNorm()).epsilon(1e-12));
	CHECK(selfAdjoint.squaredNorm() == Catch::Approx(selfAdjointDense.squaredNorm()).epsilon(1e-12));
}
