// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Andy Brown

#include <SecUtility/Hoppy/LowRankMatrix.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <type_traits>
#include <utility>


namespace
{
	template <typename Matrix, typename Coefficients, typename Vectors, typename = void>
	struct CanAddTerms : std::false_type
	{
	};

	template <typename Matrix, typename Coefficients, typename Vectors>
	struct CanAddTerms<Matrix,
	                   Coefficients,
	                   Vectors,
	                   std::void_t<decltype(std::declval<Matrix&>().addTerms(
	                           std::declval<const Coefficients&>(), std::declval<const Vectors&>()))>> : std::true_type
	{
	};

	template <typename Matrix>
	Eigen::MatrixX<typename Matrix::Scalar> DenseExpansion(const Matrix& matrix)
	{
		using Scalar = typename Matrix::Scalar;
		return matrix.leftVectors() * matrix.coefficients().template cast<Scalar>().asDiagonal()
		       * matrix.rightVectors().adjoint();
	}
}


using ComplexSelfAdjoint = Hoppy::LowRankSelfAdjointMatrixX<std::complex<double>>;
using ComplexSymmetric = Hoppy::LowRankSymmetricMatrixX<std::complex<double>>;
using FixedSelfAdjoint = Hoppy::LowRankSelfAdjointMatrix<double, 3>;
using FixedGeneralFromSelfAdjoint = decltype(std::declval<const FixedSelfAdjoint&>().toGeneral());

static_assert(std::is_base_of_v<Hoppy::LowRankMatrixBase<ComplexSelfAdjoint>, ComplexSelfAdjoint>);
static_assert(std::is_base_of_v<Hoppy::BulkLowRankMatrixBase<ComplexSelfAdjoint>, ComplexSelfAdjoint>);
static_assert(std::is_same_v<typename ComplexSelfAdjoint::Scalar, std::complex<double>>);
static_assert(std::is_same_v<typename ComplexSelfAdjoint::RealScalar, double>);
static_assert(std::is_same_v<typename ComplexSelfAdjoint::CoefficientVector, Eigen::VectorXd>);
static_assert(std::is_same_v<typename ComplexSymmetric::CoefficientVector, Eigen::VectorXcd>);
static_assert(std::is_same_v<typename ComplexSelfAdjoint::StructurePolicy,
                             Hoppy::Detail::SelfAdjointLowRankStructure>);
static_assert(std::is_same_v<typename ComplexSymmetric::StructurePolicy,
                             Hoppy::Detail::SymmetricLowRankStructure>);
static_assert(std::is_same_v<typename FixedSelfAdjoint::StructurePolicy,
                             Hoppy::Detail::SymmetricLowRankStructure>);
static_assert((Eigen::internal::traits<ComplexSelfAdjoint>::Flags & Eigen::NestByRefBit) != 0);
static_assert((ComplexSelfAdjoint::Flags & Eigen::NestByRefBit) != 0);
static_assert(!std::is_constructible_v<ComplexSelfAdjoint, std::complex<double>, Eigen::VectorXcd>);
static_assert(!std::is_constructible_v<ComplexSelfAdjoint, Eigen::VectorXcd, Eigen::MatrixXcd>);
static_assert(!CanAddTerms<ComplexSelfAdjoint, Eigen::VectorXcd, Eigen::MatrixXcd>::value);
static_assert(CanAddTerms<ComplexSelfAdjoint, Eigen::VectorXd, Eigen::MatrixXcd>::value);
static_assert(CanAddTerms<ComplexSymmetric, Eigen::VectorXcd, Eigen::MatrixXcd>::value);
static_assert(std::is_same_v<Hoppy::LowRankSymmetricMatrix<float, 3>,
                             Hoppy::LowRankSelfAdjointMatrix<float, 3>>);
static_assert(std::is_same_v<Hoppy::LowRankSymmetricMatrix<double, Eigen::Dynamic>,
                             Hoppy::LowRankSelfAdjointMatrix<double, Eigen::Dynamic>>);
static_assert(!std::is_same_v<ComplexSymmetric, ComplexSelfAdjoint>);
static_assert(std::is_same_v<FixedGeneralFromSelfAdjoint, Hoppy::LowRankMatrix<double, 3, 3>>);


TEMPLATE_TEST_CASE("Self-adjoint low-rank matrices preserve their structural expansion", "[Hoppy][LowRankMatrix]",
                   float, double, (std::complex<float>), (std::complex<double>))
{
	using Matrix = Hoppy::LowRankSelfAdjointMatrixX<TestType>;
	using RealScalar = typename Matrix::RealScalar;

	const Eigen::MatrixX<TestType> vectors = Eigen::MatrixX<TestType>::Random(4, 3);
	Eigen::VectorX<RealScalar> coefficients(3);
	coefficients << RealScalar{2}, RealScalar{-3}, RealScalar{0.5};
	const Matrix matrix(coefficients, vectors);

	CHECK(matrix.rows() == 4);
	CHECK(matrix.cols() == 4);
	REQUIRE(matrix.termCount() == 3);
	CHECK(matrix.coefficients() == coefficients);
	CHECK(matrix.leftVectors() == vectors);
	CHECK(matrix.rightVectors() == vectors);
	CHECK(matrix.leftVectorOfTerm(1) == vectors.col(1));
	CHECK(matrix.rightVectorOfTerm(1) == vectors.col(1));
	CHECK(matrix.coefficientOfTerm(1) == RealScalar{-3});

	const auto dense = DenseExpansion(matrix);
	CHECK(dense.isApprox(dense.adjoint()));

	const auto term = matrix.term(1);
	CHECK(term.leftVector == vectors.col(1));
	CHECK(term.coefficient == RealScalar{-3});
	CHECK(term.rightVector == vectors.col(1));
}


TEST_CASE("Complex symmetric low-rank matrices use transpose structure", "[Hoppy][LowRankMatrix]")
{
	const Eigen::MatrixXcd vectors = Eigen::MatrixXcd::Random(4, 2);
	Eigen::Vector2cd coefficients;
	coefficients << std::complex<double>{2.0, 1.0}, std::complex<double>{-3.0, 0.5};
	const ComplexSymmetric matrix(coefficients, vectors);

	CHECK(matrix.coefficients() == coefficients);
	CHECK(matrix.leftVectors() == vectors);
	CHECK(matrix.rightVectors() == vectors.conjugate());
	CHECK(matrix.rightVectorOfTerm(1) == vectors.col(1).conjugate());

	const Eigen::MatrixXcd dense = DenseExpansion(matrix);
	CHECK(dense.isApprox(dense.transpose()));
	CHECK_FALSE(dense.isApprox(dense.adjoint()));

	const auto general = matrix.toGeneral();
	CHECK(general.leftVectors() == vectors);
	CHECK(general.rightVectors() == vectors.conjugate());
	const Eigen::MatrixXcd generalDense =
	        general.leftVectors() * general.coefficients().asDiagonal() * general.rightVectors().adjoint();
	CHECK(generalDense.isApprox(dense));
}


TEST_CASE("Single-factor low-rank matrices support fixed, dynamic, and rank-zero dimensions",
          "[Hoppy][LowRankMatrix]")
{
	const FixedSelfAdjoint fixed;
	CHECK(fixed.rows() == 3);
	CHECK(fixed.cols() == 3);
	CHECK(fixed.termCount() == 0);

	const Hoppy::LowRankSelfAdjointMatrixX<double> dynamic(5);
	CHECK(dynamic.rows() == 5);
	CHECK(dynamic.cols() == 5);
	CHECK(dynamic.termCount() == 0);
	CHECK(dynamic.leftVectors().rows() == 5);
	CHECK(dynamic.leftVectors().cols() == 0);

	const Hoppy::LowRankSelfAdjointMatrixX<double> inferred(2.0, Eigen::Vector4d::Ones());
	CHECK(inferred.rows() == 4);
	CHECK(inferred.cols() == 4);
	CHECK(inferred.termCount() == 1);

	const Hoppy::LowRankSelfAdjointMatrixX<double> filtered(Eigen::Vector3d::Zero(),
	                                                     Eigen::MatrixXd::Random(4, 3));
	CHECK(filtered.rows() == 4);
	CHECK(filtered.cols() == 4);
	CHECK(filtered.termCount() == 0);
}


TEST_CASE("Bulk-access structured matrices materialize through factor blocks", "[Hoppy][LowRankMatrix]")
{
	FixedSelfAdjoint fixed;
	fixed.addTerm(2.0, Eigen::Vector3d{1.0, -2.0, 0.5});
	fixed.addTerm(-0.25, Eigen::Vector3d{-1.0, 3.0, 2.0});
	const Hoppy::LowRankSelfAdjointMatrixX<double> dynamic(fixed);

	CHECK(dynamic.rows() == fixed.rows());
	CHECK(dynamic.cols() == fixed.cols());
	CHECK(dynamic.termCount() == fixed.termCount());
	CHECK(dynamic.coefficients() == fixed.coefficients());
	CHECK(dynamic.leftVectors() == fixed.leftVectors());
	CHECK(dynamic.rightVectors() == fixed.rightVectors());
}


TEST_CASE("Self-adjoint low-rank insertion filters exact zeros and is alias safe", "[Hoppy][LowRankMatrix]")
{
	ComplexSelfAdjoint matrix(3);
	const Eigen::MatrixXcd vectors = Eigen::MatrixXcd::Random(3, 4);
	const Eigen::Vector4d coefficients{0, 2, 0, -3};
	matrix.addTerms(coefficients, vectors);

	REQUIRE(matrix.termCount() == 2);
	CHECK(matrix.coefficients() == Eigen::Vector2d{2, -3});
	CHECK(matrix.leftVectorOfTerm(0) == vectors.col(1));
	CHECK(matrix.leftVectorOfTerm(1) == vectors.col(3));

	const Eigen::VectorXd oldCoefficients = matrix.coefficients();
	const Eigen::MatrixXcd oldVectors = matrix.leftVectors();
	matrix.addTerms(matrix.coefficients(), matrix.leftVectors());
	REQUIRE(matrix.termCount() == 4);
	CHECK(matrix.coefficients().head(2) == oldCoefficients);
	CHECK(matrix.coefficients().tail(2) == oldCoefficients);
	CHECK(matrix.leftVectors().leftCols(2) == oldVectors);
	CHECK(matrix.leftVectors().rightCols(2) == oldVectors);

	const auto oldTermCount = matrix.termCount();
	matrix.addTerm(0.0, Eigen::VectorXcd::Zero(17));
	CHECK(matrix.termCount() == oldTermCount);
	matrix.addTerms(Eigen::Vector3d::Zero(), Eigen::MatrixXcd::Random(3, 3));
	CHECK(matrix.termCount() == oldTermCount);
}


TEST_CASE("Self-adjoint low-rank matrices convert to general factor form without dense expansion",
          "[Hoppy][LowRankMatrix]")
{
	const Eigen::MatrixXcd vectors = Eigen::MatrixXcd::Random(3, 2);
	const Eigen::Vector2d coefficients{2, -4};
	const ComplexSelfAdjoint selfAdjoint(coefficients, vectors);

	const auto general = selfAdjoint.toGeneral();
	static_assert(std::is_same_v<std::decay_t<decltype(general)>,
	                             Hoppy::LowRankMatrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>>);
	CHECK(general.rows() == 3);
	CHECK(general.cols() == 3);
	CHECK(general.termCount() == 2);
	CHECK(general.coefficients() == coefficients.cast<std::complex<double>>());
	CHECK(general.leftVectors() == vectors);
	CHECK(general.rightVectors() == vectors);

	const Eigen::MatrixXcd generalDense =
	        general.leftVectors() * general.coefficients().asDiagonal() * general.rightVectors().adjoint();
	CHECK(generalDense.isApprox(DenseExpansion(selfAdjoint)));

	const ComplexSelfAdjoint empty(3);
	const auto emptyGeneral = empty.toGeneral();
	CHECK(emptyGeneral.rows() == 3);
	CHECK(emptyGeneral.cols() == 3);
	CHECK(emptyGeneral.termCount() == 0);
}


TEST_CASE("Single-factor low-rank values support reserve, clear, copy, and move", "[Hoppy][LowRankMatrix]")
{
	Hoppy::LowRankSelfAdjointMatrixX<double> matrix(3);
	matrix.reserve(8);
	CHECK(matrix.capacity() >= 8);
	matrix.addTerm(2, Eigen::Vector3d{1, 2, 3});

	auto copy = matrix;
	CHECK(copy.coefficients() == matrix.coefficients());
	CHECK(copy.leftVectors() == matrix.leftVectors());

	auto moved = std::move(copy);
	CHECK(moved.rows() == 3);
	CHECK(moved.termCount() == 1);
	Hoppy::LowRankSelfAdjointMatrixX<double> moveAssigned;
	moveAssigned = std::move(moved);
	CHECK(moveAssigned.rows() == 3);
	CHECK(moveAssigned.termCount() == 1);

	matrix.clear();
	CHECK(matrix.rows() == 3);
	CHECK(matrix.cols() == 3);
	CHECK(matrix.termCount() == 0);
	CHECK(matrix.capacity() >= 8);
}
