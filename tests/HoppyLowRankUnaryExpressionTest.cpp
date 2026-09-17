// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Andy Brown

#define CATCH_CONFIG_DISABLE_EXCEPTIONS
#include <SecUtility/Hoppy/LowRankMatrix.hpp>

#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <type_traits>


namespace
{
	Hoppy::LowRankMatrix<std::complex<double>, 2, 3> GeneralMatrix()
	{
		Hoppy::LowRankMatrix<std::complex<double>, 2, 3> matrix;
		Eigen::Vector2cd left1;
		Eigen::Vector3cd right1;
		left1 << std::complex<double>{1, 2}, std::complex<double>{-0.5, 1};
		right1 << std::complex<double>{2, -1}, std::complex<double>{0.25, 3},
		        std::complex<double>{-2, 0.5};
		matrix.addTerm({2, 1}, left1, right1);

		Eigen::Vector2cd left2;
		Eigen::Vector3cd right2;
		left2 << std::complex<double>{3, -1}, std::complex<double>{2, 0.25};
		right2 << std::complex<double>{-1, 2}, std::complex<double>{4, -0.5},
		        std::complex<double>{1.5, 1};
		matrix.addTerm({-1, 0.5}, left2, right2);
		return matrix;
	}
}


using General = Hoppy::LowRankMatrix<std::complex<double>, 2, 3>;
using TransposeExpression = decltype(std::declval<const General&>().transpose());
using ConjugateExpression = decltype(std::declval<const General&>().conjugate());
using AdjointExpression = decltype(std::declval<const General&>().adjoint());

static_assert(TransposeExpression::RowsAtCompileTime == 3);
static_assert(TransposeExpression::ColsAtCompileTime == 2);
static_assert(ConjugateExpression::RowsAtCompileTime == 2);
static_assert(ConjugateExpression::ColsAtCompileTime == 3);
static_assert(AdjointExpression::RowsAtCompileTime == 3);
static_assert(AdjointExpression::ColsAtCompileTime == 2);
static_assert(std::is_same_v<typename TransposeExpression::StructurePolicy, void>);
static_assert((Eigen::internal::traits<TransposeExpression>::Flags & Eigen::NestByRefBit) == 0);
static_assert((TransposeExpression::Flags & Eigen::NestByRefBit) == 0);


TEST_CASE("General low-rank unary expressions match dense transformations", "[Hoppy][LowRankMatrix]")
{
	const General matrix = GeneralMatrix();
	const Eigen::MatrixXcd dense = matrix.toDense();

	CHECK(matrix.transpose().toDense().isApprox(dense.transpose()));
	CHECK(matrix.conjugate().toDense().isApprox(dense.conjugate()));
	CHECK(matrix.adjoint().toDense().isApprox(dense.adjoint()));
	CHECK(matrix.transpose().rows() == 3);
	CHECK(matrix.transpose().cols() == 2);

	CHECK(matrix.transpose().transpose().toDense().isApprox(dense));
	CHECK(matrix.conjugate().conjugate().toDense().isApprox(dense));
	CHECK(matrix.adjoint().adjoint().toDense().isApprox(dense));
}


TEST_CASE("Unary expressions preserve canonical terms", "[Hoppy][LowRankMatrix]")
{
	const General matrix = GeneralMatrix();
	const auto transpose = matrix.transpose();
	const auto conjugate = matrix.conjugate();
	const auto adjoint = matrix.adjoint();

	CHECK(transpose.coefficientOfTerm(0) == matrix.coefficientOfTerm(0));
	CHECK(transpose.leftVectorOfTerm(0) == matrix.rightVectorOfTerm(0).conjugate());
	CHECK(transpose.rightVectorOfTerm(0) == matrix.leftVectorOfTerm(0).conjugate());
	CHECK(conjugate.coefficientOfTerm(0) == std::conj(matrix.coefficientOfTerm(0)));
	CHECK(conjugate.leftVectorOfTerm(0) == matrix.leftVectorOfTerm(0).conjugate());
	CHECK(conjugate.rightVectorOfTerm(0) == matrix.rightVectorOfTerm(0).conjugate());
	CHECK(adjoint.coefficientOfTerm(0) == std::conj(matrix.coefficientOfTerm(0)));
	CHECK(adjoint.leftVectorOfTerm(0) == matrix.rightVectorOfTerm(0));
	CHECK(adjoint.rightVectorOfTerm(0) == matrix.leftVectorOfTerm(0));
}


TEST_CASE("Unary expressions safely nest lvalues and temporaries", "[Hoppy][LowRankMatrix]")
{
	General matrix = GeneralMatrix();
	auto lazyAdjoint = matrix.adjoint();
	const Eigen::Index oldTermCount = lazyAdjoint.termCount();
	matrix.addTerm({1, -2}, Eigen::Vector2cd::Ones(), Eigen::Vector3cd::Ones());
	CHECK(lazyAdjoint.termCount() == oldTermCount + 1);
	CHECK(lazyAdjoint.toDense().isApprox(matrix.toDense().adjoint()));

	const Eigen::MatrixXcd fromTemporary = GeneralMatrix().adjoint().conjugate().toDense();
	CHECK(fromTemporary.isApprox(GeneralMatrix().toDense().adjoint().conjugate()));

	const Eigen::MatrixXcd rhs = Eigen::MatrixXcd::Random(2, 3);
	CHECK(Eigen::MatrixXcd(matrix.adjoint() * rhs).isApprox(matrix.toDense().adjoint() * rhs));
}


TEST_CASE("Unary expressions materialize without dense expansion", "[Hoppy][LowRankMatrix]")
{
	const General matrix = GeneralMatrix();
	const Hoppy::LowRankMatrix<std::complex<double>, 3, 2> adjoint(matrix.adjoint());
	CHECK(adjoint.termCount() == matrix.termCount());
	CHECK(adjoint.toDense().isApprox(matrix.toDense().adjoint()));

	const Eigen::MatrixXcd vectors = Eigen::MatrixXcd::Random(3, 2);
	Eigen::Vector2cd coefficients;
	coefficients << std::complex<double>{2, 1}, std::complex<double>{-1, 0.5};
	const Hoppy::LowRankSymmetricMatrixX<std::complex<double>> symmetric(coefficients, vectors);
	const Hoppy::LowRankSymmetricMatrixX<std::complex<double>> conjugated(symmetric.conjugate());
	CHECK(conjugated.toDense().isApprox(symmetric.toDense().conjugate()));
}


TEST_CASE("Structured unary expressions preserve their structure policies", "[Hoppy][LowRankMatrix]")
{
	using Symmetric = Hoppy::LowRankSymmetricMatrixX<std::complex<double>>;
	using SelfAdjoint = Hoppy::LowRankSelfAdjointMatrixX<std::complex<double>>;
	using SymmetricTranspose = decltype(std::declval<const Symmetric&>().transpose());
	using SymmetricAdjoint = decltype(std::declval<const Symmetric&>().adjoint());
	using SelfAdjointTranspose = decltype(std::declval<const SelfAdjoint&>().transpose());
	using SelfAdjointAdjoint = decltype(std::declval<const SelfAdjoint&>().adjoint());

	static_assert(std::is_same_v<typename SymmetricTranspose::StructurePolicy,
	                             Hoppy::Detail::SymmetricLowRankStructure>);
	static_assert(std::is_same_v<typename SymmetricAdjoint::StructurePolicy,
	                             Hoppy::Detail::SymmetricLowRankStructure>);
	static_assert(std::is_same_v<typename SelfAdjointTranspose::StructurePolicy,
	                             Hoppy::Detail::SelfAdjointLowRankStructure>);
	static_assert(std::is_same_v<typename SelfAdjointAdjoint::StructurePolicy,
	                             Hoppy::Detail::SelfAdjointLowRankStructure>);

	const Eigen::MatrixXcd vectors = Eigen::MatrixXcd::Random(4, 2);
	Eigen::Vector2cd symmetricCoefficients;
	symmetricCoefficients << std::complex<double>{2, 1}, std::complex<double>{-1, 0.5};
	const Symmetric symmetric(symmetricCoefficients, vectors);
	const SelfAdjoint selfAdjoint(Eigen::Vector2d{2, -1}, vectors);
	const auto symmetricTranspose = symmetric.transpose();
	const auto selfAdjointAdjoint = selfAdjoint.adjoint();

	CHECK(symmetricTranspose.toDense().isApprox(symmetric.toDense()));
	CHECK(symmetricTranspose.coefficientOfTerm(1) == symmetric.coefficientOfTerm(1));
	CHECK(symmetricTranspose.leftVectorOfTerm(1) == symmetric.leftVectorOfTerm(1));
	CHECK(symmetricTranspose.rightVectorOfTerm(1) == symmetric.rightVectorOfTerm(1));
	CHECK(symmetric.adjoint().toDense().isApprox(symmetric.toDense().adjoint()));
	CHECK(selfAdjointAdjoint.toDense().isApprox(selfAdjoint.toDense()));
	CHECK(selfAdjointAdjoint.coefficientOfTerm(0) == selfAdjoint.coefficientOfTerm(0));
	CHECK(selfAdjointAdjoint.leftVectorOfTerm(0) == selfAdjoint.leftVectorOfTerm(0));
	CHECK(selfAdjointAdjoint.rightVectorOfTerm(0) == selfAdjoint.rightVectorOfTerm(0));
	CHECK(selfAdjoint.transpose().toDense().isApprox(selfAdjoint.toDense().transpose()));
}
