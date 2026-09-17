// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Andy Brown

#include <SecUtility/Hoppy/LowRankMatrix.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <complex>
#include <type_traits>


namespace
{
	Hoppy::LowRankMatrixX<std::complex<double>> GeneralMatrix()
	{
		Hoppy::LowRankMatrixX<std::complex<double>> matrix(3, 2);
		matrix.addTerm({2, -1}, Eigen::Vector3cd::Random(), Eigen::Vector2cd::Random());
		matrix.addTerm({-0.5, 3}, Eigen::Vector3cd::Random(), Eigen::Vector2cd::Random());
		return matrix;
	}
}


using ComplexGeneral = Hoppy::LowRankMatrixX<std::complex<double>>;
using RightScaled = decltype(std::declval<const ComplexGeneral&>() * 2.0);
using LeftScaled = decltype(2.0 * std::declval<const ComplexGeneral&>());
using Divided = decltype(std::declval<const ComplexGeneral&>() / 2.0);
static_assert(std::is_same_v<typename RightScaled::Scalar, std::complex<double>>);
static_assert(std::is_same_v<typename LeftScaled::Scalar, std::complex<double>>);
static_assert(std::is_same_v<typename Divided::Scalar, std::complex<double>>);
static_assert(std::is_same_v<typename RightScaled::StructurePolicy, void>);
static_assert((Eigen::internal::traits<RightScaled>::Flags & Eigen::NestByRefBit) == 0);
static_assert((RightScaled::Flags & Eigen::NestByRefBit) == 0);


TEST_CASE("General low-rank scalar expressions match dense scaling", "[Hoppy][LowRankMatrix]")
{
	const ComplexGeneral matrix = GeneralMatrix();
	const Eigen::MatrixXcd dense = matrix.toDense();
	const std::complex<double> factor{2, -0.5};

	CHECK((matrix * factor).toDense().isApprox(dense * factor));
	CHECK((factor * matrix).toDense().isApprox(factor * dense));
	CHECK((matrix / factor).toDense().isApprox(dense / factor));
	CHECK((matrix * factor).coefficientOfTerm(0) == matrix.coefficientOfTerm(0) * factor);
	CHECK((factor * matrix).coefficientOfTerm(0) == factor * matrix.coefficientOfTerm(0));
	CHECK((matrix / factor).coefficientOfTerm(0) == matrix.coefficientOfTerm(0) / factor);
	CHECK((matrix * factor).leftVectorOfTerm(0) == matrix.leftVectorOfTerm(0));
	CHECK((matrix * factor).rightVectorOfTerm(0) == matrix.rightVectorOfTerm(0));
}


TEST_CASE("Scalar expressions promote real low-rank values to complex", "[Hoppy][LowRankMatrix]")
{
	Hoppy::LowRankMatrixX<double> matrix(2, 3);
	matrix.addTerm(2, Eigen::Vector2d{1, -1}, Eigen::Vector3d{0.5, 2, -3});
	const std::complex<double> factor{1, 2};
	const auto scaled = matrix * factor;

	static_assert(std::is_same_v<typename std::decay_t<decltype(scaled)>::Scalar, std::complex<double>>);
	CHECK(scaled.toDense().isApprox(matrix.toDense().cast<std::complex<double>>() * factor));
	const Eigen::Vector3cd rhs = Eigen::Vector3cd::Random();
	CHECK(Eigen::VectorXcd(scaled * rhs).isApprox(scaled.toDense() * rhs));

	const Hoppy::LowRankMatrixX<std::complex<double>> materialized(scaled);
	CHECK(materialized.toDense().isApprox(scaled.toDense()));
}


TEST_CASE("Scalar expressions reference lvalues and own temporaries and factors", "[Hoppy][LowRankMatrix]")
{
	ComplexGeneral matrix = GeneralMatrix();
	std::complex<double> factor{2, 1};
	const auto scaled = matrix * factor;
	const Eigen::Index oldTermCount = scaled.termCount();
	factor = {7, -3};
	matrix.addTerm({1, 1}, Eigen::Vector3cd::Ones(), Eigen::Vector2cd::Ones());

	CHECK(scaled.termCount() == oldTermCount + 1);
	CHECK(scaled.coefficientOfTerm(0) == matrix.coefficientOfTerm(0) * std::complex<double>{2, 1});
	const ComplexGeneral temporarySource = GeneralMatrix();
	const Eigen::MatrixXcd temporaryDense = temporarySource.toDense();
	auto makeTemporaryExpression = [temporarySource]() mutable {
		return (std::move(temporarySource) * std::complex<double>{-1, 0.5}).adjoint();
	};
	const Eigen::MatrixXcd temporaryResult = makeTemporaryExpression().toDense();
	CHECK(temporaryResult.isApprox((temporaryDense * std::complex<double>{-1, 0.5}).adjoint()));
	CHECK(((matrix * 2.0) / 4.0).toDense().isApprox(matrix.toDense() / 2.0));
}


TEST_CASE("Structural policies select the result of scalar expressions", "[Hoppy][LowRankMatrix]")
{
	using Symmetric = Hoppy::LowRankSymmetricMatrixX<std::complex<double>>;
	using SelfAdjoint = Hoppy::LowRankSelfAdjointMatrixX<std::complex<double>>;
	const Eigen::MatrixXcd vectors = Eigen::MatrixXcd::Random(3, 2);
	Eigen::Vector2cd symmetricCoefficients;
	symmetricCoefficients << std::complex<double>{2, 1}, std::complex<double>{-1, 0.5};
	const Symmetric symmetric(symmetricCoefficients, vectors);
	const SelfAdjoint selfAdjoint(Eigen::Vector2d{2, -1}, vectors);

	const auto symmetricComplexScale = symmetric * std::complex<double>{1, 2};
	const auto selfAdjointRealScale = selfAdjoint / 2.0;
	const auto selfAdjointComplexScale = std::complex<double>{1, 2} * selfAdjoint;
	const auto selfAdjointComplexQuotient = selfAdjoint / std::complex<double>{1, 2};
	static_assert(std::is_same_v<typename std::decay_t<decltype(symmetricComplexScale)>::StructurePolicy,
	                             Hoppy::Detail::SymmetricLowRankStructure>);
	static_assert(std::is_same_v<typename std::decay_t<decltype(selfAdjointRealScale)>::StructurePolicy,
	                             Hoppy::Detail::SelfAdjointLowRankStructure>);
	static_assert(std::is_same_v<typename std::decay_t<decltype(selfAdjointComplexScale)>::StructurePolicy, void>);
	static_assert(std::is_same_v<typename std::decay_t<decltype(selfAdjointComplexQuotient)>::StructurePolicy, void>);

	CHECK(symmetricComplexScale.toDense().isApprox(symmetric.toDense() * std::complex<double>{1, 2}));
	CHECK(selfAdjointRealScale.toDense().isApprox(selfAdjoint.toDense() / 2.0));
	CHECK(selfAdjointComplexScale.toDense().isApprox(std::complex<double>{1, 2} * selfAdjoint.toDense()));
	CHECK(selfAdjointComplexQuotient.toDense().isApprox(selfAdjoint.toDense() / std::complex<double>{1, 2}));
	const SelfAdjoint preserved(selfAdjointRealScale);
	CHECK(preserved.toDense().isApprox(selfAdjointRealScale.toDense()));
	const Hoppy::LowRankMatrixX<std::complex<double>> generic(selfAdjointComplexScale);
	CHECK(generic.toDense().isApprox(selfAdjointComplexScale.toDense()));

	using RealSelfAdjoint = Hoppy::LowRankSelfAdjointMatrixX<double>;
	using RealComplexScale = decltype(std::declval<const RealSelfAdjoint&>() * std::complex<double>{1, 2});
	static_assert(std::is_same_v<typename RealComplexScale::StructurePolicy,
	                             Hoppy::Detail::SymmetricLowRankStructure>);
}


TEST_CASE("Scalar division follows the scalar type's division-by-zero convention", "[Hoppy][LowRankMatrix]")
{
	Hoppy::LowRankMatrixX<double> matrix(1, 1);
	matrix.addTerm(2, Eigen::Vector<double, 1>::Ones(), Eigen::Vector<double, 1>::Ones());
	const auto divided = matrix / 0.0;
	CHECK(std::isinf(divided.coefficientOfTerm(0)));
}
