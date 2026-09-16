// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Andy Brown

#include "HoppyEigenAssert.hpp"  // Must precede every Eigen header.

#include <SecUtility/Hoppy/LowRankMatrix.hpp>

#include <catch2/catch_test_macros.hpp>

#include <complex>


namespace
{
	using DynamicMatrix = Hoppy::LowRankMatrixX<double>;
	using FixedMatrix = Hoppy::LowRankMatrix<double, 2, 3>;
	using DynamicRowsMatrix = Hoppy::LowRankMatrix<double, Eigen::Dynamic, 3>;
	using DynamicSelfAdjoint = Hoppy::LowRankSelfAdjointMatrixX<double>;
	using FixedSelfAdjoint = Hoppy::LowRankSelfAdjointMatrix<double, 3>;
	using DynamicComplexSymmetric = Hoppy::LowRankSymmetricMatrixX<std::complex<double>>;
}


TEST_CASE("Single-factor low-rank dimensions follow Eigen assertion contracts", "[Hoppy][LowRankMatrix][assert]")
{
	CHECK_THROWS_AS((FixedSelfAdjoint{4}), Hoppy::Test::EigenAssertionFailure);
	CHECK_THROWS_AS((DynamicSelfAdjoint{-1}), Hoppy::Test::EigenAssertionFailure);
	CHECK_THROWS_AS((DynamicComplexSymmetric{-1}), Hoppy::Test::EigenAssertionFailure);
}


TEST_CASE("Low-rank matrix dimensions follow Eigen assertion contracts", "[Hoppy][LowRankMatrix][assert]")
{
	CHECK_THROWS_AS((FixedMatrix{3, 3}), Hoppy::Test::EigenAssertionFailure);
	CHECK_THROWS_AS((DynamicRowsMatrix{4, 2}), Hoppy::Test::EigenAssertionFailure);
	CHECK_THROWS_AS((DynamicMatrix{-1, 2}), Hoppy::Test::EigenAssertionFailure);
	CHECK_THROWS_AS((DynamicMatrix{2, -1}), Hoppy::Test::EigenAssertionFailure);
}


TEST_CASE("Low-rank term access follows Eigen assertion contracts", "[Hoppy][LowRankMatrix][assert]")
{
	DynamicMatrix matrix(2, 3);
	matrix.addTerm(2, Eigen::Vector2d{1, 2}, Eigen::Vector3d{3, 4, 5});

	CHECK_THROWS_AS(matrix.coefficientOfTerm(-1), Hoppy::Test::EigenAssertionFailure);
	CHECK_THROWS_AS(matrix.leftVectorOfTerm(1), Hoppy::Test::EigenAssertionFailure);
	CHECK_THROWS_AS(matrix.rightVectorOfTerm(1), Hoppy::Test::EigenAssertionFailure);
	CHECK_THROWS_AS(matrix.term(1), Hoppy::Test::EigenAssertionFailure);
}


TEST_CASE("Low-rank row and column access follows Eigen assertion contracts", "[Hoppy][LowRankMatrix][assert]")
{
	DynamicMatrix matrix(2, 3);
	CHECK_THROWS_AS(matrix.row(-1), Hoppy::Test::EigenAssertionFailure);
	CHECK_THROWS_AS(matrix.row(2), Hoppy::Test::EigenAssertionFailure);
	CHECK_THROWS_AS(matrix.col(-1), Hoppy::Test::EigenAssertionFailure);
	CHECK_THROWS_AS(matrix.col(3), Hoppy::Test::EigenAssertionFailure);
}


TEST_CASE("Low-rank insertion assertions precede mutation", "[Hoppy][LowRankMatrix][assert]")
{
	DynamicMatrix matrix(2, 3);
	matrix.addTerm(2, Eigen::Vector2d{1, 2}, Eigen::Vector3d{3, 4, 5});
	const Eigen::VectorXd oldCoefficients = matrix.coefficients();
	const Eigen::MatrixXd oldLeft = matrix.leftVectors();
	const Eigen::MatrixXd oldRight = matrix.rightVectors();

	CHECK_THROWS_AS(matrix.addTerm(3, Eigen::Vector3d::Ones(), Eigen::Vector3d::Ones()),
	                Hoppy::Test::EigenAssertionFailure);
	CHECK_THROWS_AS(matrix.addTerm(3, Eigen::Vector2d::Ones(), Eigen::Vector2d::Ones()),
	                Hoppy::Test::EigenAssertionFailure);
	CHECK_THROWS_AS(matrix.addTerm(3, Eigen::Matrix2d::Ones(), Eigen::Vector3d::Ones()),
	                Hoppy::Test::EigenAssertionFailure);
	CHECK_THROWS_AS(matrix.addTerms(Eigen::Matrix2d::Ones(), Eigen::MatrixXd::Ones(2, 2),
	                               Eigen::MatrixXd::Ones(3, 2)),
	                Hoppy::Test::EigenAssertionFailure);
	CHECK_THROWS_AS(matrix.addTerms(Eigen::Vector2d::Ones(), Eigen::MatrixXd::Ones(2, 1),
	                               Eigen::MatrixXd::Ones(3, 2)),
	                Hoppy::Test::EigenAssertionFailure);
	CHECK_THROWS_AS(matrix.addTerms(Eigen::Vector2d::Ones(), Eigen::MatrixXd::Ones(3, 2),
	                               Eigen::MatrixXd::Ones(3, 2)),
	                Hoppy::Test::EigenAssertionFailure);

	CHECK(matrix.termCount() == 1);
	CHECK(matrix.coefficients() == oldCoefficients);
	CHECK(matrix.leftVectors() == oldLeft);
	CHECK(matrix.rightVectors() == oldRight);
}


TEST_CASE("Low-rank reserve follows Eigen assertion contracts", "[Hoppy][LowRankMatrix][assert]")
{
	DynamicMatrix matrix(2, 3);
	CHECK_THROWS_AS(matrix.reserve(-1), Hoppy::Test::EigenAssertionFailure);
	CHECK(matrix.termCount() == 0);
}


TEST_CASE("Self-adjoint low-rank insertion assertions precede mutation", "[Hoppy][LowRankMatrix][assert]")
{
	DynamicSelfAdjoint matrix(3);
	matrix.addTerm(2, Eigen::Vector3d{1, 2, 3});
	const Eigen::VectorXd oldCoefficients = matrix.coefficients();
	const Eigen::MatrixXd oldVectors = matrix.leftVectors();

	CHECK_THROWS_AS(matrix.addTerm(3, Eigen::Vector2d::Ones()), Hoppy::Test::EigenAssertionFailure);
	CHECK_THROWS_AS(matrix.addTerm(3, Eigen::Matrix2d::Ones()), Hoppy::Test::EigenAssertionFailure);
	CHECK_THROWS_AS(matrix.addTerms(Eigen::Matrix2d::Ones(), Eigen::MatrixXd::Ones(3, 2)),
	                Hoppy::Test::EigenAssertionFailure);
	CHECK_THROWS_AS(matrix.addTerms(Eigen::Vector2d::Ones(), Eigen::MatrixXd::Ones(3, 1)),
	                Hoppy::Test::EigenAssertionFailure);
	CHECK_THROWS_AS(matrix.addTerms(Eigen::Vector2d::Ones(), Eigen::MatrixXd::Ones(2, 2)),
	                Hoppy::Test::EigenAssertionFailure);
	CHECK_THROWS_AS(matrix.reserve(-1), Hoppy::Test::EigenAssertionFailure);

	CHECK(matrix.termCount() == 1);
	CHECK(matrix.coefficients() == oldCoefficients);
	CHECK(matrix.leftVectors() == oldVectors);
}
