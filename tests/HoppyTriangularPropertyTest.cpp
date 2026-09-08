// SPDX-License-Identifier: MIT
// Triangular-compressed specification: decisions D1-D20; sections 14-20.

#include "HoppyTestSupport.hpp"
#include <SecUtility/Hoppy/TriangularCompressedMatrix.hpp>
#include <Eigen/Core>
#include <complex>
#include <random>

namespace
{
	template <typename Matrix>
	void checkFamily(const Eigen::Index dimension, std::mt19937& generator)
	{
		std::uniform_real_distribution<double> distribution(-2.0, 2.0);

		Eigen::MatrixXd denseSource(dimension, dimension);
		for (Eigen::Index index = 0; index < denseSource.size(); ++index) denseSource(index) = distribution(generator);

		const Matrix matrix = Matrix::FromModifiedDense(denseSource);
		const Eigen::MatrixXd dense = matrix.toDense();
		REQUIRE(matrix.dimension() == dimension);
		REQUIRE(matrix.storedSize() == dimension * (dimension + 1) / 2);
		Hoppy::Test::requireApprox(Matrix::FromUncheckedDense(dense).toDense(), dense);
		Hoppy::Test::requireApprox(matrix.transpose().toDense(), dense.transpose());
		Hoppy::Test::requireApprox((matrix + matrix).toDense(), dense + dense);
		Hoppy::Test::requireApprox((matrix - matrix).toDense(), Eigen::MatrixXd::Zero(dimension, dimension));
		Hoppy::Test::requireApprox((matrix * 2.0).toDense(), dense * 2.0);
		Hoppy::Test::requireApprox((matrix * matrix).toDense(), dense * dense);
		REQUIRE(matrix.sum() == Catch::Approx(dense.sum()).margin(1e-14));
		REQUIRE(matrix.squaredNorm() == Catch::Approx(dense.squaredNorm()).margin(1e-14));
	}

	template <Hoppy::TrianglePacking Packing>
	void checkPacking(std::mt19937& generator)
	{
		for (Eigen::Index dimension = 0; dimension <= 6; ++dimension)
		{
			CAPTURE(dimension, Packing);
			checkFamily<Hoppy::UpperTriangularMatrix<double, Eigen::Dynamic, Packing>>(dimension, generator);
			checkFamily<Hoppy::LowerTriangularMatrix<double, Eigen::Dynamic, Packing>>(dimension, generator);
			checkFamily<Hoppy::SymmetricMatrix<double, Eigen::Dynamic, Packing>>(dimension, generator);
			checkFamily<Hoppy::AntiSymmetricMatrix<double, Eigen::Dynamic, Packing>>(dimension, generator);
		}
	}

	template <typename Matrix>
	void checkComplexFamily(const Eigen::Index dimension, std::mt19937& generator)
	{
		using Complex = std::complex<double>;
		std::uniform_real_distribution<double> distribution(-2.0, 2.0);
		Eigen::MatrixXcd dense(dimension, dimension);
		for (Eigen::Index index = 0; index < dense.size(); ++index)
			dense(index) = Complex(distribution(generator), distribution(generator));
		const Matrix matrix = Matrix::FromModifiedDense(dense);
		const Eigen::MatrixXcd oracle = matrix.toDense();
		Hoppy::Test::requireApprox(Matrix::FromUncheckedDense(oracle).toDense(), oracle);
		Hoppy::Test::requireApprox(matrix.adjoint().toDense(), oracle.adjoint());
		Hoppy::Test::requireApprox((matrix + matrix).toDense(), oracle + oracle);
		Hoppy::Test::requireApprox((matrix * matrix).toDense(), oracle * oracle);
		REQUIRE(matrix.squaredNorm() == Catch::Approx(oracle.squaredNorm()));
	}
}

TEST_CASE("deterministic generated matrices satisfy round-trip algebra properties")
{
	std::mt19937 generator(0x5A19B2DU);
	checkPacking<Hoppy::TrianglePacking::Lower>(generator);
	checkPacking<Hoppy::TrianglePacking::Upper>(generator);
}

TEST_CASE("complex generated matrices preserve symmetric and adjoint families")
{
	std::mt19937 generator(0xC01A53EDU);
	for (Eigen::Index dimension = 0; dimension <= 6; ++dimension)
	{
		CAPTURE(dimension);
		checkComplexFamily<Hoppy::SymmetricMatrixXcd>(dimension, generator);
		checkComplexFamily<Hoppy::AntiSymmetricMatrixXcd>(dimension, generator);
		checkComplexFamily<Hoppy::HermitianMatrixXcd>(dimension, generator);
		checkComplexFamily<Hoppy::AntiHermitianMatrixXcd>(dimension, generator);
	}
}

TEST_CASE("conditioned representatives satisfy inverse properties")
{
	std::mt19937 generator(0x187A31E5U);
	for (Eigen::Index dimension = 1; dimension <= 6; ++dimension)
	{
		Eigen::MatrixXd source = Eigen::MatrixXd::NullaryExpr(dimension, dimension, [&] { return std::uniform_real_distribution<double>(-1, 1)(generator); });
		const Eigen::MatrixXd positive = source.transpose() * source + dimension * Eigen::MatrixXd::Identity(dimension, dimension);
		const auto matrix = Hoppy::SymmetricMatrixXd::FromModifiedDense(positive);
		Hoppy::Test::requireApprox((matrix * matrix.inverse()).toDense(), Eigen::MatrixXd::Identity(dimension, dimension));
	}
}
