//
// Created by Andy on 4/7/2026.
//

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <SecUtility/Math/StructuredMatrix.hpp>


using namespace SecUtility::Math;
using Catch::Approx;


namespace SecUtility::Math
{
	template <typename Scalar>
	bool IsSymmetric(const Eigen::MatrixX<Scalar>& m, double tolerance = 1e-12)
	{
		if (m.rows() != m.cols())
		{
			return false;
		}
		return (m - m.transpose()).norm() < tolerance;
	}

	template <typename Scalar>
	bool IsHermitian(const Eigen::MatrixX<Scalar>& m, double tolerance = 1e-12)
	{
		if (m.rows() != m.cols())
		{
			return false;
		}
		return (m - m.adjoint()).norm() < tolerance;
	}

	template <typename Scalar>
	bool IsAntiSymmetric(const Eigen::MatrixX<Scalar>& m, double tolerance = 1e-12)
	{
		if (m.rows() != m.cols())
		{
			return false;
		}
		return (m + m.transpose()).norm() < tolerance;
	}

	template <typename Scalar>
	bool IsAntiHermitian(const Eigen::MatrixX<Scalar>& m, double tolerance = 1e-12)
	{
		if (m.rows() != m.cols())
		{
			return false;
		}
		return (m + m.adjoint()).norm() < tolerance;
	}

	template <typename Scalar>
	bool IsUnitary(const Eigen::MatrixX<Scalar>& m, double tolerance = 1e-12)
	{
		if (m.rows() != m.cols())
		{
			return false;
		}
		const auto product = m.adjoint() * m;
		return (product - Eigen::MatrixX<Scalar>::Identity(m.cols(), m.cols())).norm() < tolerance;
	}

	template <typename Scalar>
	bool IsPositiveDefiniteHermitian(const Eigen::MatrixX<Scalar>& m, double tolerance = 1e-12)
	{
		if (!IsHermitian(m, tolerance))
		{
			return false;
		}
		Eigen::LLT<Eigen::MatrixX<Scalar>> llt(m);
		return llt.info() == Eigen::Success;
	}

	// Helper function to check if all eigenvalues are positive
	template <typename Scalar>
	bool HasAllPositiveEigenvalues(const Eigen::MatrixX<Scalar>& m)
	{
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Scalar>> eigensolver(m);
		return eigensolver.info() == Eigen::Success && (eigensolver.eigenvalues().array() > 0).all();
	}
}



TEMPLATE_TEST_CASE("RandomSymmetricMatrix", "[template]", double, (std::complex<double>))
{
	SECTION("Basic properties")
	{
		const int dim = 5;
		const auto m = RandomSymmetricMatrix<TestType>(dim);

		REQUIRE(m.rows() == dim);
		REQUIRE(m.cols() == dim);
		CHECK(IsSymmetric(m));
	}

	SECTION("Different dimensions")
	{
		for (int dim : {1, 2, 3, 10, 20})
		{
			const auto m = RandomSymmetricMatrix<TestType>(dim);
			CHECK(m.rows() == dim);
			CHECK(m.cols() == dim);
			CHECK(IsSymmetric(m));
		}
	}

	SECTION("Randomness")
	{
		const auto m1 = RandomSymmetricMatrix<TestType>(10);
		const auto m2 = RandomSymmetricMatrix<TestType>(10);

		// Two random matrices should be different
		CHECK((m1 - m2).norm() > 1e-10);
	}
}


TEMPLATE_TEST_CASE("RandomHermitianMatrix", "[template]", double, (std::complex<double>))
{
	SECTION("Basic properties")
	{
		const int dim = 5;
		const auto m = RandomHermitianMatrix<TestType>(dim);

		REQUIRE(m.rows() == dim);
		REQUIRE(m.cols() == dim);
		CHECK(IsHermitian(m));
	}

	SECTION("Different dimensions")
	{
		for (int dim : {1, 2, 3, 10})
		{
			const auto m = RandomHermitianMatrix<TestType>(dim);
			CHECK(m.rows() == dim);
			CHECK(m.cols() == dim);
			CHECK(IsHermitian(m));
		}
	}
}


TEMPLATE_TEST_CASE("RandomAntiSymmetricMatrix", "[template]", double, (std::complex<double>))
{
	SECTION("Basic properties")
	{
		const int dim = 5;
		const auto m = RandomAntiSymmetricMatrix<TestType>(dim);

		REQUIRE(m.rows() == dim);
		REQUIRE(m.cols() == dim);
		CHECK(IsAntiSymmetric(m));
	}

	SECTION("Different dimensions")
	{
		for (int dim : {1, 2, 3, 10})
		{
			const auto m = RandomAntiSymmetricMatrix<TestType>(dim);
			CHECK(IsAntiSymmetric(m));
		}
	}
}


TEMPLATE_TEST_CASE("RandomAntiHermitianMatrix", "[template]", double, (std::complex<double>))
{
	const int dim = 5;
	const auto m = RandomAntiHermitianMatrix<TestType>(dim);

	REQUIRE(m.rows() == dim);
	REQUIRE(m.cols() == dim);
	CHECK(IsAntiHermitian(m));
}


TEMPLATE_TEST_CASE("RandomUnitaryMatrix", "[template]", double, (std::complex<double>))
{
	SECTION("Basic properties")
	{
		const int dim = 5;
		const auto m = RandomUnitaryMatrix<TestType>(dim);

		REQUIRE(m.rows() == dim);
		REQUIRE(m.cols() == dim);
		CHECK(IsUnitary(m));
	}

	SECTION("Different dimensions")
	{
		for (int dim : {1, 2, 3, 10})
		{
			const auto m = RandomUnitaryMatrix<TestType>(dim);
			CHECK(IsUnitary(m, 1e-10));
		}
	}
}


TEMPLATE_TEST_CASE("RandomPositiveDefiniteHermitianMatrix", "[template]", double, (std::complex<double>))
{
	SECTION("Basic properties")
	{
		const int dim = 5;
		const auto m = RandomPositiveDefiniteHermitianMatrix<TestType>(dim);

		REQUIRE(m.rows() == dim);
		REQUIRE(m.cols() == dim);
		CHECK(IsHermitian(m));
		CHECK(IsPositiveDefiniteHermitian(m));
		CHECK(HasAllPositiveEigenvalues(m));
	}

	SECTION("Different dimensions")
	{
		for (int dim : {1, 2, 3, 10})
		{
			const auto m = RandomPositiveDefiniteHermitianMatrix<TestType>(dim);
			CHECK(IsHermitian(m));
			CHECK(IsPositiveDefiniteHermitian(m));
			CHECK(HasAllPositiveEigenvalues(m));
		}
	}

	SECTION("Positive eigenvalues")
	{
		const auto m = RandomPositiveDefiniteHermitianMatrix<TestType>(10);
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<TestType>> eigensolver(m);

		REQUIRE(eigensolver.info() == Eigen::Success);
		const auto& eigenvalues = eigensolver.eigenvalues();

		CHECK(std::all_of(eigenvalues.cbegin(), eigenvalues.cend(), [](const auto x) { return x > 0; }));
	}
}


TEMPLATE_TEST_CASE("RandomUnitaryWithGivenFirstColumn", "[template]", double, (std::complex<double>))
{
	SECTION("Basic properties")
	{
		const int dim = 5;
		Eigen::VectorX<TestType> firstCol = Eigen::VectorX<TestType>::Random(dim).normalized();

		const auto m = RandomUnitaryWithGivenFirstColumn(firstCol);

		REQUIRE(m.rows() == dim);
		REQUIRE(m.cols() == dim);
		CHECK(IsUnitary(m));

		CHECK(std::abs((m.col(0) - firstCol).norm()) < 1e-12);
	}

	SECTION("Special case: first column is [1, 0, 0, ...]")
	{
		const int dim = 5;
		Eigen::VectorX<TestType> firstCol(dim);
		firstCol.setZero();
		firstCol[0] = 1.0;

		const auto m = RandomUnitaryWithGivenFirstColumn(firstCol);

		REQUIRE(m.rows() == dim);
		REQUIRE(m.cols() == dim);
		CHECK(IsUnitary(m));

		// When first column is already [1, 0, 0, ...], should get identity
		CHECK((m - Eigen::MatrixX<TestType>::Identity(dim, dim)).norm() < 1e-14);
	}

	SECTION("Different dimensions")
	{
		for (int dim : {1, 2, 3, 10})
		{
			Eigen::VectorX<TestType> firstCol = Eigen::VectorX<TestType>::Random(dim);
			const auto m = RandomUnitaryWithGivenFirstColumn(firstCol);
			CHECK(IsUnitary(m));
		}
	}
}


TEST_CASE("RandomSparseMatrixInDenseForm")
{
	SECTION("Basic properties with default density")
	{
		const int rows = 10;
		const int cols = 10;
		const auto m = RandomSparseMatrixInDenseForm<double>(rows, cols);

		REQUIRE(m.rows() == rows);
		REQUIRE(m.cols() == cols);

		// Count non-zero elements
		int nonZeroCount = 0;
		for (int i = 0; i < rows; ++i)
		{
			for (int j = 0; j < cols; ++j)
			{
				if (std::abs(m(i, j)) > 1e-14)
				{
					nonZeroCount++;
				}
			}
		}

		// With density = 0.1, expect approximately 10% non-zero
		const double expectedDensity = 0.1;
		const double actualDensity = static_cast<double>(nonZeroCount) / (rows * cols);
		CHECK(std::abs(actualDensity - expectedDensity) < 0.15);  // Allow 15% tolerance
	}

	SECTION("Different densities")
	{
		const int rows = 20;
		const int cols = 20;

		for (double density : {0.01, 0.1, 0.5, 0.9})
		{
			const auto m = RandomSparseMatrixInDenseForm<double>(rows, cols, density);

			int nonZeroCount = 0;
			for (int i = 0; i < rows; ++i)
			{
				for (int j = 0; j < cols; ++j)
				{
					if (std::abs(m(i, j)) > 1e-14)
					{
						nonZeroCount++;
					}
				}
			}

			const double actualDensity = static_cast<double>(nonZeroCount) / (rows * cols);
			CHECK(std::abs(actualDensity - density) < 0.2);  // Allow 20% tolerance
		}
	}

	SECTION("Value ranges")
	{
		const int rows = 10;
		const int cols = 10;
		const double minVal = -5.0;
		const double maxVal = 3.0;

		const auto m = RandomSparseMatrixInDenseForm<double>(rows, cols, 0.3, maxVal, minVal);

		for (int i = 0; i < rows; ++i)
		{
			for (int j = 0; j < cols; ++j)
			{
				if (std::abs(m(i, j)) > 1e-14)
				{
					CHECK(m(i, j) >= minVal);
					CHECK(m(i, j) <= maxVal);
				}
			}
		}
	}

	SECTION("Complex values")
	{
		const int rows = 5;
		const int cols = 5;
		const auto m = RandomSparseMatrixInDenseForm<std::complex<double>>(rows, cols, 0.5);

		for (int i = 0; i < rows; ++i)
		{
			for (int j = 0; j < cols; ++j)
			{
				if (std::abs(m(i, j)) > 1e-14)
				{
					// Check both real and imaginary parts are in default range [-1, 1]
					CHECK(m(i, j).real() >= -1.0);
					CHECK(m(i, j).real() <= 1.0);
					CHECK(m(i, j).imag() >= -1.0);
					CHECK(m(i, j).imag() <= 1.0);
				}
			}
		}
	}

	SECTION("Rectangular matrices")
	{
		// Tall matrix
		const auto m1 = RandomSparseMatrixInDenseForm<double>(15, 5, 0.3);
		CHECK(m1.rows() == 15);
		CHECK(m1.cols() == 5);

		// Wide matrix
		const auto m2 = RandomSparseMatrixInDenseForm<double>(5, 15, 0.3);
		CHECK(m2.rows() == 5);
		CHECK(m2.cols() == 15);
	}

	SECTION("Zero density")
	{
		const auto m = RandomSparseMatrixInDenseForm<double>(10, 10, 0.0);
		CHECK(m.norm() < 1e-14);  // Should be essentially zero
	}

	SECTION("Full density")
	{
		const auto m = RandomSparseMatrixInDenseForm<double>(10, 10, 1.0);

		int nonZeroCount = 0;
		for (int i = 0; i < 10; ++i)
		{
			for (int j = 0; j < 10; ++j)
			{
				if (std::abs(m(i, j)) > 1e-14)
				{
					nonZeroCount++;
				}
			}
		}

		CHECK(nonZeroCount == 100);  // All elements should be non-zero
	}
}
