//
// Created by Andy on 4/7/2026.
//

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <SecUtility/Diagnostic/Exception.hpp>
#include <SecUtility/Math/Orthonormalization.hpp>


namespace SecUtility::Math
{
	// Helper function to verify orthonormality of columns
	template <typename Matrix>
	void CheckOrthonormal(
	        const Eigen::MatrixBase<Matrix>& m,
	        const typename Eigen::NumTraits<typename Eigen::internal::traits<Matrix>::Scalar>::Real tolerance = 1e-12)
	{
		for (Eigen::Index i = 0; i < m.cols(); i++)
		{
			for (Eigen::Index j = i; j < m.cols(); j++)
			{
				const typename Eigen::internal::traits<Matrix>::Scalar expected = i == j ? 1.0 : 0.0;
				CHECK(std::abs(m.col(i).dot(m.col(j)) - expected) <= tolerance);
			}
		}
	}
}


TEMPLATE_TEST_CASE("OrthogonalizeInplaceWithGramSchmidt", "[template]", double, (std::complex<double>))
{
	Eigen::MatrixX<TestType> m = Eigen::MatrixX<TestType>::Random(5, 3);

	SECTION("Full")
	{
		SecUtility::Math::OrthogonalizeInplaceWithGramSchmidt(m);
		SecUtility::Math::CheckOrthonormal(m);
	}
	SECTION("Partial")
	{
		const Eigen::MatrixX<TestType> bottom = m.bottomRows(2);
		const Eigen::MatrixX<TestType> right = m.rightCols(1);

		SecUtility::Math::OrthogonalizeInplaceWithGramSchmidt(m.topLeftCorner(3, 2));
		SecUtility::Math::CheckOrthonormal(m.topLeftCorner(3, 2));

		CHECK(bottom == m.bottomRows(2));
		CHECK(right == m.rightCols(1));
	}
}


TEMPLATE_TEST_CASE("OrthogonalizeInplaceWithClassicalGramSchmidt", "[template]", double, (std::complex<double>))
{
	Eigen::MatrixX<TestType> m = Eigen::MatrixX<TestType>::Random(6, 4);
	SecUtility::Math::OrthogonalizeInplaceWithClassicalGramSchmidt(m);
	SecUtility::Math::CheckOrthonormal(m);
}


TEMPLATE_TEST_CASE("OrthogonalizeInplaceWithModifiedGramSchmidt", "[template]", double, (std::complex<double>))
{
	Eigen::MatrixX<TestType> m = Eigen::MatrixX<TestType>::Random(6, 4);
	SecUtility::Math::OrthogonalizeInplaceWithModifiedGramSchmidt(m);
	SecUtility::Math::CheckOrthonormal(m);
}


TEMPLATE_TEST_CASE("Classical vs Modified GramSchmidt equivalence", "[template]", double, (std::complex<double>))
{
	const Eigen::MatrixX<TestType> m = Eigen::MatrixX<TestType>::Random(8, 5);

	Eigen::MatrixX<TestType> classical = m;
	Eigen::MatrixX<TestType> modified = m;

	SecUtility::Math::OrthogonalizeInplaceWithClassicalGramSchmidt(classical);
	SecUtility::Math::OrthogonalizeInplaceWithModifiedGramSchmidt(modified);

	SecUtility::Math::CheckOrthonormal(classical);
	SecUtility::Math::CheckOrthonormal(modified);

	CHECK((classical - modified).norm() < 1e-10);
}


TEMPLATE_TEST_CASE("GramSchmidt startColumn", "[template]", double, (std::complex<double>))
{
	const Eigen::MatrixX<TestType> m = Eigen::MatrixX<TestType>::Random(5, 5);
	Eigen::MatrixX<TestType> test = m;

	// First orthogonalize all columns
	SecUtility::Math::OrthogonalizeInplaceWithModifiedGramSchmidt(test, 0);
	SecUtility::Math::CheckOrthonormal(test);

	// Now test starting from column 2
	test = m;
	SecUtility::Math::OrthogonalizeInplaceWithModifiedGramSchmidt(test, 2);

	// Columns 0-1 should remain unchanged
	CHECK((test.col(0) - m.col(0)).norm() < 1e-14);
	CHECK((test.col(1) - m.col(1)).norm() < 1e-14);

	// Columns 2-4 should be orthogonal to each other
	for (Eigen::Index i = 2; i < test.cols(); i++)
	{
		for (Eigen::Index j = i; j < test.cols(); j++)
		{
			const double expected = i == j ? 1.0 : 0.0;
			CHECK(std::abs(test.col(i).dot(test.col(j)) - expected) <= 1e-12);
		}
	}
}


TEMPLATE_TEST_CASE("QR Orthogonalization w/ startColumn", "[template]", double, (std::complex<double>))
{
	const Eigen::MatrixX<TestType> m = Eigen::MatrixX<TestType>::Random(5, 5);

	// First orthogonalize all columns
	SECTION("All")
	{
		const auto test = SecUtility::Math::OrthogonalizedAndLinearDependenceRemovedWithQR(m, 0);
		SecUtility::Math::CheckOrthonormal(test);
	}

	SECTION("Partial")
	{
		// Now test starting from column 2
		const auto test = SecUtility::Math::OrthogonalizedAndLinearDependenceRemovedWithQR(m, 2);

		// Columns 0-1 should remain unchanged
		CHECK((test.col(0) - m.col(0)).norm() < 1e-14);
		CHECK((test.col(1) - m.col(1)).norm() < 1e-14);

		// Columns 2-4 should be orthogonal to each other
		for (Eigen::Index i = 2; i < test.cols(); i++)
		{
			for (Eigen::Index j = i; j < test.cols(); j++)
			{
				const double expected = i == j ? 1.0 : 0.0;
				CHECK(std::abs(test.col(i).dot(test.col(j)) - expected) <= 1e-12);
			}
		}
	}
}


TEMPLATE_TEST_CASE("GramSchmidt startColumn boundary", "[template]", double, (std::complex<double>))
{
	const Eigen::MatrixX<TestType> m = Eigen::MatrixX<TestType>::Random(5, 4);

	// Start at column 0 (default behavior)
	Eigen::MatrixX<TestType> m1 = m;
	SecUtility::Math::OrthogonalizeInplaceWithModifiedGramSchmidt(m1, 0);
	SecUtility::Math::CheckOrthonormal(m1);

	// Start at last column
	Eigen::MatrixX<TestType> m2 = m;
	SecUtility::Math::OrthogonalizeInplaceWithModifiedGramSchmidt(m2, m2.cols() - 1);
	CHECK(m2.col(m2.cols() - 1).norm() == Catch::Approx(1.0).margin(1e-12));
}


TEMPLATE_TEST_CASE("GramSchmidt linear dependence Classical", "[template]", double, (std::complex<double>))
{
	// Create a matrix with linearly dependent columns
	Eigen::MatrixX<TestType> m(5, 3);
	m.col(0) = Eigen::VectorXd::Random(5);
	m.col(1) = Eigen::VectorXd::Random(5);
	m.col(2) = 2.0 * m.col(0) + 3.0 * m.col(1);  // Linear combination

	REQUIRE_THROWS_AS(SecUtility::Math::OrthogonalizeInplaceWithClassicalGramSchmidt(m), SecUtility::RuntimeException);
}


TEMPLATE_TEST_CASE("GramSchmidt linear dependence Modified", "[template]", double, (std::complex<double>))
{
	// Create a matrix with linearly dependent columns
	Eigen::MatrixX<TestType> m(5, 3);
	m.col(0) = Eigen::VectorXd::Random(5);
	m.col(1) = Eigen::VectorXd::Random(5);
	m.col(2) = m.col(0) + m.col(1);  // Linear combination

	REQUIRE_THROWS_AS(SecUtility::Math::OrthogonalizeInplaceWithModifiedGramSchmidt(m), SecUtility::RuntimeException);
}


TEMPLATE_TEST_CASE("GramSchmidt linear dependence RemoveLinearDependence", "[template]", double, (std::complex<double>))
{
	// Create a matrix with some linearly dependent columns
	Eigen::MatrixX<TestType> m(5, 5);
	m.col(0) = Eigen::VectorXd::Random(5);
	m.col(1) = Eigen::VectorXd::Random(5);
	m.col(2) = Eigen::VectorXd::Random(5);
	m.col(3) = m.col(0) + m.col(1);  // Dependent
	m.col(4) = 2.0 * m.col(2);       // Dependent

	const Eigen::Index originalCols = m.cols();
	SecUtility::Math::OrthogonalizeAndRemoveLinearDependenceInplaceWithModifiedGramSchmidt(m);

	// Should have fewer columns now
	REQUIRE(m.cols() < originalCols);
	REQUIRE(m.cols() == 3);  // Only 3 linearly independent columns

	// All remaining columns should be orthonormal
	SecUtility::Math::CheckOrthonormal(m);
}


TEMPLATE_TEST_CASE("GramSchmidt RemoveLinearDependence all independent", "[template]", double, (std::complex<double>))
{
	// All columns are independent
	Eigen::MatrixX<TestType> m = Eigen::MatrixX<TestType>::Random(5, 4);
	const Eigen::Index originalCols = m.cols();

	SecUtility::Math::OrthogonalizeAndRemoveLinearDependenceInplaceWithModifiedGramSchmidt(m);

	// Should have the same number of columns
	REQUIRE(m.cols() == originalCols);
	SecUtility::Math::CheckOrthonormal(m);
}


TEMPLATE_TEST_CASE("GramSchmidt single column", "[template]", double, (std::complex<double>))
{
	Eigen::MatrixX<TestType> m(5, 1);
	m.col(0) = Eigen::VectorXd::Random(5);

	SecUtility::Math::OrthogonalizeInplaceWithModifiedGramSchmidt(m);

	REQUIRE(m.cols() == 1);
	CHECK(m.col(0).norm() == Catch::Approx(1.0).margin(1e-12));
}


TEMPLATE_TEST_CASE("GramSchmidt two columns", "[template]", double, (std::complex<double>))
{
	Eigen::MatrixX<TestType> m(5, 2);
	m.col(0) = Eigen::VectorXd::Random(5);
	m.col(1) = Eigen::VectorXd::Random(5);

	SecUtility::Math::OrthogonalizeInplaceWithModifiedGramSchmidt(m);

	REQUIRE(m.cols() == 2);
	SecUtility::Math::CheckOrthonormal(m);
}


TEMPLATE_TEST_CASE("GramSchmidt all columns same", "[template]", double, (std::complex<double>))
{
	// All columns are identical (highly dependent)
	Eigen::MatrixX<TestType> m(5, 4);
	m.col(0) = Eigen::VectorXd::Random(5);
	m.col(1) = m.col(0);
	m.col(2) = m.col(0);
	m.col(3) = m.col(0);

	REQUIRE_THROWS(SecUtility::Math::OrthogonalizedWithClassicalGramSchmidt(m));
	REQUIRE_THROWS(SecUtility::Math::OrthogonalizedWithModifiedGramSchmidt(m));
	REQUIRE_NOTHROW(SecUtility::Math::OrthogonalizeAndRemoveLinearDependenceInplaceWithModifiedGramSchmidt(m));

	// Should only have 1 column left
	REQUIRE(m.cols() == 1);
	CHECK(m.col(0).norm() == Catch::Approx(1.0).margin(1e-12));
}


TEMPLATE_TEST_CASE("GramSchmidt nearly dependent columns", "[template]", double, (std::complex<double>))
{
	// Create nearly dependent columns (ill-conditioned matrix)
	Eigen::MatrixX<TestType> m(5, 3);
	m.col(0) = Eigen::VectorXd::Random(5);
	m.col(1) = m.col(0) + 1e-8 * Eigen::VectorXd::Random(5);  // Nearly parallel
	m.col(2) = Eigen::VectorXd::Random(5);

	// Should not throw for nearly dependent columns
	REQUIRE_NOTHROW(SecUtility::Math::OrthogonalizeInplaceWithModifiedGramSchmidt(m));

	SecUtility::Math::CheckOrthonormal(m, 5e-8);  // Use looser tolerance
}


TEMPLATE_TEST_CASE("QR Orthogonalization: all columns same", "[template]", double, (std::complex<double>))
{
	// All columns are identical (highly dependent)
	Eigen::MatrixX<TestType> m(5, 4);
	m.col(0) = Eigen::VectorXd::Random(5);
	m.col(1) = m.col(0);
	m.col(2) = m.col(0);
	m.col(3) = m.col(0);

	const auto n = SecUtility::Math::OrthogonalizedAndLinearDependenceRemovedWithQR(m);

	// Should only have 1 column left
	REQUIRE(n.cols() == 1);
	CHECK(n.col(0).norm() == Catch::Approx(1.0).margin(1e-12));
}


TEMPLATE_TEST_CASE("QR Orthogonalization: nearly dependent columns", "[template]", double, (std::complex<double>))
{
	// Create nearly dependent columns (ill-conditioned matrix)
	Eigen::MatrixX<TestType> m(5, 3);
	m.col(0) = Eigen::VectorXd::Random(5);
	m.col(1) = m.col(0) + 1e-8 * Eigen::VectorXd::Random(5);  // Nearly parallel
	m.col(2) = Eigen::VectorXd::Random(5);

	// Should not throw for nearly dependent columns
	const auto n = SecUtility::Math::OrthogonalizedAndLinearDependenceRemovedWithQR(m);

	SecUtility::Math::CheckOrthonormal(n, 5e-8);  // Use looser tolerance
}
