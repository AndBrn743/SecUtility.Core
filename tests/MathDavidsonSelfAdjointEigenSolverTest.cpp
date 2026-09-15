// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#include <SecUtility/Math/DavidsonSelfAdjointEigenSolver.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <limits>


namespace
{
	template <typename T>
	struct DenseOperator
	{
		using Scalar = T;
		using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
		Eigen::MatrixX<Scalar> Matrix;
		Eigen::VectorX<RealScalar> DiagonalOverride;

		Eigen::Index rows() const { return Matrix.rows(); }
		Eigen::Index cols() const { return Matrix.cols(); }
		Eigen::VectorX<RealScalar> Diagonal() const
		{
			return DiagonalOverride.size() == 0 ? Matrix.diagonal().real().eval() : DiagonalOverride;
		}

		template <typename Derived>
		Eigen::MatrixX<Scalar> ApplyOn(const Eigen::MatrixBase<Derived>& vectors) const
		{
			return Matrix * vectors;
		}
	};


	template <typename Scalar>
	DenseOperator<Scalar> IdentityOperator(const Eigen::Index rows, const Eigen::Index cols)
	{
		return {Eigen::MatrixX<Scalar>::Identity(rows, cols), {}};
	}


	template <typename T>
	struct VectorOnlyCountingOperator
	{
		using Scalar = T;
		using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
		Eigen::MatrixX<Scalar> Matrix;
		mutable Eigen::Index ApplicationCount = 0;

		Eigen::Index rows() const { return Matrix.rows(); }
		Eigen::Index cols() const { return Matrix.cols(); }
		Eigen::VectorX<RealScalar> Diagonal() const { return Matrix.diagonal().real(); }

		template <typename Derived>
			requires(Derived::ColsAtCompileTime == 1)
		Eigen::VectorX<Scalar> ApplyOn(const Eigen::MatrixBase<Derived>& vector) const
		{
			ApplicationCount++;
			return Matrix * vector;
		}
	};
}


using namespace SecUtility::Math;


TEST_CASE("Davidson option and statistic defaults are stable", "[Math][Davidson]")
{
	const DavidsonEigenSolverOptions<double> options{3};
	CHECK(options.RootCount == 3);
	CHECK(options.MaximumIterationCount == 256);
	CHECK(options.InitialSubspaceDimension == 3);
	CHECK(options.MaximumSubspaceDimension == 0);
	CHECK(options.ResidualNormTolerance == 1e-7);
	CHECK(options.EigenvalueChangeTolerance == 1e-7);
	CHECK(options.PreconditionerDenominatorFloor == 1e-12);
	CHECK(options.LinearDependenceTolerance == 1e-10);

	const DavidsonEigenSolverStatistics statistics;
	CHECK(statistics.CompletedIterationCount == 0);
	CHECK(statistics.OperatorApplicationCount == 0);
	CHECK(statistics.MultipliedVectorCount == 0);
	CHECK(statistics.MaximumSubspaceDimension == 0);
	CHECK(statistics.RestartCount == 0);
	CHECK(statistics.GeneratedCorrectionVectorCount == 0);
	CHECK(statistics.RetainedCorrectionVectorCount == 0);
	CHECK(statistics.OperatorRefreshCount == 0);
}


TEMPLATE_TEST_CASE("A Davidson solver publishes an aligned initial subspace and empty Phase 3 Ritz results",
	               "[Math][Davidson]", double, (std::complex<double>))
{
	using Operator = DenseOperator<TestType>;
	DavidsonSelfAdjointEigenSolver<Operator> solver;
	CHECK(solver.Status() == DavidsonEigenSolverStatus::NotComputed);
	CHECK(solver.Eigenvalues().size() == 0);
	CHECK(solver.Eigenvectors().rows() == 0);
	CHECK(solver.Eigenvectors().cols() == 0);
	CHECK(solver.ResidualNorms().size() == 0);
	CHECK(solver.BasisVectors().size() == 0);
	CHECK(solver.BasisVectorImages().size() == 0);
	CHECK(solver.ReducedMatrix().size() == 0);
	CHECK(solver.ReducedEigenvectors().size() == 0);

	const Operator linearOperator = IdentityOperator<TestType>(4, 4);
	const Eigen::MatrixX<TestType> initialBasis = Eigen::MatrixX<TestType>::Identity(4, 2);
	const auto status = solver.Compute(linearOperator, initialBasis, DavidsonEigenSolverOptions<double>{2});

	CHECK(status == DavidsonEigenSolverStatus::IterationLimitReached);
	CHECK(solver.Status() == status);
	CHECK(solver.Eigenvalues().size() == 0);
	CHECK(solver.Eigenvectors().rows() == 4);
	CHECK(solver.Eigenvectors().cols() == 0);
	CHECK(solver.ResidualNorms().size() == 0);
	CHECK(solver.BasisVectors().rows() == 4);
	CHECK(solver.BasisVectors().cols() == 2);
	CHECK((solver.BasisVectors().adjoint() * solver.BasisVectors())
	              .isApprox(Eigen::MatrixX<TestType>::Identity(2, 2)));
	CHECK(solver.BasisVectorImages().rows() == 4);
	CHECK(solver.BasisVectorImages().cols() == 2);
	CHECK(solver.BasisVectorImages().isApprox(solver.BasisVectors()));
	CHECK(solver.ReducedMatrix().isApprox(Eigen::MatrixX<TestType>::Identity(2, 2)));
	CHECK(solver.ReducedEigenvectors().size() == 0);
	CHECK(solver.Statistics().OperatorApplicationCount == 1);
	CHECK(solver.Statistics().MultipliedVectorCount == 2);
	CHECK(solver.Statistics().MaximumSubspaceDimension == 2);
}


TEMPLATE_TEST_CASE("Davidson initial-subspace preparation orthonormalizes and removes dependent columns",
	               "[Math][Davidson]", double, (std::complex<double>))
{
	using Operator = DenseOperator<TestType>;
	const Operator linearOperator = IdentityOperator<TestType>(4, 4);
	Eigen::MatrixX<TestType> initialBasis = Eigen::MatrixX<TestType>::Zero(4, 3);
	initialBasis(0, 0) = TestType{3};
	initialBasis(1, 1) = TestType{-2};
	initialBasis(0, 2) = TestType{4};
	const Eigen::MatrixX<TestType> originalBasis = initialBasis;
	DavidsonSelfAdjointEigenSolver<Operator> solver;
	DavidsonEigenSolverOptions<double> options{1};
	options.InitialSubspaceDimension = 3;

	CHECK(solver.Compute(linearOperator, initialBasis, options) == DavidsonEigenSolverStatus::IterationLimitReached);

	REQUIRE(solver.BasisVectors().cols() == 2);
	CHECK((solver.BasisVectors().adjoint() * solver.BasisVectors())
	              .isApprox(Eigen::MatrixX<TestType>::Identity(2, 2), 1e-12));
	CHECK((solver.BasisVectors() * (solver.BasisVectors().adjoint() * originalBasis)).isApprox(originalBasis, 1e-12));
	CHECK(initialBasis.isApprox(originalBasis, 0));
	CHECK(solver.Statistics().MultipliedVectorCount == 2);
	CHECK(solver.Statistics().MaximumSubspaceDimension == 2);
}


TEST_CASE("Davidson initial-subspace rank detection respects the configured relative threshold", "[Math][Davidson]")
{
	const auto linearOperator = IdentityOperator<double>(3, 3);
	DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	DavidsonEigenSolverOptions<double> options{1};
	options.InitialSubspaceDimension = 2;
	options.LinearDependenceTolerance = 1e-8;
	Eigen::MatrixXd basis = Eigen::MatrixXd::Zero(3, 2);
	basis(0, 0) = 1;

	SECTION("A pivot above the threshold is retained")
	{
		basis(1, 1) = 1e-7;
		CHECK(solver.Compute(linearOperator, basis, options) == DavidsonEigenSolverStatus::IterationLimitReached);
		CHECK(solver.BasisVectors().cols() == 2);
	}

	SECTION("A pivot below the threshold is removed")
	{
		basis(1, 1) = 1e-9;
		CHECK(solver.Compute(linearOperator, basis, options) == DavidsonEigenSolverStatus::IterationLimitReached);
		CHECK(solver.BasisVectors().cols() == 1);
	}
}


TEMPLATE_TEST_CASE("Davidson initial vector images and projection remain aligned",
	               "[Math][Davidson]", double, (std::complex<double>))
{
	Eigen::MatrixX<TestType> matrix = Eigen::MatrixX<TestType>::Zero(4, 4);
	matrix.diagonal() << TestType{1}, TestType{2}, TestType{4}, TestType{7};
	const TestType coupling = []
	{
		if constexpr (Eigen::NumTraits<TestType>::IsComplex)
		{
			return TestType{0.25, 0.5};
		}
		return TestType{0.25};
	}();
	matrix(0, 1) = coupling;
	matrix(1, 0) = Eigen::numext::conj(coupling);
	const DenseOperator<TestType> linearOperator{matrix, {}};
	Eigen::MatrixX<TestType> initialBasis = Eigen::MatrixX<TestType>::Random(4, 3);
	DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	DavidsonEigenSolverOptions<double> options{2};
	options.InitialSubspaceDimension = 3;

	CHECK(solver.Compute(linearOperator, initialBasis, options) == DavidsonEigenSolverStatus::IterationLimitReached);

	const auto expectedImages = matrix * solver.BasisVectors();
	const auto expectedReducedMatrix = solver.BasisVectors().adjoint() * expectedImages;
	CHECK(solver.BasisVectorImages().isApprox(expectedImages, 1e-12));
	CHECK(solver.ReducedMatrix().isApprox(expectedReducedMatrix, 1e-12));
	CHECK(solver.ReducedMatrix().isApprox(solver.ReducedMatrix().adjoint(), 1e-12));
}


TEMPLATE_TEST_CASE("Davidson initial-subspace accounting distinguishes block and vector-only operators",
	               "[Math][Davidson]", double, (std::complex<double>))
{
	const Eigen::MatrixX<TestType> matrix = Eigen::MatrixX<TestType>::Identity(4, 4);
	const Eigen::MatrixX<TestType> basis = Eigen::MatrixX<TestType>::Identity(4, 3);
	DenseOperator<TestType> blockOperator{matrix, {}};
	VectorOnlyCountingOperator<TestType> vectorOperator{matrix};
	DavidsonSelfAdjointEigenSolver<decltype(blockOperator)> blockSolver;
	DavidsonSelfAdjointEigenSolver<decltype(vectorOperator)> vectorSolver;
	DavidsonEigenSolverOptions<double> options{2};
	options.InitialSubspaceDimension = 3;

	CHECK(blockSolver.Compute(blockOperator, basis, options) == DavidsonEigenSolverStatus::IterationLimitReached);
	CHECK(vectorSolver.Compute(vectorOperator, basis, options) == DavidsonEigenSolverStatus::IterationLimitReached);

	CHECK(blockSolver.BasisVectorImages().isApprox(vectorSolver.BasisVectorImages()));
	CHECK(blockSolver.Statistics().OperatorApplicationCount == 1);
	CHECK(blockSolver.Statistics().MultipliedVectorCount == 3);
	CHECK(vectorSolver.Statistics().OperatorApplicationCount == 3);
	CHECK(vectorSolver.Statistics().MultipliedVectorCount == 3);
	CHECK(vectorOperator.ApplicationCount == 3);
}


TEST_CASE("Davidson rejects an initial basis with no surviving direction", "[Math][Davidson]")
{
	const auto linearOperator = IdentityOperator<double>(3, 3);
	DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	DavidsonEigenSolverOptions<double> options{1};
	options.InitialSubspaceDimension = 2;

	CHECK_THROWS_AS(solver.Compute(linearOperator, Eigen::MatrixXd::Zero(3, 2), options),
	                SecUtility::InvalidArgumentException);
	CHECK(solver.Status() == DavidsonEigenSolverStatus::NotComputed);
	CHECK(solver.BasisVectors().size() == 0);
	CHECK(solver.BasisVectorImages().size() == 0);
	CHECK(solver.ReducedMatrix().size() == 0);
}


TEST_CASE("Davidson validation accepts boundary dimensions and automatic maximum space", "[Math][Davidson]")
{
	const auto linearOperator = IdentityOperator<double>(4, 4);
	DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;

	SECTION("One root and one initial vector")
	{
		DavidsonEigenSolverOptions<double> options{1};
		CHECK_NOTHROW(solver.Compute(linearOperator, Eigen::MatrixXd::Identity(4, 1), options));
	}

	SECTION("The full dimension may be requested")
	{
		DavidsonEigenSolverOptions<double> options{4};
		options.MaximumSubspaceDimension = 4;
		CHECK_NOTHROW(solver.Compute(linearOperator, Eigen::MatrixXd::Identity(4, 4), options));
	}

	SECTION("A supplied seed may be smaller than the requested initial space")
	{
		DavidsonEigenSolverOptions<double> options{2};
		options.InitialSubspaceDimension = 3;
		options.MaximumSubspaceDimension = 4;
		CHECK_NOTHROW(solver.Compute(linearOperator, Eigen::MatrixXd::Identity(4, 1), options));
	}
}


TEST_CASE("Davidson validation rejects inconsistent operator and basis dimensions", "[Math][Davidson]")
{
	SECTION("The operator must be square")
	{
		const auto linearOperator = IdentityOperator<double>(3, 4);
		DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;
		CHECK_THROWS_AS(solver.Compute(linearOperator, Eigen::MatrixXd::Identity(3, 1),
		                               DavidsonEigenSolverOptions<double>{1}),
		                SecUtility::InvalidArgumentException);
	}

	SECTION("The operator dimension must be positive")
	{
		const auto linearOperator = IdentityOperator<double>(0, 0);
		DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;
		CHECK_THROWS_AS(solver.Compute(linearOperator, Eigen::MatrixXd(0, 1),
		                               DavidsonEigenSolverOptions<double>{1}),
		                SecUtility::InvalidArgumentException);
	}

	SECTION("The diagonal dimension must agree")
	{
		auto linearOperator = IdentityOperator<double>(4, 4);
		linearOperator.DiagonalOverride = Eigen::VectorXd::Ones(3);
		DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;
		CHECK_THROWS_AS(solver.Compute(linearOperator, Eigen::MatrixXd::Identity(4, 1),
		                               DavidsonEigenSolverOptions<double>{1}),
		                SecUtility::InvalidArgumentException);
	}

	SECTION("The basis row count must agree")
	{
		const auto linearOperator = IdentityOperator<double>(4, 4);
		DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;
		CHECK_THROWS_AS(solver.Compute(linearOperator, Eigen::MatrixXd::Identity(3, 1),
		                               DavidsonEigenSolverOptions<double>{1}),
		                SecUtility::InvalidArgumentException);
	}

	SECTION("The basis must contain between one and dimension vectors")
	{
		const auto linearOperator = IdentityOperator<double>(4, 4);
		DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;
		CHECK_THROWS_AS(solver.Compute(linearOperator, Eigen::MatrixXd(4, 0),
		                               DavidsonEigenSolverOptions<double>{1}),
		                SecUtility::InvalidArgumentException);
		CHECK_THROWS_AS(solver.Compute(linearOperator, Eigen::MatrixXd::Ones(4, 5),
		                               DavidsonEigenSolverOptions<double>{1}),
		                SecUtility::InvalidArgumentException);
	}
}


TEST_CASE("Davidson validation rejects non-finite operator and basis data", "[Math][Davidson]")
{
	const double nan = std::numeric_limits<double>::quiet_NaN();
	const double infinity = std::numeric_limits<double>::infinity();

	auto linearOperator = IdentityOperator<double>(3, 3);
	DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	linearOperator.DiagonalOverride = Eigen::Vector3d{1, nan, 3};
	CHECK_THROWS_AS(solver.Compute(linearOperator, Eigen::MatrixXd::Identity(3, 1),
	                               DavidsonEigenSolverOptions<double>{1}),
	                SecUtility::InvalidArgumentException);

	linearOperator.DiagonalOverride = Eigen::Vector3d{1, 2, 3};
	Eigen::MatrixXd basis = Eigen::MatrixXd::Identity(3, 1);
	basis(0, 0) = infinity;
	CHECK_THROWS_AS(solver.Compute(linearOperator, basis, DavidsonEigenSolverOptions<double>{1}),
	                SecUtility::InvalidArgumentException);
}


TEST_CASE("Davidson validation rejects invalid count and subspace options", "[Math][Davidson]")
{
	const auto linearOperator = IdentityOperator<double>(4, 4);
	const Eigen::MatrixXd basis = Eigen::MatrixXd::Identity(4, 2);
	DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;

	for (const Eigen::Index rootCount : {Eigen::Index{-1}, Eigen::Index{0}, Eigen::Index{5}})
	{
		CHECK_THROWS_AS(solver.Compute(linearOperator, basis, DavidsonEigenSolverOptions<double>{rootCount}),
		                SecUtility::InvalidArgumentException);
	}

	DavidsonEigenSolverOptions<double> options{2};
	for (const Eigen::Index maximumIterationCount : {Eigen::Index{-1}, Eigen::Index{0}})
	{
		options.MaximumIterationCount = maximumIterationCount;
		CHECK_THROWS_AS(solver.Compute(linearOperator, basis, options), SecUtility::InvalidArgumentException);
	}

	for (const Eigen::Index initialSubspaceDimension : {Eigen::Index{-1}, Eigen::Index{0}, Eigen::Index{1}, Eigen::Index{5}})
	{
		options = DavidsonEigenSolverOptions<double>{2};
		options.InitialSubspaceDimension = initialSubspaceDimension;
		CHECK_THROWS_AS(solver.Compute(linearOperator, basis, options), SecUtility::InvalidArgumentException);
	}

	for (const Eigen::Index maximumSubspaceDimension : {Eigen::Index{-1}, Eigen::Index{1}, Eigen::Index{5}})
	{
		options = DavidsonEigenSolverOptions<double>{2};
		options.MaximumSubspaceDimension = maximumSubspaceDimension;
		CHECK_THROWS_AS(solver.Compute(linearOperator, basis, options), SecUtility::InvalidArgumentException);
	}

	options = DavidsonEigenSolverOptions<double>{1};
	options.InitialSubspaceDimension = 1;
	options.MaximumSubspaceDimension = 1;
	CHECK_THROWS_AS(solver.Compute(linearOperator, basis, options), SecUtility::InvalidArgumentException);
}


TEST_CASE("Davidson validation rejects every invalid numerical tolerance", "[Math][Davidson]")
{
	const auto linearOperator = IdentityOperator<double>(3, 3);
	const Eigen::MatrixXd basis = Eigen::MatrixXd::Identity(3, 1);
	DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	const double nan = std::numeric_limits<double>::quiet_NaN();
	const double infinity = std::numeric_limits<double>::infinity();

	for (const double invalid : {-1.0, 0.0, infinity, nan})
	{
		DavidsonEigenSolverOptions<double> options{1};
		options.ResidualNormTolerance = invalid;
		CHECK_THROWS_AS(solver.Compute(linearOperator, basis, options), SecUtility::InvalidArgumentException);

		options = DavidsonEigenSolverOptions<double>{1};
		options.EigenvalueChangeTolerance = invalid;
		CHECK_THROWS_AS(solver.Compute(linearOperator, basis, options), SecUtility::InvalidArgumentException);

		options = DavidsonEigenSolverOptions<double>{1};
		options.PreconditionerDenominatorFloor = invalid;
		CHECK_THROWS_AS(solver.Compute(linearOperator, basis, options), SecUtility::InvalidArgumentException);

		options = DavidsonEigenSolverOptions<double>{1};
		options.LinearDependenceTolerance = invalid;
		CHECK_THROWS_AS(solver.Compute(linearOperator, basis, options), SecUtility::InvalidArgumentException);
	}
}


TEST_CASE("A failed repeated Davidson compute clears all previous observable state", "[Math][Davidson]")
{
	auto linearOperator = IdentityOperator<double>(3, 3);
	DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	CHECK(solver.Compute(linearOperator, Eigen::MatrixXd::Identity(3, 1), DavidsonEigenSolverOptions<double>{1})
	      == DavidsonEigenSolverStatus::IterationLimitReached);

	linearOperator.DiagonalOverride = Eigen::VectorXd::Ones(2);
	CHECK_THROWS_AS(solver.Compute(linearOperator, Eigen::MatrixXd::Identity(3, 1),
	                               DavidsonEigenSolverOptions<double>{1}),
	                SecUtility::InvalidArgumentException);
	CHECK(solver.Status() == DavidsonEigenSolverStatus::NotComputed);
	CHECK(solver.Eigenvalues().size() == 0);
	CHECK(solver.Eigenvectors().size() == 0);
	CHECK(solver.ResidualNorms().size() == 0);
	CHECK(solver.BasisVectors().size() == 0);
	CHECK(solver.BasisVectorImages().size() == 0);
	CHECK(solver.ReducedMatrix().size() == 0);
	CHECK(solver.ReducedEigenvectors().size() == 0);
	CHECK(solver.Statistics().CompletedIterationCount == 0);
}
