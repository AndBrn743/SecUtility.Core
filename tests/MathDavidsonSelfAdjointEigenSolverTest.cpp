// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#include <SecUtility/Math/DavidsonSelfAdjointEigenSolver.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>


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


	struct NonFiniteImageOperator
	{
		using Scalar = double;
		Eigen::Index rows() const { return 3; }
		Eigen::Index cols() const { return 3; }
		Eigen::Vector3d Diagonal() const { return Eigen::Vector3d{1, 2, 3}; }
		Eigen::MatrixXd ApplyOn(const Eigen::MatrixXd& vectors) const
		{
			return Eigen::MatrixXd::Constant(3, vectors.cols(), std::numeric_limits<double>::quiet_NaN());
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
	CHECK(options.AdditionalRestartRitzVectorCount == 0);
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


TEMPLATE_TEST_CASE("A Davidson solver publishes aligned exact initial-space Ritz results",
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

	CHECK(status == DavidsonEigenSolverStatus::Converged);
	CHECK(solver.Status() == status);
	REQUIRE(solver.Eigenvalues().size() == 2);
	CHECK(solver.Eigenvalues().isApprox(Eigen::Vector2d::Ones()));
	CHECK(solver.Eigenvectors().rows() == 4);
	CHECK(solver.Eigenvectors().cols() == 2);
	CHECK(solver.ResidualNorms().maxCoeff() < 1e-14);
	CHECK(solver.BasisVectors().rows() == 4);
	CHECK(solver.BasisVectors().cols() == 2);
	CHECK((solver.BasisVectors().adjoint() * solver.BasisVectors())
	              .isApprox(Eigen::MatrixX<TestType>::Identity(2, 2)));
	CHECK(solver.BasisVectorImages().rows() == 4);
	CHECK(solver.BasisVectorImages().cols() == 2);
	CHECK(solver.BasisVectorImages().isApprox(solver.BasisVectors()));
	CHECK(solver.ReducedMatrix().isApprox(Eigen::MatrixX<TestType>::Identity(2, 2)));
	CHECK(solver.ReducedEigenvectors().rows() == 2);
	CHECK(solver.ReducedEigenvectors().cols() == 2);
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

	CHECK(solver.Compute(linearOperator, initialBasis, options) == DavidsonEigenSolverStatus::Converged);

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
		CHECK(solver.Compute(linearOperator, basis, options) == DavidsonEigenSolverStatus::Converged);
		CHECK(solver.BasisVectors().cols() == 2);
	}

	SECTION("A pivot below the threshold is removed")
	{
		basis(1, 1) = 1e-9;
		CHECK(solver.Compute(linearOperator, basis, options) == DavidsonEigenSolverStatus::Converged);
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
	options.MaximumIterationCount = 1;

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

	CHECK(blockSolver.Compute(blockOperator, basis, options) == DavidsonEigenSolverStatus::Converged);
	CHECK(vectorSolver.Compute(vectorOperator, basis, options) == DavidsonEigenSolverStatus::Converged);

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


TEMPLATE_TEST_CASE("Davidson Rayleigh-Ritz results match a dense reference in a rotated full space",
	               "[Math][Davidson]", double, (std::complex<double>))
{
	using RealScalar = typename Eigen::NumTraits<TestType>::Real;
	Eigen::MatrixX<TestType> matrix = Eigen::MatrixX<TestType>::Zero(4, 4);
	matrix.diagonal() << TestType{-2}, TestType{0.5}, TestType{3}, TestType{8};
	const TestType coupling = []
	{
		if constexpr (Eigen::NumTraits<TestType>::IsComplex)
		{
			return TestType{0.2, -0.35};
		}
		return TestType{0.2};
	}();
	matrix(0, 2) = coupling;
	matrix(2, 0) = Eigen::numext::conj(coupling);
	const DenseOperator<TestType> linearOperator{matrix, {}};
	Eigen::MatrixX<TestType> seed = Eigen::MatrixX<TestType>::Random(4, 4);
	const Eigen::HouseholderQR<Eigen::MatrixX<TestType>> qr(seed);
	const Eigen::MatrixX<TestType> initialBasis =
	        qr.householderQ() * Eigen::MatrixX<TestType>::Identity(4, 4);
	DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	DavidsonEigenSolverOptions<RealScalar> options{3};
	options.InitialSubspaceDimension = 4;
	options.ResidualNormTolerance = 1e-10;
	const Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<TestType>> reference(matrix);

	REQUIRE(solver.Compute(linearOperator, initialBasis, options) == DavidsonEigenSolverStatus::Converged);
	CHECK(solver.Eigenvalues().isApprox(reference.eigenvalues().head(3), 1e-10));
	CHECK(solver.ResidualNorms().maxCoeff() < 1e-10);
	CHECK((matrix * solver.Eigenvectors())
	              .isApprox(solver.Eigenvectors() * solver.Eigenvalues().template cast<TestType>().asDiagonal(), 1e-10));
	CHECK((solver.Eigenvectors().adjoint() * solver.Eigenvectors())
	              .isApprox(Eigen::MatrixX<TestType>::Identity(3, 3), 1e-10));
	CHECK(solver.Statistics().CompletedIterationCount == 1);
}


TEMPLATE_TEST_CASE("Davidson publishes reduced Ritz approximations from a partial non-invariant space",
	               "[Math][Davidson]", double, (std::complex<double>))
{
	Eigen::MatrixX<TestType> matrix = Eigen::MatrixX<TestType>::Zero(4, 4);
	matrix.diagonal() << TestType{1}, TestType{2}, TestType{4}, TestType{7};
	matrix(1, 2) = TestType{0.5};
	matrix(2, 1) = TestType{0.5};
	const DenseOperator<TestType> linearOperator{matrix, {}};
	const Eigen::MatrixX<TestType> initialBasis = Eigen::MatrixX<TestType>::Identity(4, 2);
	DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	DavidsonEigenSolverOptions<double> options{2};
	options.MaximumIterationCount = 1;

	REQUIRE(solver.Compute(linearOperator, initialBasis, options)
	        == DavidsonEigenSolverStatus::IterationLimitReached);
	REQUIRE(solver.Eigenvalues().size() == 2);
	const Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<TestType>> reducedReference(solver.ReducedMatrix());
	CHECK(solver.Eigenvalues().isApprox(reducedReference.eigenvalues(), 1e-12));
	CHECK(solver.Eigenvectors().cols() == solver.Eigenvalues().size());
	CHECK(solver.ResidualNorms().size() == solver.Eigenvalues().size());
	CHECK(solver.ResidualNorms().maxCoeff() > options.ResidualNormTolerance);
	CHECK(solver.ReducedEigenvectors().rows() == solver.ReducedMatrix().rows());
	CHECK(solver.ReducedEigenvectors().cols() == solver.Eigenvalues().size());
}


TEST_CASE("Davidson retains every available Ritz pair when fewer than the requested roots exist", "[Math][Davidson]")
{
	const auto linearOperator = IdentityOperator<double>(4, 4);
	DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	DavidsonEigenSolverOptions<double> options{3};
	options.InitialSubspaceDimension = 3;
	options.MaximumIterationCount = 1;

	REQUIRE(solver.Compute(linearOperator, Eigen::MatrixXd::Identity(4, 2), options)
	        == DavidsonEigenSolverStatus::IterationLimitReached);
	CHECK(solver.Eigenvalues().size() == 2);
	CHECK(solver.Eigenvectors().cols() == 2);
	CHECK(solver.ResidualNorms().size() == 2);
	CHECK(solver.ResidualNorms().maxCoeff() < 1e-14);
}


TEST_CASE("Davidson reports a numerical failure without publishing invalid Ritz results", "[Math][Davidson]")
{
	DavidsonSelfAdjointEigenSolver<NonFiniteImageOperator> solver;
	const auto status = solver.Compute(
	        NonFiniteImageOperator{}, Eigen::MatrixXd::Identity(3, 2), DavidsonEigenSolverOptions<double>{2});

	CHECK(status == DavidsonEigenSolverStatus::NumericalFailure);
	CHECK(solver.Status() == status);
	CHECK(solver.Statistics().CompletedIterationCount == 1);
	CHECK(solver.Eigenvalues().size() == 0);
	CHECK(solver.Eigenvectors().cols() == 0);
	CHECK(solver.ResidualNorms().size() == 0);
	CHECK(solver.ReducedEigenvectors().size() == 0);
}


TEST_CASE("Davidson signed denominator regularization preserves sign including signed zero", "[Math][Davidson]")
{
	using Detail::Davidson::RegularizeSignedDenominator;
	CHECK(RegularizeSignedDenominator(2.0, 0.5) == 2.0);
	CHECK(RegularizeSignedDenominator(-2.0, 0.5) == -2.0);
	CHECK(RegularizeSignedDenominator(0.1, 0.5) == 0.5);
	CHECK(RegularizeSignedDenominator(-0.1, 0.5) == -0.5);
	CHECK(RegularizeSignedDenominator(0.0, 0.5) == 0.5);
	CHECK(RegularizeSignedDenominator(-0.0, 0.5) == -0.5);
	CHECK_FALSE(std::signbit(RegularizeSignedDenominator(0.0, 0.5)));
	CHECK(std::signbit(RegularizeSignedDenominator(-0.0, 0.5)));
	CHECK(RegularizeSignedDenominator(0.5, 0.5) == 0.5);
	CHECK(RegularizeSignedDenominator(-0.5, 0.5) == -0.5);
}


TEMPLATE_TEST_CASE("Diagonal Davidson corrections follow r divided by theta minus D",
	               "[Math][Davidson]", double, (std::complex<double>))
{
	using namespace Detail::Davidson;
	Eigen::MatrixX<TestType> residuals(4, 2);
	residuals.col(0) << TestType{2}, TestType{4}, TestType{6}, TestType{8};
	residuals.col(1) << TestType{1}, TestType{3}, TestType{5}, TestType{7};
	const Eigen::Vector2d ritzValues{3, 10};
	const Eigen::Vector4d diagonal{1, 3, 5, 7};
	const Eigen::ArrayX<Int8> rootConvergenceIndicators{{0, 1}};
	DavidsonEigenSolverStatistics statistics;
	const DiagonalCorrectionContext<TestType> context{
	        residuals, ritzValues, diagonal, rootConvergenceIndicators, 0.5};

	const auto candidates = GenerateDiagonalCorrectionCandidates(context, statistics);

	REQUIRE(candidates.Vectors.rows() == 4);
	REQUIRE(candidates.Vectors.cols() == 1);
	CHECK(candidates.Vectors(0, 0) == residuals(0, 0) / 2.0);
	CHECK(candidates.Vectors(1, 0) == residuals(1, 0) / 0.5);
	CHECK(candidates.Vectors(2, 0) == residuals(2, 0) / -2.0);
	CHECK(candidates.Vectors(3, 0) == residuals(3, 0) / -4.0);
	REQUIRE(candidates.SourceRootIndices.size() == 1);
	CHECK(candidates.SourceRootIndices[0] == 0);
	CHECK(statistics.GeneratedCorrectionVectorCount == 1);
	CHECK(statistics.RetainedCorrectionVectorCount == 0);
	CHECK(candidates.Vectors.allFinite());
}


TEST_CASE("Diagonal Davidson correction generation skips every converged root", "[Math][Davidson]")
{
	using namespace Detail::Davidson;
	const Eigen::MatrixXd residuals = Eigen::MatrixXd::Ones(3, 2);
	const Eigen::Vector2d ritzValues{1, 2};
	const Eigen::Vector3d diagonal{0, 1, 2};
	const Eigen::ArrayX<Int8> rootConvergenceIndicators{{1, 1}};
	DavidsonEigenSolverStatistics statistics;

	const auto candidates = GenerateDiagonalCorrectionCandidates(
	        DiagonalCorrectionContext<double>{residuals, ritzValues, diagonal, rootConvergenceIndicators, 1e-3}, statistics);

	CHECK(candidates.Vectors.rows() == 3);
	CHECK(candidates.Vectors.cols() == 0);
	CHECK(candidates.SourceRootIndices.empty());
	CHECK(statistics.GeneratedCorrectionVectorCount == 0);
}


TEST_CASE("Complex Davidson corrections preserve the residual phase", "[Math][Davidson]")
{
	using namespace Detail::Davidson;
	using Scalar = std::complex<double>;
	Eigen::MatrixXcd residuals(2, 1);
	residuals << Scalar{1, 2}, Scalar{-3, 1};
	const Eigen::VectorXd ritzValues = Eigen::VectorXd::Constant(1, 2.0);
	const Eigen::Vector2d diagonal{0, 4};
	const Eigen::ArrayX<Int8> rootConvergenceIndicators = Eigen::ArrayX<Int8>::Zero(1);
	DavidsonEigenSolverStatistics statistics;
	const auto original = GenerateDiagonalCorrectionCandidates(
	        DiagonalCorrectionContext<Scalar>{residuals, ritzValues, diagonal, rootConvergenceIndicators, 1e-3}, statistics);
	const Scalar phase{0, 1};
	residuals *= phase;
	const auto rotated = GenerateDiagonalCorrectionCandidates(
	        DiagonalCorrectionContext<Scalar>{residuals, ritzValues, diagonal, rootConvergenceIndicators, 1e-3}, statistics);

	CHECK(rotated.Vectors.isApprox(phase * original.Vectors));
	CHECK(statistics.GeneratedCorrectionVectorCount == 2);
}


TEMPLATE_TEST_CASE("Davidson correction filtering projects twice and removes cross-correction dependence",
	               "[Math][Davidson]", double, (std::complex<double>))
{
	using namespace Detail::Davidson;
	const Eigen::MatrixX<TestType> basis = Eigen::MatrixX<TestType>::Identity(4, 1);
	CorrectionCandidates<TestType> candidates{Eigen::MatrixX<TestType>::Zero(4, 3), {10, 11, 12}};
	candidates.Vectors(0, 0) = TestType{1};
	candidates.Vectors(1, 1) = TestType{1};
	candidates.Vectors(1, 2) = TestType{3};
	DavidsonEigenSolverStatistics statistics;

	const auto retained = OrthogonalizeCorrectionCandidates(basis, candidates, 1e-10, statistics);

	REQUIRE(retained.Vectors.cols() == 1);
	CHECK((basis.adjoint() * retained.Vectors).norm() < 1e-14);
	CHECK((retained.Vectors.adjoint() * retained.Vectors)
	              .isApprox(Eigen::MatrixX<TestType>::Identity(1, 1), 1e-14));
	REQUIRE(retained.SourceRootIndices.size() == 1);
	CHECK(retained.SourceRootIndices[0] == 12);
	CHECK(statistics.RetainedCorrectionVectorCount == 1);
}


TEST_CASE("Davidson correction filtering handles zero and near-dependent candidates", "[Math][Davidson]")
{
	using namespace Detail::Davidson;
	const Eigen::MatrixXd basis = Eigen::MatrixXd::Identity(3, 1);
	DavidsonEigenSolverStatistics statistics;

	SECTION("Zero corrections are discarded")
	{
		const CorrectionCandidates<double> candidates{Eigen::MatrixXd::Zero(3, 2), {0, 1}};
		const auto retained = OrthogonalizeCorrectionCandidates(basis, candidates, 1e-8, statistics);
		CHECK(retained.Vectors.cols() == 0);
		CHECK(retained.SourceRootIndices.empty());
		CHECK(statistics.RetainedCorrectionVectorCount == 0);
	}

	SECTION("The relative QR threshold controls a small independent pivot")
	{
		CorrectionCandidates<double> candidates{Eigen::MatrixXd::Zero(3, 2), {0, 1}};
		candidates.Vectors(1, 0) = 1;
		candidates.Vectors(2, 1) = 1e-9;
		const auto retained = OrthogonalizeCorrectionCandidates(basis, candidates, 1e-8, statistics);
		CHECK(retained.Vectors.cols() == 1);
		CHECK(retained.SourceRootIndices == std::vector<Eigen::Index>{0});
		CHECK(statistics.RetainedCorrectionVectorCount == 1);
	}
}


TEMPLATE_TEST_CASE("Basic Davidson iteration converges to the lowest dense-reference roots",
	               "[Math][Davidson]", double, (std::complex<double>))
{
	using RealScalar = typename Eigen::NumTraits<TestType>::Real;
	Eigen::MatrixX<TestType> matrix = Eigen::MatrixX<TestType>::Zero(5, 5);
	matrix.diagonal() << TestType{1}, TestType{1.01}, TestType{2.5}, TestType{4}, TestType{7};
	const TestType coupling = []
	{
		if constexpr (Eigen::NumTraits<TestType>::IsComplex)
		{
			return TestType{0.08, 0.03};
		}
		return TestType{0.08};
	}();
	for (Eigen::Index index = 0; index + 1 < matrix.rows(); index++)
	{
		matrix(index, index + 1) = coupling;
		matrix(index + 1, index) = Eigen::numext::conj(coupling);
	}
	const DenseOperator<TestType> linearOperator{matrix, {}};
	const Eigen::MatrixX<TestType> initialBasis = Eigen::MatrixX<TestType>::Identity(5, 2);
	DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	DavidsonEigenSolverOptions<RealScalar> options{2};
	options.MaximumIterationCount = 20;
	options.ResidualNormTolerance = 1e-10;
	options.PreconditionerDenominatorFloor = 1e-8;
	const Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<TestType>> reference(matrix);

	REQUIRE(solver.Compute(linearOperator, initialBasis, options) == DavidsonEigenSolverStatus::Converged);
	CHECK(solver.Eigenvalues().isApprox(reference.eigenvalues().head(2), 1e-9));
	CHECK(solver.ResidualNorms().maxCoeff() <= options.ResidualNormTolerance);
	CHECK((matrix * solver.Eigenvectors())
	              .isApprox(solver.Eigenvectors() * solver.Eigenvalues().template cast<TestType>().asDiagonal(), 1e-9));
	CHECK(solver.BasisVectorImages().isApprox(matrix * solver.BasisVectors(), 1e-10));
	CHECK(solver.ReducedMatrix().isApprox(solver.BasisVectors().adjoint() * solver.BasisVectorImages(), 1e-10));
	CHECK(solver.Statistics().CompletedIterationCount > 1);
	CHECK(solver.Statistics().GeneratedCorrectionVectorCount
	      >= solver.Statistics().RetainedCorrectionVectorCount);
}


TEMPLATE_TEST_CASE("A controlled two-dimensional Davidson solve has exact iteration and operator statistics",
	               "[Math][Davidson]", double, (std::complex<double>))
{
	Eigen::MatrixX<TestType> matrix(2, 2);
	matrix << TestType{1}, TestType{0.2}, TestType{0.2}, TestType{2};
	const DenseOperator<TestType> linearOperator{matrix, {}};
	const Eigen::MatrixX<TestType> initialBasis = Eigen::MatrixX<TestType>::Identity(2, 1);
	DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	DavidsonEigenSolverOptions<double> options{1};
	options.ResidualNormTolerance = 1e-12;

	REQUIRE(solver.Compute(linearOperator, initialBasis, options) == DavidsonEigenSolverStatus::Converged);
	CHECK(solver.Statistics().CompletedIterationCount == 2);
	CHECK(solver.Statistics().OperatorApplicationCount == 2);
	CHECK(solver.Statistics().MultipliedVectorCount == 2);
	CHECK(solver.Statistics().MaximumSubspaceDimension == 2);
	CHECK(solver.Statistics().GeneratedCorrectionVectorCount == 1);
	CHECK(solver.Statistics().RetainedCorrectionVectorCount == 1);
	CHECK(solver.Statistics().RestartCount == 0);
	CHECK(solver.Statistics().OperatorRefreshCount == 0);
}


TEST_CASE("Basic Davidson iteration accounts for vector-only operator applications", "[Math][Davidson]")
{
	Eigen::Matrix2d matrix;
	matrix << 1, 0.2, 0.2, 2;
	VectorOnlyCountingOperator<double> linearOperator{matrix};
	DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	DavidsonEigenSolverOptions<double> options{1};
	options.ResidualNormTolerance = 1e-12;

	REQUIRE(solver.Compute(linearOperator, Eigen::MatrixXd::Identity(2, 1), options)
	        == DavidsonEigenSolverStatus::Converged);
	CHECK(solver.Statistics().OperatorApplicationCount == 2);
	CHECK(solver.Statistics().MultipliedVectorCount == 2);
	CHECK(linearOperator.ApplicationCount == 2);
}


TEST_CASE("Basic Davidson iteration exposes its latest Ritz result at the iteration limit", "[Math][Davidson]")
{
	Eigen::Matrix3d matrix;
	matrix << 1, 0.2, 0, 0.2, 2, 0.3, 0, 0.3, 4;
	const DenseOperator<double> linearOperator{matrix, {}};
	DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	DavidsonEigenSolverOptions<double> options{1};
	options.MaximumIterationCount = 1;
	options.EigenvalueChangeTolerance = 1e100;

	REQUIRE(solver.Compute(linearOperator, Eigen::MatrixXd::Identity(3, 1), options)
	        == DavidsonEigenSolverStatus::IterationLimitReached);
	CHECK(solver.Statistics().CompletedIterationCount == 1);
	CHECK(solver.Eigenvalues().size() == 1);
	CHECK(solver.Eigenvectors().cols() == 1);
	CHECK(solver.ResidualNorms().size() == 1);
	CHECK(solver.ResidualNorms()[0] > options.ResidualNormTolerance);
	CHECK(solver.Statistics().GeneratedCorrectionVectorCount == 0);
	CHECK(solver.Statistics().RetainedCorrectionVectorCount == 0);
}


TEST_CASE("Basic Davidson iteration reports bounded and dependent expansion exhaustion", "[Math][Davidson]")
{
	SECTION("A correction that cannot fit the configured space is retained only as a diagnostic candidate")
	{
		Eigen::Matrix2d matrix;
		matrix << 1, 0.2, 0.2, 2;
		const DenseOperator<double> linearOperator{matrix, {}};
		DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;
		DavidsonEigenSolverOptions<double> options{1};
		options.MaximumSubspaceDimension = 1;

		REQUIRE(solver.Compute(linearOperator, Eigen::MatrixXd::Identity(2, 1), options)
		        == DavidsonEigenSolverStatus::ExpansionSpaceExhausted);
		CHECK(solver.Statistics().CompletedIterationCount == 1);
		CHECK(solver.Statistics().GeneratedCorrectionVectorCount == 1);
		CHECK(solver.Statistics().RetainedCorrectionVectorCount == 1);
		CHECK(solver.Statistics().OperatorApplicationCount == 1);
		CHECK(solver.BasisVectors().cols() == 1);
		CHECK(solver.Statistics().RestartCount == 1);
	}

	SECTION("Exact residuals cannot manufacture missing roots")
	{
		const auto linearOperator = IdentityOperator<double>(3, 3);
		DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;
		DavidsonEigenSolverOptions<double> options{2};

		REQUIRE(solver.Compute(linearOperator, Eigen::MatrixXd::Identity(3, 1), options)
		        == DavidsonEigenSolverStatus::ExpansionSpaceExhausted);
		CHECK(solver.Eigenvalues().size() == 1);
		CHECK(solver.ResidualNorms().maxCoeff() < 1e-14);
		CHECK(solver.Statistics().GeneratedCorrectionVectorCount == 0);
		CHECK(solver.Statistics().RetainedCorrectionVectorCount == 0);
	}
}


TEST_CASE("Eigenvalue-change tolerance neither overrides nor blocks residual convergence", "[Math][Davidson]")
{
	Eigen::Matrix2d matrix;
	matrix << 1, 0.2, 0.2, 2;
	const DenseOperator<double> linearOperator{matrix, {}};
	DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;

	DavidsonEigenSolverOptions<double> looseOptions{1};
	looseOptions.MaximumIterationCount = 1;
	looseOptions.EigenvalueChangeTolerance = 1e100;
	CHECK(solver.Compute(linearOperator, Eigen::MatrixXd::Identity(2, 1), looseOptions)
	      == DavidsonEigenSolverStatus::IterationLimitReached);

	DavidsonEigenSolverOptions<double> strictOptions{1};
	strictOptions.EigenvalueChangeTolerance = 1e-100;
	CHECK(solver.Compute(linearOperator, Eigen::MatrixXd::Identity(2, 2), strictOptions)
	      == DavidsonEigenSolverStatus::Converged);
}


TEMPLATE_TEST_CASE("Davidson thick restart transforms vectors and images with identical Ritz coefficients",
	               "[Math][Davidson]", double, (std::complex<double>))
{
	using namespace Detail::Davidson;
	Eigen::MatrixX<TestType> matrix = Eigen::MatrixX<TestType>::Zero(5, 5);
	matrix.diagonal() << TestType{1}, TestType{2}, TestType{3}, TestType{5}, TestType{8};
	matrix(1, 3) = TestType{0.2};
	matrix(3, 1) = TestType{0.2};
	const DenseOperator<TestType> linearOperator{matrix, {}};
	const Eigen::MatrixX<TestType> initialBasis = Eigen::MatrixX<TestType>::Random(5, 4);
	DavidsonEigenSolverStatistics statistics;
	const auto subspace = CreateInitialSubspace(linearOperator, initialBasis, 1e-12, statistics);
	const auto analysis = AnalyzeSubspace(subspace, 1);
	const auto restarted = RestartSubspace(subspace, analysis, 1, 2, 4);

	REQUIRE(restarted.Vectors.cols() == 3);
	CHECK(restarted.Images.isApprox(matrix * restarted.Vectors, 1e-11));
	CHECK(restarted.ReducedMatrix.isApprox(restarted.Vectors.adjoint() * restarted.Images, 1e-11));
	CHECK((restarted.Vectors.adjoint() * restarted.Vectors)
	              .isApprox(Eigen::MatrixX<TestType>::Identity(3, 3), 1e-11));
	CHECK(restarted.Vectors.isApprox(
	        subspace.Vectors * analysis.AllReducedEigenvectors.leftCols(3), 1e-11));
}


TEST_CASE("Davidson thick restart retains requested roots before optional Ritz vectors", "[Math][Davidson]")
{
	using namespace Detail::Davidson;
	const auto linearOperator = IdentityOperator<double>(5, 5);
	DavidsonEigenSolverStatistics statistics;
	const auto subspace = CreateInitialSubspace(
	        linearOperator, Eigen::MatrixXd::Identity(5, 4), 1e-12, statistics);
	const auto analysis = AnalyzeSubspace(subspace, 2);

	const auto withHeadroom = RestartSubspace(subspace, analysis, 2, 100, 4);
	CHECK(withHeadroom.Vectors.cols() == 3);
	CHECK((withHeadroom.Vectors.adjoint() * withHeadroom.Vectors)
	              .isApprox(Eigen::MatrixXd::Identity(3, 3), 1e-14));

	const auto rootsOnly = RestartSubspace(subspace, analysis, 2, 100, 2);
	CHECK(rootsOnly.Vectors.cols() == 2);
	CHECK((rootsOnly.Vectors.adjoint() * rootsOnly.Vectors)
	              .isApprox(Eigen::MatrixXd::Identity(2, 2), 1e-14));
}


TEMPLATE_TEST_CASE("Bounded Davidson iteration converges through repeated thick restarts",
	               "[Math][Davidson]", double, (std::complex<double>))
{
	using RealScalar = typename Eigen::NumTraits<TestType>::Real;
	Eigen::MatrixX<TestType> matrix = Eigen::MatrixX<TestType>::Zero(8, 8);
	for (Eigen::Index index = 0; index < matrix.rows(); index++)
	{
		matrix(index, index) = static_cast<RealScalar>(index + 1);
	}
	const TestType coupling = []
	{
		if constexpr (Eigen::NumTraits<TestType>::IsComplex)
		{
			return TestType{0.08, 0.02};
		}
		return TestType{0.08};
	}();
	for (Eigen::Index index = 0; index + 1 < matrix.rows(); index++)
	{
		matrix(index, index + 1) = coupling;
		matrix(index + 1, index) = Eigen::numext::conj(coupling);
	}
	const DenseOperator<TestType> linearOperator{matrix, {}};
	const Eigen::MatrixX<TestType> initialBasis = Eigen::MatrixX<TestType>::Identity(8, 1);
	DavidsonEigenSolverOptions<RealScalar> boundedOptions{1};
	boundedOptions.MaximumIterationCount = 200;
	boundedOptions.MaximumSubspaceDimension = 3;
	boundedOptions.ResidualNormTolerance = 1e-9;
	DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> boundedSolver;
	DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> unboundedSolver;
	DavidsonEigenSolverOptions<RealScalar> unboundedOptions = boundedOptions;
	unboundedOptions.MaximumSubspaceDimension = 0;

	REQUIRE(boundedSolver.Compute(linearOperator, initialBasis, boundedOptions)
	        == DavidsonEigenSolverStatus::Converged);
	REQUIRE(unboundedSolver.Compute(linearOperator, initialBasis, unboundedOptions)
	        == DavidsonEigenSolverStatus::Converged);
	CHECK(boundedSolver.Eigenvalues().isApprox(unboundedSolver.Eigenvalues(), 1e-8));
	CHECK(boundedSolver.ResidualNorms().maxCoeff() <= boundedOptions.ResidualNormTolerance);
	CHECK(boundedSolver.Statistics().RestartCount >= 2);
	CHECK(boundedSolver.Statistics().MaximumSubspaceDimension <= boundedOptions.MaximumSubspaceDimension);
	CHECK(boundedSolver.BasisVectors().cols() <= boundedOptions.MaximumSubspaceDimension);
	CHECK(boundedSolver.BasisVectorImages().isApprox(matrix * boundedSolver.BasisVectors(), 1e-9));
	CHECK(boundedSolver.ReducedMatrix().isApprox(
	        boundedSolver.BasisVectors().adjoint() * boundedSolver.BasisVectorImages(), 1e-9));
}


TEMPLATE_TEST_CASE("Additional restart Ritz vectors preserve convergence and the configured dimension bound",
	               "[Math][Davidson]", double, (std::complex<double>))
{
	Eigen::MatrixX<TestType> matrix = Eigen::MatrixX<TestType>::Zero(6, 6);
	for (Eigen::Index index = 0; index < matrix.rows(); index++)
	{
		matrix(index, index) = static_cast<double>(index + 1);
	}
	for (Eigen::Index index = 0; index + 1 < matrix.rows(); index++)
	{
		matrix(index, index + 1) = TestType{0.1};
		matrix(index + 1, index) = TestType{0.1};
	}
	const DenseOperator<TestType> linearOperator{matrix, {}};
	DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	DavidsonEigenSolverOptions<double> options{1};
	options.MaximumIterationCount = 200;
	options.MaximumSubspaceDimension = 4;
	options.AdditionalRestartRitzVectorCount = 2;
	options.ResidualNormTolerance = 1e-9;

	REQUIRE(solver.Compute(linearOperator, Eigen::MatrixX<TestType>::Identity(6, 1), options)
	        == DavidsonEigenSolverStatus::Converged);
	CHECK(solver.Statistics().RestartCount > 0);
	CHECK(solver.Statistics().MaximumSubspaceDimension <= 4);
	CHECK(solver.ResidualNorms().maxCoeff() <= options.ResidualNormTolerance);
}


TEMPLATE_TEST_CASE("The explicit diagonal Davidson strategy matches the default solver path",
	               "[Math][Davidson]", double, (std::complex<double>))
{
	Eigen::MatrixX<TestType> matrix(3, 3);
	matrix << TestType{1}, TestType{0.2}, TestType{0}, TestType{0.2}, TestType{2}, TestType{0.3}, TestType{0},
	        TestType{0.3}, TestType{4};
	const DenseOperator<TestType> linearOperator{matrix, {}};
	const Eigen::MatrixX<TestType> basis = Eigen::MatrixX<TestType>::Identity(3, 1);
	DavidsonEigenSolverOptions<double> options{1};
	options.ResidualNormTolerance = 1e-10;
	DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> defaultSolver;
	DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> explicitSolver;

	REQUIRE(defaultSolver.Compute(linearOperator, basis, options) == DavidsonEigenSolverStatus::Converged);
	REQUIRE(explicitSolver.Compute(linearOperator, basis, options, DiagonalDavidsonCorrection{})
	        == DavidsonEigenSolverStatus::Converged);
	CHECK(explicitSolver.Eigenvalues().isApprox(defaultSolver.Eigenvalues(), 1e-12));
	CHECK(explicitSolver.ResidualNorms().isApprox(defaultSolver.ResidualNorms(), 1e-12));
	CHECK(explicitSolver.Statistics().GeneratedCorrectionVectorCount
	      == defaultSolver.Statistics().GeneratedCorrectionVectorCount);
}


TEMPLATE_TEST_CASE("A stateful Davidson correction strategy receives an immutable complete context",
	               "[Math][Davidson]", double, (std::complex<double>))
{
	Eigen::MatrixX<TestType> matrix(2, 2);
	matrix << TestType{1}, TestType{0.2}, TestType{0.2}, TestType{2};
	const DenseOperator<TestType> linearOperator{matrix, {}};
	DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	DavidsonEigenSolverOptions<double> options{1};
	Eigen::Index callCount = 0;
	auto strategy = [&callCount, &matrix](const DavidsonCorrectionContext<TestType>& context)
	{
		callCount++;
		CHECK(context.RitzVectors.rows() == matrix.rows());
		CHECK(context.RitzVectors.cols() == context.RitzValues.size());
		CHECK(context.Residuals.rows() == matrix.rows());
		CHECK(context.Residuals.cols() == context.RitzValues.size());
		CHECK(context.OperatorDiagonal.isApprox(matrix.diagonal().real()));
		CHECK(context.RootConvergenceIndicators.size() == context.RitzValues.size());
		CHECK((context.RootConvergenceIndicators == 0).all());
		CHECK(context.BasisVectors.rows() == matrix.rows());
		CHECK(context.DenominatorFloor > 0);
		return DiagonalDavidsonCorrection{}(context);
	};

	REQUIRE(solver.Compute(linearOperator, Eigen::MatrixX<TestType>::Identity(2, 1), options, strategy)
	        == DavidsonEigenSolverStatus::Converged);
	CHECK(callCount == 1);
}


TEST_CASE("A temporary Davidson correction strategy remains alive for the whole compute call", "[Math][Davidson]")
{
	Eigen::Matrix2d matrix;
	matrix << 1, 0.2, 0.2, 2;
	const DenseOperator<double> linearOperator{matrix, {}};
	DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	const auto callCount = std::make_shared<Eigen::Index>(0);

	struct Strategy
	{
		std::shared_ptr<Eigen::Index> CallCount;
		DavidsonCorrectionCandidates<double> operator()(const DavidsonCorrectionContext<double>& context)
		{
			++*CallCount;
			return DiagonalDavidsonCorrection{}(context);
		}
	};

	REQUIRE(solver.Compute(linearOperator,
	                       Eigen::MatrixXd::Identity(2, 1),
	                       DavidsonEigenSolverOptions<double>{1},
	                       Strategy{callCount}) == DavidsonEigenSolverStatus::Converged);
	CHECK(*callCount == 1);
}


TEMPLATE_TEST_CASE("Olsen Davidson correction follows the reference correction formula",
	               "[Math][Davidson]", double, (std::complex<double>))
{
	Eigen::MatrixX<TestType> ritzVectors = Eigen::MatrixX<TestType>::Zero(3, 1);
	ritzVectors.col(0) << TestType{0.5}, TestType{0.5}, TestType{0.7071067811865475};
	Eigen::MatrixX<TestType> residuals(3, 1);
	residuals.col(0) << TestType{1}, TestType{-2}, TestType{0.5};
	const Eigen::VectorXd ritzValues = Eigen::VectorXd::Constant(1, 2.0);
	const Eigen::Vector3d diagonal{0, 3, 5};
	const Eigen::ArrayX<Int8> indicators{{0}};
	const Eigen::MatrixX<TestType> basis = ritzVectors;
	const DavidsonCorrectionContext<TestType> context{
	        ritzVectors, residuals, ritzValues, diagonal, indicators, basis, 1e-6};
	Eigen::VectorX<TestType> preconditionedResidual(3);
	Eigen::VectorX<TestType> preconditionedRitzVector(3);
	for (Eigen::Index index = 0; index < 3; index++)
	{
		const double denominator = ritzValues[0] - diagonal[index];
		preconditionedResidual[index] = residuals(index, 0) / denominator;
		preconditionedRitzVector[index] = ritzVectors(index, 0) / denominator;
	}
	const TestType epsilon = ritzVectors.col(0).dot(preconditionedResidual)
	                         / ritzVectors.col(0).dot(preconditionedRitzVector);
	const Eigen::VectorX<TestType> expected = preconditionedResidual + epsilon * ritzVectors.col(0);

	const auto candidates = OlsenDavidsonCorrection{}(context);

	REQUIRE(candidates.Vectors.cols() == 1);
	CHECK(candidates.Vectors.col(0).isApprox(expected, 1e-12));
	CHECK(candidates.SourceRootIndices == std::vector<Eigen::Index>{0});
	const TestType phase = []
	{
		if constexpr (Eigen::NumTraits<TestType>::IsComplex)
		{
			return TestType{0, 1};
		}
		return TestType{-1};
	}();
	const Eigen::MatrixX<TestType> rotatedRitzVectors = phase * ritzVectors;
	const Eigen::MatrixX<TestType> rotatedResiduals = phase * residuals;
	const DavidsonCorrectionContext<TestType> rotatedContext{rotatedRitzVectors,
	                                                         rotatedResiduals,
	                                                         ritzValues,
	                                                         diagonal,
	                                                         indicators,
	                                                         basis,
	                                                         1e-6};
	const auto rotatedCandidates = OlsenDavidsonCorrection{}(rotatedContext);
	CHECK(rotatedCandidates.Vectors.isApprox(phase * candidates.Vectors, 1e-12));
}


TEST_CASE("Olsen Davidson correction regularizes a vanishing Olsen denominator", "[Math][Davidson]")
{
	const double inverseSqrtTwo = 1 / std::sqrt(2.0);
	Eigen::MatrixXd ritzVectors(2, 1);
	ritzVectors << inverseSqrtTwo, inverseSqrtTwo;
	Eigen::MatrixXd residuals(2, 1);
	residuals << 1, -1;
	const Eigen::VectorXd ritzValues = Eigen::VectorXd::Zero(1);
	const Eigen::Vector2d diagonal{-1, 1};
	const Eigen::ArrayX<Int8> indicators{{0}};
	const DavidsonCorrectionContext<double> context{
	        ritzVectors, residuals, ritzValues, diagonal, indicators, ritzVectors, 1e-6};

	const auto candidates = OlsenDavidsonCorrection{}(context);

	REQUIRE(candidates.Vectors.cols() == 1);
	CHECK(candidates.Vectors.allFinite());
}


TEMPLATE_TEST_CASE("Olsen Davidson correction converges end to end",
	               "[Math][Davidson]", double, (std::complex<double>))
{
	Eigen::MatrixX<TestType> matrix(3, 3);
	matrix << TestType{1}, TestType{0.2}, TestType{0}, TestType{0.2}, TestType{2}, TestType{0.3}, TestType{0},
	        TestType{0.3}, TestType{4};
	const DenseOperator<TestType> linearOperator{matrix, {}};
	DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	DavidsonEigenSolverOptions<double> options{1};
	options.ResidualNormTolerance = 1e-10;
	const Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<TestType>> reference(matrix);

	REQUIRE(solver.Compute(
	                linearOperator, Eigen::MatrixX<TestType>::Identity(3, 1), options, OlsenDavidsonCorrection{})
	        == DavidsonEigenSolverStatus::Converged);
	CHECK(solver.Eigenvalues().isApprox(reference.eigenvalues().head(1), 1e-9));
	CHECK(solver.ResidualNorms().maxCoeff() <= options.ResidualNormTolerance);
}


TEST_CASE("Davidson rejects malformed custom correction results", "[Math][Davidson]")
{
	Eigen::Matrix2d matrix;
	matrix << 1, 0.2, 0.2, 2;
	const DenseOperator<double> linearOperator{matrix, {}};
	const Eigen::MatrixXd basis = Eigen::MatrixXd::Identity(2, 1);
	DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;

	SECTION("Row count")
	{
		auto strategy = [](const DavidsonCorrectionContext<double>&)
		{ return DavidsonCorrectionCandidates<double>{Eigen::MatrixXd::Zero(1, 1), {0}}; };
		CHECK_THROWS_AS(solver.Compute(linearOperator, basis, DavidsonEigenSolverOptions<double>{1}, strategy),
		                SecUtility::InvalidArgumentException);
	}
	SECTION("Source count")
	{
		auto strategy = [](const DavidsonCorrectionContext<double>&)
		{ return DavidsonCorrectionCandidates<double>{Eigen::MatrixXd::Zero(2, 1), {}}; };
		CHECK_THROWS_AS(solver.Compute(linearOperator, basis, DavidsonEigenSolverOptions<double>{1}, strategy),
		                SecUtility::InvalidArgumentException);
	}
	SECTION("Source index")
	{
		auto strategy = [](const DavidsonCorrectionContext<double>&)
		{ return DavidsonCorrectionCandidates<double>{Eigen::MatrixXd::Zero(2, 1), {1}}; };
		CHECK_THROWS_AS(solver.Compute(linearOperator, basis, DavidsonEigenSolverOptions<double>{1}, strategy),
		                SecUtility::InvalidArgumentException);
	}
	SECTION("Finite values")
	{
		auto strategy = [](const DavidsonCorrectionContext<double>&)
		{
			return DavidsonCorrectionCandidates<double>{
			        Eigen::MatrixXd::Constant(2, 1, std::numeric_limits<double>::quiet_NaN()), {0}};
		};
		CHECK_THROWS_AS(solver.Compute(linearOperator, basis, DavidsonEigenSolverOptions<double>{1}, strategy),
		                SecUtility::InvalidArgumentException);
	}
}


TEST_CASE("Davidson propagates custom correction exceptions and handles valid zero output", "[Math][Davidson]")
{
	Eigen::Matrix2d matrix;
	matrix << 1, 0.2, 0.2, 2;
	const DenseOperator<double> linearOperator{matrix, {}};
	const Eigen::MatrixXd basis = Eigen::MatrixXd::Identity(2, 1);
	DavidsonSelfAdjointEigenSolver<decltype(linearOperator)> solver;

	auto throwing = [](const DavidsonCorrectionContext<double>&) -> DavidsonCorrectionCandidates<double>
	{ throw std::runtime_error("custom correction failure"); };
	CHECK_THROWS_AS(solver.Compute(linearOperator, basis, DavidsonEigenSolverOptions<double>{1}, throwing),
	                std::runtime_error);

	auto zero = [](const DavidsonCorrectionContext<double>& context)
	{
		return DavidsonCorrectionCandidates<double>{Eigen::MatrixXd::Zero(context.Residuals.rows(), 1), {0}};
	};
	CHECK(solver.Compute(linearOperator, basis, DavidsonEigenSolverOptions<double>{1}, zero)
	      == DavidsonEigenSolverStatus::ExpansionSpaceExhausted);
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

	options = DavidsonEigenSolverOptions<double>{2};
	options.AdditionalRestartRitzVectorCount = -1;
	CHECK_THROWS_AS(solver.Compute(linearOperator, basis, options), SecUtility::InvalidArgumentException);

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
	      == DavidsonEigenSolverStatus::Converged);

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
