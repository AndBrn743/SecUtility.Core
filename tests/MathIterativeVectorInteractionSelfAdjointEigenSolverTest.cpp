// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#include <SecUtility/Math/IterativeVectorInteractionSelfAdjointEigenSolver.hpp>
#include <SecUtility/Math/MatrixFreeLinearOperator.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <limits>
#include <type_traits>


namespace
{
	template <typename T>
	struct DenseSelfAdjointLinearOperator
	{
		using Scalar = T;
		using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

		Eigen::MatrixX<Scalar> Matrix;

		[[nodiscard]] Eigen::Index rows() const noexcept
		{
			return Matrix.rows();
		}

		[[nodiscard]] Eigen::Index cols() const noexcept
		{
			return Matrix.cols();
		}

		[[nodiscard]] Eigen::VectorX<RealScalar> Diagonal() const
		{
			return Matrix.diagonal().real();
		}

		template <typename Derived>
		[[nodiscard]] auto ApplyOn(const Eigen::MatrixBase<Derived>& vectors) const
		{
			return (Matrix * vectors).eval();
		}
	};


	template <typename Scalar>
	DenseSelfAdjointLinearOperator<Scalar> IdentityOperator(const Eigen::Index dimension)
	{
		return {Eigen::MatrixX<Scalar>::Identity(dimension, dimension)};
	}


	template <typename T>
	struct VectorOnlySelfAdjointLinearOperator
	{
		using Scalar = T;
		using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

		Eigen::MatrixX<Scalar> Matrix;

		[[nodiscard]] Eigen::Index rows() const noexcept { return Matrix.rows(); }
		[[nodiscard]] Eigen::Index cols() const noexcept { return Matrix.cols(); }
		[[nodiscard]] Eigen::VectorX<RealScalar> Diagonal() const { return Matrix.diagonal().real(); }
		template <typename Derived>
			requires(Derived::ColsAtCompileTime == 1)
		[[nodiscard]] Eigen::VectorX<Scalar> ApplyOn(const Eigen::MatrixBase<Derived>& vector) const
		{
			return Matrix * vector;
		}
	};


	struct OperatorWithoutDiagonal
	{
		using Scalar = double;

		[[nodiscard]] Eigen::Index rows() const noexcept { return 3; }
		[[nodiscard]] Eigen::Index cols() const noexcept { return 3; }
		[[nodiscard]] Eigen::VectorXd ApplyOn(const Eigen::VectorXd& vector) const { return vector; }
	};


	struct ComplexDiagonalOperator
	{
		using Scalar = std::complex<double>;

		[[nodiscard]] Eigen::Index rows() const noexcept { return 3; }
		[[nodiscard]] Eigen::Index cols() const noexcept { return 3; }
		[[nodiscard]] Eigen::VectorXcd Diagonal() const { return Eigen::VectorXcd::Ones(3); }
		[[nodiscard]] Eigen::VectorXcd ApplyOn(const Eigen::VectorXcd& vector) const { return vector; }
	};


	struct OperatorWithoutScalar
	{
		[[nodiscard]] Eigen::Index rows() const noexcept { return 3; }
		[[nodiscard]] Eigen::Index cols() const noexcept { return 3; }
		[[nodiscard]] Eigen::VectorXd Diagonal() const { return Eigen::VectorXd::Ones(3); }
		[[nodiscard]] Eigen::VectorXd ApplyOn(const Eigen::VectorXd& vector) const { return vector; }
	};
}


using namespace SecUtility::Math;


static_assert(SelfAdjointLinearOperator<DenseSelfAdjointLinearOperator<double>>);
static_assert(SelfAdjointLinearOperator<DenseSelfAdjointLinearOperator<std::complex<double>>>);
static_assert(BlockSelfAdjointLinearOperator<DenseSelfAdjointLinearOperator<double>>);
static_assert(BlockSelfAdjointLinearOperator<DenseSelfAdjointLinearOperator<std::complex<double>>>);
static_assert(SelfAdjointLinearOperator<VectorOnlySelfAdjointLinearOperator<double>>);
static_assert(!BlockSelfAdjointLinearOperator<VectorOnlySelfAdjointLinearOperator<double>>);
static_assert(!SelfAdjointLinearOperator<OperatorWithoutDiagonal>);
static_assert(!SelfAdjointLinearOperator<ComplexDiagonalOperator>);
static_assert(!SelfAdjointLinearOperator<OperatorWithoutScalar>);
static_assert(!SelfAdjointLinearOperator<DenseSelfAdjointLinearOperator<int>>);


TEMPLATE_TEST_CASE("iVI public result contract", "[Math][iVI]", double, (std::complex<double>))
{
	using Operator = DenseSelfAdjointLinearOperator<TestType>;
	using Solver = IterativeVectorInteractionSelfAdjointEigenSolver<Operator>;
	using RealScalar = typename Eigen::NumTraits<TestType>::Real;

	Solver solver;
	STATIC_CHECK(std::is_same_v<typename Solver::Scalar, TestType>);
	STATIC_CHECK(std::is_same_v<typename Solver::RealScalar, RealScalar>);
	CHECK(solver.Status() == InteriorEigenSolverStatus::NotComputed);
	CHECK(solver.Eigenvalues().size() == 0);
	CHECK(solver.Eigenvectors().size() == 0);
	CHECK(solver.ResidualNorms().size() == 0);
	CHECK(solver.Statistics().CompletedIterationCount == 0);
	CHECK(solver.Statistics().MultipliedVectorCount == 0);
	CHECK(solver.Statistics().OperatorApplicationCount == 0);
	CHECK(solver.Statistics().MaximumExpansionSpaceSize == 0);
	CHECK(solver.Statistics().ExplicitImageRecalculationCount == 0);
	CHECK(solver.Statistics().GeneralizedSolveCount == 0);
}


TEMPLATE_TEST_CASE("iVI operator contract and input validation", "[Math][iVI]", double, (std::complex<double>))
{
	using RealScalar = typename Eigen::NumTraits<TestType>::Real;
	const auto operatorMatrix = IdentityOperator<TestType>(4);
	const EigenvalueInterval<RealScalar> interval{-1, 1};
	InteriorEigenSolverOptions<RealScalar> options;
	options.MaximumEigenpairCount = 2;

	SECTION("A valid problem is accepted")
	{
		CHECK_NOTHROW(ValidateInteriorEigenSolverInput(operatorMatrix, interval, options));
		CHECK(interval.IsContaining(RealScalar{-1}));
		CHECK(interval.IsContaining(RealScalar{1}));
		CHECK_FALSE(interval.IsContaining(RealScalar{2}));
	}

	SECTION("A reversed interval is rejected")
	{
		CHECK_THROWS_AS(
		        ValidateInteriorEigenSolverInput(operatorMatrix, EigenvalueInterval<RealScalar>{1, -1}, options),
		        SecUtility::InvalidArgumentException);
	}

	SECTION("Non-finite interval bounds are rejected")
	{
		const auto infinity = std::numeric_limits<RealScalar>::infinity();
		CHECK_THROWS_AS(
		        ValidateInteriorEigenSolverInput(
		                operatorMatrix, EigenvalueInterval<RealScalar>{-infinity, 1}, options),
		        SecUtility::InvalidArgumentException);
	}

	SECTION("A non-positive eigenpair capacity is rejected")
	{
		options.MaximumEigenpairCount = 0;
		CHECK_THROWS_AS(ValidateInteriorEigenSolverInput(operatorMatrix, interval, options), SecUtility::InvalidArgumentException);
	}

	SECTION("An eigenpair capacity larger than the dimension is rejected")
	{
		options.MaximumEigenpairCount = 5;
		CHECK_THROWS_AS(ValidateInteriorEigenSolverInput(operatorMatrix, interval, options), SecUtility::InvalidArgumentException);
	}

	SECTION("Invalid iteration and retained-vector counts are rejected")
	{
		options.MaximumIterationCount = 0;
		CHECK_THROWS_AS(ValidateInteriorEigenSolverInput(operatorMatrix, interval, options), SecUtility::InvalidArgumentException);

		options.MaximumIterationCount = 1;
		options.AdditionalRitzVectorCount = -1;
		CHECK_THROWS_AS(ValidateInteriorEigenSolverInput(operatorMatrix, interval, options), SecUtility::InvalidArgumentException);
	}

	SECTION("Invalid numerical controls are rejected")
	{
		options.GeneralizedSolveInterval = 0;
		CHECK_THROWS_AS(ValidateInteriorEigenSolverInput(operatorMatrix, interval, options), SecUtility::InvalidArgumentException);

		options.GeneralizedSolveInterval = 1;
		options.ResidualNormTolerance = 0;
		CHECK_THROWS_AS(ValidateInteriorEigenSolverInput(operatorMatrix, interval, options), SecUtility::InvalidArgumentException);

		options.ResidualNormTolerance = std::numeric_limits<RealScalar>::quiet_NaN();
		CHECK_THROWS_AS(ValidateInteriorEigenSolverInput(operatorMatrix, interval, options), SecUtility::InvalidArgumentException);
	}
}


TEST_CASE("iVI validation rejects inconsistent operator dimensions", "[Math][iVI]")
{
	InteriorEigenSolverOptions<double> options;
	options.MaximumEigenpairCount = 2;

	SECTION("The operator must be square")
	{
		const DenseSelfAdjointLinearOperator<double> operatorMatrix{Eigen::MatrixXd::Zero(3, 4)};
		CHECK_THROWS_AS(
		        ValidateInteriorEigenSolverInput(operatorMatrix, EigenvalueInterval<double>{-1, 1}, options),
		        SecUtility::InvalidArgumentException);
	}

	SECTION("The operator must have a positive dimension")
	{
		const DenseSelfAdjointLinearOperator<double> operatorMatrix{Eigen::MatrixXd::Zero(0, 0)};
		CHECK_THROWS_AS(
		        ValidateInteriorEigenSolverInput(operatorMatrix, EigenvalueInterval<double>{-1, 1}, options),
		        SecUtility::InvalidArgumentException);
	}

	SECTION("The diagonal must match the operator dimension")
	{
		struct IncorrectDiagonalOperator
		{
			using Scalar = double;
			[[nodiscard]] Eigen::Index rows() const noexcept { return 4; }
			[[nodiscard]] Eigen::Index cols() const noexcept { return 4; }
			[[nodiscard]] Eigen::VectorXd Diagonal() const { return Eigen::VectorXd::Ones(3); }
			[[nodiscard]] Eigen::VectorXd ApplyOn(const Eigen::VectorXd& vector) const { return vector; }
		};

		CHECK_THROWS_AS(
		        ValidateInteriorEigenSolverInput(
		                IncorrectDiagonalOperator{}, EigenvalueInterval<double>{-1, 1}, options),
		        SecUtility::InvalidArgumentException);
	}
}


TEST_CASE("The existing matrix-free wrapper requires a diagonal-aware adapter for iVI", "[Math][iVI]")
{
	const MatrixFreeLinearOperator operatorWithoutDiagonal([](const Eigen::VectorXd& vector) { return vector; }, 3);
	STATIC_CHECK_FALSE(SelfAdjointLinearOperator<decltype(operatorWithoutDiagonal)>);
}


TEMPLATE_TEST_CASE("iVI projection and symmetric orthonormalization", "[Math][iVI]", double, (std::complex<double>))
{
	using namespace Detail::IterativeVectorInteraction;
	using RealScalar = typename Eigen::NumTraits<TestType>::Real;
	constexpr RealScalar tolerance = static_cast<RealScalar>(1e-12);

	const Eigen::MatrixX<TestType> basis = Eigen::MatrixX<TestType>::Identity(4, 1);
	Eigen::MatrixX<TestType> candidates = Eigen::MatrixX<TestType>::Zero(4, 3);
	candidates(0, 0) = TestType{2};
	candidates(1, 0) = TestType{1};
	candidates(2, 1) = TestType{1};
	candidates.col(2) = candidates.col(1);

	const auto projected = ProjectAgainstBasis(basis, candidates);
	CHECK((basis.adjoint() * projected).norm() < tolerance);
	CHECK(projected(1, 0) == TestType{1});
	CHECK(projected(2, 1) == TestType{1});
	CHECK(ProjectAgainstBasis(Eigen::MatrixX<TestType>(4, 0), candidates).isApprox(candidates));

	const auto orthonormalized = OrthogonalizeAndRemoveLinearDependence(basis, candidates, tolerance);
	REQUIRE(orthonormalized.rows() == 4);
	REQUIRE(orthonormalized.cols() == 2);
	CHECK((basis.adjoint() * orthonormalized).norm() < tolerance);
	CHECK((orthonormalized.adjoint() * orthonormalized - Eigen::MatrixX<TestType>::Identity(2, 2)).norm()
	      < tolerance);

	const Eigen::MatrixX<TestType> emptyCandidates(4, 0);
	CHECK(OrthogonalizeAndRemoveLinearDependence(basis, emptyCandidates, tolerance).cols() == 0);
	const Eigen::MatrixX<TestType> zeroCandidates = Eigen::MatrixX<TestType>::Zero(4, 2);
	CHECK(SymmetricallyOrthonormalize(zeroCandidates, tolerance).cols() == 0);
}


TEST_CASE("iVI complex orthonormalization is invariant to column phase", "[Math][iVI]")
{
	using namespace Detail::IterativeVectorInteraction;
	const std::complex<double> phase = std::polar(1.0, 0.73);
	Eigen::MatrixXcd vectors = Eigen::MatrixXcd::Identity(3, 2);
	vectors.col(1) *= phase;

	const auto orthonormalized = SymmetricallyOrthonormalize(vectors, 1e-12);
	REQUIRE(orthonormalized.cols() == 2);
	CHECK((orthonormalized.adjoint() * orthonormalized - Eigen::MatrixXcd::Identity(2, 2)).norm() < 1e-12);
}


TEMPLATE_TEST_CASE("iVI vector and image transformations remain aligned", "[Math][iVI]", double, (std::complex<double>))
{
	using namespace Detail::IterativeVectorInteraction;
	Eigen::MatrixX<TestType> matrix = Eigen::MatrixX<TestType>::Zero(3, 3);
	matrix.diagonal() << TestType{1}, TestType{2}, TestType{4};
	Eigen::MatrixX<TestType> vectors = Eigen::MatrixX<TestType>::Identity(3, 3);
	VectorImagePair<TestType> source{vectors, matrix * vectors};

	Eigen::MatrixX<TestType> coefficients = Eigen::MatrixX<TestType>::Zero(3, 2);
	coefficients(0, 0) = TestType{1} / std::sqrt(2.0);
	coefficients(1, 0) = TestType{1} / std::sqrt(2.0);
	coefficients(1, 1) = TestType{1} / std::sqrt(2.0);
	coefficients(2, 1) = TestType{1} / std::sqrt(2.0);

	const auto transformed = TransformVectorImagePair(source, coefficients);
	CHECK(transformed.Vectors.isApprox(vectors * coefficients));
	CHECK(transformed.Images.isApprox(matrix * transformed.Vectors));

	const auto reducedMatrix = FormReducedMatrix(transformed);
	CHECK(reducedMatrix.isApprox(reducedMatrix.adjoint()));
	CHECK(reducedMatrix.isApprox(transformed.Vectors.adjoint() * matrix * transformed.Vectors));
}


TEMPLATE_TEST_CASE("iVI residual and absolute preconditioner kernels", "[Math][iVI]", double, (std::complex<double>))
{
	using namespace Detail::IterativeVectorInteraction;
	using RealScalar = typename Eigen::NumTraits<TestType>::Real;

	Eigen::MatrixX<TestType> vectors = Eigen::MatrixX<TestType>::Zero(3, 2);
	vectors(0, 0) = TestType{1};
	vectors(1, 1) = TestType{1};
	Eigen::MatrixX<TestType> images = vectors;
	images(2, 0) = TestType{3};
	images(2, 1) = TestType{-4};
	const Eigen::VectorX<RealScalar> eigenvalues{{1, 1}};

	const auto residuals = CalculateResiduals(VectorImagePair<TestType>{vectors, images}, eigenvalues);
	const auto residualNorms = CalculateColumnNorms(residuals);
	CHECK(residualNorms.isApprox(Eigen::VectorX<RealScalar>{{3, 4}}));
	CHECK(CalculateColumnNorms(Eigen::MatrixX<TestType>(3, 0)).size() == 0);

	const Eigen::VectorX<RealScalar> diagonal{{RealScalar{0.5}, RealScalar{2}, RealScalar{1}}};
	const auto corrections = ApplyAbsoluteDiagonalPreconditioner(residuals, eigenvalues, diagonal, RealScalar{0.25});
	CHECK(corrections(2, 0) == TestType{12});
	CHECK(corrections(2, 1) == TestType{-16});
	CHECK(corrections(0, 0) == TestType{0});
	CHECK(corrections(1, 1) == TestType{0});
}


TEMPLATE_TEST_CASE("iVI operator application selects block or column fallback", "[Math][iVI]", double, (std::complex<double>))
{
	using namespace Detail::IterativeVectorInteraction;
	Eigen::MatrixX<TestType> matrix = Eigen::MatrixX<TestType>::Identity(3, 3);
	matrix.diagonal() << TestType{1}, TestType{2}, TestType{3};
	const Eigen::MatrixX<TestType> vectors = Eigen::MatrixX<TestType>::Random(3, 2);

	InteriorEigenSolverStatistics blockStatistics;
	const DenseSelfAdjointLinearOperator<TestType> blockOperator{matrix};
	const auto blockImages = ApplyOperator(blockOperator, vectors, blockStatistics);
	CHECK(blockImages.isApprox(matrix * vectors));
	CHECK(blockStatistics.MultipliedVectorCount == 2);
	CHECK(blockStatistics.OperatorApplicationCount == 1);

	InteriorEigenSolverStatistics columnStatistics;
	const VectorOnlySelfAdjointLinearOperator<TestType> columnOperator{matrix};
	const auto columnImages = ApplyOperator(columnOperator, vectors, columnStatistics);
	CHECK(columnImages.isApprox(blockImages));
	CHECK(columnStatistics.MultipliedVectorCount == 2);
	CHECK(columnStatistics.OperatorApplicationCount == 2);

	const Eigen::MatrixX<TestType> emptyVectors(3, 0);
	CHECK(ApplyOperator(blockOperator, emptyVectors, blockStatistics).cols() == 0);
	CHECK(blockStatistics.MultipliedVectorCount == 2);
	CHECK(blockStatistics.OperatorApplicationCount == 1);
}
