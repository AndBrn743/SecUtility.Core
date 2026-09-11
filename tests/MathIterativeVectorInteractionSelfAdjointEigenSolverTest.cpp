// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#include <SecUtility/Math/IterativeVectorInteractionSelfAdjointEigenSolver.hpp>
#include <SecUtility/Math/MatrixFreeLinearOperator.hpp>
#include <catch2/catch_approx.hpp>
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


	template <typename Scalar>
	Eigen::MatrixX<Scalar> CoupledHermitianMatrix(const Eigen::Index dimension)
	{
		Eigen::MatrixX<Scalar> matrix = Eigen::MatrixX<Scalar>::Zero(dimension, dimension);
		for (Eigen::Index index = 0; index < dimension; index++)
		{
			matrix(index, index) = static_cast<double>(index);
		}
		for (Eigen::Index index = 0; index + 1 < dimension; index++)
		{
			const Scalar coupling = []
			{
				if constexpr (Eigen::NumTraits<Scalar>::IsComplex)
				{
					return Scalar{0.18, 0.07};
				}
				else
				{
					return Scalar{0.18};
				}
			}();
			matrix(index, index + 1) = coupling;
			if constexpr (Eigen::NumTraits<Scalar>::IsComplex)
			{
				matrix(index + 1, index) = std::conj(coupling);
			}
			else
			{
				matrix(index + 1, index) = coupling;
			}
		}
		return matrix;
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


TEST_CASE("iVI population matching follows permuted real Ritz vectors", "[Math][iVI]")
{
	using namespace Detail::IterativeVectorInteraction;
	Eigen::MatrixXd reducedEigenvectors = Eigen::MatrixXd::Zero(4, 4);
	reducedEigenvectors(2, 0) = 1;
	reducedEigenvectors(0, 1) = -1;
	reducedEigenvectors(3, 2) = 1;
	reducedEigenvectors(1, 3) = 1;

	const auto populations = FormRitzPopulationMatrix(reducedEigenvectors, 4);
	const auto matches = MatchRitzVectors(populations, 2, 4);
	REQUIRE(matches.size() == 4);
	CHECK(matches[0].PreviousIndex == 2);
	CHECK(matches[1].PreviousIndex == 0);
	CHECK(matches[2].PreviousIndex == 3);
	CHECK(matches[3].PreviousIndex == 1);
	CHECK(matches[1].IsPrimaryVectorContinuation);
	CHECK(matches[0].IsRetainedVectorContinuation);
	CHECK(matches[0].IsWeakMatch);
	CHECK_FALSE(matches[1].IsWeakMatch);
}


TEST_CASE("iVI population matching is invariant to complex phase", "[Math][iVI]")
{
	using namespace Detail::IterativeVectorInteraction;
	Eigen::MatrixXcd reducedEigenvectors = Eigen::MatrixXcd::Zero(3, 3);
	reducedEigenvectors(1, 0) = std::polar(1.0, 0.4);
	reducedEigenvectors(2, 1) = std::polar(1.0, -1.2);
	reducedEigenvectors(0, 2) = std::polar(1.0, 2.3);

	const auto populations = FormRitzPopulationMatrix(reducedEigenvectors, 3);
	const auto matches = MatchRitzVectors(populations, 3, 3);
	CHECK(populations.isApprox(Eigen::Matrix3d{{0, 1, 0}, {0, 0, 1}, {1, 0, 0}}));
	CHECK(matches[0].PreviousIndex == 1);
	CHECK(matches[1].PreviousIndex == 2);
	CHECK(matches[2].PreviousIndex == 0);
}


TEST_CASE("iVI populations are squared coefficient magnitudes", "[Math][iVI]")
{
	using Detail::IterativeVectorInteraction::FormRitzPopulationMatrix;
	Eigen::MatrixXcd reducedEigenvectors(2, 2);
	reducedEigenvectors << std::polar(0.8, 0.4), std::polar(0.6, -0.7),
	        std::polar(0.6, 1.1), std::polar(0.8, 2.2);

	const auto populations = FormRitzPopulationMatrix(reducedEigenvectors, 2);
	const Eigen::Matrix2d expectedPopulations{{0.64, 0.36}, {0.36, 0.64}};
	CHECK(populations.isApprox(expectedPopulations, 1e-12));
}


TEST_CASE("iVI population matching handles rotations and deterministic ties", "[Math][iVI]")
{
	using namespace Detail::IterativeVectorInteraction;
	const Eigen::MatrixXd equalPopulations = Eigen::MatrixXd::Constant(2, 2, 0.5);
	const auto matches = MatchRitzVectors(equalPopulations, 2, 2);
	CHECK(matches[0].PreviousIndex == 0);
	CHECK(matches[1].PreviousIndex == 1);
	CHECK(matches[0].IsWeakMatch);
	CHECK(matches[1].IsWeakMatch);

	const Eigen::Matrix<double, 3, 2> populations{{0.1, 0.8}, {0.9, 0.1}, {0.2, 0.2}};
	const auto unmatched = MatchRitzVectors(Eigen::MatrixXd{populations}, 2, 2);
	CHECK(unmatched[0].PreviousIndex == 1);
	CHECK(unmatched[1].PreviousIndex == 0);
	CHECK_FALSE(unmatched[2].IsMatched());
}


TEST_CASE("iVI Hungarian matching maximizes total population", "[Math][iVI]")
{
	using namespace Detail::IterativeVectorInteraction;
	const Eigen::MatrixXd populations{{0.90, 0.80}, {0.85, 0.00}};

	const auto greedyMatches = MatchRitzVectors_Greedy(populations, 2, 2);
	CHECK(greedyMatches[0].PreviousIndex == 0);
	CHECK(greedyMatches[1].PreviousIndex == 1);

	const auto hungarianMatches = MatchRitzVectors_Hungarian(populations, 2, 2);
	CHECK(hungarianMatches[0].PreviousIndex == 1);
	CHECK(hungarianMatches[1].PreviousIndex == 0);
	CHECK(hungarianMatches[0].Population + hungarianMatches[1].Population
	      > greedyMatches[0].Population + greedyMatches[1].Population);

	const auto defaultMatches = MatchRitzVectors(populations, 2, 2);
	CHECK(defaultMatches[0].PreviousIndex == hungarianMatches[0].PreviousIndex);
	CHECK(defaultMatches[1].PreviousIndex == hungarianMatches[1].PreviousIndex);
}


TEST_CASE("iVI matchers handle an empty previous Ritz set", "[Math][iVI]")
{
	using namespace Detail::IterativeVectorInteraction;
	const Eigen::MatrixXd populations(3, 0);
	const auto greedyMatches = MatchRitzVectors_Greedy(populations, 0, 0);
	const auto hungarianMatches = MatchRitzVectors_Hungarian(populations, 0, 0);
	REQUIRE(greedyMatches.size() == 3);
	REQUIRE(hungarianMatches.size() == 3);
	CHECK_FALSE(greedyMatches[0].IsMatched());
	CHECK_FALSE(hungarianMatches[0].IsMatched());
}


TEST_CASE("iVI interval selection tracks matches and retains neighboring Ritz vectors", "[Math][iVI]")
{
	using namespace Detail::IterativeVectorInteraction;
	const Eigen::VectorXd currentEigenvalues{{-2.0, -0.1, 0.4, 1.1, 3.0}};
	const Eigen::VectorXd previousEigenvalues{{-0.11, 0.39, 1.08, 3.1}};
	std::vector<RitzVectorMatch<double>> matches(5);
	for (Eigen::Index index = 0; index < 5; index++)
	{
		matches[static_cast<std::size_t>(index)].CurrentIndex = index;
	}
	for (Eigen::Index index = 1; index < 5; index++)
	{
		auto& ref_match = matches[static_cast<std::size_t>(index)];
		ref_match.PreviousIndex = index - 1;
		ref_match.Population = 0.9;
		ref_match.IsPrimaryVectorContinuation = true;
		ref_match.IsRetainedVectorContinuation = true;
		ref_match.IsWeakMatch = false;
	}

	const auto selection = SelectInteriorRitzVectors(
	        currentEigenvalues, matches, previousEigenvalues, EigenvalueInterval<double>{-0.1, 1.1}, 3, 2);
	CHECK(selection.IntervalRitzIndices == std::vector<Eigen::Index>{1, 2, 3});
	CHECK(selection.AdditionalRitzIndices == std::vector<Eigen::Index>{4, 0});
	CHECK(selection.MaximumMatchedEigenvalueChange == Catch::Approx(0.02));
	CHECK_FALSE(selection.HasUnmatchedIntervalRitzVector);
	CHECK_FALSE(selection.IsMaximumEigenpairCountExceeded);
}


TEST_CASE("iVI interval selection reports new roots and capacity exhaustion", "[Math][iVI]")
{
	using namespace Detail::IterativeVectorInteraction;
	const Eigen::VectorXd eigenvalues{{-0.5, 0.0, 0.5}};
	std::vector<RitzVectorMatch<double>> matches(3);
	for (Eigen::Index index = 0; index < 3; index++)
	{
		matches[static_cast<std::size_t>(index)].CurrentIndex = index;
	}

	const auto selection = SelectInteriorRitzVectors(
	        eigenvalues, matches, Eigen::VectorXd{}, EigenvalueInterval<double>{-0.5, 0.5}, 2, 0);
	CHECK(selection.IntervalRitzIndices == std::vector<Eigen::Index>{0, 1, 2});
	CHECK(selection.HasUnmatchedIntervalRitzVector);
	CHECK(selection.IsMaximumEigenpairCountExceeded);
	CHECK(std::isinf(selection.MaximumMatchedEigenvalueChange));

	const auto emptySelection = SelectInteriorRitzVectors(
	        eigenvalues, matches, Eigen::VectorXd{}, EigenvalueInterval<double>{2, 3}, 2, 1);
	CHECK(emptySelection.IntervalRitzIndices.empty());
	CHECK(emptySelection.AdditionalRitzIndices == std::vector<Eigen::Index>{2});
	CHECK(std::isinf(emptySelection.MaximumMatchedEigenvalueChange));
}


TEST_CASE("iVI interval selection retains a near-degenerate boundary continuation", "[Math][iVI]")
{
	using namespace Detail::IterativeVectorInteraction;
	const Eigen::VectorXd eigenvalues{{-1.0, -0.500004, -0.5, 0.2, 2.0}};
	std::vector<RitzVectorMatch<double>> matches(5);
	for (Eigen::Index index = 0; index < 5; index++)
	{
		matches[static_cast<std::size_t>(index)].CurrentIndex = index;
	}

	const auto selection = SelectInteriorRitzVectors(
	        eigenvalues, matches, Eigen::VectorXd{}, EigenvalueInterval<double>{-0.5, 0.5}, 2, 1, 1e-5);
	CHECK(selection.IntervalRitzIndices == std::vector<Eigen::Index>{2, 3});
	CHECK(selection.AdditionalRitzIndices == std::vector<Eigen::Index>{1, 0});
}


TEST_CASE("iVI interval selection retains chained upper-boundary continuations", "[Math][iVI]")
{
	using namespace Detail::IterativeVectorInteraction;
	const Eigen::VectorXd eigenvalues{{-2.0, -0.2, 0.5, 0.500004, 0.500009, 2.0}};
	std::vector<RitzVectorMatch<double>> matches(6);
	for (Eigen::Index index = 0; index < 6; index++)
	{
		matches[static_cast<std::size_t>(index)].CurrentIndex = index;
	}

	const auto selection = SelectInteriorRitzVectors(
	        eigenvalues, matches, Eigen::VectorXd{}, EigenvalueInterval<double>{-0.5, 0.5}, 2, 1, 1e-5);
	CHECK(selection.IntervalRitzIndices == std::vector<Eigen::Index>{1, 2});
	CHECK(selection.AdditionalRitzIndices == std::vector<Eigen::Index>{3, 4, 0});
}


TEST_CASE("iVI primary Ritz space uses interval headroom within the available space", "[Math][iVI]")
{
	using Detail::IterativeVectorInteraction::CalculatePrimaryRitzVectorCount;
	CHECK(CalculatePrimaryRitzVectorCount(1, 0, 20) == 6);
	CHECK(CalculatePrimaryRitzVectorCount(3, 0, 20) == 9);
	CHECK(CalculatePrimaryRitzVectorCount(3, 2, 20) == 11);
	CHECK(CalculatePrimaryRitzVectorCount(5, 0, 8) == 8);
}


TEMPLATE_TEST_CASE("iVI selected Ritz pairs are permuted to the front", "[Math][iVI]", double, (std::complex<double>))
{
	using namespace Detail::IterativeVectorInteraction;
	using RealScalar = Eigen::NumTraits<TestType>::Real;
	const Eigen::VectorX<RealScalar> eigenvalues{{-2, -1, 0, 1}};
	const Eigen::MatrixX<TestType> eigenvectors = Eigen::MatrixX<TestType>::Identity(4, 4);
	InteriorRitzSelection<RealScalar> selection;
	selection.IntervalRitzIndices = {2, 3};
	selection.AdditionalRitzIndices = {1};

	const auto permuted = PermuteSelectedRitzPairsToTheFront(eigenvalues, eigenvectors, selection);
	// CHECK(permuted.OriginalIndices == std::vector<Eigen::Index>{2, 3, 1, 0});
	CHECK(permuted.Eigenvalues.isApprox(Eigen::VectorX<RealScalar>{{0, 1, -1, -2}}));
	CHECK(permuted.Eigenvectors.col(0).isApprox(eigenvectors.col(2)));
	CHECK(permuted.Eigenvectors.col(3).isApprox(eigenvectors.col(0)));
}


TEMPLATE_TEST_CASE("iVI solves all diagonal eigenpairs in an inclusive interval", "[Math][iVI]", double, (std::complex<double>))
{
	using RealScalar = Eigen::NumTraits<TestType>::Real;
	Eigen::MatrixX<TestType> matrix = Eigen::MatrixX<TestType>::Zero(8, 8);
	for (Eigen::Index index = 0; index < 8; index++)
	{
		matrix(index, index) = static_cast<RealScalar>(index - 3);
	}
	const DenseSelfAdjointLinearOperator<TestType> linearOperator{matrix};
	InteriorEigenSolverOptions<RealScalar> options;
	options.MaximumEigenpairCount = 3;

	IterativeVectorInteractionSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	solver.Compute(linearOperator, EigenvalueInterval<RealScalar>{-1, 1}, options);
	CHECK(solver.Status() == InteriorEigenSolverStatus::Converged);
	CHECK(solver.Eigenvalues().isApprox(Eigen::VectorX<RealScalar>{{-1, 0, 1}}));
	CHECK(solver.ResidualNorms().isZero());
	CHECK((solver.Eigenvectors().adjoint() * solver.Eigenvectors()
	       - Eigen::MatrixX<TestType>::Identity(3, 3))
	              .norm()
	      < 1e-12);
}


TEST_CASE("iVI reports when the interval exceeds the configured eigenpair capacity", "[Math][iVI]")
{
	Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(8, 8);
	matrix.diagonal() << -3, -2, -1, 0, 1, 2, 3, 4;
	const DenseSelfAdjointLinearOperator<double> linearOperator{matrix};
	InteriorEigenSolverOptions<double> options;
	options.MaximumEigenpairCount = 2;

	IterativeVectorInteractionSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	solver.Compute(linearOperator, EigenvalueInterval<double>{-1, 1}, options);
	CHECK(solver.Status() == InteriorEigenSolverStatus::MaximumEigenpairCountExceeded);
	CHECK(solver.Eigenvalues().isApprox(Eigen::Vector3d{-1, 0, 1}));
}


TEST_CASE("iVI reports an empty interval when exact retained vectors cannot expand", "[Math][iVI]")
{
	Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(8, 8);
	matrix.diagonal() << -3, -2, -1, 0, 1, 2, 3, 4;
	const DenseSelfAdjointLinearOperator<double> linearOperator{matrix};
	InteriorEigenSolverOptions<double> options;
	options.MaximumEigenpairCount = 2;

	IterativeVectorInteractionSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	solver.Compute(linearOperator, EigenvalueInterval<double>{10, 11}, options);
	CHECK(solver.Status() == InteriorEigenSolverStatus::NoEigenpairsFound);
	CHECK(solver.Eigenvalues().size() == 0);
	CHECK(solver.Eigenvectors().cols() == 0);
	CHECK(solver.ResidualNorms().size() == 0);
}


TEST_CASE("iVI solves a dense complex Hermitian problem", "[Math][iVI]")
{
	using Scalar = std::complex<double>;
	Eigen::MatrixXcd matrix = Eigen::MatrixXcd::Zero(4, 4);
	matrix.diagonal() << Scalar{-2}, Scalar{-0.5}, Scalar{1}, Scalar{3};
	matrix(0, 1) = Scalar{0.2, 0.1};
	matrix(1, 0) = std::conj(matrix(0, 1));
	matrix(2, 3) = Scalar{-0.15, 0.3};
	matrix(3, 2) = std::conj(matrix(2, 3));
	const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> referenceSolver(matrix);
	const DenseSelfAdjointLinearOperator<Scalar> linearOperator{matrix};
	InteriorEigenSolverOptions<double> options;
	options.MaximumEigenpairCount = 4;

	IterativeVectorInteractionSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	solver.Compute(linearOperator, EigenvalueInterval<double>{-10, 10}, options);
	CHECK(solver.Status() == InteriorEigenSolverStatus::Converged);
	CHECK(solver.Eigenvalues().isApprox(referenceSolver.eigenvalues(), 1e-12));
	CHECK(solver.ResidualNorms().maxCoeff() < 1e-12);
}


TEMPLATE_TEST_CASE("iVI expands and restarts for an interior eigenpair", "[Math][iVI]", double, (std::complex<double>))
{
	using RealScalar = Eigen::NumTraits<TestType>::Real;
	const Eigen::MatrixX<TestType> matrix = CoupledHermitianMatrix<TestType>(10);
	const Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<TestType>> referenceSolver(matrix);
	const RealScalar lowerBound = (referenceSolver.eigenvalues()[4] + referenceSolver.eigenvalues()[5]) / 2;
	const RealScalar upperBound = (referenceSolver.eigenvalues()[5] + referenceSolver.eigenvalues()[6]) / 2;
	const DenseSelfAdjointLinearOperator<TestType> linearOperator{matrix};
	InteriorEigenSolverOptions<RealScalar> options;
	options.MaximumEigenpairCount = 1;
	options.MaximumIterationCount = 50;
	options.EigenvalueChangeTolerance = 1e-10;
	options.ResidualNormTolerance = 1e-10;

	IterativeVectorInteractionSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	solver.Compute(linearOperator, EigenvalueInterval<RealScalar>{lowerBound, upperBound}, options);
	REQUIRE(solver.Status() == InteriorEigenSolverStatus::Converged);
	REQUIRE(solver.Eigenvalues().size() == 1);
	CHECK(solver.Eigenvalues()[0] == Catch::Approx(referenceSolver.eigenvalues()[5]).margin(1e-10));
	CHECK(solver.ResidualNorms()[0] < 1e-10);
	CHECK(solver.Statistics().CompletedIterationCount > 1);
	CHECK(solver.Statistics().MultipliedVectorCount < matrix.rows() * solver.Statistics().CompletedIterationCount);
}


TEST_CASE("iVI exposes a partial result at the iteration limit", "[Math][iVI]")
{
	const Eigen::MatrixXd matrix = CoupledHermitianMatrix<double>(10);
	const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> referenceSolver(matrix);
	const double lowerBound = (referenceSolver.eigenvalues()[4] + referenceSolver.eigenvalues()[5]) / 2;
	const double upperBound = (referenceSolver.eigenvalues()[5] + referenceSolver.eigenvalues()[6]) / 2;
	const DenseSelfAdjointLinearOperator<double> linearOperator{matrix};
	InteriorEigenSolverOptions<double> options;
	options.MaximumEigenpairCount = 1;
	options.MaximumIterationCount = 1;
	options.ResidualNormTolerance = 1e-14;

	IterativeVectorInteractionSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	solver.Compute(linearOperator, EigenvalueInterval<double>{lowerBound, upperBound}, options);
	CHECK(solver.Status() == InteriorEigenSolverStatus::IterationLimitReached);
	CHECK(solver.Eigenvalues().size() == 1);
	CHECK(solver.ResidualNorms().size() == 1);
}


TEST_CASE("iVI uses the column-wise operator fallback in an end-to-end solve", "[Math][iVI]")
{
	const Eigen::MatrixXd matrix = Eigen::VectorXd{{-2, -1, 0, 1, 2}}.asDiagonal();
	const VectorOnlySelfAdjointLinearOperator<double> linearOperator{matrix};
	InteriorEigenSolverOptions<double> options;
	options.MaximumEigenpairCount = 1;

	IterativeVectorInteractionSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	solver.Compute(linearOperator, EigenvalueInterval<double>{0, 0}, options);
	CHECK(solver.Status() == InteriorEigenSolverStatus::Converged);
	CHECK(solver.Eigenvalues().isApprox(Eigen::VectorXd::Zero(1)));
	CHECK(solver.Statistics().OperatorApplicationCount == matrix.rows());
	CHECK(solver.Statistics().MultipliedVectorCount == matrix.rows());
}
