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


	template <typename T>
	struct BlockOnlySelfAdjointLinearOperator
	{
		using Scalar = T;
		using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

		Eigen::MatrixX<Scalar> Matrix;

		[[nodiscard]] Eigen::Index rows() const noexcept { return Matrix.rows(); }
		[[nodiscard]] Eigen::Index cols() const noexcept { return Matrix.cols(); }
		[[nodiscard]] Eigen::VectorX<RealScalar> Diagonal() const { return Matrix.diagonal().real(); }
		template <typename Derived>
			requires(Derived::ColsAtCompileTime != 1)
		[[nodiscard]] Eigen::MatrixX<Scalar> ApplyOn(const Eigen::MatrixBase<Derived>& vectors) const
		{
			return Matrix * vectors;
		}
	};


	template <typename T>
	struct TridiagonalSelfAdjointLinearOperator
	{
		using Scalar = T;
		using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

		Eigen::VectorX<RealScalar> DiagonalElements;
		Scalar UpperDiagonalElement;

		[[nodiscard]] Eigen::Index rows() const noexcept { return DiagonalElements.size(); }
		[[nodiscard]] Eigen::Index cols() const noexcept { return DiagonalElements.size(); }
		[[nodiscard]] const Eigen::VectorX<RealScalar>& Diagonal() const noexcept { return DiagonalElements; }

		template <typename Derived>
		[[nodiscard]] Eigen::MatrixX<Scalar> ApplyOn(const Eigen::MatrixBase<Derived>& vectors) const
		{
			Eigen::MatrixX<Scalar> images = DiagonalElements.template cast<Scalar>().asDiagonal() * vectors;
			for (Eigen::Index rowIndex = 0; rowIndex + 1 < rows(); rowIndex++)
			{
				images.row(rowIndex) += UpperDiagonalElement * vectors.row(rowIndex + 1);
				images.row(rowIndex + 1) += SecUtility::Math::Conj(UpperDiagonalElement) * vectors.row(rowIndex);
			}
			return images;
		}
	};


	template <typename Scalar>
	Eigen::MatrixX<Scalar> FormDenseMatrix(const TridiagonalSelfAdjointLinearOperator<Scalar>& linearOperator)
	{
		Eigen::MatrixX<Scalar> matrix = linearOperator.DiagonalElements.template cast<Scalar>().asDiagonal();
		for (Eigen::Index index = 0; index + 1 < linearOperator.rows(); index++)
		{
			matrix(index, index + 1) = linearOperator.UpperDiagonalElement;
			matrix(index + 1, index) = SecUtility::Math::Conj(linearOperator.UpperDiagonalElement);
		}
		return matrix;
	}


	struct HubAndBandSelfAdjointLinearOperator
	{
		using Scalar = double;
		static constexpr Eigen::Index HubSize = 11;

		Eigen::Index Dimension;

		[[nodiscard]] Eigen::Index rows() const noexcept { return Dimension; }
		[[nodiscard]] Eigen::Index cols() const noexcept { return Dimension; }
		[[nodiscard]] Eigen::VectorXd Diagonal() const
		{
			Eigen::VectorXd diagonal(Dimension);
			for (Eigen::Index index = 0; index < Dimension; index++)
			{
				diagonal[index] = std::sqrt(static_cast<double>(index));
			}
			return diagonal;
		}

		template <typename Derived>
		[[nodiscard]] Eigen::MatrixXd ApplyOn(const Eigen::MatrixBase<Derived>& vectors) const
		{
			Eigen::MatrixXd images = Diagonal().asDiagonal() * vectors;
			const auto allRowSum = vectors.colwise().sum().eval();
			const auto hubRowSum = vectors.topRows(HubSize).colwise().sum().eval();
			for (Eigen::Index rowIndex = 0; rowIndex < Dimension; rowIndex++)
			{
				if (rowIndex < HubSize)
				{
					images.row(rowIndex) -= allRowSum - vectors.row(rowIndex);
					continue;
				}

				images.row(rowIndex) -= hubRowSum;
				const Eigen::Index firstBandColumn = SecUtility::Math::Max(HubSize, rowIndex - 2);
				const Eigen::Index lastBandColumn = SecUtility::Math::Min(Dimension - 1, rowIndex + 2);
				for (Eigen::Index columnIndex = firstBandColumn; columnIndex <= lastBandColumn; columnIndex++)
				{
					if (columnIndex != rowIndex)
					{
						images.row(rowIndex) -= vectors.row(columnIndex);
					}
				}
			}
			return images;
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


	struct InitiallyInconsistentIdentityOperator
	{
		using Scalar = double;
		mutable Eigen::Index ApplicationCount = 0;

		[[nodiscard]] Eigen::Index rows() const noexcept { return 8; }
		[[nodiscard]] Eigen::Index cols() const noexcept { return 8; }
		[[nodiscard]] Eigen::VectorXd Diagonal() const { return Eigen::VectorXd::Ones(8); }

		template <typename Derived>
		[[nodiscard]] Eigen::MatrixXd ApplyOn(const Eigen::MatrixBase<Derived>& vectors) const
		{
			Eigen::MatrixXd images = vectors;
			if (ApplicationCount++ == 0)
			{
				images.row(0) += vectors.row(1);
			}
			return images;
		}
	};
}


using namespace SecUtility::Math;


static_assert(SelfAdjointLinearOperator<DenseSelfAdjointLinearOperator<double>>);
static_assert(SelfAdjointLinearOperator<DenseSelfAdjointLinearOperator<std::complex<double>>>);
static_assert(BlockSelfAdjointLinearOperator<DenseSelfAdjointLinearOperator<double>>);
static_assert(BlockSelfAdjointLinearOperator<DenseSelfAdjointLinearOperator<std::complex<double>>>);
static_assert(SelfAdjointLinearOperator<VectorOnlySelfAdjointLinearOperator<double>>);
static_assert(!BlockSelfAdjointLinearOperator<VectorOnlySelfAdjointLinearOperator<double>>);
static_assert(!ScalarSelfAdjointLinearOperator<BlockOnlySelfAdjointLinearOperator<double>>);
static_assert(BlockSelfAdjointLinearOperator<BlockOnlySelfAdjointLinearOperator<double>>);
static_assert(SelfAdjointLinearOperator<BlockOnlySelfAdjointLinearOperator<double>>);
static_assert(SelfAdjointLinearOperator<BlockOnlySelfAdjointLinearOperator<std::complex<double>>>);
static_assert(!SelfAdjointLinearOperator<OperatorWithoutDiagonal>);
static_assert(!SelfAdjointLinearOperator<ComplexDiagonalOperator>);
static_assert(!SelfAdjointLinearOperator<OperatorWithoutScalar>);
static_assert(!SelfAdjointLinearOperator<DenseSelfAdjointLinearOperator<int>>);
static_assert(!SelfAdjointLinearOperator<DenseSelfAdjointLinearOperator<std::complex<int>>>);
static_assert(SecUtility::is_bitmask_v<IterativeVectorInteractionSubspaceExtension>);
static_assert(((IterativeVectorInteractionSubspaceExtension::AdditionalRitzVectors
                | IterativeVectorInteractionSubspaceExtension::PreviousRitzVectors)
               & IterativeVectorInteractionSubspaceExtension::PreviousRitzVectors)
              == IterativeVectorInteractionSubspaceExtension::PreviousRitzVectors);
static_assert((IterativeVectorInteractionSubspaceExtension::AdditionalRitzVectors
               | IterativeVectorInteractionSubspaceExtension::CorrectionVectorImages
               | IterativeVectorInteractionSubspaceExtension::PreconditionedOffDiagonalCorrectionImages
               | IterativeVectorInteractionSubspaceExtension::PreviousRitzVectors)
              == IterativeVectorInteractionSubspaceExtension::All);


TEMPLATE_TEST_CASE("iVI public result contract", "[Math][iVI]", double, (std::complex<double>))
{
	using Operator = DenseSelfAdjointLinearOperator<TestType>;
	using Solver = IterativeVectorInteractionSelfAdjointEigenSolver<Operator>;
	using RealScalar = typename Eigen::NumTraits<TestType>::Real;

	Solver solver;
	STATIC_CHECK(std::is_same_v<typename Solver::Scalar, TestType>);
	STATIC_CHECK(std::is_same_v<typename Solver::RealScalar, RealScalar>);
	static_assert(!std::is_default_constructible_v<InteriorEigenSolverOptions<RealScalar>>);
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
	CHECK(solver.Statistics().GeneratedCorrectionVectorCount == 0);
	CHECK(solver.Statistics().RetainedCorrectionVectorCount == 0);
	CHECK(solver.Statistics().GeneratedCorrectionImageVectorCount == 0);
	CHECK(solver.Statistics().RetainedCorrectionImageVectorCount == 0);
	CHECK(solver.Statistics().GeneratedOffDiagonalCorrectionVectorCount == 0);
	CHECK(solver.Statistics().RetainedOffDiagonalCorrectionVectorCount == 0);
	CHECK(solver.Statistics().RetainedAdditionalRitzVectorCount == 0);
	CHECK(solver.Statistics().RecycledVectorCount == 0);
	CHECK(solver.Statistics().CurrentFrozenVectorCount == 0);
	CHECK(solver.Statistics().MaximumFrozenVectorCount == 0);
}


TEMPLATE_TEST_CASE("iVI operator contract and input validation", "[Math][iVI]", double, (std::complex<double>))
{
	using Detail::IterativeVectorInteraction::ValidateInteriorEigenSolverInput;
	using RealScalar = typename Eigen::NumTraits<TestType>::Real;
	const auto operatorMatrix = IdentityOperator<TestType>(4);
	const EigenvalueInterval<RealScalar> interval{-1, 1};
	InteriorEigenSolverOptions<RealScalar> options{2};

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
		options.EigenpairCountLimit = 0;
		CHECK_THROWS_AS(ValidateInteriorEigenSolverInput(operatorMatrix, interval, options), SecUtility::InvalidArgumentException);
	}

	SECTION("An eigenpair capacity larger than the dimension is rejected")
	{
		options.EigenpairCountLimit = 5;
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

		options.ResidualNormTolerance = RealScalar{1e-8};
		options.FreezingCoefficientTolerance = 0;
		CHECK_THROWS_AS(ValidateInteriorEigenSolverInput(operatorMatrix, interval, options), SecUtility::InvalidArgumentException);

		options.FreezingCoefficientTolerance = RealScalar{1e-8};
		options.FreezingResidualNormTolerance = std::numeric_limits<RealScalar>::infinity();
		CHECK_THROWS_AS(ValidateInteriorEigenSolverInput(operatorMatrix, interval, options), SecUtility::InvalidArgumentException);

		options.FreezingResidualNormTolerance = RealScalar{-1};
		CHECK_THROWS_AS(ValidateInteriorEigenSolverInput(operatorMatrix, interval, options), SecUtility::InvalidArgumentException);

		options.FreezingResidualNormTolerance = RealScalar{};
		options.ReducedMatrixAsymmetryTolerance = 0;
		CHECK_THROWS_AS(ValidateInteriorEigenSolverInput(operatorMatrix, interval, options), SecUtility::InvalidArgumentException);
	}
}


TEST_CASE("iVI subspace-extension flags and validation", "[Math][iVI]")
{
	using Detail::IterativeVectorInteraction::IsSubspaceExtensionEnabled;
	using Detail::IterativeVectorInteraction::ValidateInteriorEigenSolverInput;
	using Underlying = std::underlying_type_t<IterativeVectorInteractionSubspaceExtension>;
	constexpr Underlying allExtensionBits =
	        static_cast<Underlying>(IterativeVectorInteractionSubspaceExtension::All);
	const auto operatorMatrix = IdentityOperator<double>(4);
	const EigenvalueInterval<double> interval{-1, 1};

	SECTION("The default preserves the v1 extension configuration")
	{
		const InteriorEigenSolverOptions<double> options{2};
		CHECK(IsSubspaceExtensionEnabled(
		        options.SubspaceExtensions, IterativeVectorInteractionSubspaceExtension::AdditionalRitzVectors));
		CHECK(IsSubspaceExtensionEnabled(
		        options.SubspaceExtensions, IterativeVectorInteractionSubspaceExtension::PreviousRitzVectors));
		CHECK_FALSE(IsSubspaceExtensionEnabled(
		        options.SubspaceExtensions, IterativeVectorInteractionSubspaceExtension::CorrectionVectorImages));
		CHECK_FALSE(IsSubspaceExtensionEnabled(
		        options.SubspaceExtensions,
		        IterativeVectorInteractionSubspaceExtension::PreconditionedOffDiagonalCorrectionImages));
	}

	SECTION("Every combination of known flags is valid")
	{
		for (Underlying bits = 0; bits <= allExtensionBits; ++bits)
		{
			InteriorEigenSolverOptions<double> options{2};
			options.SubspaceExtensions = static_cast<IterativeVectorInteractionSubspaceExtension>(bits);
			CHECK_NOTHROW(ValidateInteriorEigenSolverInput(operatorMatrix, interval, options));
		}
	}

	SECTION("Unknown flags are rejected")
	{
		InteriorEigenSolverOptions<double> options{2};
		options.SubspaceExtensions = static_cast<IterativeVectorInteractionSubspaceExtension>(allExtensionBits + 1);
		CHECK_THROWS_AS(
		        ValidateInteriorEigenSolverInput(operatorMatrix, interval, options),
		        SecUtility::InvalidArgumentException);
	}

	SECTION("Controls for disabled extensions are ignored")
	{
		InteriorEigenSolverOptions<double> options{2};
		options.SubspaceExtensions = IterativeVectorInteractionSubspaceExtension::None;
		options.AdditionalRitzVectorCount = -1;
		options.PreviousRitzVectorRecyclingTolerance = -1;
		options.RecyclingRefinementTolerance = -1;
		CHECK_NOTHROW(ValidateInteriorEigenSolverInput(operatorMatrix, interval, options));
	}

	SECTION("Invalid controls for enabled extensions are rejected")
	{
		InteriorEigenSolverOptions<double> options{2};
		options.PreviousRitzVectorRecyclingTolerance = -1;
		CHECK_THROWS_AS(
		        ValidateInteriorEigenSolverInput(operatorMatrix, interval, options),
		        SecUtility::InvalidArgumentException);

		options.PreviousRitzVectorRecyclingTolerance = 1e-8;
		options.RecyclingRefinementTolerance = -1;
		CHECK_THROWS_AS(
		        ValidateInteriorEigenSolverInput(operatorMatrix, interval, options),
		        SecUtility::InvalidArgumentException);
	}
}


TEST_CASE("iVI validation rejects inconsistent operator dimensions", "[Math][iVI]")
{
	using Detail::IterativeVectorInteraction::ValidateInteriorEigenSolverInput;
	InteriorEigenSolverOptions<double> options{2};

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


TEMPLATE_TEST_CASE("iVI projection and QR orthonormalization", "[Math][iVI]", double, (std::complex<double>))
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
	REQUIRE(orthonormalized.Vectors.rows() == 4);
	REQUIRE(orthonormalized.Vectors.cols() == 2);
	REQUIRE(orthonormalized.SourceCoefficients.rows() == candidates.cols());
	REQUIRE(orthonormalized.SourceCoefficients.cols() == 2);
	CHECK((basis.adjoint() * orthonormalized.Vectors).norm() < tolerance);
	CHECK((orthonormalized.Vectors.adjoint() * orthonormalized.Vectors
	       - Eigen::MatrixX<TestType>::Identity(2, 2))
	              .norm()
	      < tolerance);
	const auto twiceProjectedCandidates = ProjectAgainstBasis(basis, ProjectAgainstBasis(basis, candidates));
	CHECK((twiceProjectedCandidates * orthonormalized.SourceCoefficients)
	              .isApprox(orthonormalized.Vectors, tolerance));

	const Eigen::MatrixX<TestType> emptyCandidates(4, 0);
	const auto emptyDirections = OrthogonalizeAndRemoveLinearDependence(basis, emptyCandidates, tolerance);
	CHECK(emptyDirections.Vectors.cols() == 0);
	CHECK(emptyDirections.SourceCoefficients.rows() == 0);
	CHECK(emptyDirections.SourceCoefficients.cols() == 0);
	const Eigen::MatrixX<TestType> zeroCandidates = Eigen::MatrixX<TestType>::Zero(4, 2);
	const auto zeroDirections = OrthonormalizeAndRemoveLinearDependenceWithQR(zeroCandidates, tolerance);
	CHECK(zeroDirections.Vectors.cols() == 0);
	CHECK(zeroDirections.SourceCoefficients.rows() == 2);
	CHECK(zeroDirections.SourceCoefficients.cols() == 0);
}


TEST_CASE("iVI complex orthonormalization is invariant to column phase", "[Math][iVI]")
{
	using namespace Detail::IterativeVectorInteraction;
	const std::complex<double> phase = std::polar(1.0, 0.73);
	Eigen::MatrixXcd vectors = Eigen::MatrixXcd::Identity(3, 2);
	vectors.col(1) *= phase;

	const auto orthonormalized = OrthonormalizeAndRemoveLinearDependenceWithQR(vectors, 1e-12);
	REQUIRE(orthonormalized.Vectors.cols() == 2);
	CHECK((orthonormalized.Vectors.adjoint() * orthonormalized.Vectors - Eigen::MatrixXcd::Identity(2, 2)).norm()
	      < 1e-12);
	CHECK((vectors * orthonormalized.SourceCoefficients).isApprox(orthonormalized.Vectors, 1e-12));
}


TEMPLATE_TEST_CASE("iVI pivoted QR removes dependent correction directions", "[Math][iVI]", double,
                   (std::complex<double>))
{
	using namespace Detail::IterativeVectorInteraction;
	Eigen::MatrixX<TestType> vectors = Eigen::MatrixX<TestType>::Zero(4, 3);
	vectors(0, 0) = TestType{1e-3};
	vectors(1, 1) = TestType{2};
	vectors.col(2) = TestType{3} * vectors.col(1);

	const auto orthonormalized = OrthonormalizeAndRemoveLinearDependenceWithQR(vectors, 1e-12);
	REQUIRE(orthonormalized.Vectors.cols() == 2);
	CHECK(orthonormalized.SourcePivotIndices == std::vector<Eigen::Index>{2, 0});
	CHECK((orthonormalized.Vectors.adjoint() * orthonormalized.Vectors
	       - Eigen::MatrixX<TestType>::Identity(2, 2))
	              .norm()
	      < 1e-12);
	CHECK((orthonormalized.Vectors * orthonormalized.Vectors.adjoint() * vectors - vectors).norm() < 1e-12);
	CHECK((vectors * orthonormalized.SourceCoefficients).isApprox(orthonormalized.Vectors, 1e-12));
}


TEMPLATE_TEST_CASE("iVI correction QR applies the squared-magnitude dependence tolerance",
	               "[Math][iVI]",
	               double,
	               (std::complex<double>))
{
	using namespace Detail::IterativeVectorInteraction;
	Eigen::MatrixX<TestType> vectors = Eigen::MatrixX<TestType>::Zero(4, 3);
	vectors(0, 0) = TestType{1};
	vectors(1, 1) = TestType{2e-4};
	vectors(2, 2) = TestType{5e-5};

	const auto orthonormalized = OrthonormalizeAndRemoveLinearDependenceWithQR(vectors, 1e-8);
	REQUIRE(orthonormalized.Vectors.cols() == 2);
	REQUIRE(orthonormalized.SourceCoefficients.rows() == 3);
	REQUIRE(orthonormalized.SourceCoefficients.cols() == 2);
	CHECK(orthonormalized.SourcePivotIndices == std::vector<Eigen::Index>{0, 1});
	CHECK((vectors * orthonormalized.SourceCoefficients).isApprox(orthonormalized.Vectors, 1e-12));
	CHECK((orthonormalized.Vectors.adjoint() * orthonormalized.Vectors
	       - Eigen::MatrixX<TestType>::Identity(2, 2))
	              .norm()
	      < 1e-12);
}


TEMPLATE_TEST_CASE("iVI coefficient-space QR uses the vector-space metric", "[Math][iVI]", double,
                   (std::complex<double>))
{
	using namespace Detail::IterativeVectorInteraction;
	Eigen::MatrixX<TestType> expansionVectors = Eigen::MatrixX<TestType>::Zero(4, 4);
	expansionVectors.diagonal() << TestType{2}, TestType{3}, TestType{4}, TestType{5};
	Eigen::MatrixX<TestType> basisCoefficients = Eigen::MatrixX<TestType>::Zero(4, 1);
	basisCoefficients(0, 0) = TestType{0.5};
	Eigen::MatrixX<TestType> candidateCoefficients = Eigen::MatrixX<TestType>::Zero(4, 3);
	candidateCoefficients(0, 0) = TestType{2};
	candidateCoefficients(1, 0) = TestType{1} / TestType{3};
	candidateCoefficients(2, 0) = TestType{1} / TestType{4};
	candidateCoefficients(2, 1) = TestType{1} / TestType{4};
	candidateCoefficients(3, 1) = TestType{1} / TestType{5};
	candidateCoefficients.col(2) = candidateCoefficients.col(1);
	if constexpr (!std::is_same_v<TestType, double>)
	{
		candidateCoefficients(2, 0) *= TestType{0, 1};
	}

	const auto orthonormalizedCoefficients = OrthogonalizeCoefficientDirectionsInVectorSpace(
	        expansionVectors, basisCoefficients, candidateCoefficients, 1e-12);
	const Eigen::MatrixX<TestType> basisVectors = expansionVectors * basisCoefficients;
	const Eigen::MatrixX<TestType> orthonormalizedVectors = expansionVectors * orthonormalizedCoefficients;

	REQUIRE(orthonormalizedCoefficients.cols() == 2);
	CHECK((basisVectors.adjoint() * orthonormalizedVectors).norm() < 1e-12);
	CHECK((orthonormalizedVectors.adjoint() * orthonormalizedVectors
	       - Eigen::MatrixX<TestType>::Identity(2, 2))
	              .norm()
	      < 1e-12);
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


TEMPLATE_TEST_CASE("iVI solves standard and generalized reduced eigenproblems",
	               "[Math][iVI]",
	               double,
	               (std::complex<double>))
{
	using namespace Detail::IterativeVectorInteraction;
	using RealScalar = Eigen::NumTraits<TestType>::Real;
	const TestType phase = []
	{
		if constexpr (Eigen::NumTraits<TestType>::IsComplex)
		{
			return std::polar(RealScalar{1}, RealScalar{0.41});
		}
		else
		{
			return TestType{1};
		}
	}();
	const Eigen::MatrixX<TestType> matrix =
	        Eigen::VectorX<RealScalar>{{RealScalar{1}, RealScalar{3}, RealScalar{7}}}.template cast<TestType>().asDiagonal();
	Eigen::MatrixX<TestType> vectors = Eigen::MatrixX<TestType>::Zero(3, 2);
	vectors(0, 0) = phase;
	vectors(1, 1) = TestType{2};
	const VectorImagePair<TestType> vectorImagePair{vectors, matrix * vectors};

	Eigen::VectorX<RealScalar> generalizedEigenvalues;
	Eigen::MatrixX<TestType> generalizedEigenvectors;
	REQUIRE(SolveReducedSelfAdjointEigenproblem(
	                vectorImagePair, true, generalizedEigenvalues, generalizedEigenvectors)
	        == Eigen::Success);
	CHECK(generalizedEigenvalues.isApprox(Eigen::VectorX<RealScalar>{{RealScalar{1}, RealScalar{3}}}));
	CHECK((generalizedEigenvectors.adjoint() * vectors.adjoint() * vectors * generalizedEigenvectors)
	              .isApprox(Eigen::MatrixX<TestType>::Identity(2, 2)));

	Eigen::VectorX<RealScalar> standardEigenvalues;
	Eigen::MatrixX<TestType> standardEigenvectors;
	REQUIRE(SolveReducedSelfAdjointEigenproblem(
	                vectorImagePair, false, standardEigenvalues, standardEigenvectors)
	        == Eigen::Success);
	CHECK(standardEigenvalues.isApprox(Eigen::VectorX<RealScalar>{{RealScalar{1}, RealScalar{12}}}));
}


TEMPLATE_TEST_CASE("iVI detects material reduced-matrix asymmetry",
	               "[Math][iVI]",
	               double,
	               (std::complex<double>))
{
	using namespace Detail::IterativeVectorInteraction;
	using RealScalar = Eigen::NumTraits<TestType>::Real;
	Eigen::MatrixX<TestType> reducedMatrix = Eigen::MatrixX<TestType>::Zero(2, 2);
	reducedMatrix(0, 0) = TestType{1};
	reducedMatrix(1, 1) = TestType{2};
	reducedMatrix(0, 1) = TestType{1};

	CHECK(DoesReducedMatrixRequireExplicitImages(reducedMatrix, 0.1));
	CHECK_FALSE(DoesReducedMatrixRequireExplicitImages(reducedMatrix, 0.6));
	CHECK(DoesReducedMatrixRequireExplicitImages((RealScalar{100} * reducedMatrix).eval(), 0.1));
	CHECK_FALSE(DoesReducedMatrixRequireExplicitImages((RealScalar{100} * reducedMatrix).eval(), 0.6));

	Eigen::MatrixX<TestType> largeReducedMatrix = Eigen::MatrixX<TestType>::Zero(20, 20);
	largeReducedMatrix.diagonal().setConstant(TestType{2});
	largeReducedMatrix(0, 1) = TestType{1};
	CHECK(DoesReducedMatrixRequireExplicitImages(largeReducedMatrix, 0.1));
	CHECK_FALSE(DoesReducedMatrixRequireExplicitImages(Eigen::MatrixX<TestType>(0, 0), 0.1));
}


TEMPLATE_TEST_CASE("iVI explicitly refreshes images",
	               "[Math][iVI]",
	               double,
	               (std::complex<double>))
{
	using namespace Detail::IterativeVectorInteraction;
	const auto linearOperator = IdentityOperator<TestType>(3);
	VectorImagePair<TestType> vectorImagePair{
	        Eigen::MatrixX<TestType>::Identity(3, 2), Eigen::MatrixX<TestType>::Zero(3, 2)};
	InteriorEigenSolverStatistics statistics;

	RecalculateImages(linearOperator, vectorImagePair, statistics);
	CHECK(vectorImagePair.Images.isApprox(vectorImagePair.Vectors));
	CHECK(statistics.ExplicitImageRecalculationCount == 1);
	CHECK(statistics.MultipliedVectorCount == vectorImagePair.Vectors.cols());
}


TEST_CASE("iVI refreshes retained images after detecting inconsistent projected images", "[Math][iVI]")
{
	const InitiallyInconsistentIdentityOperator linearOperator;
	InteriorEigenSolverOptions<double> options{8};
	IterativeVectorInteractionSelfAdjointEigenSolver<InitiallyInconsistentIdentityOperator> solver;
	(void)solver.Compute(linearOperator, EigenvalueInterval<double>{0, 2}, options);

	CHECK(solver.Status() == InteriorEigenSolverStatus::Converged);
	CHECK(solver.Statistics().ExplicitImageRecalculationCount == 1);
	CHECK(solver.Statistics().OperatorApplicationCount == 2);
	CHECK(solver.ResidualNorms().isZero());
}


TEST_CASE("iVI reports a singular generalized reduced problem", "[Math][iVI]")
{
	using namespace Detail::IterativeVectorInteraction;
	Eigen::MatrixXd vectors(2, 2);
	vectors << 1, 1, 0, 0;
	const VectorImagePair<double> vectorImagePair{vectors, vectors};
	Eigen::VectorXd eigenvalues;
	Eigen::MatrixXd eigenvectors;
	CHECK(SolveReducedSelfAdjointEigenproblem(vectorImagePair, true, eigenvalues, eigenvectors)
	      != Eigen::Success);
}


TEMPLATE_TEST_CASE("iVI rejects a numerically rank-deficient generalized reduced problem",
	               "[Math][iVI]",
	               double,
	               (std::complex<double>))
{
	using namespace Detail::IterativeVectorInteraction;
	using RealScalar = Eigen::NumTraits<TestType>::Real;
	const RealScalar scale = RealScalar{1e4};
	Eigen::MatrixX<TestType> vectors = Eigen::MatrixX<TestType>::Zero(2, 2);
	vectors(0, 0) = static_cast<TestType>(scale);
	vectors(1, 1) = static_cast<TestType>(scale * std::sqrt(std::numeric_limits<RealScalar>::epsilon()));
	const VectorImagePair<TestType> vectorImagePair{vectors, vectors};
	Eigen::VectorX<RealScalar> eigenvalues;
	Eigen::MatrixX<TestType> eigenvectors;

	CHECK(SolveReducedSelfAdjointEigenproblem(vectorImagePair, true, eigenvalues, eigenvectors)
	      == Eigen::NumericalIssue);
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


TEMPLATE_TEST_CASE("iVI source-coefficient-weighted eigenvalues follow the reference convention",
	               "[Math][iVI]",
	               double,
	               (std::complex<double>))
{
	using namespace Detail::IterativeVectorInteraction;
	using RealScalar = typename Eigen::NumTraits<TestType>::Real;
	const RealScalar inverseSquareRootOfTwo = RealScalar{1} / std::sqrt(RealScalar{2});
	Eigen::MatrixX<TestType> sourceCoefficients = Eigen::MatrixX<TestType>::Zero(2, 2);
	sourceCoefficients(0, 0) = TestType{2};
	sourceCoefficients(0, 1) = TestType{inverseSquareRootOfTwo};
	sourceCoefficients(1, 1) = TestType{inverseSquareRootOfTwo};
	if constexpr (Eigen::NumTraits<TestType>::IsComplex)
	{
		sourceCoefficients(1, 1) *= TestType{0, 1};
	}
	const Eigen::VectorX<RealScalar> sourceEigenvalues{{2, 10}};

	const auto weightedEigenvalues =
	        CalculateSourceCoefficientWeightedEigenvalues(sourceCoefficients, sourceEigenvalues);

	REQUIRE(weightedEigenvalues.size() == 2);
	CHECK(weightedEigenvalues[0] == Catch::Approx(8));
	CHECK(weightedEigenvalues[1] == Catch::Approx(6));

	const TestType phase = []
	{
		if constexpr (Eigen::NumTraits<TestType>::IsComplex)
		{
			return std::polar(RealScalar{1}, RealScalar{0.73});
		}
		return TestType{-1};
	}();
	sourceCoefficients.col(1) *= phase;
	CHECK(CalculateSourceCoefficientWeightedEigenvalues(sourceCoefficients, sourceEigenvalues)
	              .isApprox(weightedEigenvalues));
	CHECK(CalculateSourceCoefficientWeightedEigenvalues(Eigen::MatrixX<TestType>(2, 0), sourceEigenvalues).size()
	      == 0);
}


TEMPLATE_TEST_CASE("iVI forms preconditioned off-diagonal correction vectors",
	               "[Math][iVI]",
	               double,
	               (std::complex<double>))
{
	using namespace Detail::IterativeVectorInteraction;
	using RealScalar = typename Eigen::NumTraits<TestType>::Real;
	const Eigen::VectorX<RealScalar> diagonal{{1, 2, 4}};
	const Eigen::MatrixX<TestType> corrections = Eigen::MatrixX<TestType>::Identity(3, 2);
	Eigen::MatrixX<TestType> offDiagonalActions = Eigen::MatrixX<TestType>::Zero(3, 2);
	offDiagonalActions(0, 0) = TestType{0.5};
	offDiagonalActions(2, 0) = TestType{3};
	offDiagonalActions(0, 1) = TestType{-2};
	const Eigen::MatrixX<TestType> correctionImages =
	        diagonal.template cast<TestType>().asDiagonal() * corrections + offDiagonalActions;
	const Eigen::MatrixX<TestType> sourceCoefficients = Eigen::MatrixX<TestType>::Identity(2, 2);
	const Eigen::VectorX<RealScalar> sourceEigenvalues{{1, 2}};

	CHECK(FormOffDiagonalCorrectionVectors(corrections, correctionImages, diagonal)
	              .isApprox(offDiagonalActions));
	const auto preconditioned = FormPreconditionedOffDiagonalCorrectionVectors(corrections,
	                                                                           correctionImages,
	                                                                           sourceCoefficients,
	                                                                           sourceEigenvalues,
	                                                                           diagonal,
	                                                                           RealScalar{0.25});
	CHECK(preconditioned(2, 0) == TestType{1});
	CHECK(preconditioned(0, 1) == TestType{-2});
	CHECK(preconditioned(0, 0) == TestType{2});
	CHECK(preconditioned(1, 1) == TestType{0});

	const Eigen::MatrixX<TestType> diagonalImages =
	        diagonal.template cast<TestType>().asDiagonal() * corrections;
	CHECK(FormPreconditionedOffDiagonalCorrectionVectors(corrections,
	                                                     diagonalImages,
	                                                     sourceCoefficients,
	                                                     sourceEigenvalues,
	                                                     diagonal,
	                                                     RealScalar{0.25})
	              .isZero());
}


TEMPLATE_TEST_CASE("iVI correction statistics distinguish generated and retained vectors",
	               "[Math][iVI]",
	               double,
	               (std::complex<double>))
{
	using namespace Detail::IterativeVectorInteraction;
	using RealScalar = typename Eigen::NumTraits<TestType>::Real;
	Eigen::MatrixX<TestType> matrix = Eigen::MatrixX<TestType>::Zero(3, 3);
	matrix << TestType{1}, TestType{0}, TestType{1},
	          TestType{0}, TestType{2}, TestType{1},
	          TestType{1}, TestType{1}, TestType{3};
	const DenseSelfAdjointLinearOperator<TestType> linearOperator{matrix};
	const Eigen::MatrixX<TestType> retainedVectors = Eigen::MatrixX<TestType>::Identity(3, 2);
	VectorImagePair<TestType> retainedSpace{retainedVectors, matrix * retainedVectors};

	InteriorIterationAnalysis<TestType> analysis;
	analysis.OrderedRitzPairs.Eigenvalues = Eigen::VectorX<RealScalar>{{1, 2}};
	analysis.RetainedVectorCount = 2;
	analysis.PrimaryVectorCount = 2;
	InteriorEigenSolverOptions<RealScalar> options{2};
	InteriorIterationState<TestType> state;
	InteriorEigenSolverStatistics statistics;
	const Eigen::VectorX<RealScalar> diagonal = matrix.diagonal().real();

	const auto result = ExpandForNextIteration(
	        linearOperator, diagonal, analysis, options, retainedSpace, state, statistics);

	CHECK(result == ExpansionResult::Expanded);
	CHECK(statistics.GeneratedCorrectionVectorCount == 2);
	CHECK(statistics.RetainedCorrectionVectorCount == 1);
	CHECK(statistics.MultipliedVectorCount == 1);
	CHECK(statistics.OperatorApplicationCount == 1);
	CHECK(state.ExpansionSpace.Vectors.cols() == 3);
	CHECK(state.ExpansionSpace.Images.isApprox(matrix * state.ExpansionSpace.Vectors));
	CHECK(statistics.GeneratedCorrectionImageVectorCount == 0);
	CHECK(statistics.RetainedCorrectionImageVectorCount == 0);
	CHECK(statistics.GeneratedOffDiagonalCorrectionVectorCount == 0);
	CHECK(statistics.RetainedOffDiagonalCorrectionVectorCount == 0);

	const auto identityOperator = IdentityOperator<TestType>(3);
	InteriorIterationAnalysis<TestType> convergedAnalysis;
	convergedAnalysis.OrderedRitzPairs.Eigenvalues = Eigen::VectorX<RealScalar>{{1}};
	convergedAnalysis.RetainedVectorCount = 1;
	convergedAnalysis.PrimaryVectorCount = 1;
	const Eigen::MatrixX<TestType> convergedVector = Eigen::MatrixX<TestType>::Identity(3, 1);
	const VectorImagePair<TestType> convergedSpace{convergedVector, convergedVector};
	InteriorIterationState<TestType> convergedState;
	InteriorEigenSolverStatistics convergedStatistics;
	const Eigen::VectorX<RealScalar> identityDiagonal = identityOperator.Diagonal();

	const auto convergedResult = ExpandForNextIteration(identityOperator,
	                                                    identityDiagonal,
	                                                    convergedAnalysis,
	                                                    options,
	                                                    convergedSpace,
	                                                    convergedState,
	                                                    convergedStatistics);
	CHECK(convergedResult == ExpansionResult::NoIndependentCorrections);
	CHECK(convergedStatistics.GeneratedCorrectionVectorCount == 1);
	CHECK(convergedStatistics.RetainedCorrectionVectorCount == 0);
	CHECK(convergedStatistics.MultipliedVectorCount == 0);
	CHECK(convergedStatistics.OperatorApplicationCount == 0);
}


TEMPLATE_TEST_CASE("iVI appends correction-vector images as independent expansion directions",
	               "[Math][iVI]",
	               double,
	               (std::complex<double>))
{
	using namespace Detail::IterativeVectorInteraction;
	using RealScalar = typename Eigen::NumTraits<TestType>::Real;
	const TestType firstCoupling = []
	{
		if constexpr (Eigen::NumTraits<TestType>::IsComplex)
		{
			return TestType{0.5, 0.2};
		}
		return TestType{0.5};
	}();
	const TestType secondCoupling = []
	{
		if constexpr (Eigen::NumTraits<TestType>::IsComplex)
		{
			return TestType{0.3, -0.1};
		}
		return TestType{0.3};
	}();
	Eigen::MatrixX<TestType> matrix = Eigen::MatrixX<TestType>::Zero(3, 3);
	matrix.diagonal() << TestType{1}, TestType{2}, TestType{3};
	matrix(0, 1) = firstCoupling;
	matrix(1, 0) = Conj(firstCoupling);
	matrix(1, 2) = secondCoupling;
	matrix(2, 1) = Conj(secondCoupling);
	const DenseSelfAdjointLinearOperator<TestType> linearOperator{matrix};
	const Eigen::MatrixX<TestType> retainedVectors = Eigen::MatrixX<TestType>::Identity(3, 1);
	VectorImagePair<TestType> retainedSpace{retainedVectors, matrix * retainedVectors};

	InteriorIterationAnalysis<TestType> analysis;
	analysis.OrderedRitzPairs.Eigenvalues = Eigen::VectorX<RealScalar>{{1}};
	analysis.RetainedVectorCount = 1;
	analysis.PrimaryVectorCount = 1;
	InteriorEigenSolverOptions<RealScalar> options{1};
	options.SubspaceExtensions = IterativeVectorInteractionSubspaceExtension::CorrectionVectorImages;
	InteriorIterationState<TestType> state;
	InteriorEigenSolverStatistics statistics;
	const Eigen::VectorX<RealScalar> diagonal = matrix.diagonal().real();

	const auto result = ExpandForNextIteration(
	        linearOperator, diagonal, analysis, options, retainedSpace, state, statistics);

	CHECK(result == ExpansionResult::Expanded);
	CHECK(state.ExpansionSpace.Vectors.cols() == 3);
	CHECK((state.ExpansionSpace.Vectors.adjoint() * state.ExpansionSpace.Vectors)
	              .isApprox(Eigen::MatrixX<TestType>::Identity(3, 3), 1e-12));
	CHECK(state.ExpansionSpace.Images.isApprox(matrix * state.ExpansionSpace.Vectors, 1e-12));
	CHECK(statistics.GeneratedCorrectionVectorCount == 1);
	CHECK(statistics.RetainedCorrectionVectorCount == 1);
	CHECK(statistics.GeneratedCorrectionImageVectorCount == 1);
	CHECK(statistics.RetainedCorrectionImageVectorCount == 1);
	CHECK(statistics.MultipliedVectorCount == 2);
	CHECK(statistics.OperatorApplicationCount == 2);
}


TEMPLATE_TEST_CASE("iVI discards correction-vector images contained in the expansion space",
	               "[Math][iVI]",
	               double,
	               (std::complex<double>))
{
	using namespace Detail::IterativeVectorInteraction;
	const auto linearOperator = IdentityOperator<TestType>(3);
	VectorImagePair<TestType> expansionSpace{Eigen::MatrixX<TestType>::Identity(3, 3),
	                                         Eigen::MatrixX<TestType>::Identity(3, 3)};
	const Eigen::MatrixX<TestType> correctionImages = Eigen::MatrixX<TestType>::Identity(3, 2);
	InteriorEigenSolverStatistics statistics;

	const AuxiliaryExpansionCandidates<TestType> candidates{correctionImages, 2};
	AppendAuxiliaryExpansionVectors(linearOperator, candidates, 1e-12, expansionSpace, statistics);

	CHECK(expansionSpace.Vectors.cols() == 3);
	CHECK(statistics.RetainedCorrectionImageVectorCount == 0);
	CHECK(statistics.MultipliedVectorCount == 0);
	CHECK(statistics.OperatorApplicationCount == 0);
}


TEMPLATE_TEST_CASE("iVI integrates and jointly filters auxiliary correction directions",
	               "[Math][iVI]",
	               double,
	               (std::complex<double>))
{
	using namespace Detail::IterativeVectorInteraction;
	using RealScalar = typename Eigen::NumTraits<TestType>::Real;
	Eigen::MatrixX<TestType> matrix = Eigen::MatrixX<TestType>::Zero(3, 3);
	matrix.diagonal() << TestType{1}, TestType{2}, TestType{3};
	matrix(0, 1) = TestType{0.5};
	matrix(1, 0) = TestType{0.5};
	matrix(1, 2) = TestType{0.25};
	matrix(2, 1) = TestType{0.25};
	const DenseSelfAdjointLinearOperator<TestType> linearOperator{matrix};
	const Eigen::MatrixX<TestType> retainedVectors = Eigen::MatrixX<TestType>::Identity(3, 1);
	const VectorImagePair<TestType> retainedSpace{retainedVectors, matrix * retainedVectors};
	InteriorIterationAnalysis<TestType> analysis;
	analysis.OrderedRitzPairs.Eigenvalues = Eigen::VectorX<RealScalar>{{1}};
	analysis.RetainedVectorCount = 1;
	analysis.PrimaryVectorCount = 1;
	const Eigen::VectorX<RealScalar> diagonal = matrix.diagonal().real();

	const auto expandWith = [&](const IterativeVectorInteractionSubspaceExtension extensions)
	{
		InteriorEigenSolverOptions<RealScalar> options{1};
		options.SubspaceExtensions = extensions;
		InteriorIterationState<TestType> state;
		InteriorEigenSolverStatistics statistics;
		CHECK(ExpandForNextIteration(
		              linearOperator, diagonal, analysis, options, retainedSpace, state, statistics)
		      == ExpansionResult::Expanded);
		return std::pair{std::move(state), statistics};
	};

	auto [offDiagonalState, offDiagonalStatistics] = expandWith(
	        IterativeVectorInteractionSubspaceExtension::PreconditionedOffDiagonalCorrectionImages);
	CHECK(offDiagonalState.ExpansionSpace.Vectors.cols() == 3);
	CHECK(offDiagonalState.ExpansionSpace.Images.isApprox(
	        matrix * offDiagonalState.ExpansionSpace.Vectors, 1e-12));
	CHECK(offDiagonalStatistics.GeneratedCorrectionImageVectorCount == 0);
	CHECK(offDiagonalStatistics.RetainedCorrectionImageVectorCount == 0);
	CHECK(offDiagonalStatistics.GeneratedOffDiagonalCorrectionVectorCount == 1);
	CHECK(offDiagonalStatistics.RetainedOffDiagonalCorrectionVectorCount == 1);
	CHECK(offDiagonalStatistics.MultipliedVectorCount == 2);

	auto [combinedState, combinedStatistics] = expandWith(
	        IterativeVectorInteractionSubspaceExtension::CorrectionVectorImages
	        | IterativeVectorInteractionSubspaceExtension::PreconditionedOffDiagonalCorrectionImages);
	CHECK(combinedState.ExpansionSpace.Vectors.cols() == 3);
	CHECK(combinedState.ExpansionSpace.Images.isApprox(matrix * combinedState.ExpansionSpace.Vectors, 1e-12));
	CHECK(combinedStatistics.GeneratedCorrectionImageVectorCount == 1);
	CHECK(combinedStatistics.GeneratedOffDiagonalCorrectionVectorCount == 1);
	CHECK(combinedStatistics.RetainedCorrectionImageVectorCount
	              + combinedStatistics.RetainedOffDiagonalCorrectionVectorCount
	      == 1);
	CHECK(combinedStatistics.MultipliedVectorCount == 2);
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

	InteriorEigenSolverStatistics blockOnlyStatistics;
	const BlockOnlySelfAdjointLinearOperator<TestType> blockOnlyOperator{matrix};
	const auto blockOnlyImages = ApplyOperator(blockOnlyOperator, vectors, blockOnlyStatistics);
	CHECK(blockOnlyImages.isApprox(matrix * vectors));
	CHECK(blockOnlyStatistics.MultipliedVectorCount == 2);
	CHECK(blockOnlyStatistics.OperatorApplicationCount == 1);

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


TEMPLATE_TEST_CASE("iVI sorts public eigenpairs and preserves their alignment",
	               "[Math][iVI]",
	               double,
	               (std::complex<double>))
{
	using namespace Detail::IterativeVectorInteraction;
	using RealScalar = typename Eigen::NumTraits<TestType>::Real;
	Eigen::VectorX<RealScalar> eigenvalues{{2, -1, 1}};
	Eigen::MatrixX<TestType> eigenvectors = Eigen::MatrixX<TestType>::Zero(3, 3);
	eigenvectors(0, 0) = TestType{20};
	eigenvectors(1, 1) = TestType{-10};
	eigenvectors(2, 2) = TestType{10};
	Eigen::VectorX<RealScalar> residualNorms{{0.2, 0.1, 0.3}};

	SortEigenpairsInAscendingOrder(eigenvalues, eigenvectors, residualNorms);

	CHECK(eigenvalues.isApprox(Eigen::VectorX<RealScalar>{{-1, 1, 2}}));
	CHECK(eigenvectors.col(0).isApprox(Eigen::VectorX<TestType>{{0, -10, 0}}));
	CHECK(eigenvectors.col(1).isApprox(Eigen::VectorX<TestType>{{0, 0, 10}}));
	CHECK(eigenvectors.col(2).isApprox(Eigen::VectorX<TestType>{{20, 0, 0}}));
	CHECK(residualNorms.isApprox(Eigen::VectorX<RealScalar>{{0.1, 0.3, 0.2}}));
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
	CHECK_FALSE(selection.HasExcessIntervalRitzCandidates);
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
	CHECK(selection.HasExcessIntervalRitzCandidates);
	CHECK(std::isinf(selection.MaximumMatchedEigenvalueChange));

	const auto emptySelection = SelectInteriorRitzVectors(
	        eigenvalues, matches, Eigen::VectorXd{}, EigenvalueInterval<double>{2, 3}, 2, 1);
	CHECK(emptySelection.IntervalRitzIndices.empty());
	CHECK(emptySelection.AdditionalRitzIndices == std::vector<Eigen::Index>{2});
	CHECK(std::isinf(emptySelection.MaximumMatchedEigenvalueChange));
}


TEST_CASE("iVI demotes unvalidated transient interval overflow", "[Math][iVI]")
{
	using namespace Detail::IterativeVectorInteraction;
	InteriorRitzSelection<double> selection;
	selection.IntervalRitzIndices = {0, 1, 2};
	selection.AdditionalRitzIndices = {3};
	selection.HasExcessIntervalRitzCandidates = true;
	const Eigen::VectorXd residualNorms{{1e-3, 1e-9, 1e-8, 2e-2}};

	DemoteUnvalidatedExcessIntervalRitzVectors(selection, residualNorms, 2, 1e-7);
	CHECK(selection.IntervalRitzIndices == std::vector<Eigen::Index>{1, 2});
	CHECK(selection.AdditionalRitzIndices == std::vector<Eigen::Index>{0, 3});
	CHECK(selection.HasExcessIntervalRitzCandidates);
	CHECK_FALSE(selection.AreExcessIntervalRitzVectorsValidated);
}


TEST_CASE("iVI preserves residual-validated interval overflow", "[Math][iVI]")
{
	using namespace Detail::IterativeVectorInteraction;
	InteriorRitzSelection<double> selection;
	selection.IntervalRitzIndices = {0, 1, 2};
	selection.HasExcessIntervalRitzCandidates = true;
	const Eigen::VectorXd residualNorms{{1e-9, 2e-9, 3e-9}};

	DemoteUnvalidatedExcessIntervalRitzVectors(selection, residualNorms, 2, 1e-7);
	CHECK(selection.IntervalRitzIndices == std::vector<Eigen::Index>{0, 1, 2});
	CHECK(selection.AdditionalRitzIndices.empty());
	CHECK(selection.HasExcessIntervalRitzCandidates);
	CHECK(selection.AreExcessIntervalRitzVectorsValidated);
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
	InteriorEigenSolverOptions<RealScalar> options{3};

	IterativeVectorInteractionSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	CHECK(solver.Compute(linearOperator, EigenvalueInterval<RealScalar>{-1, 1}, options)
	      == InteriorEigenSolverStatus::Converged);
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
	InteriorEigenSolverOptions<double> options{2};

	IterativeVectorInteractionSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	CHECK(solver.Compute(linearOperator, EigenvalueInterval<double>{-1, 1}, options)
	      == InteriorEigenSolverStatus::EigenpairCountLimitExceeded);
	CHECK(solver.Status() == InteriorEigenSolverStatus::EigenpairCountLimitExceeded);
	CHECK(solver.Eigenvalues().isApprox(Eigen::Vector3d{-1, 0, 1}));
}


TEST_CASE("iVI reports an empty interval when exact retained vectors cannot expand", "[Math][iVI]")
{
	Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(8, 8);
	matrix.diagonal() << -3, -2, -1, 0, 1, 2, 3, 4;
	const DenseSelfAdjointLinearOperator<double> linearOperator{matrix};
	InteriorEigenSolverOptions<double> options{2};

	IterativeVectorInteractionSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	CHECK(solver.Compute(linearOperator, EigenvalueInterval<double>{10, 11}, options)
	      == InteriorEigenSolverStatus::ExpansionSpaceExhausted);
	CHECK(solver.Status() == InteriorEigenSolverStatus::ExpansionSpaceExhausted);
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
	InteriorEigenSolverOptions<double> options{4};

	IterativeVectorInteractionSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	(void)solver.Compute(linearOperator, EigenvalueInterval<double>{-10, 10}, options);
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
	InteriorEigenSolverOptions<RealScalar> options{1};
	options.MaximumIterationCount = 50;
	options.EigenvalueChangeTolerance = 1e-10;
	options.ResidualNormTolerance = 1e-10;

	IterativeVectorInteractionSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	(void)solver.Compute(linearOperator, EigenvalueInterval<RealScalar>{lowerBound, upperBound}, options);
	REQUIRE(solver.Status() == InteriorEigenSolverStatus::Converged);
	REQUIRE(solver.Eigenvalues().size() == 1);
	CHECK(solver.Eigenvalues()[0] == Catch::Approx(referenceSolver.eigenvalues()[5]).margin(1e-10));
	CHECK(solver.ResidualNorms()[0] < 1e-10);
	CHECK(solver.Statistics().CompletedIterationCount > 1);
	CHECK(solver.Statistics().GeneralizedSolveCount
	      == 1 + (solver.Statistics().CompletedIterationCount - 1) / options.GeneralizedSolveInterval);
	CHECK(solver.Statistics().MultipliedVectorCount < matrix.rows() * solver.Statistics().CompletedIterationCount);
}


TEMPLATE_TEST_CASE("iVI correction-vector-image expansion converges to the dense reference",
	               "[Math][iVI]",
	               double,
	               (std::complex<double>))
{
	using RealScalar = Eigen::NumTraits<TestType>::Real;
	const Eigen::MatrixX<TestType> matrix = CoupledHermitianMatrix<TestType>(10);
	const Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<TestType>> referenceSolver(matrix);
	const RealScalar lowerBound = (referenceSolver.eigenvalues()[4] + referenceSolver.eigenvalues()[5]) / 2;
	const RealScalar upperBound = (referenceSolver.eigenvalues()[5] + referenceSolver.eigenvalues()[6]) / 2;
	const DenseSelfAdjointLinearOperator<TestType> linearOperator{matrix};
	InteriorEigenSolverOptions<RealScalar> options{1};
	options.MaximumIterationCount = 50;
	options.EigenvalueChangeTolerance = 1e-10;
	options.ResidualNormTolerance = 1e-10;
	options.SubspaceExtensions = IterativeVectorInteractionSubspaceExtension::AdditionalRitzVectors
	                             | IterativeVectorInteractionSubspaceExtension::CorrectionVectorImages;

	IterativeVectorInteractionSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	(void)solver.Compute(linearOperator, EigenvalueInterval<RealScalar>{lowerBound, upperBound}, options);

	REQUIRE(solver.Status() == InteriorEigenSolverStatus::Converged);
	REQUIRE(solver.Eigenvalues().size() == 1);
	CHECK(solver.Eigenvalues()[0] == Catch::Approx(referenceSolver.eigenvalues()[5]).margin(1e-10));
	CHECK(solver.ResidualNorms()[0] < 1e-10);
	CHECK(solver.Statistics().GeneratedCorrectionImageVectorCount
	      == solver.Statistics().RetainedCorrectionVectorCount);
	CHECK(solver.Statistics().RetainedCorrectionImageVectorCount > 0);
	CHECK(solver.Statistics().RetainedCorrectionImageVectorCount
	      <= solver.Statistics().GeneratedCorrectionImageVectorCount);
}


TEMPLATE_TEST_CASE("iVI off-diagonal correction expansion converges to the dense reference",
	               "[Math][iVI]",
	               double,
	               (std::complex<double>))
{
	using RealScalar = Eigen::NumTraits<TestType>::Real;
	const Eigen::MatrixX<TestType> matrix = CoupledHermitianMatrix<TestType>(10);
	const Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<TestType>> referenceSolver(matrix);
	const RealScalar lowerBound = (referenceSolver.eigenvalues()[4] + referenceSolver.eigenvalues()[5]) / 2;
	const RealScalar upperBound = (referenceSolver.eigenvalues()[5] + referenceSolver.eigenvalues()[6]) / 2;
	const DenseSelfAdjointLinearOperator<TestType> linearOperator{matrix};
	InteriorEigenSolverOptions<RealScalar> options{1};
	options.MaximumIterationCount = 50;
	options.EigenvalueChangeTolerance = 1e-10;
	options.ResidualNormTolerance = 1e-10;
	options.SubspaceExtensions = IterativeVectorInteractionSubspaceExtension::AdditionalRitzVectors
	                             | IterativeVectorInteractionSubspaceExtension::PreconditionedOffDiagonalCorrectionImages;

	IterativeVectorInteractionSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	(void)solver.Compute(linearOperator, EigenvalueInterval<RealScalar>{lowerBound, upperBound}, options);

	REQUIRE(solver.Status() == InteriorEigenSolverStatus::Converged);
	REQUIRE(solver.Eigenvalues().size() == 1);
	CHECK(solver.Eigenvalues()[0] == Catch::Approx(referenceSolver.eigenvalues()[5]).margin(1e-10));
	CHECK(solver.ResidualNorms()[0] < 1e-10);
	CHECK(solver.Statistics().GeneratedOffDiagonalCorrectionVectorCount
	      == solver.Statistics().RetainedCorrectionVectorCount);
	CHECK(solver.Statistics().RetainedOffDiagonalCorrectionVectorCount > 0);
	CHECK(solver.Statistics().RetainedOffDiagonalCorrectionVectorCount
	      <= solver.Statistics().GeneratedOffDiagonalCorrectionVectorCount);
}


TEST_CASE("iVI exposes a partial result at the iteration limit", "[Math][iVI]")
{
	const Eigen::MatrixXd matrix = CoupledHermitianMatrix<double>(10);
	const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> referenceSolver(matrix);
	const double lowerBound = (referenceSolver.eigenvalues()[4] + referenceSolver.eigenvalues()[5]) / 2;
	const double upperBound = (referenceSolver.eigenvalues()[5] + referenceSolver.eigenvalues()[6]) / 2;
	const DenseSelfAdjointLinearOperator<double> linearOperator{matrix};
	InteriorEigenSolverOptions<double> options{1};
	options.MaximumIterationCount = 1;
	options.ResidualNormTolerance = 1e-14;

	IterativeVectorInteractionSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	(void)solver.Compute(linearOperator, EigenvalueInterval<double>{lowerBound, upperBound}, options);
	CHECK(solver.Status() == InteriorEigenSolverStatus::IterationLimitReached);
	CHECK(solver.Eigenvalues().size() == 1);
	CHECK(solver.ResidualNorms().size() == 1);
}


TEST_CASE("iVI uses the column-wise operator fallback in an end-to-end solve", "[Math][iVI]")
{
	const Eigen::MatrixXd matrix = Eigen::VectorXd{{-2, -1, 0, 1, 2}}.asDiagonal();
	const VectorOnlySelfAdjointLinearOperator<double> linearOperator{matrix};
	InteriorEigenSolverOptions<double> options{1};

	IterativeVectorInteractionSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	(void)solver.Compute(linearOperator, EigenvalueInterval<double>{0, 0}, options);
	CHECK(solver.Status() == InteriorEigenSolverStatus::Converged);
	CHECK(solver.Eigenvalues().isApprox(Eigen::VectorXd::Zero(1)));
	CHECK(solver.Statistics().OperatorApplicationCount == matrix.rows());
	CHECK(solver.Statistics().MultipliedVectorCount == matrix.rows());
}


TEMPLATE_TEST_CASE("iVI recovers discarded previous Ritz directions in coefficient space",
	               "[Math][iVI]",
	               double,
	               (std::complex<double>))
{
	using namespace Detail::IterativeVectorInteraction;
	using RealScalar = Eigen::NumTraits<TestType>::Real;
	const RealScalar inverseSquareRootOfTwo = RealScalar{1} / std::sqrt(RealScalar{2});
	Eigen::MatrixX<TestType> selectedCoefficients = Eigen::MatrixX<TestType>::Zero(3, 1);
	selectedCoefficients(0, 0) = inverseSquareRootOfTwo;
	selectedCoefficients(1, 0) = inverseSquareRootOfTwo;

	const auto recyclingCoefficients =
	        FormPreviousRitzVectorRecyclingCoefficients(Eigen::MatrixX<TestType>::Identity(3, 3).eval(),
	                                                      selectedCoefficients,
	                                                      2,
	                                                      RealScalar{1e-10},
	                                                      RealScalar{1e-2});
	REQUIRE(recyclingCoefficients.rows() == 3);
	REQUIRE(recyclingCoefficients.cols() == 1);
	CHECK((selectedCoefficients.adjoint() * recyclingCoefficients).norm() < 1e-12);
	CHECK((recyclingCoefficients.adjoint() * recyclingCoefficients - Eigen::MatrixX<TestType>::Identity(1, 1)).norm()
	      < 1e-12);

	const Eigen::MatrixX<TestType> fullPreviousSpace = Eigen::MatrixX<TestType>::Identity(3, 2);
	CHECK((fullPreviousSpace * recyclingCoefficients.topRows(2) - recyclingCoefficients).norm() < 1e-12);
}


TEST_CASE("iVI recycling discards directions already retained by Ritz selection", "[Math][iVI]")
{
	using namespace Detail::IterativeVectorInteraction;
	const Eigen::MatrixXd selectedCoefficients = Eigen::MatrixXd::Identity(3, 2);
	const auto recyclingCoefficients = FormPreviousRitzVectorRecyclingCoefficients(
	        Eigen::MatrixXd::Identity(3, 3).eval(), selectedCoefficients, 2, 1e-10, 1e-2);
	CHECK(recyclingCoefficients.cols() == 0);
}


TEMPLATE_TEST_CASE("iVI recycling preserves selected generalized Ritz vectors",
	               "[Math][iVI]",
	               double,
	               (std::complex<double>))
{
	using namespace Detail::IterativeVectorInteraction;
	Eigen::MatrixX<TestType> expansionVectors = Eigen::MatrixX<TestType>::Zero(3, 3);
	expansionVectors.diagonal() << TestType{2}, TestType{3}, TestType{4};
	Eigen::MatrixX<TestType> selectedCoefficients = Eigen::MatrixX<TestType>::Zero(3, 2);
	selectedCoefficients(1, 0) = TestType{1} / TestType{3};
	selectedCoefficients(2, 1) = TestType{1} / TestType{4};

	InteriorIterationAnalysis<TestType> analysis;
	analysis.OrderedRitzPairs.Eigenvalues = Eigen::Vector3d{{1, 2, 3}};
	analysis.OrderedRitzPairs.Eigenvectors = Eigen::MatrixX<TestType>::Zero(3, 3);
	analysis.OrderedRitzPairs.Eigenvectors.leftCols(2) = selectedCoefficients;
	analysis.RetainedVectorCount = 2;
	InteriorIterationState<TestType> state;
	state.ExpansionSpace = {expansionVectors, expansionVectors};
	state.PreviousPrimaryVectorCount = 1;
	state.IsPreviousRitzVectorRecyclingActive = true;
	InteriorEigenSolverOptions<double> options{2};
	options.IsPreviousRitzVectorRecyclingDynamicallyEnabled = false;
	InteriorEigenSolverStatistics statistics;

	const Eigen::MatrixX<TestType> collapseCoefficients =
	        FormCollapseCoefficients(analysis, 1, options, state, statistics);

	REQUIRE(collapseCoefficients.cols() == 3);
	CHECK(collapseCoefficients.leftCols(2).isApprox(selectedCoefficients));
	const Eigen::MatrixX<TestType> retainedVectors = expansionVectors * collapseCoefficients;
	CHECK((retainedVectors.adjoint() * retainedVectors)
	              .isApprox(Eigen::MatrixX<TestType>::Identity(3, 3), 1e-12));
	CHECK(statistics.RecycledVectorCount == 1);

	options.SubspaceExtensions = IterativeVectorInteractionSubspaceExtension::AdditionalRitzVectors;
	InteriorEigenSolverStatistics disabledStatistics;
	const Eigen::MatrixX<TestType> coefficientsWithoutRecycling =
	        FormCollapseCoefficients(analysis, 1, options, state, disabledStatistics);
	CHECK(coefficientsWithoutRecycling.cols() == 2);
	CHECK(coefficientsWithoutRecycling.isApprox(selectedCoefficients));
	CHECK(disabledStatistics.RecycledVectorCount == 0);
}


TEST_CASE("iVI recycling activation follows interior convergence progress", "[Math][iVI]")
{
	using Detail::IterativeVectorInteraction::IsPreviousRitzVectorRecyclingActiveFor;
	CHECK_FALSE(IsPreviousRitzVectorRecyclingActiveFor(true, 0.2));
	CHECK_FALSE(IsPreviousRitzVectorRecyclingActiveFor(true, 1e-6));
	CHECK(IsPreviousRitzVectorRecyclingActiveFor(false, 0.005));
	CHECK(IsPreviousRitzVectorRecyclingActiveFor(true, 0.05));
	CHECK_FALSE(IsPreviousRitzVectorRecyclingActiveFor(false, 0.05));
}


TEST_CASE("iVI appends previous Ritz directions when dynamic recycling is disabled", "[Math][iVI]")
{
	const Eigen::MatrixXd matrix = CoupledHermitianMatrix<double>(10);
	const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> referenceSolver(matrix);
	const double lowerBound = (referenceSolver.eigenvalues()[4] + referenceSolver.eigenvalues()[5]) / 2;
	const double upperBound = (referenceSolver.eigenvalues()[5] + referenceSolver.eigenvalues()[6]) / 2;
	const DenseSelfAdjointLinearOperator<double> linearOperator{matrix};
	InteriorEigenSolverOptions<double> options{1};
	options.MaximumIterationCount = 2;
	options.EigenvalueChangeTolerance = 1e-15;
	options.ResidualNormTolerance = 1e-15;
	options.SubspaceExtensions = IterativeVectorInteractionSubspaceExtension::AdditionalRitzVectors
	                             | IterativeVectorInteractionSubspaceExtension::PreviousRitzVectors;
	options.IsPreviousRitzVectorRecyclingDynamicallyEnabled = false;

	IterativeVectorInteractionSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	(void)solver.Compute(linearOperator, EigenvalueInterval<double>{lowerBound, upperBound}, options);
	CHECK(solver.Status() == InteriorEigenSolverStatus::IterationLimitReached);
	CHECK(solver.Statistics().CompletedIterationCount == 2);
	CHECK(solver.Statistics().RecycledVectorCount > 0);
}


TEMPLATE_TEST_CASE("iVI can solve without optional subspace extensions",
	               "[Math][iVI]",
	               double,
	               (std::complex<double>))
{
	using RealScalar = typename Eigen::NumTraits<TestType>::Real;
	Eigen::MatrixX<TestType> matrix = Eigen::MatrixX<TestType>::Zero(6, 6);
	matrix.diagonal() << TestType{-3}, TestType{-2}, TestType{-1}, TestType{1}, TestType{2}, TestType{3};
	const DenseSelfAdjointLinearOperator<TestType> linearOperator{matrix};
	InteriorEigenSolverOptions<RealScalar> options{1};
	options.SubspaceExtensions = IterativeVectorInteractionSubspaceExtension::None;

	IterativeVectorInteractionSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	REQUIRE(solver.Compute(linearOperator, EigenvalueInterval<RealScalar>{-1.5, -0.5}, options)
	        == InteriorEigenSolverStatus::Converged);
	REQUIRE(solver.Eigenvalues().size() == 1);
	CHECK(solver.Eigenvalues()[0] == Catch::Approx(-1));
	CHECK(solver.Statistics().RecycledVectorCount == 0);
	CHECK(solver.Statistics().RetainedAdditionalRitzVectorCount == 0);
	CHECK(solver.Statistics().GeneratedCorrectionImageVectorCount == 0);
	CHECK(solver.Statistics().RetainedCorrectionImageVectorCount == 0);
	CHECK(solver.Statistics().GeneratedOffDiagonalCorrectionVectorCount == 0);
	CHECK(solver.Statistics().RetainedOffDiagonalCorrectionVectorCount == 0);

	options.SubspaceExtensions = IterativeVectorInteractionSubspaceExtension::AdditionalRitzVectors;
	IterativeVectorInteractionSelfAdjointEigenSolver<decltype(linearOperator)> solverWithAdditionalRitzVectors;
	REQUIRE(solverWithAdditionalRitzVectors.Compute(
	                linearOperator, EigenvalueInterval<RealScalar>{-1.5, -0.5}, options)
	        == InteriorEigenSolverStatus::Converged);
	CHECK(solverWithAdditionalRitzVectors.Statistics().RetainedAdditionalRitzVectorCount == 5);
}


TEMPLATE_TEST_CASE("iVI freezes only stable interval Ritz vectors",
	               "[Math][iVI]",
	               double,
	               (std::complex<double>))
{
	using namespace Detail::IterativeVectorInteraction;
	using RealScalar = Eigen::NumTraits<TestType>::Real;
	const RealScalar inverseSquareRootOfTwo = RealScalar{1} / std::sqrt(RealScalar{2});
	const TestType phase = []
	{
		if constexpr (Eigen::NumTraits<TestType>::IsComplex)
		{
			return TestType{0, 1};
		}
		else
		{
			return TestType{1};
		}
	}();
	Eigen::MatrixX<TestType> reducedEigenvectors = Eigen::MatrixX<TestType>::Zero(4, 3);
	reducedEigenvectors(0, 0) = phase;
	reducedEigenvectors(0, 1) = inverseSquareRootOfTwo;
	reducedEigenvectors(2, 1) = inverseSquareRootOfTwo;
	reducedEigenvectors(1, 2) = TestType{1};
	const Eigen::VectorX<RealScalar> residualNorms{{RealScalar{1e-9}, RealScalar{1e-9}, RealScalar{1e-3}}};
	const std::vector<Eigen::Index> intervalRitzIndices{0, 1, 2};

	const auto isFrozen = DetermineFrozenRitzVectors(
	        reducedEigenvectors, residualNorms, intervalRitzIndices, 2, RealScalar{1e-8}, RealScalar{1e-7});
	CHECK(isFrozen == std::vector<bool>{true, false, false});

	const auto isFrozenWithoutPreviousPrimaryVectors = DetermineFrozenRitzVectors(
	        reducedEigenvectors, residualNorms, intervalRitzIndices, 0, RealScalar{1e-8}, RealScalar{1e-7});
	CHECK(isFrozenWithoutPreviousPrimaryVectors == std::vector<bool>{false, false, false});
}


TEST_CASE("iVI freezing can be disabled", "[Math][iVI]")
{
	const Eigen::MatrixXd matrix = CoupledHermitianMatrix<double>(10);
	const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> referenceSolver(matrix);
	const double lowerBound = (referenceSolver.eigenvalues()[4] + referenceSolver.eigenvalues()[5]) / 2;
	const double upperBound = (referenceSolver.eigenvalues()[5] + referenceSolver.eigenvalues()[6]) / 2;
	const DenseSelfAdjointLinearOperator<double> linearOperator{matrix};
	InteriorEigenSolverOptions<double> options{1};
	options.MaximumIterationCount = 2;
	options.EigenvalueChangeTolerance = 1e-15;
	options.ResidualNormTolerance = 1e-15;
	options.FreezingCoefficientTolerance = 1;
	options.FreezingResidualNormTolerance = 1e6;
	options.IsFreezingEnabled = false;

	IterativeVectorInteractionSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	(void)solver.Compute(linearOperator, EigenvalueInterval<double>{lowerBound, upperBound}, options);
	CHECK(solver.Statistics().CurrentFrozenVectorCount == 0);
	CHECK(solver.Statistics().MaximumFrozenVectorCount == 0);
}


TEST_CASE("iVI skips corrections for frozen Ritz vectors", "[Math][iVI]")
{
	const Eigen::MatrixXd matrix = CoupledHermitianMatrix<double>(10);
	const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> referenceSolver(matrix);
	const double lowerBound = (referenceSolver.eigenvalues()[4] + referenceSolver.eigenvalues()[5]) / 2;
	const double upperBound = (referenceSolver.eigenvalues()[5] + referenceSolver.eigenvalues()[6]) / 2;
	const DenseSelfAdjointLinearOperator<double> linearOperator{matrix};
	InteriorEigenSolverOptions<double> options{1};
	options.MaximumIterationCount = 2;
	options.EigenvalueChangeTolerance = 1e-15;
	options.ResidualNormTolerance = 1e-15;
	options.FreezingCoefficientTolerance = 1;
	options.FreezingResidualNormTolerance = 1e6;

	IterativeVectorInteractionSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	(void)solver.Compute(linearOperator, EigenvalueInterval<double>{lowerBound, upperBound}, options);
	CHECK(solver.Statistics().CurrentFrozenVectorCount == 1);
	CHECK(solver.Statistics().MaximumFrozenVectorCount == 1);
}


TEMPLATE_TEST_CASE("iVI can use a generalized reduced solve on every iteration",
	               "[Math][iVI]",
	               double,
	               (std::complex<double>))
{
	using RealScalar = Eigen::NumTraits<TestType>::Real;
	const Eigen::MatrixX<TestType> matrix = CoupledHermitianMatrix<TestType>(10);
	const Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<TestType>> referenceSolver(matrix);
	const RealScalar lowerBound = (referenceSolver.eigenvalues()[4] + referenceSolver.eigenvalues()[5]) / 2;
	const RealScalar upperBound = (referenceSolver.eigenvalues()[5] + referenceSolver.eigenvalues()[6]) / 2;
	const DenseSelfAdjointLinearOperator<TestType> linearOperator{matrix};
	InteriorEigenSolverOptions<RealScalar> options{1};
	options.MaximumIterationCount = 2;
	options.GeneralizedSolveInterval = 1;
	options.EigenvalueChangeTolerance = RealScalar{1e-15};
	options.ResidualNormTolerance = RealScalar{1e-15};
	options.IsFreezingEnabled = false;

	IterativeVectorInteractionSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	(void)solver.Compute(linearOperator, EigenvalueInterval<RealScalar>{lowerBound, upperBound}, options);
	CHECK(solver.Statistics().CompletedIterationCount == 2);
	CHECK(solver.Statistics().GeneralizedSolveCount == solver.Statistics().CompletedIterationCount);
}


TEMPLATE_TEST_CASE("iVI finds a complete interior set for a large matrix-free Hermitian operator",
	               "[Math][iVI][Validation]",
	               double,
	               (std::complex<double>))
{
	using RealScalar = Eigen::NumTraits<TestType>::Real;
	constexpr Eigen::Index dimension = 256;
	Eigen::VectorX<RealScalar> diagonal(dimension);
	for (Eigen::Index index = 0; index < dimension; index++)
	{
		diagonal[index] = RealScalar{0.1} * static_cast<RealScalar>(index - dimension / 2);
	}
	const TestType coupling = []
	{
		if constexpr (Eigen::NumTraits<TestType>::IsComplex)
		{
			return TestType{0.025, 0.01};
		}
		else
		{
			return TestType{0.025};
		}
	}();
	const TridiagonalSelfAdjointLinearOperator<TestType> linearOperator{std::move(diagonal), coupling};
	const Eigen::MatrixX<TestType> referenceMatrix = FormDenseMatrix(linearOperator);
	const Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<TestType>> referenceSolver(referenceMatrix);
	REQUIRE(referenceSolver.info() == Eigen::Success);
	const Eigen::Index firstExpectedIndex = dimension / 2 - 2;
	constexpr Eigen::Index expectedEigenpairCount = 3;
	const RealScalar lowerBound =
	        (referenceSolver.eigenvalues()[firstExpectedIndex - 1]
	         + referenceSolver.eigenvalues()[firstExpectedIndex])
	        / 2;
	const RealScalar upperBound =
	        (referenceSolver.eigenvalues()[firstExpectedIndex + expectedEigenpairCount - 1]
	         + referenceSolver.eigenvalues()[firstExpectedIndex + expectedEigenpairCount])
	        / 2;

	InteriorEigenSolverOptions<RealScalar> options{expectedEigenpairCount};
	options.MaximumIterationCount = 100;
	options.EigenvalueChangeTolerance = RealScalar{1e-8};
	options.ResidualNormTolerance = RealScalar{1e-8};
	options.FreezingResidualNormTolerance = options.ResidualNormTolerance;
	// The rank threshold applies to squared correction magnitudes. Keep it below the
	// requested residual scale without admitting roundoff-only correction directions.
	options.LinearDependenceTolerance = RealScalar{1e-14};
	IterativeVectorInteractionSelfAdjointEigenSolver<decltype(linearOperator)> solver;
	(void)solver.Compute(linearOperator, EigenvalueInterval<RealScalar>{lowerBound, upperBound}, options);

	INFO("iterations: " << solver.Statistics().CompletedIterationCount);
	INFO("maximum expansion size: " << solver.Statistics().MaximumExpansionSpaceSize);
	INFO("current frozen vectors: " << solver.Statistics().CurrentFrozenVectorCount);
	INFO("maximum frozen vectors: " << solver.Statistics().MaximumFrozenVectorCount);
	INFO("eigenvalues: " << solver.Eigenvalues().transpose());
	INFO("residual norms: " << solver.ResidualNorms().transpose());
	REQUIRE(solver.Status() == InteriorEigenSolverStatus::Converged);
	REQUIRE(solver.Eigenvalues().size() == expectedEigenpairCount);
	CHECK(std::ranges::is_sorted(solver.Eigenvalues()));
	Eigen::VectorX<RealScalar> actualEigenvalues = solver.Eigenvalues();
	std::sort(actualEigenvalues.begin(), actualEigenvalues.end());
	CHECK(actualEigenvalues.isApprox(
	        referenceSolver.eigenvalues().segment(firstExpectedIndex, expectedEigenpairCount), RealScalar{1e-8}));
	CHECK(solver.ResidualNorms().maxCoeff() <= options.ResidualNormTolerance);
	CHECK((solver.Eigenvectors().adjoint() * solver.Eigenvectors())
	              .isApprox(Eigen::MatrixX<TestType>::Identity(expectedEigenpairCount, expectedEigenpairCount),
	                        RealScalar{1e-9}));
	CHECK(solver.Statistics().MaximumExpansionSpaceSize < dimension / 4);
}


TEST_CASE("iVI finds the complete interval set for a matrix-free hub-and-band operator",
	      "[Math][iVI][Validation]")
{
	constexpr Eigen::Index dimension = 512;
	const HubAndBandSelfAdjointLinearOperator linearOperator{dimension};
	const Eigen::MatrixXd referenceMatrix =
	        linearOperator.ApplyOn(Eigen::MatrixXd::Identity(dimension, dimension));
	const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> referenceSolver(referenceMatrix);
	REQUIRE(referenceSolver.info() == Eigen::Success);
	const EigenvalueInterval<double> interval{15, 16};
	std::vector<double> expectedEigenvalues;
	for (const double eigenvalue : referenceSolver.eigenvalues())
	{
		if (interval.IsContaining(eigenvalue))
		{
			expectedEigenvalues.push_back(eigenvalue);
		}
	}
	REQUIRE_FALSE(expectedEigenvalues.empty());

	InteriorEigenSolverOptions<double> options{static_cast<Eigen::Index>(expectedEigenvalues.size())};
	options.MaximumIterationCount = 200;
	// BDF terminates this problem using eigenvalue movement and then applies a much
	// looser residual-quality filter. This remains strict enough to validate each pair.
	options.ResidualNormTolerance = 1e-3;
	IterativeVectorInteractionSelfAdjointEigenSolver<HubAndBandSelfAdjointLinearOperator> solver;
	(void)solver.Compute(linearOperator, interval, options);

	INFO("expected eigenpair count: " << expectedEigenvalues.size());
	INFO("iterations: " << solver.Statistics().CompletedIterationCount);
	INFO("maximum expansion size: " << solver.Statistics().MaximumExpansionSpaceSize);
	INFO("current frozen vectors: " << solver.Statistics().CurrentFrozenVectorCount);
	INFO("maximum frozen vectors: " << solver.Statistics().MaximumFrozenVectorCount);
	INFO("returned eigenvalues: " << solver.Eigenvalues().transpose());
	INFO("residual norms: " << solver.ResidualNorms().transpose());
	REQUIRE(solver.Status() == InteriorEigenSolverStatus::Converged);
	REQUIRE(solver.Eigenvalues().size() == static_cast<Eigen::Index>(expectedEigenvalues.size()));
	CHECK(std::ranges::is_sorted(solver.Eigenvalues()));
	Eigen::VectorXd actualEigenvalues = solver.Eigenvalues();
	std::ranges::sort(actualEigenvalues);
	CHECK(std::equal(actualEigenvalues.begin(),
	                 actualEigenvalues.end(),
	                 expectedEigenvalues.begin(),
	                 [](const double actual, const double expected)
	                 { return actual == Catch::Approx(expected).margin(1e-7); }));
	CHECK(solver.ResidualNorms().maxCoeff() <= options.ResidualNormTolerance);
	CHECK((solver.Eigenvectors().adjoint() * solver.Eigenvectors())
	              .isApprox(Eigen::MatrixXd::Identity(
	                                static_cast<Eigen::Index>(expectedEigenvalues.size()),
	                                static_cast<Eigen::Index>(expectedEigenvalues.size())),
	                        1e-8));
	CHECK(solver.Statistics().MaximumExpansionSpaceSize < dimension);
}
