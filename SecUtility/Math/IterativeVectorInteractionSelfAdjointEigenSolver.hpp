// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#if __cplusplus < 202002L
#error "IterativeVectorInteractionSelfAdjointEigenSolver requires C++20 or later"
#endif

#include <SecUtility/Diagnostic/Exception.hpp>
#include <SecUtility/Math/Core.hpp>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>


namespace SecUtility::Math
{
	namespace Detail
	{
		template <typename T>
		inline constexpr bool IsSupportedEigenSolverScalar = std::floating_point<T>;

		template <std::floating_point T>
		inline constexpr bool IsSupportedEigenSolverScalar<std::complex<T>> = true;
	}


	template <typename RealScalar>
	struct EigenvalueInterval
	{
		RealScalar LowerBound{};
		RealScalar UpperBound{};

		[[nodiscard]] constexpr bool IsValid() const noexcept
		{
			return LowerBound <= UpperBound;
		}

		[[nodiscard]] constexpr bool IsContaining(const RealScalar value) const noexcept
		{
			return value >= LowerBound && value <= UpperBound;
		}
	};


	template <typename RealScalar>
	struct InteriorEigenSolverOptions
	{
		explicit constexpr InteriorEigenSolverOptions(const Eigen::Index eigenpairCountLimit) noexcept
		    : EigenpairCountLimit(eigenpairCountLimit)
		{
		}

		// This limits the expected result count, but an exceeded result may expose every
		// residual-validated candidate so the overflow can be inspected.
		Eigen::Index EigenpairCountLimit;
		Eigen::Index MaximumIterationCount = 1000;
		// Zero selects an automatic count derived from EigenpairCountLimit.
		Eigen::Index AdditionalRitzVectorCount = 0;
		Eigen::Index GeneralizedSolveInterval = 5;

		RealScalar EigenvalueChangeTolerance = static_cast<RealScalar>(1e-7);
		RealScalar ResidualNormTolerance = static_cast<RealScalar>(1e-7);
		RealScalar PreconditionerDenominatorFloor = static_cast<RealScalar>(1e-2);
		RealScalar LinearDependenceTolerance = static_cast<RealScalar>(1e-10);
		RealScalar PreviousRitzVectorRecyclingTolerance = static_cast<RealScalar>(1e-8);
		RealScalar RecyclingRefinementTolerance = static_cast<RealScalar>(1e-2);
		RealScalar FreezingCoefficientTolerance = static_cast<RealScalar>(1e-8);
		// Zero uses ResidualNormTolerance.
		RealScalar FreezingResidualNormTolerance{};
		RealScalar ReducedMatrixAsymmetryTolerance = static_cast<RealScalar>(1e-6);

		bool IsPreviousRitzVectorRecyclingEnabled = true;
		bool IsPreviousRitzVectorRecyclingDynamicallyEnabled = true;
		bool IsFreezingEnabled = true;
	};


	enum class InteriorEigenSolverStatus
	{
		NotComputed,
		Converged,
		EigenpairCountLimitExceeded,
		IterationLimitReached,
		ExpansionSpaceExhausted,
		NumericalFailure
	};


	struct InteriorEigenSolverStatistics
	{
		Eigen::Index CompletedIterationCount = 0;
		Eigen::Index MultipliedVectorCount = 0;
		Eigen::Index OperatorApplicationCount = 0;
		Eigen::Index MaximumExpansionSpaceSize = 0;
		Eigen::Index ExplicitImageRecalculationCount = 0;
		Eigen::Index GeneralizedSolveCount = 0;
		Eigen::Index RecycledVectorCount = 0;
		Eigen::Index CurrentFrozenVectorCount = 0;
		Eigen::Index MaximumFrozenVectorCount = 0;
	};


	namespace Detail::IterativeVectorInteraction
	{
		template <typename Scalar>
		struct VectorImagePair
		{
			Eigen::MatrixX<Scalar> Vectors;
			Eigen::MatrixX<Scalar> Images;
		};


		template <typename RealScalar>
		struct RitzVectorMatch
		{
			Eigen::Index PreviousIndex = Eigen::Index{-1};
			Eigen::Index CurrentIndex = Eigen::Index{-1};
			RealScalar Population{};
			bool IsPrimaryVectorContinuation = false;
			bool IsRetainedVectorContinuation = false;
			bool IsWeakMatch = true;

			[[nodiscard]] bool IsMatched() const noexcept
			{
				return PreviousIndex >= 0;
			}
		};


		template <typename RealScalar>
		struct InteriorRitzSelection
		{
			std::vector<Eigen::Index> IntervalRitzIndices;
			std::vector<Eigen::Index> AdditionalRitzIndices;
			RealScalar MaximumMatchedEigenvalueChange = std::numeric_limits<RealScalar>::infinity();
			bool HasUnmatchedIntervalRitzVector = false;
			bool IsEigenpairCountLimitExceeded = false;
		};


		template <typename Scalar>
		struct RitzPairs
		{
			Eigen::VectorX<typename Eigen::NumTraits<Scalar>::Real> Eigenvalues;
			Eigen::MatrixX<Scalar> Eigenvectors;
		};
	}


	template <typename Operator>
	using LinearOperatorScalar = std::remove_cvref_t<Operator>::Scalar;

	template <typename Operator>
	using LinearOperatorRealScalar = Eigen::NumTraits<LinearOperatorScalar<Operator>>::Real;


	template <typename Operator>
	concept SelfAdjointLinearOperatorBase = requires(const std::remove_reference_t<Operator>& linearOperator) {
		typename LinearOperatorScalar<Operator>;
		typename LinearOperatorRealScalar<Operator>;
		requires Detail::IsSupportedEigenSolverScalar<LinearOperatorScalar<Operator>>;

		{ linearOperator.rows() } -> std::convertible_to<Eigen::Index>;
		{ linearOperator.cols() } -> std::convertible_to<Eigen::Index>;

		{ linearOperator.Diagonal().size() } -> std::convertible_to<Eigen::Index>;
		{ linearOperator.Diagonal()[Eigen::Index{}] } -> std::convertible_to<LinearOperatorRealScalar<Operator>>;
	};


	template <typename Operator>
	concept ScalarSelfAdjointLinearOperator =
	        SelfAdjointLinearOperatorBase<Operator>
	        && requires(const std::remove_reference_t<Operator>& linearOperator,
	                    const Eigen::VectorX<LinearOperatorScalar<Operator>>& vector) {
		           { linearOperator.ApplyOn(vector).size() } -> std::convertible_to<Eigen::Index>;
		           {
			           linearOperator.ApplyOn(vector)[Eigen::Index{}]
		           } -> std::convertible_to<LinearOperatorScalar<Operator>>;
	           };


	template <typename Operator>
	concept BlockSelfAdjointLinearOperator =
	        SelfAdjointLinearOperatorBase<Operator>
	        && requires(const std::remove_reference_t<Operator>& linearOperator,
	                    const Eigen::MatrixX<LinearOperatorScalar<Operator>>& vectors) {
		           { linearOperator.ApplyOn(vectors).rows() } -> std::convertible_to<Eigen::Index>;
		           { linearOperator.ApplyOn(vectors).cols() } -> std::convertible_to<Eigen::Index>;
		           {
			           linearOperator.ApplyOn(vectors)(Eigen::Index{}, Eigen::Index{})
		           } -> std::convertible_to<LinearOperatorScalar<Operator>>;
	           };


	template <typename Operator>
	concept SelfAdjointLinearOperator =
	        ScalarSelfAdjointLinearOperator<Operator> || BlockSelfAdjointLinearOperator<Operator>;


	namespace Detail::IterativeVectorInteraction
	{
		template <typename Scalar>
		Eigen::MatrixX<typename Eigen::NumTraits<Scalar>::Real> FormRitzPopulationMatrix(
		        const Eigen::MatrixX<Scalar>& reducedEigenvectors, const Eigen::Index previousRetainedVectorCount)
		{
			return reducedEigenvectors.topRows(previousRetainedVectorCount).transpose().cwiseAbs2();
		}


		template <typename RealScalar>
		std::vector<RitzVectorMatch<RealScalar>> FormRitzVectorMatches(
		        const Eigen::MatrixX<RealScalar>& populations,
		        const std::vector<Eigen::Index>& previousIndexForCurrentIndex,
		        const Eigen::Index previousPrimaryVectorCount,
		        const Eigen::Index previousRetainedVectorCount,
		        const RealScalar weakMatchPopulationThreshold)
		{
			std::vector<RitzVectorMatch<RealScalar>> matches(static_cast<std::size_t>(populations.rows()));
			for (Eigen::Index currentIndex = 0; currentIndex < populations.rows(); currentIndex++)
			{
				auto& match = matches[static_cast<std::size_t>(currentIndex)];
				match.CurrentIndex = currentIndex;
				match.PreviousIndex = previousIndexForCurrentIndex[static_cast<std::size_t>(currentIndex)];
				if (!match.IsMatched())
				{
					continue;
				}
				match.Population = populations(currentIndex, match.PreviousIndex);
				match.IsPrimaryVectorContinuation = match.PreviousIndex < previousPrimaryVectorCount;
				match.IsRetainedVectorContinuation = match.PreviousIndex < previousRetainedVectorCount;
				match.IsWeakMatch =
				        match.Population <= weakMatchPopulationThreshold || !match.IsPrimaryVectorContinuation;
			}
			return matches;
		}

		template <typename RealScalar>
		std::vector<RitzVectorMatch<RealScalar>> MatchRitzVectors_Greedy(
		        const Eigen::MatrixX<RealScalar>& populations,
		        const Eigen::Index previousPrimaryVectorCount,
		        const Eigen::Index previousRetainedVectorCount,
		        const RealScalar weakMatchPopulationThreshold = RealScalar{0.5})
		{
			std::vector<Eigen::Index> sortedPopulationIndices(static_cast<std::size_t>(populations.size()));
			std::iota(sortedPopulationIndices.begin(), sortedPopulationIndices.end(), Eigen::Index{0});
			std::ranges::stable_sort(sortedPopulationIndices,
			                         std::greater<>{},
			                         [&populations](const Eigen::Index index) { return populations.data()[index]; });

			std::vector isCurrentIndexAssigned(static_cast<std::size_t>(populations.rows()), false);
			std::vector isPreviousIndexAssigned(static_cast<std::size_t>(populations.cols()), false);
			std::vector previousIndexForCurrentIndex(static_cast<std::size_t>(populations.rows()), Eigen::Index{-1});
			Eigen::Index assignmentCount = 0;
			const Eigen::Index maximumAssignmentCount = Min(populations.rows(), populations.cols());
			for (const Eigen::Index populationIndex : sortedPopulationIndices)
			{
				const Eigen::Index currentIndex = populationIndex % populations.rows();
				const Eigen::Index previousIndex = populationIndex / populations.rows();
				if (isCurrentIndexAssigned[static_cast<std::size_t>(currentIndex)]
				    || isPreviousIndexAssigned[static_cast<std::size_t>(previousIndex)])
				{
					continue;
				}

				isCurrentIndexAssigned[static_cast<std::size_t>(currentIndex)] = true;
				isPreviousIndexAssigned[static_cast<std::size_t>(previousIndex)] = true;
				previousIndexForCurrentIndex[static_cast<std::size_t>(currentIndex)] = previousIndex;
				if (++assignmentCount == maximumAssignmentCount)
				{
					break;
				}
			}
			return FormRitzVectorMatches(populations,
			                             previousIndexForCurrentIndex,
			                             previousPrimaryVectorCount,
			                             previousRetainedVectorCount,
			                             weakMatchPopulationThreshold);
		}


		template <typename RealScalar>
		std::vector<RitzVectorMatch<RealScalar>> MatchRitzVectors_Hungarian(
		        const Eigen::MatrixX<RealScalar>& populations,
		        const Eigen::Index previousPrimaryVectorCount,
		        const Eigen::Index previousRetainedVectorCount,
		        const RealScalar weakMatchPopulationThreshold = RealScalar{0.5})
		{
			std::vector previousIndexForCurrentIndex(static_cast<std::size_t>(populations.rows()), Eigen::Index{-1});
			if (populations.rows() == 0 || populations.cols() == 0)
			{
				return FormRitzVectorMatches(populations,
				                             previousIndexForCurrentIndex,
				                             previousPrimaryVectorCount,
				                             previousRetainedVectorCount,
				                             weakMatchPopulationThreshold);
			}

			const bool isCurrentSideSmaller = populations.rows() <= populations.cols();
			const Eigen::Index workerCount = isCurrentSideSmaller ? populations.rows() : populations.cols();
			const Eigen::Index jobCount = isCurrentSideSmaller ? populations.cols() : populations.rows();
			const auto populationAt =
			        [&populations, isCurrentSideSmaller](const Eigen::Index workerIndex, const Eigen::Index jobIndex)
			{ return isCurrentSideSmaller ? populations(workerIndex, jobIndex) : populations(jobIndex, workerIndex); };

			std::vector<RealScalar> workerPotentials(static_cast<std::size_t>(workerCount + 1));
			std::vector<RealScalar> jobPotentials(static_cast<std::size_t>(jobCount + 1));
			std::vector<Eigen::Index> workerForJob(static_cast<std::size_t>(jobCount + 1));
			std::vector<Eigen::Index> precedingJob(static_cast<std::size_t>(jobCount + 1));
			for (Eigen::Index worker = 1; worker <= workerCount; worker++)
			{
				workerForJob[0] = worker;
				std::vector<RealScalar> minimumReducedCosts(static_cast<std::size_t>(jobCount + 1),
				                                            std::numeric_limits<RealScalar>::infinity());
				std::vector<std::uint8_t> isJobInAlternatingTree(static_cast<std::size_t>(jobCount + 1));
				Eigen::Index currentJob = 0;
				do
				{
					isJobInAlternatingTree[static_cast<std::size_t>(currentJob)] = true;
					const Eigen::Index currentWorker = workerForJob[static_cast<std::size_t>(currentJob)];
					RealScalar smallestReducedCost = std::numeric_limits<RealScalar>::infinity();
					Eigen::Index nextJob = 0;
					for (Eigen::Index job = 1; job <= jobCount; job++)
					{
						if (isJobInAlternatingTree[static_cast<std::size_t>(job)])
						{
							continue;
						}
						const RealScalar reducedCost = -populationAt(currentWorker - 1, job - 1)
						                               - workerPotentials[static_cast<std::size_t>(currentWorker)]
						                               - jobPotentials[static_cast<std::size_t>(job)];
						if (reducedCost < minimumReducedCosts[static_cast<std::size_t>(job)])
						{
							minimumReducedCosts[static_cast<std::size_t>(job)] = reducedCost;
							precedingJob[static_cast<std::size_t>(job)] = currentJob;
						}
						if (minimumReducedCosts[static_cast<std::size_t>(job)] < smallestReducedCost)
						{
							smallestReducedCost = minimumReducedCosts[static_cast<std::size_t>(job)];
							nextJob = job;
						}
					}

					for (Eigen::Index job = 0; job <= jobCount; job++)
					{
						if (isJobInAlternatingTree[static_cast<std::size_t>(job)])
						{
							workerPotentials[static_cast<std::size_t>(workerForJob[static_cast<std::size_t>(job)])] +=
							        smallestReducedCost;
							jobPotentials[static_cast<std::size_t>(job)] -= smallestReducedCost;
						}
						else
						{
							minimumReducedCosts[static_cast<std::size_t>(job)] -= smallestReducedCost;
						}
					}
					currentJob = nextJob;
				} while (workerForJob[static_cast<std::size_t>(currentJob)] != 0);

				do
				{
					const Eigen::Index previousJob = precedingJob[static_cast<std::size_t>(currentJob)];
					workerForJob[static_cast<std::size_t>(currentJob)] =
					        workerForJob[static_cast<std::size_t>(previousJob)];
					currentJob = previousJob;
				} while (currentJob != 0);
			}

			for (Eigen::Index job = 1; job <= jobCount; job++)
			{
				const Eigen::Index worker = workerForJob[static_cast<std::size_t>(job)];
				if (worker == 0)
				{
					continue;
				}
				const Eigen::Index currentIndex = isCurrentSideSmaller ? worker - 1 : job - 1;
				const Eigen::Index previousIndex = isCurrentSideSmaller ? job - 1 : worker - 1;
				previousIndexForCurrentIndex[static_cast<std::size_t>(currentIndex)] = previousIndex;
			}
			return FormRitzVectorMatches(populations,
			                             previousIndexForCurrentIndex,
			                             previousPrimaryVectorCount,
			                             previousRetainedVectorCount,
			                             weakMatchPopulationThreshold);
		}


		template <typename RealScalar>
		std::vector<RitzVectorMatch<RealScalar>> MatchRitzVectors(
		        const Eigen::MatrixX<RealScalar>& populations,
		        const Eigen::Index previousPrimaryVectorCount,
		        const Eigen::Index previousRetainedVectorCount,
		        const RealScalar weakMatchPopulationThreshold = RealScalar{0.5})
		{
			return MatchRitzVectors_Hungarian(
			        populations, previousPrimaryVectorCount, previousRetainedVectorCount, weakMatchPopulationThreshold);
		}


		template <typename RealScalar>
		void UpdateInteriorRitzConvergence(InteriorRitzSelection<RealScalar>& ref_selection,
		                                   const Eigen::VectorX<RealScalar>& currentEigenvalues,
		                                   const std::vector<RitzVectorMatch<RealScalar>>& matches,
		                                   const Eigen::VectorX<RealScalar>& previousEigenvalues)
		{
			ref_selection.MaximumMatchedEigenvalueChange = std::numeric_limits<RealScalar>::infinity();
			ref_selection.HasUnmatchedIntervalRitzVector = false;
			RealScalar maximumChange{};
			for (const Eigen::Index currentIndex : ref_selection.IntervalRitzIndices)
			{
				const auto& match = matches[static_cast<std::size_t>(currentIndex)];
				if (!match.IsMatched() || match.PreviousIndex >= previousEigenvalues.size() || match.IsWeakMatch)
				{
					ref_selection.HasUnmatchedIntervalRitzVector = true;
					continue;
				}
				maximumChange = Max(maximumChange,
				                    Abs(currentEigenvalues[currentIndex] - previousEigenvalues[match.PreviousIndex]));
			}
			if (!ref_selection.IntervalRitzIndices.empty() && !ref_selection.HasUnmatchedIntervalRitzVector)
			{
				ref_selection.MaximumMatchedEigenvalueChange = maximumChange;
			}
		}


		template <typename RealScalar>
		InteriorRitzSelection<RealScalar> SelectInteriorRitzVectors(
		        const Eigen::VectorX<RealScalar>& currentEigenvalues,
		        const std::vector<RitzVectorMatch<RealScalar>>& matches,
		        const Eigen::VectorX<RealScalar>& previousEigenvalues,
		        const EigenvalueInterval<RealScalar>& interval,
		        const Eigen::Index maximumEigenpairCount,
		        const Eigen::Index additionalRitzVectorCount,
		        const RealScalar boundaryDegeneracyTolerance = static_cast<RealScalar>(1e-5))
		{
			InteriorRitzSelection<RealScalar> selection;

			// Exact interval membership determines which Ritz pairs may be returned to the caller.
			for (Eigen::Index currentIndex = 0; currentIndex < currentEigenvalues.size(); currentIndex++)
			{
				if (interval.IsContaining(currentEigenvalues[currentIndex]))
				{
					selection.IntervalRitzIndices.push_back(currentIndex);
				}
			}

			selection.IsEigenpairCountLimitExceeded =
			        static_cast<Eigen::Index>(selection.IntervalRitzIndices.size()) > maximumEigenpairCount;

			// Compare each target with its previous identity match. A new or weakly matched target prevents
			// convergence.
			UpdateInteriorRitzConvergence(selection, currentEigenvalues, matches, previousEigenvalues);

			// Retain complete near-degenerate clusters straddling either interval boundary.
			std::vector<bool> isBoundaryContinuation(static_cast<std::size_t>(currentEigenvalues.size()), false);
			if (!selection.IntervalRitzIndices.empty())
			{
				Eigen::Index adjacentIndex = selection.IntervalRitzIndices.front();
				for (Eigen::Index index = adjacentIndex - 1;
				     index >= 0
				     && currentEigenvalues[adjacentIndex] - currentEigenvalues[index] <= boundaryDegeneracyTolerance;
				     index--)
				{
					isBoundaryContinuation[static_cast<std::size_t>(index)] = true;
					selection.AdditionalRitzIndices.push_back(index);
					adjacentIndex = index;
				}

				adjacentIndex = selection.IntervalRitzIndices.back();
				for (Eigen::Index index = adjacentIndex + 1;
				     index < currentEigenvalues.size()
				     && currentEigenvalues[index] - currentEigenvalues[adjacentIndex] <= boundaryDegeneracyTolerance;
				     index++)
				{
					isBoundaryContinuation[static_cast<std::size_t>(index)] = true;
					selection.AdditionalRitzIndices.push_back(index);
					adjacentIndex = index;
				}
			}

			// The remaining out-of-interval Ritz vectors are candidates for additional retention.
			std::vector<Eigen::Index> candidates;
			candidates.reserve(static_cast<std::size_t>(currentEigenvalues.size()));
			for (Eigen::Index currentIndex = 0; currentIndex < currentEigenvalues.size(); currentIndex++)
			{
				if (!interval.IsContaining(currentEigenvalues[currentIndex])
				    && !isBoundaryContinuation[static_cast<std::size_t>(currentIndex)])
				{
					candidates.push_back(currentIndex);
				}
			}
			const auto distanceFromInterval = [&currentEigenvalues, &interval](const Eigen::Index index)
			{
				return currentEigenvalues[index] < interval.LowerBound
				               ? interval.LowerBound - currentEigenvalues[index]
				               : currentEigenvalues[index] - interval.UpperBound;
			};

			// Preserve known vector identities first, then prefer candidates closest to the interval.
			std::stable_sort(candidates.begin(),
			                 candidates.end(),
			                 [&matches, &distanceFromInterval](const Eigen::Index lhs, const Eigen::Index rhs)
			                 {
				                 const bool isLhsContinuation =
				                         matches[static_cast<std::size_t>(lhs)].IsRetainedVectorContinuation;
				                 const bool isRhsContinuation =
				                         matches[static_cast<std::size_t>(rhs)].IsRetainedVectorContinuation;
				                 if (isLhsContinuation != isRhsContinuation)
				                 {
					                 return isLhsContinuation;
				                 }
				                 return distanceFromInterval(lhs) < distanceFromInterval(rhs);
			                 });

			// Boundary-cluster continuations do not consume the configured additional-vector allowance.
			const Eigen::Index retainedCandidateCount =
			        Min(additionalRitzVectorCount, static_cast<Eigen::Index>(candidates.size()));
			selection.AdditionalRitzIndices.insert(selection.AdditionalRitzIndices.end(),
			                                       candidates.begin(),
			                                       candidates.begin() + retainedCandidateCount);
			return selection;
		}


		template <typename Scalar>
		RitzPairs<Scalar> PermuteSelectedRitzPairsToTheFront(
		        const Eigen::VectorX<typename Eigen::NumTraits<Scalar>::Real>& eigenvalues,
		        const Eigen::MatrixX<Scalar>& eigenvectors,
		        const InteriorRitzSelection<typename Eigen::NumTraits<Scalar>::Real>& selection)
		{
			std::vector<Eigen::Index> order;
			order.reserve(static_cast<std::size_t>(eigenvalues.size()));
			order.insert(order.end(), selection.IntervalRitzIndices.begin(), selection.IntervalRitzIndices.end());
			order.insert(order.end(), selection.AdditionalRitzIndices.begin(), selection.AdditionalRitzIndices.end());
			std::vector<bool> isSelected(static_cast<std::size_t>(eigenvalues.size()), false);
			for (const Eigen::Index index : order)
			{
				isSelected[static_cast<std::size_t>(index)] = true;
			}
			for (Eigen::Index index = 0; index < eigenvalues.size(); index++)
			{
				if (!isSelected[static_cast<std::size_t>(index)])
				{
					order.push_back(index);
				}
			}

			RitzPairs<Scalar> result{Eigen::VectorX<typename Eigen::NumTraits<Scalar>::Real>(eigenvalues.size()),
			                         Eigen::MatrixX<Scalar>(eigenvectors.rows(), eigenvectors.cols())};
			for (Eigen::Index destinationIndex = 0; destinationIndex < eigenvalues.size(); destinationIndex++)
			{
				const Eigen::Index sourceIndex = order[static_cast<std::size_t>(destinationIndex)];
				result.Eigenvalues[destinationIndex] = eigenvalues[sourceIndex];
				result.Eigenvectors.col(destinationIndex) = eigenvectors.col(sourceIndex);
			}
			return result;
		}


		inline Eigen::Index CalculatePrimaryRitzVectorCount(const Eigen::Index intervalEigenpairCount,
		                                                    const Eigen::Index frozenVectorCount,
		                                                    const Eigen::Index availableRitzVectorCount)
		{
			const Eigen::Index requestedCount =
			        Max(6 + frozenVectorCount, 3 * intervalEigenpairCount + frozenVectorCount);
			return Min(requestedCount, availableRitzVectorCount);
		}


		template <typename RealScalar>
		bool IsPreviousRitzVectorRecyclingActiveFor(const bool isCurrentlyActive,
		                                            const RealScalar maximumMatchedEigenvalueChange)
		{
			if (maximumMatchedEigenvalueChange > static_cast<RealScalar>(0.1)
			    || maximumMatchedEigenvalueChange < static_cast<RealScalar>(1e-5))
			{
				return false;
			}
			if (maximumMatchedEigenvalueChange < static_cast<RealScalar>(0.01))
			{
				return true;
			}
			return isCurrentlyActive;
		}


		template <typename Scalar>
		std::vector<bool> DetermineFrozenRitzVectors(
		        const Eigen::MatrixX<Scalar>& reducedEigenvectors,
		        const Eigen::VectorX<typename Eigen::NumTraits<Scalar>::Real>& residualNorms,
		        const std::vector<Eigen::Index>& intervalRitzIndices,
		        const Eigen::Index previousPrimaryVectorCount,
		        const typename Eigen::NumTraits<Scalar>::Real coefficientTolerance,
		        const typename Eigen::NumTraits<Scalar>::Real residualNormTolerance)
		{
			std::vector<bool> isFrozen(static_cast<std::size_t>(reducedEigenvectors.cols()), false);
			if (previousPrimaryVectorCount == 0)
			{
				return isFrozen;
			}

			for (const Eigen::Index currentIndex : intervalRitzIndices)
			{
				const auto primaryPopulation =
				        reducedEigenvectors.col(currentIndex).head(previousPrimaryVectorCount).squaredNorm();
				isFrozen[static_cast<std::size_t>(currentIndex)] =
				        primaryPopulation >= typename Eigen::NumTraits<Scalar>::Real{1} - coefficientTolerance
				        && residualNorms[currentIndex] <= residualNormTolerance;
			}
			return isFrozen;
		}


		template <typename RealScalar>
		void DemoteUnvalidatedExcessIntervalRitzVectors(InteriorRitzSelection<RealScalar>& ref_selection,
		                                                const Eigen::VectorX<RealScalar>& residualNorms,
		                                                const Eigen::Index maximumEigenpairCount,
		                                                const RealScalar residualNormTolerance)
		{
			if (!ref_selection.IsEigenpairCountLimitExceeded)
			{
				return;
			}
			const bool areAllIntervalCandidatesValidated =
			        std::ranges::all_of(ref_selection.IntervalRitzIndices,
			                            [&residualNorms, residualNormTolerance](const Eigen::Index index)
			                            { return residualNorms[index] <= residualNormTolerance; });
			if (areAllIntervalCandidatesValidated)
			{
				return;
			}

			std::stable_sort(ref_selection.IntervalRitzIndices.begin(),
			                 ref_selection.IntervalRitzIndices.end(),
			                 [&residualNorms](const Eigen::Index lhs, const Eigen::Index rhs)
			                 { return residualNorms[lhs] < residualNorms[rhs]; });
			ref_selection.AdditionalRitzIndices.insert(ref_selection.AdditionalRitzIndices.begin(),
			                                           ref_selection.IntervalRitzIndices.begin()
			                                                   + maximumEigenpairCount,
			                                           ref_selection.IntervalRitzIndices.end());
			ref_selection.IntervalRitzIndices.resize(static_cast<std::size_t>(maximumEigenpairCount));
			ref_selection.IsEigenpairCountLimitExceeded = false;
		}


		template <SelfAdjointLinearOperator Operator>
		Eigen::MatrixX<LinearOperatorScalar<Operator>> ApplyOperator(
		        const Operator& linearOperator,
		        const Eigen::MatrixX<LinearOperatorScalar<Operator>>& vectors,
		        InteriorEigenSolverStatistics& ref_statistics)
		{
			using Scalar = LinearOperatorScalar<Operator>;
			Eigen::MatrixX<Scalar> images(linearOperator.rows(), vectors.cols());

			if (vectors.cols() == 0)
			{
				return images;
			}

			if constexpr (BlockSelfAdjointLinearOperator<Operator>)
			{
				images = linearOperator.ApplyOn(vectors);
				ref_statistics.OperatorApplicationCount++;
			}
			else
			{
				for (Eigen::Index columnIndex = 0; columnIndex < vectors.cols(); columnIndex++)
				{
					images.col(columnIndex) = linearOperator.ApplyOn(vectors.col(columnIndex));
					ref_statistics.OperatorApplicationCount++;
				}
			}

			ref_statistics.MultipliedVectorCount += vectors.cols();
			return images;
		}


		template <SelfAdjointLinearOperator Operator>
		void RecalculateImages(const Operator& linearOperator,
		                       VectorImagePair<LinearOperatorScalar<Operator>>& ref_vectorImagePair,
		                       InteriorEigenSolverStatistics& ref_statistics)
		{
			ref_vectorImagePair.Images = ApplyOperator(linearOperator, ref_vectorImagePair.Vectors, ref_statistics);
			ref_statistics.ExplicitImageRecalculationCount++;
		}


		template <typename Scalar>
		Eigen::MatrixX<Scalar> ProjectAgainstBasis(const Eigen::MatrixX<Scalar>& basis,
		                                           const Eigen::MatrixX<Scalar>& candidates)
		{
			if (basis.cols() == 0)
			{
				return candidates;
			}
			return candidates - basis * (basis.adjoint() * candidates);
		}


		template <typename Scalar>
		Eigen::MatrixX<Scalar> SymmetricallyOrthonormalize(
		        const Eigen::MatrixX<Scalar>& vectors,
		        const typename Eigen::NumTraits<Scalar>::Real relativeLinearDependenceTolerance)
		{
			using RealScalar = Eigen::NumTraits<Scalar>::Real;
			if (vectors.cols() == 0)
			{
				return vectors;
			}

			const Eigen::MatrixX<Scalar> gramMatrix = vectors.adjoint() * vectors;
			const Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Scalar>> eigenSolver(gramMatrix);
			if (eigenSolver.info() != Eigen::Success)
			{
				throw OperationFailedException("Symmetric orthonormalization failed to diagonalize the Gram matrix");
			}

			const auto& gramEigenvalues = eigenSolver.eigenvalues();
			const RealScalar largestEigenvalue = gramEigenvalues.cend()[-1];
			const RealScalar rejectionThreshold =
			        relativeLinearDependenceTolerance * Max(RealScalar{1}, largestEigenvalue);
			const auto firstAcceptedIndex = static_cast<Eigen::Index>(
			        std::distance(gramEigenvalues.begin(),
			                      std::find_if(gramEigenvalues.begin(),
			                                   gramEigenvalues.end(),
			                                   [rejectionThreshold](const RealScalar eigenvalue)
			                                   { return eigenvalue >= rejectionThreshold; })));
			const Eigen::Index acceptedCount = vectors.cols() - firstAcceptedIndex;
			if (acceptedCount == 0)
			{
				return Eigen::MatrixX<Scalar>(vectors.rows(), 0);
			}

			const auto acceptedEigenvalues = gramEigenvalues.tail(acceptedCount);
			const auto acceptedEigenvectors = eigenSolver.eigenvectors().rightCols(acceptedCount);
			return vectors * acceptedEigenvectors * acceptedEigenvalues.cwiseInverse().cwiseSqrt().asDiagonal();
		}


		template <typename Scalar>
		Eigen::MatrixX<Scalar> OrthogonalizeAndRemoveLinearDependence(
		        const Eigen::MatrixX<Scalar>& basis,
		        const Eigen::MatrixX<Scalar>& candidates,
		        const typename Eigen::NumTraits<Scalar>::Real relativeLinearDependenceTolerance)
		{
			return SymmetricallyOrthonormalize(ProjectAgainstBasis(basis, candidates),
			                                   relativeLinearDependenceTolerance);
		}


		template <typename Scalar>
		Eigen::MatrixX<Scalar> FormPreviousRitzVectorRecyclingCoefficients(
		        const Eigen::MatrixX<Scalar>& selectedRitzCoefficients,
		        const Eigen::Index previousPrimaryVectorCount,
		        const typename Eigen::NumTraits<Scalar>::Real linearDependenceTolerance,
		        const typename Eigen::NumTraits<Scalar>::Real refinementTolerance)
		{
			Eigen::MatrixX<Scalar> previousPrimaryCoefficients =
			        Eigen::MatrixX<Scalar>::Zero(selectedRitzCoefficients.rows(), previousPrimaryVectorCount);
			previousPrimaryCoefficients.topRows(previousPrimaryVectorCount).setIdentity();
			Eigen::MatrixX<Scalar> recyclingCoefficients = OrthogonalizeAndRemoveLinearDependence(
			        selectedRitzCoefficients, previousPrimaryCoefficients, linearDependenceTolerance);
			return OrthogonalizeAndRemoveLinearDependence(
			        selectedRitzCoefficients, recyclingCoefficients, refinementTolerance);
		}


		template <typename Scalar>
		VectorImagePair<Scalar> TransformVectorImagePair(const VectorImagePair<Scalar>& source,
		                                                 const Eigen::MatrixX<Scalar>& coefficients)
		{
			return {source.Vectors * coefficients, source.Images * coefficients};
		}


		template <typename Scalar>
		Eigen::ComputationInfo SymmetricallyOrthonormalizeVectorImagePair(VectorImagePair<Scalar>& ref_vectorImagePair)
		{
			const Eigen::MatrixX<Scalar> gramMatrix =
			        ref_vectorImagePair.Vectors.adjoint() * ref_vectorImagePair.Vectors;
			const Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Scalar>> gramEigenSolver(gramMatrix,
			                                                                            Eigen::EigenvaluesOnly);
			if (gramEigenSolver.info() != Eigen::Success)
			{
				return gramEigenSolver.info();
			}
			const auto& gramEigenvalues = gramEigenSolver.eigenvalues();
			const auto minimumAcceptableGramEigenvalue =
			        std::numeric_limits<typename Eigen::NumTraits<Scalar>::Real>::epsilon()
			        * static_cast<Eigen::NumTraits<Scalar>::Real>(gramMatrix.rows()) * gramEigenvalues.cend()[-1];
			if (gramEigenvalues[0] <= minimumAcceptableGramEigenvalue)
			{
				return Eigen::NumericalIssue;
			}
			const Eigen::LLT<Eigen::MatrixX<Scalar>> choleskyDecomposition(gramMatrix);
			if (choleskyDecomposition.info() != Eigen::Success)
			{
				return choleskyDecomposition.info();
			}
			const Eigen::MatrixX<Scalar> inverseUpperFactor = choleskyDecomposition.matrixU().solve(
			        Eigen::MatrixX<Scalar>::Identity(gramMatrix.rows(), gramMatrix.cols()));
			ref_vectorImagePair = TransformVectorImagePair(ref_vectorImagePair, inverseUpperFactor);
			return Eigen::Success;
		}


		template <typename Scalar>
		Eigen::MatrixX<Scalar> FormReducedMatrix(const VectorImagePair<Scalar>& vectorImagePair)
		{
			return vectorImagePair.Vectors.adjoint() * vectorImagePair.Images;
		}


		template <typename Scalar>
		bool DoesReducedMatrixRequireExplicitImages(const Eigen::MatrixX<Scalar>& reducedMatrix,
		                                            const typename Eigen::NumTraits<Scalar>::Real asymmetryTolerance)
		{
			if (reducedMatrix.size() == 0)
			{
				return false;
			}
			const auto maximumAsymmetry = (reducedMatrix - reducedMatrix.adjoint()).cwiseAbs().maxCoeff();
			const auto scaledTolerance =
			        asymmetryTolerance * static_cast<Eigen::NumTraits<Scalar>::Real>(reducedMatrix.size());
			return maximumAsymmetry > scaledTolerance;
		}


		template <typename Scalar>
		Eigen::ComputationInfo SolveReducedSelfAdjointEigenproblem(
		        const VectorImagePair<Scalar>& vectorImagePair,
		        const bool isGeneralizedSolve,
		        Eigen::VectorX<typename Eigen::NumTraits<Scalar>::Real>& out_eigenvalues,
		        Eigen::MatrixX<Scalar>& out_eigenvectors)
		{
			const Eigen::MatrixX<Scalar> reducedMatrix = FormReducedMatrix(vectorImagePair);
			if (isGeneralizedSolve)
			{
				const Eigen::MatrixX<Scalar> overlapMatrix =
				        vectorImagePair.Vectors.adjoint() * vectorImagePair.Vectors;
				const Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Scalar>> overlapEigenSolver(overlapMatrix,
				                                                                               Eigen::EigenvaluesOnly);
				if (overlapEigenSolver.info() != Eigen::Success)
				{
					return overlapEigenSolver.info();
				}
				const auto& overlapEigenvalues = overlapEigenSolver.eigenvalues();
				const auto minimumAcceptableOverlapEigenvalue =
				        std::numeric_limits<typename Eigen::NumTraits<Scalar>::Real>::epsilon()
				        * static_cast<Eigen::NumTraits<Scalar>::Real>(overlapMatrix.rows())
				        * overlapEigenvalues.cend()[-1];
				if (overlapEigenvalues[0] <= minimumAcceptableOverlapEigenvalue)
				{
					return Eigen::NumericalIssue;
				}
				const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixX<Scalar>> eigenSolver(reducedMatrix,
				                                                                                   overlapMatrix);
				if (eigenSolver.info() == Eigen::Success)
				{
					out_eigenvalues = eigenSolver.eigenvalues();
					out_eigenvectors = eigenSolver.eigenvectors();
				}
				return eigenSolver.info();
			}

			const Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Scalar>> eigenSolver(reducedMatrix);
			if (eigenSolver.info() == Eigen::Success)
			{
				out_eigenvalues = eigenSolver.eigenvalues();
				out_eigenvectors = eigenSolver.eigenvectors();
			}
			return eigenSolver.info();
		}


		template <typename Scalar>
		Eigen::MatrixX<Scalar> CalculateResiduals(
		        const VectorImagePair<Scalar>& vectorImagePair,
		        const Eigen::VectorX<typename Eigen::NumTraits<Scalar>::Real>& eigenvalues)
		{
			return vectorImagePair.Images - vectorImagePair.Vectors * eigenvalues.asDiagonal();
		}


		template <typename Scalar>
		Eigen::VectorX<typename Eigen::NumTraits<Scalar>::Real> CalculateColumnNorms(
		        const Eigen::MatrixX<Scalar>& vectors)
		{
			using RealScalar = Eigen::NumTraits<Scalar>::Real;
			Eigen::VectorX<RealScalar> norms(vectors.cols());
			for (Eigen::Index columnIndex = 0; columnIndex < vectors.cols(); columnIndex++)
			{
				norms[columnIndex] = vectors.col(columnIndex).norm();
			}
			return norms;
		}


		template <typename Scalar>
		void SortEigenpairsInAscendingOrder(Eigen::VectorX<typename Eigen::NumTraits<Scalar>::Real>& ref_eigenvalues,
		                                    Eigen::MatrixX<Scalar>& ref_eigenvectors,
		                                    Eigen::VectorX<typename Eigen::NumTraits<Scalar>::Real>& ref_residualNorms)
		{
			std::vector<Eigen::Index> indices(static_cast<std::size_t>(ref_eigenvalues.size()));
			std::iota(indices.begin(), indices.end(), Eigen::Index{});
			std::ranges::stable_sort(
			        indices, {}, [&ref_eigenvalues](const Eigen::Index index) { return ref_eigenvalues[index]; });

			const auto eigenvalues = ref_eigenvalues.eval();
			const auto eigenvectors = ref_eigenvectors.eval();
			const auto residualNorms = ref_residualNorms.eval();
			for (Eigen::Index destinationIndex = 0; destinationIndex < ref_eigenvalues.size(); destinationIndex++)
			{
				const Eigen::Index sourceIndex = indices[static_cast<std::size_t>(destinationIndex)];
				ref_eigenvalues[destinationIndex] = eigenvalues[sourceIndex];
				ref_eigenvectors.col(destinationIndex) = eigenvectors.col(sourceIndex);
				ref_residualNorms[destinationIndex] = residualNorms[sourceIndex];
			}
		}


		template <typename Scalar>
		Eigen::MatrixX<Scalar> ApplyAbsoluteDiagonalPreconditioner(
		        const Eigen::MatrixX<Scalar>& residuals,
		        const Eigen::VectorX<typename Eigen::NumTraits<Scalar>::Real>& eigenvalues,
		        const Eigen::VectorX<typename Eigen::NumTraits<Scalar>::Real>& diagonal,
		        const typename Eigen::NumTraits<Scalar>::Real denominatorFloor)
		{
			using RealScalar = Eigen::NumTraits<Scalar>::Real;
			Eigen::MatrixX<Scalar> corrections(residuals.rows(), residuals.cols());
			for (Eigen::Index columnIndex = 0; columnIndex < residuals.cols(); columnIndex++)
			{
				for (Eigen::Index rowIndex = 0; rowIndex < residuals.rows(); rowIndex++)
				{
					const RealScalar denominator =
					        Max(Abs(diagonal[rowIndex] - eigenvalues[columnIndex]), denominatorFloor);
					corrections(rowIndex, columnIndex) = residuals(rowIndex, columnIndex) / denominator;
				}
			}
			return corrections;
		}


		template <typename Scalar>
		struct InteriorIterationState
		{
			using RealScalar = Eigen::NumTraits<Scalar>::Real;

			VectorImagePair<Scalar> ExpansionSpace;
			Eigen::VectorX<RealScalar> PreviousRetainedEigenvalues;
			Eigen::Index PreviousPrimaryVectorCount = 0;
			Eigen::Index PreviousRetainedVectorCount = 0;
			bool IsPreviousRitzVectorRecyclingActive = false;
		};


		template <typename Scalar>
		struct InteriorIterationAnalysis
		{
			using RealScalar = Eigen::NumTraits<Scalar>::Real;

			Eigen::ComputationInfo ComputationInfo = Eigen::Success;
			InteriorRitzSelection<RealScalar> Selection;
			RitzPairs<Scalar> OrderedRitzPairs;
			Eigen::Index FrozenVectorCount = 0;
			Eigen::Index IntervalEigenpairCount = 0;
			Eigen::Index RetainedVectorCount = 0;
			Eigen::Index PrimaryVectorCount = 0;
			bool IsGeneralizedSolve = false;
			bool AreExplicitImagesRequired = false;
		};


		enum class ExpansionResult
		{
			Expanded,
			NoIndependentCorrections
		};


		template <SelfAdjointLinearOperator Operator>
		VectorImagePair<LinearOperatorScalar<Operator>> CreateInitialExpansionSpace(
		        const Operator& linearOperator,
		        const Eigen::VectorX<LinearOperatorRealScalar<Operator>>& diagonal,
		        const EigenvalueInterval<LinearOperatorRealScalar<Operator>>& interval,
		        const Eigen::Index eigenpairCountLimit,
		        InteriorEigenSolverStatistics& ref_statistics)
		{
			using Scalar = LinearOperatorScalar<Operator>;
			using RealScalar = LinearOperatorRealScalar<Operator>;
			const RealScalar intervalCenter = (interval.LowerBound + interval.UpperBound) / RealScalar{2};
			std::vector<Eigen::Index> coordinateIndices(static_cast<std::size_t>(linearOperator.rows()));
			std::iota(coordinateIndices.begin(), coordinateIndices.end(), Eigen::Index{});
			std::ranges::stable_sort(coordinateIndices,
			                         {},
			                         [&diagonal, intervalCenter](const Eigen::Index index)
			                         { return Abs(diagonal[index] - intervalCenter); });

			const Eigen::Index initialVectorCount =
			        Min(linearOperator.rows(), Max(Eigen::Index{6}, Eigen::Index{3} * eigenpairCountLimit));
			Eigen::MatrixX<Scalar> vectors = Eigen::MatrixX<Scalar>::Zero(linearOperator.rows(), initialVectorCount);
			for (Eigen::Index columnIndex = 0; columnIndex < initialVectorCount; columnIndex++)
			{
				vectors(coordinateIndices[static_cast<std::size_t>(columnIndex)], columnIndex) = Scalar{1};
			}
			Eigen::MatrixX<Scalar> images = ApplyOperator(linearOperator, vectors, ref_statistics);
			return {std::move(vectors), std::move(images)};
		}


		template <typename Scalar>
		InteriorIterationAnalysis<Scalar> AnalyzeExpansionSpace(
		        const InteriorIterationState<Scalar>& state,
		        const EigenvalueInterval<typename Eigen::NumTraits<Scalar>::Real>& interval,
		        const InteriorEigenSolverOptions<typename Eigen::NumTraits<Scalar>::Real>& options,
		        const Eigen::Index iterationIndex,
		        const typename Eigen::NumTraits<Scalar>::Real freezingResidualNormTolerance)
		{
			using RealScalar = Eigen::NumTraits<Scalar>::Real;
			InteriorIterationAnalysis<Scalar> analysis;
			analysis.AreExplicitImagesRequired = DoesReducedMatrixRequireExplicitImages(
			        FormReducedMatrix(state.ExpansionSpace), options.ReducedMatrixAsymmetryTolerance);
			analysis.IsGeneralizedSolve = iterationIndex % options.GeneralizedSolveInterval == 0;

			// Solve the projected problem in the current expansion basis.
			Eigen::VectorX<RealScalar> currentEigenvalues;
			Eigen::MatrixX<Scalar> reducedEigenvectors;
			analysis.ComputationInfo = SolveReducedSelfAdjointEigenproblem(
			        state.ExpansionSpace, analysis.IsGeneralizedSolve, currentEigenvalues, reducedEigenvectors);
			if (analysis.ComputationInfo != Eigen::Success)
			{
				return analysis;
			}

			// Match Ritz identities, then select interval targets and nearby retained directions.
			const Eigen::MatrixX<RealScalar> populations =
			        FormRitzPopulationMatrix(reducedEigenvectors, state.PreviousRetainedVectorCount);
			const auto matches =
			        MatchRitzVectors(populations, state.PreviousPrimaryVectorCount, state.PreviousRetainedVectorCount);
			const Eigen::Index requestedAdditionalCount =
			        options.AdditionalRitzVectorCount == 0
			                ? Max(Eigen::Index{6}, Eigen::Index{3} * options.EigenpairCountLimit)
			                : options.AdditionalRitzVectorCount;
			analysis.Selection = SelectInteriorRitzVectors(currentEigenvalues,
			                                               matches,
			                                               state.PreviousRetainedEigenvalues,
			                                               interval,
			                                               options.EigenpairCountLimit,
			                                               requestedAdditionalCount);
			const VectorImagePair<Scalar> allRitzPairs =
			        TransformVectorImagePair(state.ExpansionSpace, reducedEigenvectors);
			const Eigen::VectorX<RealScalar> allResidualNorms =
			        CalculateColumnNorms(CalculateResiduals(allRitzPairs, currentEigenvalues));
			DemoteUnvalidatedExcessIntervalRitzVectors(
			        analysis.Selection, allResidualNorms, options.EigenpairCountLimit, options.ResidualNormTolerance);
			UpdateInteriorRitzConvergence(
			        analysis.Selection, currentEigenvalues, matches, state.PreviousRetainedEigenvalues);

			if (options.IsFreezingEnabled)
			{
				// Move frozen interval targets to the leading columns used by the next cycle.
				const auto isCurrentRitzVectorFrozen =
				        DetermineFrozenRitzVectors(reducedEigenvectors,
				                                   allResidualNorms,
				                                   analysis.Selection.IntervalRitzIndices,
				                                   state.PreviousPrimaryVectorCount,
				                                   options.FreezingCoefficientTolerance,
				                                   freezingResidualNormTolerance);
				std::stable_sort(analysis.Selection.IntervalRitzIndices.begin(),
				                 analysis.Selection.IntervalRitzIndices.end(),
				                 [&isCurrentRitzVectorFrozen](const Eigen::Index lhs, const Eigen::Index rhs)
				                 {
					                 return isCurrentRitzVectorFrozen[static_cast<std::size_t>(lhs)]
					                        && !isCurrentRitzVectorFrozen[static_cast<std::size_t>(rhs)];
				                 });
				analysis.FrozenVectorCount = static_cast<Eigen::Index>(
				        std::ranges::count_if(analysis.Selection.IntervalRitzIndices,
				                              [&isCurrentRitzVectorFrozen](const Eigen::Index index)
				                              { return isCurrentRitzVectorFrozen[static_cast<std::size_t>(index)]; }));
			}
			analysis.OrderedRitzPairs =
			        PermuteSelectedRitzPairsToTheFront(currentEigenvalues, reducedEigenvectors, analysis.Selection);
			analysis.IntervalEigenpairCount = static_cast<Eigen::Index>(analysis.Selection.IntervalRitzIndices.size());
			analysis.RetainedVectorCount = analysis.IntervalEigenpairCount
			                               + static_cast<Eigen::Index>(analysis.Selection.AdditionalRitzIndices.size());
			analysis.PrimaryVectorCount = CalculatePrimaryRitzVectorCount(
			        analysis.IntervalEigenpairCount, analysis.FrozenVectorCount, analysis.RetainedVectorCount);
			return analysis;
		}


		template <typename Scalar>
		Eigen::MatrixX<Scalar> FormCollapseCoefficients(
		        const InteriorIterationAnalysis<Scalar>& analysis,
		        const Eigen::Index iterationIndex,
		        const InteriorEigenSolverOptions<typename Eigen::NumTraits<Scalar>::Real>& options,
		        InteriorIterationState<Scalar>& ref_state,
		        InteriorEigenSolverStatistics& ref_statistics)
		{
			Eigen::MatrixX<Scalar> coefficients =
			        analysis.OrderedRitzPairs.Eigenvectors.leftCols(analysis.RetainedVectorCount);
			if (iterationIndex > 0 && options.IsPreviousRitzVectorRecyclingEnabled
			    && options.IsPreviousRitzVectorRecyclingDynamicallyEnabled)
			{
				ref_state.IsPreviousRitzVectorRecyclingActive =
				        IsPreviousRitzVectorRecyclingActiveFor(ref_state.IsPreviousRitzVectorRecyclingActive,
				                                               analysis.Selection.MaximumMatchedEigenvalueChange);
			}
			if (iterationIndex == 0 || !ref_state.IsPreviousRitzVectorRecyclingActive
			    || ref_state.PreviousPrimaryVectorCount == 0)
			{
				return coefficients;
			}

			const Eigen::MatrixX<Scalar> recyclingCoefficients =
			        FormPreviousRitzVectorRecyclingCoefficients(coefficients,
			                                                    ref_state.PreviousPrimaryVectorCount,
			                                                    options.PreviousRitzVectorRecyclingTolerance,
			                                                    options.RecyclingRefinementTolerance);
			const Eigen::Index selectedColumnCount = coefficients.cols();
			coefficients.conservativeResize(Eigen::NoChange, selectedColumnCount + recyclingCoefficients.cols());
			coefficients.rightCols(recyclingCoefficients.cols()) = recyclingCoefficients;
			ref_statistics.RecycledVectorCount += recyclingCoefficients.cols();
			return coefficients;
		}


		template <typename Scalar>
		void PublishIntervalEigenpairs(const VectorImagePair<Scalar>& retainedSpace,
		                               const RitzPairs<Scalar>& orderedRitzPairs,
		                               const Eigen::Index intervalEigenpairCount,
		                               Eigen::VectorX<typename Eigen::NumTraits<Scalar>::Real>& ref_eigenvalues,
		                               Eigen::MatrixX<Scalar>& ref_eigenvectors,
		                               Eigen::VectorX<typename Eigen::NumTraits<Scalar>::Real>& ref_residualNorms)
		{
			if (intervalEigenpairCount == 0)
			{
				ref_eigenvalues.resize(0);
				ref_eigenvectors.resize(retainedSpace.Vectors.rows(), 0);
				ref_residualNorms.resize(0);
				return;
			}

			const VectorImagePair<Scalar> intervalPairs{retainedSpace.Vectors.leftCols(intervalEigenpairCount),
			                                            retainedSpace.Images.leftCols(intervalEigenpairCount)};
			ref_eigenvalues = orderedRitzPairs.Eigenvalues.head(intervalEigenpairCount);
			ref_eigenvectors = intervalPairs.Vectors;
			ref_residualNorms = CalculateColumnNorms(CalculateResiduals(intervalPairs, ref_eigenvalues));
			SortEigenpairsInAscendingOrder(ref_eigenvalues, ref_eigenvectors, ref_residualNorms);
		}


		template <SelfAdjointLinearOperator Operator>
		ExpansionResult ExpandForNextIteration(
		        const Operator& linearOperator,
		        const Eigen::VectorX<LinearOperatorRealScalar<Operator>>& diagonal,
		        const InteriorIterationAnalysis<LinearOperatorScalar<Operator>>& analysis,
		        const InteriorEigenSolverOptions<LinearOperatorRealScalar<Operator>>& options,
		        VectorImagePair<LinearOperatorScalar<Operator>> retainedSpace,
		        InteriorIterationState<LinearOperatorScalar<Operator>>& ref_state,
		        InteriorEigenSolverStatistics& ref_statistics)
		{
			using Scalar = LinearOperatorScalar<Operator>;
			ref_state.PreviousRetainedEigenvalues =
			        analysis.OrderedRitzPairs.Eigenvalues.head(analysis.RetainedVectorCount);
			ref_state.PreviousPrimaryVectorCount = analysis.PrimaryVectorCount;
			ref_state.PreviousRetainedVectorCount = analysis.RetainedVectorCount;
			ref_state.ExpansionSpace = std::move(retainedSpace);

			const Eigen::Index activePrimaryVectorCount = analysis.PrimaryVectorCount - analysis.FrozenVectorCount;
			const VectorImagePair<Scalar> primaryPairs{
			        ref_state.ExpansionSpace.Vectors.middleCols(analysis.FrozenVectorCount, activePrimaryVectorCount),
			        ref_state.ExpansionSpace.Images.middleCols(analysis.FrozenVectorCount, activePrimaryVectorCount)};
			const Eigen::VectorX<LinearOperatorRealScalar<Operator>> primaryEigenvalues =
			        ref_state.PreviousRetainedEigenvalues.segment(analysis.FrozenVectorCount, activePrimaryVectorCount);
			const Eigen::MatrixX<Scalar> residuals = CalculateResiduals(primaryPairs, primaryEigenvalues);
			const Eigen::MatrixX<Scalar> unorthogonalizedCorrections = ApplyAbsoluteDiagonalPreconditioner(
			        residuals, primaryEigenvalues, diagonal, options.PreconditionerDenominatorFloor);
			const Eigen::MatrixX<Scalar> corrections = OrthogonalizeAndRemoveLinearDependence(
			        ref_state.ExpansionSpace.Vectors, unorthogonalizedCorrections, options.LinearDependenceTolerance);
			if (corrections.cols() == 0)
			{
				return ExpansionResult::NoIndependentCorrections;
			}

			const Eigen::MatrixX<Scalar> correctionImages = ApplyOperator(linearOperator, corrections, ref_statistics);
			const Eigen::Index oldExpansionSize = ref_state.ExpansionSpace.Vectors.cols();
			ref_state.ExpansionSpace.Vectors.conservativeResize(Eigen::NoChange, oldExpansionSize + corrections.cols());
			ref_state.ExpansionSpace.Images.conservativeResize(Eigen::NoChange, oldExpansionSize + corrections.cols());
			ref_state.ExpansionSpace.Vectors.rightCols(corrections.cols()) = corrections;
			ref_state.ExpansionSpace.Images.rightCols(corrections.cols()) = correctionImages;
			return ExpansionResult::Expanded;
		}
	}


	template <SelfAdjointLinearOperator Operator>
	class IterativeVectorInteractionSelfAdjointEigenSolver
	{
	public:
		using Scalar = LinearOperatorScalar<Operator>;
		using RealScalar = LinearOperatorRealScalar<Operator>;

		// Returns and stores the terminal status. Non-converged runs retain the latest
		// interval Ritz approximations for diagnosis.
		[[nodiscard]] InteriorEigenSolverStatus Compute(const Operator& linearOperator,
		                                                const EigenvalueInterval<RealScalar>& interval,
		                                                const InteriorEigenSolverOptions<RealScalar>& options);

		[[nodiscard]] InteriorEigenSolverStatus Status() const noexcept
		{
			return m_Status;
		}

		// Eigenvalues are ascending. Column i and residual norm i belong to eigenvalue i.
		[[nodiscard]] const Eigen::VectorX<RealScalar>& Eigenvalues() const noexcept
		{
			return m_Eigenvalues;
		}

		[[nodiscard]] const Eigen::MatrixX<Scalar>& Eigenvectors() const noexcept
		{
			return m_Eigenvectors;
		}

		[[nodiscard]] const Eigen::VectorX<RealScalar>& ResidualNorms() const noexcept
		{
			return m_ResidualNorms;
		}

		[[nodiscard]] const InteriorEigenSolverStatistics& Statistics() const noexcept
		{
			return m_Statistics;
		}

	private:
		InteriorEigenSolverStatus m_Status = InteriorEigenSolverStatus::NotComputed;
		Eigen::VectorX<RealScalar> m_Eigenvalues;
		Eigen::MatrixX<Scalar> m_Eigenvectors;
		Eigen::VectorX<RealScalar> m_ResidualNorms;
		InteriorEigenSolverStatistics m_Statistics;
	};


	namespace Detail::IterativeVectorInteraction
	{
		template <SelfAdjointLinearOperator Operator>
		void ValidateInteriorEigenSolverInput(
		        const Operator& linearOperator,
		        const EigenvalueInterval<LinearOperatorRealScalar<Operator>>& interval,
		        const InteriorEigenSolverOptions<LinearOperatorRealScalar<Operator>>& options)
		{
			if (linearOperator.rows() != linearOperator.cols())
			{
				throw InvalidArgumentException("The self-adjoint linear operator must be square");
			}
			if (linearOperator.rows() <= 0)
			{
				throw InvalidArgumentException("The self-adjoint linear operator must have a positive dimension");
			}
			if (linearOperator.Diagonal().size() != linearOperator.rows())
			{
				throw InvalidArgumentException("The linear operator diagonal has an inconsistent dimension");
			}
			if (!interval.IsValid())
			{
				throw InvalidArgumentException("The eigenvalue interval lower bound must not exceed its upper bound");
			}
			if (!std::isfinite(interval.LowerBound) || !std::isfinite(interval.UpperBound))
			{
				throw InvalidArgumentException("The eigenvalue interval bounds must be finite");
			}
			if (options.EigenpairCountLimit <= 0)
			{
				throw InvalidArgumentException("EigenpairCountLimit must be positive");
			}
			if (options.EigenpairCountLimit > linearOperator.rows())
			{
				throw InvalidArgumentException("EigenpairCountLimit must not exceed the operator dimension");
			}
			if (options.MaximumIterationCount <= 0)
			{
				throw InvalidArgumentException("MaximumIterationCount must be positive");
			}
			if (options.AdditionalRitzVectorCount < 0)
			{
				throw InvalidArgumentException("AdditionalRitzVectorCount must not be negative");
			}
			if (options.GeneralizedSolveInterval <= 0)
			{
				throw InvalidArgumentException("GeneralizedSolveInterval must be positive");
			}
			if (!std::isfinite(options.EigenvalueChangeTolerance) || options.EigenvalueChangeTolerance <= 0
			    || !std::isfinite(options.ResidualNormTolerance) || options.ResidualNormTolerance <= 0
			    || !std::isfinite(options.PreconditionerDenominatorFloor) || options.PreconditionerDenominatorFloor <= 0
			    || !std::isfinite(options.LinearDependenceTolerance) || options.LinearDependenceTolerance <= 0
			    || !std::isfinite(options.PreviousRitzVectorRecyclingTolerance)
			    || options.PreviousRitzVectorRecyclingTolerance <= 0
			    || !std::isfinite(options.RecyclingRefinementTolerance) || options.RecyclingRefinementTolerance <= 0
			    || !std::isfinite(options.FreezingCoefficientTolerance) || options.FreezingCoefficientTolerance <= 0
			    || !std::isfinite(options.FreezingResidualNormTolerance) || options.FreezingResidualNormTolerance < 0
			    || !std::isfinite(options.ReducedMatrixAsymmetryTolerance)
			    || options.ReducedMatrixAsymmetryTolerance <= 0)
			{
				throw InvalidArgumentException("All numerical tolerances must be positive");
			}
		}
	}


	template <SelfAdjointLinearOperator Operator>
	InteriorEigenSolverStatus IterativeVectorInteractionSelfAdjointEigenSolver<Operator>::Compute(
	        const Operator& linearOperator,
	        const EigenvalueInterval<RealScalar>& interval,
	        const InteriorEigenSolverOptions<RealScalar>& options)
	{
		using namespace Detail::IterativeVectorInteraction;
		Detail::IterativeVectorInteraction::ValidateInteriorEigenSolverInput(linearOperator, interval, options);

		m_Status = InteriorEigenSolverStatus::NotComputed;
		m_Eigenvalues.resize(0);
		m_Eigenvectors.resize(linearOperator.rows(), 0);
		m_ResidualNorms.resize(0);
		m_Statistics = {};

		const Eigen::VectorX<RealScalar> diagonal = linearOperator.Diagonal();
		const RealScalar freezingResidualNormTolerance = options.FreezingResidualNormTolerance == RealScalar{}
		                                                         ? options.ResidualNormTolerance
		                                                         : options.FreezingResidualNormTolerance;

		InteriorIterationState<Scalar> state;
		state.ExpansionSpace = CreateInitialExpansionSpace(
		        linearOperator, diagonal, interval, options.EigenpairCountLimit, m_Statistics);
		state.IsPreviousRitzVectorRecyclingActive = options.IsPreviousRitzVectorRecyclingEnabled;
		for (Eigen::Index iterationIndex = 0; iterationIndex < options.MaximumIterationCount; iterationIndex++)
		{
			m_Statistics.CompletedIterationCount = iterationIndex + 1;
			m_Statistics.MaximumExpansionSpaceSize =
			        Max(m_Statistics.MaximumExpansionSpaceSize, state.ExpansionSpace.Vectors.cols());

			const auto analysis =
			        AnalyzeExpansionSpace(state, interval, options, iterationIndex, freezingResidualNormTolerance);
			if (analysis.ComputationInfo != Eigen::Success)
			{
				m_Status = InteriorEigenSolverStatus::NumericalFailure;
				return m_Status;
			}
			m_Statistics.GeneralizedSolveCount += static_cast<Eigen::Index>(analysis.IsGeneralizedSolve);
			m_Statistics.CurrentFrozenVectorCount = analysis.FrozenVectorCount;
			m_Statistics.MaximumFrozenVectorCount =
			        Max(m_Statistics.MaximumFrozenVectorCount, analysis.FrozenVectorCount);

			const Eigen::MatrixX<Scalar> collapseCoefficients =
			        FormCollapseCoefficients(analysis, iterationIndex, options, state, m_Statistics);
			VectorImagePair<Scalar> retainedSpace =
			        TransformVectorImagePair(state.ExpansionSpace, collapseCoefficients);
			if (analysis.IsGeneralizedSolve
			    && SymmetricallyOrthonormalizeVectorImagePair(retainedSpace) != Eigen::Success)
			{
				m_Status = InteriorEigenSolverStatus::NumericalFailure;
				return m_Status;
			}
			if (analysis.AreExplicitImagesRequired)
			{
				RecalculateImages(linearOperator, retainedSpace, m_Statistics);
			}

			// Materialize inspectable results before evaluating any terminal condition.
			PublishIntervalEigenpairs(retainedSpace,
			                          analysis.OrderedRitzPairs,
			                          analysis.IntervalEigenpairCount,
			                          m_Eigenvalues,
			                          m_Eigenvectors,
			                          m_ResidualNorms);

			const bool hasResidualConverged =
			        analysis.IntervalEigenpairCount > 0 && m_ResidualNorms.maxCoeff() <= options.ResidualNormTolerance;
			const bool hasEigenvalueConverged =
			        (iterationIndex == 0 && hasResidualConverged)
			        || (iterationIndex > 0 && !analysis.Selection.HasUnmatchedIntervalRitzVector
			            && analysis.Selection.MaximumMatchedEigenvalueChange <= options.EigenvalueChangeTolerance);
			// An unconverged Ritz value may drift through an interval temporarily. Treat the
			// capacity as exceeded only after residuals validate the entire candidate set.
			if (analysis.Selection.IsEigenpairCountLimitExceeded && hasResidualConverged)
			{
				m_Status = InteriorEigenSolverStatus::EigenpairCountLimitExceeded;
				return m_Status;
			}
			if (hasResidualConverged && hasEigenvalueConverged)
			{
				m_Status = InteriorEigenSolverStatus::Converged;
				return m_Status;
			}
			if (iterationIndex + 1 == options.MaximumIterationCount)
			{
				m_Status = InteriorEigenSolverStatus::IterationLimitReached;
				return m_Status;
			}

			const ExpansionResult expansionResult = ExpandForNextIteration(
			        linearOperator, diagonal, analysis, options, std::move(retainedSpace), state, m_Statistics);
			if (expansionResult == ExpansionResult::NoIndependentCorrections)
			{
				m_Status = InteriorEigenSolverStatus::ExpansionSpaceExhausted;
				return m_Status;
			}
		}
		return m_Status;
	}
}  // namespace SecUtility::Math
