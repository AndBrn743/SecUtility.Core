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
		Eigen::Index MaximumEigenpairCount = 0;
		Eigen::Index MaximumIterationCount = 1000;
		// Zero selects the automatic policy: retain as many additional Ritz vectors as current primary vectors.
		Eigen::Index AdditionalRitzVectorCount = 0;
		Eigen::Index GeneralizedSolveInterval = 5;

		RealScalar EigenvalueChangeTolerance = static_cast<RealScalar>(1e-7);
		RealScalar ResidualNormTolerance = static_cast<RealScalar>(1e-7);
		RealScalar PreconditionerDenominatorFloor = static_cast<RealScalar>(1e-2);
		RealScalar LinearDependenceTolerance = static_cast<RealScalar>(1e-10);
		RealScalar PreviousRitzVectorRecyclingTolerance = static_cast<RealScalar>(1e-8);
		RealScalar RecyclingRefinementTolerance = static_cast<RealScalar>(1e-2);
		RealScalar FreezingCoefficientTolerance = static_cast<RealScalar>(1e-8);
		RealScalar FreezingResidualNormTolerance = static_cast<RealScalar>(1e-7);

		bool IsPreviousRitzVectorRecyclingEnabled = true;
		bool IsPreviousRitzVectorRecyclingDynamicallyEnabled = true;
		bool IsFreezingEnabled = true;
	};


	enum class InteriorEigenSolverStatus
	{
		NotComputed,
		Converged,
		NoEigenpairsFound,
		MaximumEigenpairCountExceeded,
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
			bool IsMaximumEigenpairCountExceeded = false;
		};


		template <typename Scalar>
		struct PermutedRitzPairs  // consider name to RitzPairs
		{
			Eigen::VectorX<typename Eigen::NumTraits<Scalar>::Real> Eigenvalues;
			Eigen::MatrixX<Scalar> Eigenvectors;
			// std::vector<Eigen::Index> OriginalIndices;
		};
	}


	template <typename Operator>
	using LinearOperatorScalar = std::remove_cvref_t<Operator>::Scalar;

	template <typename Operator>
	using LinearOperatorRealScalar = Eigen::NumTraits<LinearOperatorScalar<Operator>>::Real;


	template <typename Operator>
	concept SelfAdjointLinearOperator = requires(
	        const std::remove_reference_t<Operator>& linearOperator,
	        const Eigen::VectorX<LinearOperatorScalar<Operator>>& vector)
	{
		typename LinearOperatorScalar<Operator>;
		typename LinearOperatorRealScalar<Operator>;
		requires !Eigen::NumTraits<LinearOperatorScalar<Operator>>::IsInteger;

		{ linearOperator.rows() } -> std::convertible_to<Eigen::Index>;
		{ linearOperator.cols() } -> std::convertible_to<Eigen::Index>;

		{ linearOperator.Diagonal().size() } -> std::convertible_to<Eigen::Index>;
		{ linearOperator.Diagonal()[Eigen::Index{}] } -> std::convertible_to<LinearOperatorRealScalar<Operator>>;

		{ linearOperator.ApplyOn(vector).size() } -> std::convertible_to<Eigen::Index>;
		{ linearOperator.ApplyOn(vector)[Eigen::Index{}] } -> std::convertible_to<LinearOperatorScalar<Operator>>;
	};


	template <typename Operator>
	concept BlockSelfAdjointLinearOperator = SelfAdjointLinearOperator<Operator>
	                                      && requires(
	                                              const std::remove_reference_t<Operator>& linearOperator,
	                                              const Eigen::MatrixX<LinearOperatorScalar<Operator>>& vectors)
	{
		{ linearOperator.ApplyOn(vectors).rows() } -> std::convertible_to<Eigen::Index>;
		{ linearOperator.ApplyOn(vectors).cols() } -> std::convertible_to<Eigen::Index>;
		{ linearOperator.ApplyOn(vectors)(Eigen::Index{}, Eigen::Index{}) }
		        -> std::convertible_to<LinearOperatorScalar<Operator>>;
	};


	namespace Detail::IterativeVectorInteraction
	{
		template <typename Scalar>
		Eigen::MatrixX<typename Eigen::NumTraits<Scalar>::Real> FormRitzPopulationMatrix(
		        const Eigen::MatrixX<Scalar>& reducedEigenvectors,
		        const Eigen::Index previousRetainedVectorCount)
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
				match.IsWeakMatch = match.Population <= weakMatchPopulationThreshold
				                    || !match.IsPrimaryVectorContinuation;
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
			std::ranges::stable_sort(
			        sortedPopulationIndices,
			        std::greater<>{},
			        [&populations](const Eigen::Index index) { return populations.data()[index]; });

			std::vector isCurrentIndexAssigned(static_cast<std::size_t>(populations.rows()), false);
			std::vector isPreviousIndexAssigned(static_cast<std::size_t>(populations.cols()), false);
			std::vector previousIndexForCurrentIndex( static_cast<std::size_t>(populations.rows()), Eigen::Index{-1});
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
			const auto populationAt = [&populations, isCurrentSideSmaller](const Eigen::Index workerIndex,
			                                                                       const Eigen::Index jobIndex)
			{
				return isCurrentSideSmaller ? populations(workerIndex, jobIndex)
				                            : populations(jobIndex, workerIndex);
			};

			std::vector<RealScalar> workerPotentials(static_cast<std::size_t>(workerCount + 1));
			std::vector<RealScalar> jobPotentials(static_cast<std::size_t>(jobCount + 1));
			std::vector<Eigen::Index> workerForJob(static_cast<std::size_t>(jobCount + 1));
			std::vector<Eigen::Index> precedingJob(static_cast<std::size_t>(jobCount + 1));
			for (Eigen::Index worker = 1; worker <= workerCount; worker++)
			{
				workerForJob[0] = worker;
				std::vector<RealScalar> minimumReducedCosts(
				        static_cast<std::size_t>(jobCount + 1), std::numeric_limits<RealScalar>::infinity());
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
						const RealScalar reducedCost =
						        -populationAt(currentWorker - 1, job - 1)
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
							workerPotentials[static_cast<std::size_t>(workerForJob[static_cast<std::size_t>(job)])]
							        += smallestReducedCost;
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
			return MatchRitzVectors_Hungarian(populations,
			                                   previousPrimaryVectorCount,
			                                   previousRetainedVectorCount,
			                                   weakMatchPopulationThreshold);
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

			selection.IsMaximumEigenpairCountExceeded =
			        static_cast<Eigen::Index>(selection.IntervalRitzIndices.size()) > maximumEigenpairCount;

			// Compare each target with its previous identity match. A new or weakly matched target prevents convergence.
			RealScalar maximumChange{};
			for (const Eigen::Index currentIndex : selection.IntervalRitzIndices)
			{
				const auto& match = matches[static_cast<std::size_t>(currentIndex)];
				if (!match.IsMatched() || match.PreviousIndex >= previousEigenvalues.size() || match.IsWeakMatch)
				{
					selection.HasUnmatchedIntervalRitzVector = true;
					continue;
				}
				maximumChange = Max(
				        maximumChange,
				        Abs(currentEigenvalues[currentIndex] - previousEigenvalues[match.PreviousIndex]));
			}
			if (!selection.IntervalRitzIndices.empty() && !selection.HasUnmatchedIntervalRitzVector)
			{
				selection.MaximumMatchedEigenvalueChange = maximumChange;
			}

			// Retain complete near-degenerate clusters straddling either interval boundary.
			std::vector<bool> isBoundaryContinuation(static_cast<std::size_t>(currentEigenvalues.size()), false);
			if (!selection.IntervalRitzIndices.empty())
			{
				Eigen::Index adjacentIndex = selection.IntervalRitzIndices.front();
				for (Eigen::Index index = adjacentIndex - 1;
				     index >= 0 && currentEigenvalues[adjacentIndex] - currentEigenvalues[index]
				                           <= boundaryDegeneracyTolerance;
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
				                 const bool isLhsContinuation = matches[static_cast<std::size_t>(lhs)].IsRetainedVectorContinuation;
				                 const bool isRhsContinuation = matches[static_cast<std::size_t>(rhs)].IsRetainedVectorContinuation;
				                 if (isLhsContinuation != isRhsContinuation)
				                 {
					                 return isLhsContinuation;
				                 }
				                 return distanceFromInterval(lhs) < distanceFromInterval(rhs);
			                 });

			// Boundary-cluster continuations do not consume the configured additional-vector allowance.
			const Eigen::Index retainedCandidateCount = Min(
			        additionalRitzVectorCount, static_cast<Eigen::Index>(candidates.size()));
			selection.AdditionalRitzIndices.insert(selection.AdditionalRitzIndices.end(),
			                                           candidates.begin(),
			                                           candidates.begin() + retainedCandidateCount);
			return selection;
		}


		template <typename Scalar>
		PermutedRitzPairs<Scalar> PermuteSelectedRitzPairsToTheFront(
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

			PermutedRitzPairs<Scalar> result{
			        Eigen::VectorX<typename Eigen::NumTraits<Scalar>::Real>(eigenvalues.size()),
			        Eigen::MatrixX<Scalar>(eigenvectors.rows(), eigenvectors.cols())  /*,
			        order*/};
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
			const Eigen::Index requestedCount = Max(
			        6 + frozenVectorCount, 3 * intervalEigenpairCount + frozenVectorCount);
			return Min(requestedCount, availableRitzVectorCount);
		}


		template <typename RealScalar>
		bool IsPreviousRitzVectorRecyclingActiveFor(
		        const bool isCurrentlyActive,
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
			const auto firstAcceptedIndex = static_cast<Eigen::Index>(std::distance(
			        gramEigenvalues.begin(),
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
			return SymmetricallyOrthonormalize(
			        ProjectAgainstBasis(basis, candidates), relativeLinearDependenceTolerance);
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
		Eigen::MatrixX<Scalar> FormReducedMatrix(const VectorImagePair<Scalar>& vectorImagePair)
		{
			return vectorImagePair.Vectors.adjoint() * vectorImagePair.Images;
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
				const Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Scalar>> overlapEigenSolver(
				        overlapMatrix, Eigen::EigenvaluesOnly);
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
				const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixX<Scalar>> eigenSolver(
				        reducedMatrix, overlapMatrix);
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
	}


	template <SelfAdjointLinearOperator Operator>
	class IterativeVectorInteractionSelfAdjointEigenSolver
	{
	public:
		using Scalar = LinearOperatorScalar<Operator>;
		using RealScalar = LinearOperatorRealScalar<Operator>;

		void Compute(const Operator& linearOperator,
		             const EigenvalueInterval<RealScalar>& interval,
		             const InteriorEigenSolverOptions<RealScalar>& options);

		[[nodiscard]] InteriorEigenSolverStatus Status() const noexcept
		{
			return m_Status;
		}

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
		if (options.MaximumEigenpairCount <= 0)
		{
			throw InvalidArgumentException("MaximumEigenpairCount must be positive");
		}
		if (options.MaximumEigenpairCount > linearOperator.rows())
		{
			throw InvalidArgumentException("MaximumEigenpairCount must not exceed the operator dimension");
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
		    || !std::isfinite(options.PreconditionerDenominatorFloor)
		    || options.PreconditionerDenominatorFloor <= 0 || !std::isfinite(options.LinearDependenceTolerance)
		    || options.LinearDependenceTolerance <= 0
		    || !std::isfinite(options.PreviousRitzVectorRecyclingTolerance)
		    || options.PreviousRitzVectorRecyclingTolerance <= 0
		    || !std::isfinite(options.RecyclingRefinementTolerance)
		    || options.RecyclingRefinementTolerance <= 0 || !std::isfinite(options.FreezingCoefficientTolerance)
		    || options.FreezingCoefficientTolerance <= 0 || !std::isfinite(options.FreezingResidualNormTolerance)
		    || options.FreezingResidualNormTolerance <= 0)
		{
			throw InvalidArgumentException("All numerical tolerances must be positive");
		}
	}


	template <SelfAdjointLinearOperator Operator>
	void IterativeVectorInteractionSelfAdjointEigenSolver<Operator>::Compute(
	        const Operator& linearOperator,
	        const EigenvalueInterval<RealScalar>& interval,
	        const InteriorEigenSolverOptions<RealScalar>& options)
	{
		using namespace Detail::IterativeVectorInteraction;
		ValidateInteriorEigenSolverInput(linearOperator, interval, options);

		m_Status = InteriorEigenSolverStatus::NotComputed;
		m_Eigenvalues.resize(0);
		m_Eigenvectors.resize(linearOperator.rows(), 0);
		m_ResidualNorms.resize(0);
		m_Statistics = {};

		const Eigen::VectorX<RealScalar> diagonal = linearOperator.Diagonal();

		// Start from coordinate vectors whose diagonal estimates are closest to the requested interval.
		const RealScalar intervalCenter = (interval.LowerBound + interval.UpperBound) / RealScalar{2};
		std::vector<Eigen::Index> coordinateIndices(static_cast<std::size_t>(linearOperator.rows()));
		std::iota(coordinateIndices.begin(), coordinateIndices.end(), Eigen::Index{0});
		std::ranges::stable_sort(
		        coordinateIndices,
		        {},
		        [&diagonal, intervalCenter](const Eigen::Index index)
		        { return Abs(diagonal[index] - intervalCenter); });

		const Eigen::Index initialVectorCount = Min(
		        linearOperator.rows(), Max(Eigen::Index{6}, Eigen::Index{3} * options.MaximumEigenpairCount));
		Eigen::MatrixX<Scalar> initialVectors = Eigen::MatrixX<Scalar>::Zero(linearOperator.rows(), initialVectorCount);
		for (Eigen::Index columnIndex = 0; columnIndex < initialVectorCount; columnIndex++)
		{
			initialVectors(coordinateIndices[static_cast<std::size_t>(columnIndex)], columnIndex) = Scalar{1};
		}
		VectorImagePair<Scalar> expansionSpace{
		        initialVectors, ApplyOperator(linearOperator, initialVectors, m_Statistics)};

		Eigen::VectorX<RealScalar> previousRetainedEigenvalues;
		Eigen::Index previousPrimaryVectorCount = 0;
		Eigen::Index previousRetainedVectorCount = 0;
		Eigen::Index frozenVectorCount = 0;
		bool isPreviousRitzVectorRecyclingActive = options.IsPreviousRitzVectorRecyclingEnabled;
		for (Eigen::Index iterationIndex = 0; iterationIndex < options.MaximumIterationCount; iterationIndex++)
		{
			m_Statistics.CompletedIterationCount = iterationIndex + 1;
			m_Statistics.MaximumExpansionSpaceSize =
			        Max(m_Statistics.MaximumExpansionSpaceSize, expansionSpace.Vectors.cols());

			// Solve in the current expansion space and match its Ritz vectors to the preceding collapsed space.
			const bool isGeneralizedSolve = iterationIndex % options.GeneralizedSolveInterval == 0;
			Eigen::VectorX<RealScalar> currentEigenvalues;
			Eigen::MatrixX<Scalar> reducedEigenvectors;
			if (SolveReducedSelfAdjointEigenproblem(
			            expansionSpace, isGeneralizedSolve, currentEigenvalues, reducedEigenvectors)
			    != Eigen::Success)
			{
				m_Status = InteriorEigenSolverStatus::NumericalFailure;
				return;
			}
			m_Statistics.GeneralizedSolveCount += static_cast<Eigen::Index>(isGeneralizedSolve);
			const Eigen::MatrixX<RealScalar> populations =
			        FormRitzPopulationMatrix(reducedEigenvectors, previousRetainedVectorCount);
			const auto matches = MatchRitzVectors(
			        populations, previousPrimaryVectorCount, previousRetainedVectorCount);
			const Eigen::Index requestedAdditionalCount = options.AdditionalRitzVectorCount == 0
			                                                          ? Max(Eigen::Index{6},
			                                                                Eigen::Index{3}
			                                                                        * options.MaximumEigenpairCount)
			                                                          : options.AdditionalRitzVectorCount;
			// Select exact interval results and nearby directions that keep the next cycle productive.
			auto selection = SelectInteriorRitzVectors(currentEigenvalues,
			                                                       matches,
			                                                       previousRetainedEigenvalues,
			                                                       interval,
			                                                       options.MaximumEigenpairCount,
			                                                       requestedAdditionalCount);
			const VectorImagePair<Scalar> allRitzPairs =
			        TransformVectorImagePair(expansionSpace, reducedEigenvectors);
			const Eigen::VectorX<RealScalar> allResidualNorms =
			        CalculateColumnNorms(CalculateResiduals(allRitzPairs, currentEigenvalues));
			const std::vector<bool> isCurrentRitzVectorFrozen = options.IsFreezingEnabled
			                                                              ? DetermineFrozenRitzVectors(
			                                                                        reducedEigenvectors,
			                                                                        allResidualNorms,
			                                                                        selection.IntervalRitzIndices,
			                                                                        previousPrimaryVectorCount,
			                                                                        options.FreezingCoefficientTolerance,
			                                                                        options.FreezingResidualNormTolerance)
			                                                              : std::vector<bool>(
			                                                                        static_cast<std::size_t>(
			                                                                                currentEigenvalues.size()),
			                                                                        false);
			std::stable_sort(selection.IntervalRitzIndices.begin(),
			                 selection.IntervalRitzIndices.end(),
			                 [&isCurrentRitzVectorFrozen](const Eigen::Index lhs, const Eigen::Index rhs)
			                 {
				                 return isCurrentRitzVectorFrozen[static_cast<std::size_t>(lhs)]
				                        && !isCurrentRitzVectorFrozen[static_cast<std::size_t>(rhs)];
			                 });
			frozenVectorCount = static_cast<Eigen::Index>(std::ranges::count_if(
			        selection.IntervalRitzIndices,
			        [&isCurrentRitzVectorFrozen](const Eigen::Index index)
			        { return isCurrentRitzVectorFrozen[static_cast<std::size_t>(index)]; }));
			m_Statistics.CurrentFrozenVectorCount = frozenVectorCount;
			m_Statistics.MaximumFrozenVectorCount =
			        Max(m_Statistics.MaximumFrozenVectorCount, frozenVectorCount);
			const auto orderedRitzPairs =
			        PermuteSelectedRitzPairsToTheFront(currentEigenvalues, reducedEigenvectors, selection);
			const auto intervalEigenpairCount = static_cast<Eigen::Index>(selection.IntervalRitzIndices.size());
			const Eigen::Index selectedRetainedVectorCount = intervalEigenpairCount
			                                                       + static_cast<Eigen::Index>(
			                                                               selection.AdditionalRitzIndices.size());
			const Eigen::Index primaryVectorCount = CalculatePrimaryRitzVectorCount(
			        intervalEigenpairCount, frozenVectorCount, selectedRetainedVectorCount);
			Eigen::MatrixX<Scalar> collapseCoefficients =
			        orderedRitzPairs.Eigenvectors.leftCols(selectedRetainedVectorCount);
			if (iterationIndex > 0 && options.IsPreviousRitzVectorRecyclingEnabled
			    && options.IsPreviousRitzVectorRecyclingDynamicallyEnabled)
			{
				isPreviousRitzVectorRecyclingActive = IsPreviousRitzVectorRecyclingActiveFor(
				        isPreviousRitzVectorRecyclingActive, selection.MaximumMatchedEigenvalueChange);
			}
			if (iterationIndex > 0 && isPreviousRitzVectorRecyclingActive && previousPrimaryVectorCount > 0)
			{
				const Eigen::MatrixX<Scalar> recyclingCoefficients = FormPreviousRitzVectorRecyclingCoefficients(
				        collapseCoefficients,
				        previousPrimaryVectorCount,
				        options.PreviousRitzVectorRecyclingTolerance,
				        options.RecyclingRefinementTolerance);
				const Eigen::Index selectedColumnCount = collapseCoefficients.cols();
				collapseCoefficients.conservativeResize(
				        Eigen::NoChange, selectedColumnCount + recyclingCoefficients.cols());
				collapseCoefficients.rightCols(recyclingCoefficients.cols()) = recyclingCoefficients;
				m_Statistics.RecycledVectorCount += recyclingCoefficients.cols();
			}
			VectorImagePair<Scalar> retainedSpace = TransformVectorImagePair(expansionSpace, collapseCoefficients);

			// Materialize inspectable results before evaluating any terminal condition.
			if (intervalEigenpairCount > 0)
			{
				const VectorImagePair<Scalar> intervalPairs{
				        retainedSpace.Vectors.leftCols(intervalEigenpairCount),
				        retainedSpace.Images.leftCols(intervalEigenpairCount)};
				m_Eigenvalues = orderedRitzPairs.Eigenvalues.head(intervalEigenpairCount);
				m_Eigenvectors = intervalPairs.Vectors;
				m_ResidualNorms = CalculateColumnNorms(CalculateResiduals(intervalPairs, m_Eigenvalues));
			}
			else
			{
				m_Eigenvalues.resize(0);
				m_Eigenvectors.resize(linearOperator.rows(), 0);
				m_ResidualNorms.resize(0);
			}

			if (selection.IsMaximumEigenpairCountExceeded)
			{
				m_Status = InteriorEigenSolverStatus::MaximumEigenpairCountExceeded;
				return;
			}
			const bool hasResidualConverged = intervalEigenpairCount > 0
			                                  && m_ResidualNorms.maxCoeff() <= options.ResidualNormTolerance;
			const bool hasEigenvalueConverged =
			        (iterationIndex == 0 && hasResidualConverged)
			        || (iterationIndex > 0 && !selection.HasUnmatchedIntervalRitzVector
			            && selection.MaximumMatchedEigenvalueChange <= options.EigenvalueChangeTolerance);
			if (hasResidualConverged && hasEigenvalueConverged)
			{
				m_Status = InteriorEigenSolverStatus::Converged;
				return;
			}
			if (iterationIndex + 1 == options.MaximumIterationCount)
			{
				m_Status = intervalEigenpairCount == 0 ? InteriorEigenSolverStatus::NoEigenpairsFound
				                                             : InteriorEigenSolverStatus::IterationLimitReached;
				return;
			}

			// Collapse vectors and images together; only genuinely new corrections require operator applications.
			previousRetainedEigenvalues = orderedRitzPairs.Eigenvalues.head(selectedRetainedVectorCount);
			previousPrimaryVectorCount = primaryVectorCount;
			previousRetainedVectorCount = selectedRetainedVectorCount;
			expansionSpace = std::move(retainedSpace);

			const Eigen::Index activePrimaryVectorCount = primaryVectorCount - frozenVectorCount;
			const VectorImagePair<Scalar> primaryPairs{
			        expansionSpace.Vectors.middleCols(frozenVectorCount, activePrimaryVectorCount),
			        expansionSpace.Images.middleCols(frozenVectorCount, activePrimaryVectorCount)};
			const Eigen::VectorX<RealScalar> primaryEigenvalues =
			        previousRetainedEigenvalues.segment(frozenVectorCount, activePrimaryVectorCount);
			const Eigen::MatrixX<Scalar> residuals = CalculateResiduals(primaryPairs, primaryEigenvalues);
			const Eigen::MatrixX<Scalar> unorthogonalizedCorrections = ApplyAbsoluteDiagonalPreconditioner(
			        residuals, primaryEigenvalues, diagonal, options.PreconditionerDenominatorFloor);
			const Eigen::MatrixX<Scalar> corrections = OrthogonalizeAndRemoveLinearDependence(
			        expansionSpace.Vectors, unorthogonalizedCorrections, options.LinearDependenceTolerance);
			if (corrections.cols() == 0)
			{
				m_Status = intervalEigenpairCount == 0 ? InteriorEigenSolverStatus::NoEigenpairsFound
				                                             : InteriorEigenSolverStatus::ExpansionSpaceExhausted;
				return;
			}

			const Eigen::MatrixX<Scalar> correctionImages = ApplyOperator(linearOperator, corrections, m_Statistics);
			const Eigen::Index oldExpansionSize = expansionSpace.Vectors.cols();
			expansionSpace.Vectors.conservativeResize(Eigen::NoChange, oldExpansionSize + corrections.cols());
			expansionSpace.Images.conservativeResize(Eigen::NoChange, oldExpansionSize + corrections.cols());
			expansionSpace.Vectors.rightCols(corrections.cols()) = corrections;
			expansionSpace.Images.rightCols(corrections.cols()) = correctionImages;
		}
	}
}  // namespace SecUtility::Math
