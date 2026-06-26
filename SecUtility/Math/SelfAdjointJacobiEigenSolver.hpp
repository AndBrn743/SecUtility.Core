// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Andy Brown

#pragma once

#include <Eigen/Dense>
#include <SecUtility/Collection/RoundRobinOrdering.hpp>
#include <SecUtility/Math/Core.hpp>
#include <SecUtility/Math/Regin.hpp>
#include <SecUtility/Math/RotationOfPointsInThePlane.hpp>
#include <SecUtility/Threading/ThreadPool.hpp>
#include <future>
#include <map>
#include <numeric>
#include <utility>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-literal-operator"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
#include <range/v3/view/iota.hpp>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif


#if !defined(SECUTILITY_CACHE_LINE_SIZE_IN_BYTES)
#define SECUTILITY_CACHE_LINE_SIZE_IN_BYTES (64)
#endif


namespace SecUtility::Math
{
	/// <summary>
	/// Jacobi-rotation based diagonalizer (EVD solver) for self-adjoint (aka Hermitian) matrices which also provide
	/// block-diagonalization capability
	/// </summary>
	template <typename Scalar>
	class SelfAdjointJacobiEigenSolver
	{
		struct DefaultIndexPairFilter
		{
			bool operator()(const Eigen::Index i, const Eigen::Index j) const noexcept
			{
				return i >= 0 && j >= 0;
			}
		};

		using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
		using IndexPair = std::pair<Eigen::Index, Eigen::Index>;
		using LocalIndexPair = std::pair<std::int8_t, std::int8_t>;
		using SineCosinePair = std::pair<Scalar, RealScalar>;
		static constexpr Eigen::Index PaddingSize = SECUTILITY_CACHE_LINE_SIZE_IN_BYTES / sizeof(Scalar);


	public:
		template <typename ParallelizationOrdering, typename IndexPairFilter>
		SelfAdjointJacobiEigenSolver(/*COPY*/ Eigen::MatrixX<Scalar> matrix,
		                             ParallelizationOrdering parallelizationOrdering,
		                             const RealScalar tolerance = 1e-10,
		                             const Eigen::Index maxIterations = 64,
		                             IndexPairFilter filter = {})
		    : m_TransformedMatrix(std::move(matrix))
		{
			static_assert(
			        std::is_base_of_v<AbstractRoundRobinOrdering<ParallelizationOrdering>, ParallelizationOrdering>);
			Solve(parallelizationOrdering, tolerance, maxIterations, filter);
		}

		template <typename ParallelizationOrdering>
		SelfAdjointJacobiEigenSolver(const Eigen::MatrixX<Scalar>& matrix,
		                             ParallelizationOrdering parallelizationOrdering,
		                             const RealScalar tolerance = 1e-10,
		                             const Eigen::Index maxIterations = 64)
		    : SelfAdjointJacobiEigenSolver{matrix,
		                                   std::move(parallelizationOrdering),
		                                   tolerance,
		                                   maxIterations,
		                                   DefaultIndexPairFilter{}}
		{
			/* NO CODE */
		}

		explicit SelfAdjointJacobiEigenSolver(const Eigen::MatrixX<Scalar>& matrix,
		                                      const RealScalar tolerance = 1e-10,
		                                      const Eigen::Index maxIterations = 64)
		    : SelfAdjointJacobiEigenSolver{matrix,
		                                   RoundRobinOrdering{ranges::views::iota(Eigen::Index{0}, matrix.rows()), -1},
		                                   tolerance,
		                                   maxIterations}
		{
			/* NO CODE */
		}

	protected:
		template <typename ParallelizationOrdering, typename IndexPairFilter = DefaultIndexPairFilter>
		void Solve(ParallelizationOrdering parallelizationOrdering,
		           const RealScalar tolerance = 1e-10,
		           const Eigen::Index maxIterations = 64,
		           IndexPairFilter filter = {})
		{
			const Eigen::Index dimension = m_TransformedMatrix.rows();
			if (IsPaddingNeededFor(dimension))
			{
				m_TransformedMatrix.conservativeResize(dimension + PaddingSize, dimension + PaddingSize);
			}
			m_TransformationMatrix =
			        Eigen::MatrixX<Scalar>::Identity(m_TransformedMatrix.rows(), m_TransformedMatrix.cols());

			std::vector<std::pair<Eigen::Index, Eigen::Index>> rotationIndexPairs;
			rotationIndexPairs.reserve(dimension / 2);

			for (Eigen::Index iter = 0; iter < maxIterations; iter++)
			{
				bool hasConverged = true;

				// std::cout << "SWEEP" << std::setw(3) << iter << "\n";
				for (Eigen::Index delta = 0; delta < parallelizationOrdering.Period(); delta++)
				{
					rotationIndexPairs.clear();

					if (delta > 0)
					{
						parallelizationOrdering.NextCycle();
					}
					for (const auto [p, q] : parallelizationOrdering)
					{
						if (filter(p, q) && std::abs(m_TransformedMatrix(p, q)) > tolerance)
						{
							rotationIndexPairs.emplace_back(p, q);
						}
					}

					hasConverged &= rotationIndexPairs.empty();
					PerformJacobiRotations(m_TransformedMatrix, m_TransformationMatrix, rotationIndexPairs);
				}

				if (hasConverged)
				{
					break;
				}
			}

			if (IsPaddingNeededFor(dimension))
			{
				m_TransformedMatrix.conservativeResize(dimension, dimension);
				m_TransformationMatrix.conservativeResize(dimension, dimension);
			}
		}

		template <typename RoundRobinOrdering>
		Eigen::Index SweepOffDiagonalBlocks(const RealScalar tolerance,
		                                    const Eigen::Index dimension,
		                                    const Eigen::Index blockSize,
		                                    RoundRobinOrdering rro)
		{
			Eigen::Index nonZeroCount = 0;

			for (Eigen::Index robinIndex = 0; robinIndex < rro.Period(); robinIndex++)
			{
				std::vector<std::future<Eigen::Index>> futures;
				// futures.reserve(threadCount);
				std::map<std::pair<Eigen::Index, Eigen::Index>, std::vector<std::pair<IndexPair, SineCosinePair>>>
				        pivots;

#if __cplusplus >= 202002L
				for (const auto [blockRowIndex, blockColumnIndex] : rro)
				{
#else
				for (const auto blockRowColumnIndex : rro)
				{
					const auto blockRowIndex = std::get<0>(blockRowColumnIndex);
					const auto blockColumnIndex = std::get<1>(blockRowColumnIndex);
#endif
					if (blockRowIndex == -1 || blockColumnIndex == -1)
					{
						continue;
					}

					futures.emplace_back(Threading::MasterThreadPool().Submit(
					        [tolerance,
					         blockRowIndex,
					         blockColumnIndex,
					         blockSize,
					         dimension,
					         &localPivots = pivots[{blockRowIndex, blockColumnIndex}],
					         this]
					        {
						        Eigen::Index nonZeroElementCount = 0;
						        localPivots.reserve(dimension * dimension);

						        for (Eigen::Index q = blockColumnIndex * blockSize;
						             q < (blockColumnIndex + 1) * blockSize && q < dimension;
						             q++)
						        {
							        for (Eigen::Index p = blockRowIndex * blockSize;
							             p < (blockRowIndex + 1) * blockSize && p < dimension;
							             p++)
							        {
								        if (std::abs(m_TransformedMatrix(p, q)) > tolerance)
								        {
									        nonZeroElementCount++;

									        // perform column rotation
									        const auto [sin, cos] = JacobiRotationalSinCosPairWithoutZeroCheck(
									                m_TransformedMatrix(p, q),
									                m_TransformedMatrix.diagonal()[p],
									                m_TransformedMatrix.diagonal()[q]);
									        localPivots.emplace_back(std::pair{p, q}, std::pair{sin, cos});

									        PerformsRotationOfPointsInThePlane(
									                m_TransformedMatrix.col(p), m_TransformedMatrix.col(q), sin, cos);
									        PerformsRotationOfPointsInThePlane(m_TransformationMatrix.col(p),
									                                           m_TransformationMatrix.col(q),
									                                           sin,
									                                           cos);
								        }
							        }
						        }

						        return nonZeroElementCount;
					        }));
				}

				const auto nonZerosInOffDiagonalBlocks =
				        std::accumulate(futures.begin(),
				                        futures.end(),
				                        Eigen::Index{0},
				                        [](const auto s, auto&& f) { return s + f.get(); });

				if (nonZerosInOffDiagonalBlocks == 0)
				{
					rro.NextCycle();
					continue;
				}

				nonZeroCount += nonZerosInOffDiagonalBlocks;
				futures.clear();

				m_TransformedMatrix.adjointInPlace();
				for (const auto [blockRowIndex, blockColumnIndex] : rro)
				{
					if (blockRowIndex == -1 || blockColumnIndex == -1)
					{
						continue;
					}

					futures.emplace_back(Threading::MasterThreadPool().Submit(
					        [&localPivots = pivots[{blockRowIndex, blockColumnIndex}], this]
					        {
						        for (const auto& [pq, sc] : localPivots)
						        {
							        const auto [p, q] = pq;
							        const auto [sin, cos] = sc;

							        PerformsRotationOfPointsInThePlane(
							                m_TransformedMatrix.col(p), m_TransformedMatrix.col(q), sin, cos);
						        }

						        return Eigen::Index{};  // we should use std::future<void> instead
					        }));
				}
				std::for_each(futures.begin(), futures.end(), [](auto&& f) { f.get(); });
				futures.clear();

				rro.NextCycle();
			}

			return nonZeroCount;
		}

		Eigen::Index SweepDiagonalBlocks(const RealScalar tolerance,
		                                 const Eigen::Index dimension,
		                                 const Eigen::Index blockSize,
		                                 const Eigen::Index oneDimensionalBlockCount)
		{
			std::vector<std::future<Eigen::Index>> futures;
			std::map<Eigen::Index, std::vector<std::pair<IndexPair, SineCosinePair>>> diagonalPivots;

			for (Eigen::Index diagonalBlockIndex = 0; diagonalBlockIndex < oneDimensionalBlockCount;
			     diagonalBlockIndex++)
			{
				futures.emplace_back(Threading::MasterThreadPool().Submit(
				        [diagonalBlockIndex,
				         blockSize,
				         dimension,
				         &localPivots = diagonalPivots[diagonalBlockIndex],
				         tolerance,
				         this]
				        {
					        Eigen::Index nonZeroElementCount = 0;
					        localPivots.reserve(Regin1<2>(blockSize - 1));

					        for (Eigen::Index p = diagonalBlockIndex * blockSize;
					             p < (diagonalBlockIndex + 1) * blockSize && p < dimension;
					             p++)
					        {
						        for (Eigen::Index q = diagonalBlockIndex * blockSize; q < p; q++)
						        {
							        if (std::abs(m_TransformedMatrix(p, q)) > tolerance)
							        {
								        nonZeroElementCount++;

								        const auto [sin, cos] = JacobiRotationalSinCosPairWithoutZeroCheck(
								                m_TransformedMatrix(p, q),
								                m_TransformedMatrix.diagonal()[p],
								                m_TransformedMatrix.diagonal()[q]);
								        localPivots.emplace_back(std::pair{p, q}, std::pair{sin, cos});

								        PerformsRotationOfPointsInThePlane(
								                m_TransformedMatrix.col(p), m_TransformedMatrix.col(q), sin, cos);
								        PerformsRotationOfPointsInThePlane(
								                m_TransformationMatrix.col(p), m_TransformationMatrix.col(q), sin, cos);
							        }
						        }
					        }

					        return nonZeroElementCount;
				        }));
			}

			const auto nonZeroCount = std::accumulate(futures.begin(),
			                                          futures.end(),
			                                          Eigen::Index{0},
			                                          [](const auto s, auto&& f) { return s + f.get(); });

			if (nonZeroCount == 0)
			{
				return nonZeroCount;
			}

			futures.clear();

			m_TransformedMatrix.adjointInPlace();
			for (Eigen::Index diagonalBlockIndex = 0; diagonalBlockIndex < oneDimensionalBlockCount;
			     diagonalBlockIndex++)
			{
				futures.emplace_back(Threading::MasterThreadPool().Submit(
				        [&localPivots = diagonalPivots[diagonalBlockIndex], this]
				        {
					        for (const auto& [pq, sc] : localPivots)
					        {
						        const auto [p, q] = pq;
						        const auto [sin, cos] = sc;

						        PerformsRotationOfPointsInThePlane(
						                m_TransformedMatrix.col(p), m_TransformedMatrix.col(q), sin, cos);
					        }

					        return Eigen::Index{};  // we should use std::future<void> instead
				        }));
			}
			std::for_each(futures.begin(), futures.end(), [](auto&& f) { f.get(); });

			return nonZeroCount;
		}

		void SolveByBlock(const RealScalar tolerance = 1e-10, const Eigen::Index maxIterations = 64)
		{
			const auto dimension = m_TransformedMatrix.rows();
			// const auto blockSize = (dimension + 2 * threadCount - 1) / (2 * threadCount);
			const auto blockSize = PaddingSize * 8;
			const auto oneDimensionalBlockCount = (dimension + blockSize - 1) / blockSize;
			if (IsPaddingNeededFor(dimension))
			{
				m_TransformedMatrix.conservativeResize(dimension + PaddingSize, dimension + PaddingSize);
			}
			m_TransformationMatrix =
			        Eigen::MatrixX<Scalar>::Identity(m_TransformedMatrix.rows(), m_TransformedMatrix.cols());
			RoundRobinOrdering rro{ranges::views::iota(Eigen::Index{0}, oneDimensionalBlockCount), -1};

			// std::cout << "std::thread::hardware_concurrency(): " << std::thread ::hardware_concurrency() <<
			// std::endl; std::cout << "blockSize: " << blockSize << std::endl; std::cout << "oneDimensionalBlockCount:
			// " << oneDimensionalBlockCount << std::endl; std::cout << "rro.PairsPerCycle(): " << rro.PairsPerCycle()
			// << std::endl;

			// std::cout << "m_TransformedMatrix:\n" << m_TransformedMatrix << std::endl;

			for (Eigen::Index iter = 0; iter < maxIterations; iter++)
			{
				// std::cout << "SWEEP: " << iter << std::endl;
				Eigen::Index nonZeroCount = SweepOffDiagonalBlocks(tolerance, dimension, blockSize, rro);
				nonZeroCount += SweepDiagonalBlocks(tolerance, dimension, blockSize, oneDimensionalBlockCount);
				// std::cout << "    nonZeroCount: " << nonZeroCount << '\n';

				// std::cout << "m_TransformedMatrix:\n" << m_TransformedMatrix << std::endl;

				if (nonZeroCount == 0)
				{
					break;
				}
			}
			// std::cout << std::endl;

			if (IsPaddingNeededFor(dimension))
			{
				m_TransformedMatrix.conservativeResize(dimension, dimension);
				m_TransformationMatrix.conservativeResize(dimension, dimension);
			}
		}


	public:
		const Eigen::MatrixX<Scalar>& TransformedMatrix() const noexcept
		{
			return m_TransformedMatrix;
		}

		const Eigen::MatrixX<Scalar>& TransformationMatrix() const noexcept
		{
			return m_TransformationMatrix;
		}

		decltype(auto) UnsortedEigenvalues() const noexcept
		{
			return m_TransformedMatrix.diagonal();
		}

		const Eigen::MatrixX<Scalar>& UnsortedEigenvector() const noexcept
		{
			return m_TransformationMatrix;
		}

		typename Eigen::NumTraits<Scalar>::Real Error() const noexcept
		{
			return (m_TransformedMatrix
			        - Eigen::MatrixX<Scalar>::Identity(m_TransformedMatrix.rows(), m_TransformedMatrix.cols()))
			        .cwiseAbs()
			        .maxCoeff();
		}


	private:
		static void PerformJacobiRotations(Eigen::MatrixX<Scalar>& ref_matrix,
		                                   Eigen::MatrixX<Scalar>& ref_eigenvectors,
		                                   const std::vector<std::pair<Eigen::Index, Eigen::Index>>& rotationIndexPairs)
		{
			// I know that the recommended thing to do is `default(none)`, but I've made every that need to be
			// and can be `private` in the smallest block scoped already, so it should do the job already

			const auto rotationalSinCosPairs = [&rotationIndexPairs, &ref_matrix]
			{
				std::vector<std::pair<Scalar, RealScalar>> scPairs(rotationIndexPairs.size());

				// compute sin/cos in parallel first
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) default(shared)  // NOLINT(*-use-default-none)
#endif
				for (std::size_t i = 0; i < rotationIndexPairs.size(); ++i)
				{
					const auto [p, q] = rotationIndexPairs[i];
					scPairs[i] = JacobiRotationalSinCosPairWithoutZeroCheck(
					        ref_matrix(p, q), ref_matrix.diagonal()[p], ref_matrix.diagonal()[q]);
				}

				return scPairs;
			}();


			// apply column rotations: do only for pairs that are independent concurrently
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) default(shared)  // NOLINT(*-use-default-none)
#endif
			for (std::size_t i = 0; i < rotationIndexPairs.size(); i++)
			{
				const auto [p, q] = rotationIndexPairs[i];
				const auto [sin, cos] = rotationalSinCosPairs[i];
				PerformsRotationOfPointsInThePlane(ref_matrix.col(p), ref_matrix.col(q), sin, cos);
				PerformsRotationOfPointsInThePlane(ref_eigenvectors.col(p), ref_eigenvectors.col(q), sin, cos);
			}

			// apply row rotations similarly, with care about independence
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) default(shared)  // NOLINT(*-use-default-none)
#endif
			for (std::size_t i = 0; i < rotationIndexPairs.size(); i++)
			{
				const auto [p, q] = rotationIndexPairs[i];
				const auto [sin, cos] = rotationalSinCosPairs[i];
				PerformsRotationOfPointsInThePlane(ref_matrix.row(p), ref_matrix.row(q), Conj(sin), cos);
			}
		}


		static EIGEN_ALWAYS_INLINE std::pair<Scalar, RealScalar> JacobiRotationalSinCosPairWithoutZeroCheck(
		        const Scalar aPQ, const Scalar aPP, const Scalar aQQ)
		{
			if constexpr (std::is_same_v<Scalar, RealScalar>)
			{
				const auto beta = (aQQ - aPP) / (2 * aPQ);
				const auto tan = -1 / (beta + CopySignToTheLeft(Sqrt(1 + beta * beta), beta));
				const auto cos = 1 / Sqrt(1 + tan * tan);
				const auto sin = tan * cos;

				return {sin, cos};
			}
			else
			{
				const auto r = Abs(aPQ);

				const auto tau = (Re(aQQ) - Re(aPP)) / (2 * r);
				const auto t = -1 / (tau + CopySignToTheLeft(Sqrt(1 + tau * tau), tau));
				const auto c = 1 / Sqrt(1 + t * t);
				const auto s = c * t * Conj(aPQ) / r;  // equivalent to std::polar(c * t, -std::arg(aPQ))

				return {s, c};
			}
		}

		constexpr static bool IsPaddingNeededFor(const Eigen::Index dimension) noexcept
		{
			return IsPowerOfTwo(dimension) && dimension > 64;
		}

		Eigen::MatrixX<Scalar> m_TransformedMatrix;
		Eigen::MatrixX<Scalar> m_TransformationMatrix;
	};
}
