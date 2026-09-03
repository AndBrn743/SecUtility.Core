// SPDX-License-Identifier: MIT

#pragma once

#include <Eigen/Core>

#if EIGEN_WORLD_VERSION != 3 || EIGEN_MAJOR_VERSION != 5 || EIGEN_MINOR_VERSION != 0
#error "Hoppy requires Eigen 5.0.0; use a supported Eigen release or update Hoppy's Eigen integration."
#endif

namespace Hoppy
{
	struct Column
	{};
	struct Row
	{};

	struct DenseBlockPolicy;

	template <typename TScalar, typename TBlockPolicy = DenseBlockPolicy>
	class BlockDiagonalMatrix;

	template <typename TScalar, typename TOrientation = Column>
	class BlockVector;

	template <typename Derived>
	class BlockExpressionBase;

	template <typename Derived>
	class BlockDiagonalMatrixExpr;

	template <typename Derived>
	class BlockVectorExpr;

	namespace Detail
	{
		template <typename Source, bool Writable>
		class DiagonalReturnType;
		struct BlockDiagonalStorage;
		struct BlockVectorStorage;
		struct BlockDiagonalShape;
		struct BlockVectorShape;
	}  // namespace Detail
}  // namespace Hoppy
