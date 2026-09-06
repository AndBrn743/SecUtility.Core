// SPDX-License-Identifier: MIT

#pragma once

#include <Eigen/Core>

#include <complex>

#if EIGEN_WORLD_VERSION != 3 || EIGEN_MAJOR_VERSION != 5 || EIGEN_MINOR_VERSION != 0
#error "Hoppy requires Eigen 5.0.0; use a supported Eigen release or update Hoppy's Eigen integration."
#endif

namespace Hoppy
{
	namespace BlockVectorOrientation
	{
		struct Column
		{};
		struct Row
		{};
	}  // namespace BlockVectorOrientation

	struct DenseBlockPolicy;

	template <typename TScalar, typename TBlockPolicy = DenseBlockPolicy>
	class BlockDiagonalMatrix;

	template <typename TScalar, typename TOrientation = BlockVectorOrientation::Column>
	class BlockVector;

	template <typename Derived>
	class BlockExpressionBase;

	template <typename Derived>
	class BlockDiagonalMatrixExpr;

	template <typename Derived>
	class BlockVectorExpr;

	using BlockDiagonalMatrixXd = BlockDiagonalMatrix<double>;
	using BlockDiagonalMatrixXf = BlockDiagonalMatrix<float>;
	using BlockDiagonalMatrixXcd = BlockDiagonalMatrix<std::complex<double>>;
	using BlockDiagonalMatrixXcf = BlockDiagonalMatrix<std::complex<float>>;
	using BlockDiagonalMatrixXi = BlockDiagonalMatrix<int>;
	using BlockDiagonalMatrixXl = BlockDiagonalMatrix<long>;

	using BlockVectorXd = BlockVector<double>;
	using BlockVectorXf = BlockVector<float>;
	using BlockVectorXcd = BlockVector<std::complex<double>>;
	using BlockVectorXcf = BlockVector<std::complex<float>>;
	using BlockVectorXi = BlockVector<int>;
	using BlockVectorXl = BlockVector<long>;

	using BlockRowVectorXd = BlockVector<double, BlockVectorOrientation::Row>;
	using BlockRowVectorXf = BlockVector<float, BlockVectorOrientation::Row>;
	using BlockRowVectorXcd = BlockVector<std::complex<double>, BlockVectorOrientation::Row>;
	using BlockRowVectorXcf = BlockVector<std::complex<float>, BlockVectorOrientation::Row>;
	using BlockRowVectorXi = BlockVector<int, BlockVectorOrientation::Row>;
	using BlockRowVectorXl = BlockVector<long, BlockVectorOrientation::Row>;

	namespace Detail
	{
		template <typename Matrix, typename Transform, bool Back>
		class CongruenceExpression;
		template <typename Source, bool Writable>
		class DiagonalReturnType;
		struct BlockDiagonalStorage;
		struct BlockVectorStorage;
		struct BlockDiagonalShape;
		struct BlockVectorShape;
	}  // namespace Detail
}  // namespace Hoppy
