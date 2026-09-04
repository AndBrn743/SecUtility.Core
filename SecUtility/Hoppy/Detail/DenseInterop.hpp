// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/BlockDiagonalMatrixExpr.hpp>
#include <SecUtility/Hoppy/Detail/Traits.hpp>

#include <Eigen/Core>

#include <type_traits>

namespace Hoppy
{
	template <typename MatrixDerived, typename DenseDerived,
	          typename = std::enable_if_t<DenseDerived::ColsAtCompileTime != 1>,
	          typename = typename Eigen::ScalarBinaryOpTraits<
	                  typename Eigen::internal::traits<MatrixDerived>::Scalar,
	                  typename DenseDerived::Scalar>::ReturnType,
	          typename = void>
	auto operator*(const BlockDiagonalMatrixExpr<MatrixDerived>& matrix,
	               const Eigen::MatrixBase<DenseDerived>& dense)
	{
		return Eigen::Product<MatrixDerived, DenseDerived>(matrix.derived(), dense.derived());
	}

	template <typename DenseDerived, typename MatrixDerived,
	          typename = std::enable_if_t<DenseDerived::RowsAtCompileTime != 1>,
	          typename = typename Eigen::ScalarBinaryOpTraits<
	                  typename DenseDerived::Scalar,
	                  typename Eigen::internal::traits<MatrixDerived>::Scalar>::ReturnType,
	          typename = void>
	auto operator*(const Eigen::MatrixBase<DenseDerived>& dense,
	               const BlockDiagonalMatrixExpr<MatrixDerived>& matrix)
	{
		return Eigen::Product<DenseDerived, MatrixDerived>(dense.derived(), matrix.derived());
	}
}  // namespace Hoppy

namespace Eigen::internal
{
	template <typename Lhs, typename Rhs, int ProductTag>
	struct generic_product_impl<Lhs, Rhs, Hoppy::Detail::BlockDiagonalShape, DenseShape, ProductTag>
	    : generic_product_impl_base<
	              Lhs, Rhs,
	              generic_product_impl<Lhs, Rhs, Hoppy::Detail::BlockDiagonalShape, DenseShape, ProductTag>>
	{
		using Scalar = typename Product<Lhs, Rhs>::Scalar;

		template <typename Destination>
		static void scaleAndAddTo(Destination& destination, const Lhs& lhs, const Rhs& rhs,
		                          const Scalar& alpha)
		{
			for (Index index = 0; index < lhs.blockCount(); ++index)
			{
				const auto offset = lhs.blockOffset(index);
				const auto dimension = lhs.dimensionOfBlock(index);
				destination.middleRows(offset, dimension).noalias()
				        += alpha * (lhs[index] * rhs.middleRows(offset, dimension));
			}
		}
	};

	template <typename Lhs, typename Rhs, int ProductTag>
	struct generic_product_impl<Lhs, Rhs, DenseShape, Hoppy::Detail::BlockDiagonalShape, ProductTag>
	    : generic_product_impl_base<
	              Lhs, Rhs,
	              generic_product_impl<Lhs, Rhs, DenseShape, Hoppy::Detail::BlockDiagonalShape, ProductTag>>
	{
		using Scalar = typename Product<Lhs, Rhs>::Scalar;

		template <typename Destination>
		static void scaleAndAddTo(Destination& destination, const Lhs& lhs, const Rhs& rhs,
		                          const Scalar& alpha)
		{
			for (Index index = 0; index < rhs.blockCount(); ++index)
			{
				const auto offset = rhs.blockOffset(index);
				const auto dimension = rhs.dimensionOfBlock(index);
				destination.middleCols(offset, dimension).noalias()
				        += alpha * (lhs.middleCols(offset, dimension) * rhs[index]);
			}
		}
	};
}  // namespace Eigen::internal
