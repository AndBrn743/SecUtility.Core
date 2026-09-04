// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/BlockDiagonalMatrixExpr.hpp>
#include <SecUtility/Hoppy/Detail/Traits.hpp>

#include <Eigen/Core>

#include <type_traits>

namespace Hoppy::Detail
{
	template <typename Dense, typename Block, bool DenseFirst>
	class DenseBlockDiagonalSum;
	template <typename Dense, typename Block, bool DenseFirst>
	class DenseBlockDiagonalDifference;
}

template <typename Dense, typename Block, bool DenseFirst>
struct Eigen::internal::traits<Hoppy::Detail::DenseBlockDiagonalSum<Dense, Block, DenseFirst>>
{
	using DenseScalar = typename traits<Dense>::Scalar;
	using BlockScalar = typename traits<Block>::Scalar;
	using Operation = std::conditional_t<DenseFirst, scalar_sum_op<DenseScalar, BlockScalar>,
	                                     scalar_sum_op<BlockScalar, DenseScalar>>;
	using Scalar = typename std::conditional_t<
	        DenseFirst, Eigen::ScalarBinaryOpTraits<DenseScalar, BlockScalar, Operation>,
	        Eigen::ScalarBinaryOpTraits<BlockScalar, DenseScalar, Operation>>::ReturnType;
	using ReturnType = Eigen::MatrixX<Scalar>;
};

template <typename Dense, typename Block, bool DenseFirst>
struct Eigen::internal::traits<Hoppy::Detail::DenseBlockDiagonalDifference<Dense, Block, DenseFirst>>
{
	using DenseScalar = typename traits<Dense>::Scalar;
	using BlockScalar = typename traits<Block>::Scalar;
	using Operation = std::conditional_t<DenseFirst, scalar_difference_op<DenseScalar, BlockScalar>,
	                                     scalar_difference_op<BlockScalar, DenseScalar>>;
	using Scalar = typename std::conditional_t<
	        DenseFirst, Eigen::ScalarBinaryOpTraits<DenseScalar, BlockScalar, Operation>,
	        Eigen::ScalarBinaryOpTraits<BlockScalar, DenseScalar, Operation>>::ReturnType;
	using ReturnType = Eigen::MatrixX<Scalar>;
};

namespace Hoppy::Detail
{
	template <typename Dense, typename Block, bool DenseFirst>
	class DenseBlockDiagonalSum
	    : public Eigen::ReturnByValue<DenseBlockDiagonalSum<Dense, Block, DenseFirst>>
	{
	public:
		using Scalar = typename Eigen::internal::traits<DenseBlockDiagonalSum>::Scalar;
		DenseBlockDiagonalSum(const Dense& dense, const Block& block) : m_Dense(dense), m_Block(block)
		{
			eigen_assert(m_Dense.rows() == m_Block.rows() && m_Dense.cols() == m_Block.cols());
		}
		Eigen::Index rows() const { return m_Dense.rows(); }
		Eigen::Index cols() const { return m_Dense.cols(); }

		template <typename Destination>
		void evalTo(Destination& destination) const
		{
			assertNoOverlap(m_Block, destination);
			destination = m_Dense;
			for (Eigen::Index index = 0; index < m_Block.blockCount(); ++index)
			{
				const auto offset = m_Block.blockOffset(index);
				const auto dimension = m_Block.dimensionOfBlock(index);
				destination.block(offset, offset, dimension, dimension) += m_Block[index];
			}
		}

	private:
		typename Eigen::internal::ref_selector<Dense>::type m_Dense;
		typename Eigen::internal::ref_selector<Block>::type m_Block;
	};

	template <typename Dense, typename Block, bool DenseFirst>
	class DenseBlockDiagonalDifference
	    : public Eigen::ReturnByValue<DenseBlockDiagonalDifference<Dense, Block, DenseFirst>>
	{
	public:
		using Scalar = typename Eigen::internal::traits<DenseBlockDiagonalDifference>::Scalar;
		DenseBlockDiagonalDifference(const Dense& dense, const Block& block) : m_Dense(dense), m_Block(block)
		{
			eigen_assert(m_Dense.rows() == m_Block.rows() && m_Dense.cols() == m_Block.cols());
		}
		Eigen::Index rows() const { return m_Dense.rows(); }
		Eigen::Index cols() const { return m_Dense.cols(); }

		template <typename Destination>
		void evalTo(Destination& destination) const
		{
			assertNoOverlap(m_Block, destination);
			if constexpr (DenseFirst)
				destination = m_Dense;
			else
				destination = -m_Dense;
			for (Eigen::Index index = 0; index < m_Block.blockCount(); ++index)
			{
				const auto offset = m_Block.blockOffset(index);
				const auto dimension = m_Block.dimensionOfBlock(index);
				if constexpr (DenseFirst)
					destination.block(offset, offset, dimension, dimension) -= m_Block[index];
				else
					destination.block(offset, offset, dimension, dimension) += m_Block[index];
			}
		}

	private:
		typename Eigen::internal::ref_selector<Dense>::type m_Dense;
		typename Eigen::internal::ref_selector<Block>::type m_Block;
	};
}  // namespace Hoppy::Detail

namespace Hoppy
{
	template <typename DenseDerived, typename BlockDerived,
	          typename = typename Eigen::ScalarBinaryOpTraits<
	                  typename DenseDerived::Scalar,
	                  typename Eigen::internal::traits<BlockDerived>::Scalar,
	                  Eigen::internal::scalar_sum_op<
	                          typename DenseDerived::Scalar,
	                          typename Eigen::internal::traits<BlockDerived>::Scalar>>::ReturnType>
	auto operator+(const Eigen::MatrixBase<DenseDerived>& dense,
	               const BlockDiagonalMatrixExpr<BlockDerived>& block)
	{
		return Detail::DenseBlockDiagonalSum<DenseDerived, BlockDerived, true>(dense.derived(),
		                                                                           block.derived());
	}
	template <typename BlockDerived, typename DenseDerived,
	          typename = typename Eigen::ScalarBinaryOpTraits<
	                  typename Eigen::internal::traits<BlockDerived>::Scalar,
	                  typename DenseDerived::Scalar,
	                  Eigen::internal::scalar_sum_op<
	                          typename Eigen::internal::traits<BlockDerived>::Scalar,
	                          typename DenseDerived::Scalar>>::ReturnType>
	auto operator+(const BlockDiagonalMatrixExpr<BlockDerived>& block,
	               const Eigen::MatrixBase<DenseDerived>& dense)
	{
		return Detail::DenseBlockDiagonalSum<DenseDerived, BlockDerived, false>(dense.derived(),
		                                                                            block.derived());
	}
	template <typename DenseDerived, typename BlockDerived,
	          typename = typename Eigen::ScalarBinaryOpTraits<
	                  typename DenseDerived::Scalar,
	                  typename Eigen::internal::traits<BlockDerived>::Scalar,
	                  Eigen::internal::scalar_difference_op<
	                          typename DenseDerived::Scalar,
	                          typename Eigen::internal::traits<BlockDerived>::Scalar>>::ReturnType>
	auto operator-(const Eigen::MatrixBase<DenseDerived>& dense,
	               const BlockDiagonalMatrixExpr<BlockDerived>& block)
	{
		return Detail::DenseBlockDiagonalDifference<DenseDerived, BlockDerived, true>(dense.derived(),
		                                                                                  block.derived());
	}
	template <typename BlockDerived, typename DenseDerived,
	          typename = typename Eigen::ScalarBinaryOpTraits<
	                  typename Eigen::internal::traits<BlockDerived>::Scalar,
	                  typename DenseDerived::Scalar,
	                  Eigen::internal::scalar_difference_op<
	                          typename Eigen::internal::traits<BlockDerived>::Scalar,
	                          typename DenseDerived::Scalar>>::ReturnType>
	auto operator-(const BlockDiagonalMatrixExpr<BlockDerived>& block,
	               const Eigen::MatrixBase<DenseDerived>& dense)
	{
		return Detail::DenseBlockDiagonalDifference<DenseDerived, BlockDerived, false>(dense.derived(),
		                                                                                   block.derived());
	}

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
