// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/BlockExpressionBase.hpp>
#include <SecUtility/Hoppy/ForwardDeclarations.hpp>

#include <Eigen/Core>

#include <type_traits>

namespace Hoppy
{
	template <typename Derived>
	class BlockVectorExpr : public BlockExpressionBase<Derived>
	{
	public:
		using Base = BlockExpressionBase<Derived>;
		using Scalar = typename Base::Scalar;
		using Orientation = typename Eigen::internal::traits<Derived>::Orientation;
		using DensePlain = std::conditional_t<std::is_same_v<Orientation, Row>,
		                                      Eigen::Matrix<Scalar, 1, Eigen::Dynamic>, Eigen::VectorX<Scalar>>;
		using Base::derived;

		BlockVector<Scalar, Orientation> eval() const { return BlockVector<Scalar, Orientation>(derived()); }

		DensePlain toDense() const
		{
			DensePlain result(this->rows(), this->cols());
			evalTo(result);
			return result;
		}

		template <typename Destination>
		void evalTo(Eigen::MatrixBase<Destination>& destination) const
		{
			Detail::assertNoOverlap(derived(), destination.derived());
			destination.derived().resize(this->rows(), this->cols());
			for (Eigen::Index index = 0; index < this->blockCount(); ++index)
			{
				const auto offset = this->blockOffset(index);
				const auto dimension = this->dimensionOfBlock(index);
				if constexpr (std::is_same_v<Orientation, Row>)
					destination.derived().row(0).segment(offset, dimension) = derived()[index];
				else
					destination.derived().col(0).segment(offset, dimension) = derived()[index];
			}
		}
	};
}  // namespace Hoppy
