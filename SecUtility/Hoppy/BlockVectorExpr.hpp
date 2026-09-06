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
		using RealScalar = typename Base::RealScalar;
		using Orientation = typename Eigen::internal::traits<Derived>::Orientation;
		using DensePlain = std::conditional_t<std::is_same_v<Orientation, BlockVectorOrientation::Row>,
		                                      Eigen::Matrix<Scalar, 1, Eigen::Dynamic>, Eigen::VectorX<Scalar>>;
		using Base::derived;

		auto operator+() const&;
		auto operator-() const&;
		auto transpose() const&;
		auto conjugate() const&;
		auto adjoint() const&;
		auto real() const&;
		auto imag() const&;
		template <typename NewScalar> auto cast() const&;
		template <typename OtherDerived,
		          typename = std::enable_if_t<std::is_same_v<Orientation,
		                                                     typename Eigen::internal::traits<OtherDerived>::Orientation>>,
		          typename = typename Eigen::ScalarBinaryOpTraits<
		                  Scalar, typename Eigen::internal::traits<OtherDerived>::Scalar,
		                  Eigen::internal::scalar_sum_op<Scalar,
		                                                 typename Eigen::internal::traits<OtherDerived>::Scalar>>::ReturnType>
		auto operator+(const BlockVectorExpr<OtherDerived>& other) const&;
		template <typename OtherDerived,
		          typename = std::enable_if_t<std::is_same_v<Orientation,
		                                                     typename Eigen::internal::traits<OtherDerived>::Orientation>>,
		          typename = typename Eigen::ScalarBinaryOpTraits<
		                  Scalar, typename Eigen::internal::traits<OtherDerived>::Scalar,
		                  Eigen::internal::scalar_difference_op<
		                          Scalar, typename Eigen::internal::traits<OtherDerived>::Scalar>>::ReturnType>
		auto operator-(const BlockVectorExpr<OtherDerived>& other) const&;
		template <typename TOtherScalar,
		          typename = typename Eigen::ScalarBinaryOpTraits<
		                  Scalar, TOtherScalar,
		                  Eigen::internal::scalar_product_op<Scalar, TOtherScalar>>::ReturnType>
		auto operator*(const TOtherScalar& scalar) const&;
		template <typename TOtherScalar,
		          typename = typename Eigen::ScalarBinaryOpTraits<
		                  Scalar, TOtherScalar,
		                  Eigen::internal::scalar_quotient_op<Scalar, TOtherScalar>>::ReturnType>
		auto operator/(const TOtherScalar& scalar) const&;
		template <typename T = Orientation, typename = std::enable_if_t<std::is_same_v<T, BlockVectorOrientation::Column>>>
		auto asDiagonal() const&;
		template <typename OtherDerived,
		          typename OtherScalar = typename Eigen::internal::traits<OtherDerived>::Scalar,
		          typename Operation = Eigen::internal::scalar_conj_product_op<Scalar, OtherScalar>,
		          typename ResultScalar = typename Eigen::ScalarBinaryOpTraits<
		                  Scalar, OtherScalar, Operation>::ReturnType>
		ResultScalar dot(const BlockVectorExpr<OtherDerived>& other) const
		{
			eigen_assert(this->totalDimension() == other.totalDimension());
			ResultScalar result{};
			Operation operation;
			Eigen::Index lhsBlock = 0;
			Eigen::Index rhsBlock = 0;
			Eigen::Index lhsIndex = 0;
			Eigen::Index rhsIndex = 0;
			while (lhsBlock < this->blockCount())
			{
				const auto lhsRemaining = this->dimensionOfBlock(lhsBlock) - lhsIndex;
				const auto rhsRemaining = other.dimensionOfBlock(rhsBlock) - rhsIndex;
				const auto count = (std::min)(lhsRemaining, rhsRemaining);
				for (Eigen::Index index = 0; index < count; ++index)
					result += operation(derived()[lhsBlock].coeff(lhsIndex + index),
					                    other.derived()[rhsBlock].coeff(rhsIndex + index));
				lhsIndex += count;
				rhsIndex += count;
				if (lhsIndex == this->dimensionOfBlock(lhsBlock))
				{
					++lhsBlock;
					lhsIndex = 0;
				}
				if (rhsIndex == other.dimensionOfBlock(rhsBlock))
				{
					++rhsBlock;
					rhsIndex = 0;
				}
			}
			return result;
		}

		template <typename S = Scalar,
		          typename = std::enable_if_t<
		                  std::is_floating_point_v<typename Eigen::NumTraits<S>::Real>>>
		BlockVector<Scalar, Orientation> normalized() const
		{
			BlockVector<Scalar, Orientation> result(derived());
			const RealScalar squaredNorm = result.squaredNorm();
			if (squaredNorm > RealScalar{})
			{
				using std::sqrt;
				result /= sqrt(squaredNorm);
			}
			return result;
		}

		RealScalar absSum() const { RealScalar result{}; for (Eigen::Index i = 0; i < this->blockCount(); ++i) result += derived()[i].cwiseAbs().sum(); return result; }

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
				if constexpr (std::is_same_v<Orientation, BlockVectorOrientation::Row>)
					destination.derived().row(0).segment(offset, dimension) = derived()[index];
				else
					destination.derived().col(0).segment(offset, dimension) = derived()[index];
			}
		}
	};
}  // namespace Hoppy
