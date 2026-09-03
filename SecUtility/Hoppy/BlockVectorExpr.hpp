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
		using DensePlain = std::conditional_t<std::is_same_v<Orientation, Row>,
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
		template <typename NewScalar> auto cast() const&& = delete;
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
		template <typename OtherDerived> auto operator+(const BlockVectorExpr<OtherDerived>& other) const&& = delete;
		template <typename OtherDerived> auto operator-(const BlockVectorExpr<OtherDerived>& other) const&& = delete;
		template <typename TOtherScalar> auto operator*(const TOtherScalar& scalar) const&& = delete;
		template <typename TOtherScalar> auto operator/(const TOtherScalar& scalar) const&& = delete;
		template <typename T = Orientation, typename = std::enable_if_t<std::is_same_v<T, Column>>>
		auto asDiagonal() const&;
		auto operator+() const&& = delete;
		auto operator-() const&& = delete;
		auto transpose() const&& = delete;
		auto conjugate() const&& = delete;
		auto adjoint() const&& = delete;
		auto real() const&& = delete;
		auto imag() const&& = delete;

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
				if constexpr (std::is_same_v<Orientation, Row>)
					destination.derived().row(0).segment(offset, dimension) = derived()[index];
				else
					destination.derived().col(0).segment(offset, dimension) = derived()[index];
			}
		}
	};
}  // namespace Hoppy
