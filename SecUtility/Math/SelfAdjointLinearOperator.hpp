// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#if __cplusplus < 202002L
#error "SelfAdjointLinearOperator requires C++20 or later"
#endif

#include <Eigen/Core>

#include <complex>
#include <concepts>
#include <type_traits>


namespace SecUtility::Math
{
	namespace Detail
	{
		template <typename T>
		inline constexpr bool IsSupportedEigenSolverScalar = std::floating_point<T>;

		template <std::floating_point T>
		inline constexpr bool IsSupportedEigenSolverScalar<std::complex<T>> = true;
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
		           { linearOperator.ApplyOn(vector)[Eigen::Index{}] }
		           -> std::convertible_to<LinearOperatorScalar<Operator>>;
	           };


	template <typename Operator>
	concept BlockSelfAdjointLinearOperator =
	        SelfAdjointLinearOperatorBase<Operator>
	        && requires(const std::remove_reference_t<Operator>& linearOperator,
	                    const Eigen::MatrixX<LinearOperatorScalar<Operator>>& vectors) {
		           { linearOperator.ApplyOn(vectors).rows() } -> std::convertible_to<Eigen::Index>;
		           { linearOperator.ApplyOn(vectors).cols() } -> std::convertible_to<Eigen::Index>;
		           { linearOperator.ApplyOn(vectors)(Eigen::Index{}, Eigen::Index{}) }
		           -> std::convertible_to<LinearOperatorScalar<Operator>>;
	           };


	template <typename Operator>
	concept SelfAdjointLinearOperator =
	        ScalarSelfAdjointLinearOperator<Operator> || BlockSelfAdjointLinearOperator<Operator>;


	/// Applies an operator to a block. Block-capable operators are invoked once; scalar-only operators once per column.
	/// An empty block does not invoke the operator.
	template <SelfAdjointLinearOperator Operator>
	[[nodiscard]] Eigen::MatrixX<LinearOperatorScalar<Operator>> ApplySelfAdjointLinearOperator(
	        const Operator& linearOperator, const Eigen::MatrixX<LinearOperatorScalar<Operator>>& vectors)
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
		}
		else
		{
			for (Eigen::Index columnIndex = 0; columnIndex < vectors.cols(); columnIndex++)
			{
				images.col(columnIndex) = linearOperator.ApplyOn(vectors.col(columnIndex));
			}
		}
		return images;
	}
}
