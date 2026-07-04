// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <Eigen/Dense>
#include <SecUtility/Diagnostic/Exception.hpp>
#include <SecUtility/Math/Core.hpp>
#include <SecUtility/Math/QuadratureGrid.hpp>

namespace SecUtility::Math
{
	template <typename Scalar, int SizeAtCompileTime>
	struct OrthogonalPolynomialRecurrence
	{
		Eigen::Vector<Scalar, SizeAtCompileTime> Alphas;  // diagonal of Jacobi matrix
		Eigen::Vector<Scalar, SizeAtCompileTime> Gammas;  // off-diagonal of Jacobi matrix, gamma[0] = 0
	};

	namespace Detail
	{
		template <typename Scalar, int OrderAtCompileTime, int GridSizeAtCompileTime>
		OrthogonalPolynomialRecurrence<Scalar, OrderAtCompileTime> ConstructOrthogonalPolynomialRecurrenceImpl(
		        const QuadratureGrid<Scalar, GridSizeAtCompileTime>& weightedGrid, const Eigen::Index runtimeOrder)
		{
			static_assert(GridSizeAtCompileTime == Eigen::Dynamic || OrderAtCompileTime == Eigen::Dynamic
			                      || OrderAtCompileTime <= GridSizeAtCompileTime,
			              "Order must not exceed GridSize");

			eigen_assert(runtimeOrder >= 1);

		// this impl is using the Lanczos algorithm

			const auto Dot = [&weightedGrid](const auto& a, const auto& b)
			{ return a.cwiseProduct(b).cwiseProduct(weightedGrid.Weights()).sum(); };

			const Eigen::Index n = weightedGrid.Size();

			OrthogonalPolynomialRecurrence<Scalar, OrderAtCompileTime> rule{};
			if constexpr (OrderAtCompileTime == Eigen::Dynamic)
			{
				rule.Alphas.resize(runtimeOrder);  // garbage, it's fine
				rule.Gammas.resize(runtimeOrder);  // garbage, it's fine
			}
			rule.Gammas[0] = 0;  // this one need to be zeroed out

			Eigen::Vector<Scalar, GridSizeAtCompileTime> previousBasis =
			        Eigen::Vector<Scalar, GridSizeAtCompileTime>::Zero(n);
			Eigen::Vector<Scalar, GridSizeAtCompileTime> candidate =
			        Eigen::Vector<Scalar, GridSizeAtCompileTime>::Zero(n);
			Eigen::Vector<Scalar, GridSizeAtCompileTime> residual =
			        Eigen::Vector<Scalar, GridSizeAtCompileTime>::Zero(n);

			const Scalar zerothMoment = weightedGrid.Weights().sum();
			Eigen::Vector<Scalar, GridSizeAtCompileTime> currentBasis =
			        Eigen::Vector<Scalar, GridSizeAtCompileTime>::Ones(n) / Sqrt(zerothMoment);

			for (Eigen::Index k = 0; k < runtimeOrder; ++k)
			{
				candidate = weightedGrid.Nodes().cwiseProduct(currentBasis) - rule.Gammas[k] * previousBasis;
				rule.Alphas[k] = Dot(currentBasis, candidate);
				residual = candidate - rule.Alphas[k] * currentBasis;
				const auto nextGamma = Sqrt(Dot(residual, residual));

				if (k + 1 == runtimeOrder || nextGamma == 0)
				{
					if (nextGamma == 0 && k + 1 != runtimeOrder)
					{
						if constexpr (OrderAtCompileTime == Eigen::Dynamic)
						{
							rule.Alphas.conservativeResize(k + 1);
							rule.Gammas.conservativeResize(k + 1);
						}
						else
						{
							throw OperationFailedException{
							        "Lanczos terminated early: next gamma is 0 before reaching the requested Order"};
						}
					}

					break;
				}

				rule.Gammas[k + 1] = nextGamma;

				previousBasis = currentBasis;
				currentBasis = residual / nextGamma;
			}

			return rule;
		}
	}  // namespace Detail

	/// Fixed-Order overload. Order is the first explicit template parameter so callers can write
	/// ConstructOrthogonalPolynomialRecurrence<Scalar, 50>(grid); GridSize is deduced from the argument.
	/// Throws OperationFailedException if Lanczos terminates early (next gamma == 0) before reaching Order.
	template <int OrderAtCompileTime, typename Scalar, int GridSizeAtCompileTime>
	OrthogonalPolynomialRecurrence<Scalar, OrderAtCompileTime> ConstructOrthogonalPolynomialRecurrence(
	        const QuadratureGrid<Scalar, GridSizeAtCompileTime>& weightedGrid)
	{
		return Detail::ConstructOrthogonalPolynomialRecurrenceImpl<Scalar, OrderAtCompileTime, GridSizeAtCompileTime>(
		        weightedGrid, OrderAtCompileTime == Eigen::Dynamic ? weightedGrid.Size() : OrderAtCompileTime);
	}

	/// Dynamic overload — kept for backward compatibility. Output is dynamic-sized; conservativeResize
	/// is used on early termination.
	template <typename Scalar, int GridSizeAtCompileTime>
	OrthogonalPolynomialRecurrence<Scalar> ConstructOrthogonalPolynomialRecurrence(
	        const QuadratureGrid<Scalar, GridSizeAtCompileTime>& weightedGrid, const Eigen::Index order)
	{
		return Detail::ConstructOrthogonalPolynomialRecurrenceImpl<Scalar, Eigen::Dynamic, GridSizeAtCompileTime>(
		        weightedGrid, order);
	}
}
