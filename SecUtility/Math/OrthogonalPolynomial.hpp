// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <SecUtility/Math/Core.hpp>
#include <SecUtility/Math/QuadratureGrid.hpp>

namespace SecUtility::Math
{
	template <typename Scalar>
	struct OrthogonalPolynomialRecurrence
	{
		Eigen::VectorX<Scalar> Alphas;  // diagonal of Jacobi matrix
		Eigen::VectorX<Scalar> Gammas;  // off-diagonal of Jacobi matrix, gamma[0] = 0
	};

	template <typename Scalar>
	OrthogonalPolynomialRecurrence<Scalar> ConstructOrthogonalPolynomialRecurrence(
	        const QuadratureGrid<Scalar>& weightedGrid, const int order)
	{
		// this impl is using the Lanczos algorithm

		const auto Dot = [&weightedGrid](const Eigen::VectorX<Scalar>& a, const Eigen::VectorX<Scalar>& b)
		{ return a.cwiseProduct(b).cwiseProduct(weightedGrid.Weights()).sum(); };

		const int n = weightedGrid.Size();

		OrthogonalPolynomialRecurrence<Scalar> rule{};
		rule.Alphas.resize(order);  // garbage, it's fine
		rule.Gammas.resize(order);  // garbage, it's fine
		rule.Gammas[0] = 0;         // this one need to be zeroed out

		Eigen::VectorX<Scalar> previousBasis = Eigen::VectorX<Scalar>::Zero(n);
		Eigen::VectorX<Scalar> candidate = Eigen::VectorX<Scalar>::Zero(n);
		Eigen::VectorX<Scalar> residual = Eigen::VectorX<Scalar>::Zero(n);

		const Scalar zerothMoment = weightedGrid.Weights().sum();
		Eigen::VectorX<Scalar> currentBasis = Eigen::VectorX<Scalar>::Ones(n) / Sqrt(zerothMoment);

		for (Eigen::Index k = 0; k < order; ++k)
		{
			candidate = weightedGrid.Nodes().cwiseProduct(currentBasis) - rule.Gammas[k] * previousBasis;
			rule.Alphas[k] = Dot(currentBasis, candidate);
			residual = candidate - rule.Alphas[k] * currentBasis;
			const auto nextGamma = Sqrt(Dot(residual, residual));

			if (k + 1 == order || nextGamma == 0)
			{
				if (nextGamma == 0 && k + 1 != order)
				{
					rule.Alphas.conservativeResize(k + 1);
					rule.Gammas.conservativeResize(k + 1);
				}

				break;
			}

			rule.Gammas[k + 1] = nextGamma;

			previousBasis = currentBasis;
			currentBasis = residual / nextGamma;
		}

		return rule;
	}
}
