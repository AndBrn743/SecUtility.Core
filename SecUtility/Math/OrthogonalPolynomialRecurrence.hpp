// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <Eigen/Core>
#include <SecUtility/Math/Core.hpp>

namespace SecUtility::Math
{
	template <typename Scalar, int SizeAtCompileTime = Eigen::Dynamic>
	struct OrthogonalPolynomialRecurrence
	{
		Eigen::Vector<Scalar, SizeAtCompileTime> Alphas;  // diagonal of Jacobi matrix
		Eigen::Vector<Scalar, SizeAtCompileTime> Gammas;  // off-diagonal of Jacobi matrix, gamma[0] = 0
	};

	/// Evaluates lambda_n(x) = zerothMoment / sum_{j=0}^{n-1} p_j(x)^2,
	/// where p_0=1 and p_j follow the supplied three-term recurrence.
	/// At Gaussian quadrature nodes this is the quadrature weight; x need not be a node.
	/// The recurrence must be nonempty and unreduced: Gammas[1..n-1] are nonzero,
	/// Gammas[0] is zero, and Alphas/Gammas have matching sizes.
	template <typename Scalar, int SizeAtCompileTime>
	Scalar EvaluateChristoffelFunction(const OrthogonalPolynomialRecurrence<Scalar, SizeAtCompileTime>& recurrence,
	                                   const Scalar x,
	                                   const Scalar zerothMoment = Scalar{1})
	{
		eigen_assert(recurrence.Alphas.size() == recurrence.Gammas.size());
		eigen_assert(recurrence.Alphas.size() >= 1);
		Scalar previous = 0;
		Scalar current = 1;
		Scalar result = zerothMoment;
		for (Eigen::Index j = 0; j + 1 < recurrence.Alphas.size(); ++j)
		{
			// previous/current are p_{j-1}/sqrt(S), p_j/sqrt(S), where
			// S=sum_{k=0}^j p_k^2, and result=mu_0/S. Normalize each step
			// without forming large p_k^2 or dividing by a tiny gamma.
			const Scalar numerator = (x - recurrence.Alphas[j]) * current - recurrence.Gammas[j] * previous;
			const Scalar gamma = recurrence.Gammas[j + 1];
			eigen_assert(gamma != Scalar{0});
			const Scalar norm = Hypotenuse(gamma, numerator);
			const Scalar ratio = Abs(gamma) / norm;
			result *= ratio;
			result *= ratio;
			previous = current * ratio;
			current = gamma < Scalar{0} ? -numerator / norm : numerator / norm;
		}
		return result;
	}
}  // namespace SecUtility::Math
