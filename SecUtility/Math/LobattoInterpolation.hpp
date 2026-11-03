// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <Eigen/Core>
#include <SecUtility/Math/Constant.hpp>
#include <SecUtility/Math/Core.hpp>

namespace SecUtility::Math
{
	/// Returns degree+1 ascending Chebyshev-Lobatto nodes on [-1,1].
	/// degree must be positive. Endpoints and the even-degree midpoint are exact.
	template <typename Scalar>
	Eigen::VectorX<Scalar> ChebyshevLobattoNodes(const Eigen::Index degree)
	{
		eigen_assert(degree > 0);

		Eigen::VectorX<Scalar> nodes(degree + 1);
		for (Eigen::Index i = 0; i <= degree; ++i)  // upper is inclusive
		{
			if (i == 0)
			{
				nodes[i] = Scalar{-1};
			}
			else if (i == degree)
			{
				nodes[i] = Scalar{1};
			}
			else if (2 * i == degree)
			{
				nodes[i] = Scalar{0};
			}
			else
			{
				nodes[i] = -Cos(i * Constant::Pi<Scalar> / degree);
			}
		}
		return nodes;
	}

	/// Shared immutable nodes, initialized once per scalar/degree specialization.
	template <typename Scalar, int DegreeAtCompileTime>
	const Eigen::Vector<Scalar, DegreeAtCompileTime + 1>& ChebyshevLobattoNodes()
	{
		static_assert(DegreeAtCompileTime > 0);
		static const Eigen::Vector<Scalar, DegreeAtCompileTime + 1> nodes =
		        ChebyshevLobattoNodes<Scalar>(DegreeAtCompileTime);
		return nodes;
	}

	/// Barycentric interpolation weights for ChebyshevLobattoNodes<Scalar>(degree).
	/// degree must be positive. These are proportional weights, with alternating
	/// signs and half-sized endpoints, rather than numerical integration weights.
	template <typename Scalar>
	Eigen::VectorX<Scalar> ChebyshevLobattoBarycentricWeights(const Eigen::Index degree)
	{
		eigen_assert(degree > 0);
		Eigen::VectorX<Scalar> weights(degree + 1);
		for (Eigen::Index i = 0; i <= degree; ++i)
		{
			weights[i] = MinusOneToThePowerOf<Scalar>(i) * (i == 0 || i == degree ? Scalar{0.5} : Scalar{1});
		}
		return weights;
	}

	/// Shared immutable barycentric weights for the matching fixed-degree nodes.
	template <typename Scalar, int DegreeAtCompileTime>
	const Eigen::Vector<Scalar, DegreeAtCompileTime + 1>& ChebyshevLobattoBarycentricWeights()
	{
		static_assert(DegreeAtCompileTime > 0);
		static const Eigen::Vector<Scalar, DegreeAtCompileTime + 1> weights =
		        ChebyshevLobattoBarycentricWeights<Scalar>(DegreeAtCompileTime);
		return weights;
	}

	/// Interpolates the columns of a (DegreeAtCompileTime+1)-row sample matrix.
	/// x must be finite; the interpolation interval is [-1,1]. Outside that interval
	/// the same polynomial is extrapolated, without an accuracy guarantee.
	/// Row i contains the values at ChebyshevLobattoNodes<Scalar, DegreeAtCompileTime>()[i].
	/// Storage order, strides, and matrix expressions are handled by Eigen.
	/// Nodes and corresponding weights are shared internally. The caller owns
	/// interval mapping. At least one sample column is required. The result is an
	/// owning column vector whose compile-time size follows the sample column count.
	/// Exact node matches return the corresponding sample row directly. Scalar is
	/// taken from SampleMatrix; the coordinate is converted to that type.
	/// Arithmetic is unscaled: finite inputs and a finite mathematical result do not
	/// guarantee finite intermediate reciprocals, products, or sums, especially near nodes.
	template <int DegreeAtCompileTime, typename SampleMatrix>
	Eigen::Vector<typename SampleMatrix::Scalar, SampleMatrix::ColsAtCompileTime> InterpolateChebyshevLobatto(
	        const typename SampleMatrix::Scalar x, const Eigen::MatrixBase<SampleMatrix>& samples)
	{
		static_assert(DegreeAtCompileTime > 0);
		static_assert(SampleMatrix::RowsAtCompileTime == Eigen::Dynamic
		              || SampleMatrix::RowsAtCompileTime == DegreeAtCompileTime + 1,
		              "Lobatto interpolation needs DegreeAtCompileTime+1 sample rows");
		static_assert(SampleMatrix::ColsAtCompileTime == Eigen::Dynamic || SampleMatrix::ColsAtCompileTime > 0,
		              "Lobatto interpolation needs at least one sample column");
		eigen_assert(samples.rows() == DegreeAtCompileTime + 1);
		eigen_assert(samples.cols() > 0);
		using Scalar = typename SampleMatrix::Scalar;
		const auto& nodes = ChebyshevLobattoNodes<Scalar, DegreeAtCompileTime>();
		const auto& weights = ChebyshevLobattoBarycentricWeights<Scalar, DegreeAtCompileTime>();

		using ValueVector = Eigen::Vector<Scalar, SampleMatrix::ColsAtCompileTime>;
		ValueVector numerators = ValueVector::Zero(samples.cols());

		Scalar denominator = 0;
		for (int i = 0; i <= DegreeAtCompileTime; ++i)
		{
			const auto values = samples.row(i).transpose();
			if (x == nodes[i])
			{
				return values;
			}
			// Possible hardening: first scan for exact matches and the nearest-node distance d,
			// then use weights[i] * (d / (x - nodes[i])) in both accumulations below. Compute
			// the distance ratio before multiplying; scaling an overflowing reciprocal is too late.
			// The common factor cancels in the final quotient, and the ratio has magnitude <= 1.
			// This costs an extra node scan and bounds the terms, reducing reciprocal/product
			// overflow near nodes; sums of extreme sample values can still overflow.
			const Scalar weight = weights[i] / (x - nodes[i]);
			denominator += weight;
			numerators += weight * values;
		}
		return numerators / denominator;
	}
}  // namespace SecUtility::Math
