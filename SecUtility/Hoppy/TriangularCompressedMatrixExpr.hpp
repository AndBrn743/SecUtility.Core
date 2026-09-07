// SPDX-License-Identifier: MIT

#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <type_traits>

namespace Hoppy
{
	template <typename Derived>
	class TriangularCompressedMatrixExpr
	{
	public:
		const Derived& derived() const noexcept { return static_cast<const Derived&>(*this); }
		Derived& derived() noexcept { return static_cast<Derived&>(*this); }

		auto transpose() const&;
		auto transpose() &&;
		auto conjugate() const&;
		auto conjugate() &&;
		auto adjoint() const&;
		auto adjoint() &&;

		auto diagonal() &;
		auto diagonal() const&;
		auto diagonal() && = delete;
		template <unsigned int Mode> auto triangularView() &;
		template <unsigned int Mode> auto triangularView() const&;
		template <unsigned int Mode> auto triangularView() && = delete;
		auto block(Eigen::Index row, Eigen::Index column, Eigen::Index rows, Eigen::Index columns) &;
		auto block(Eigen::Index row, Eigen::Index column, Eigen::Index rows, Eigen::Index columns) const&;
		template <int Rows, int Columns> auto block(Eigen::Index row, Eigen::Index column) &;
		template <int Rows, int Columns> auto block(Eigen::Index row, Eigen::Index column) const&;
		auto block(Eigen::Index, Eigen::Index, Eigen::Index, Eigen::Index) && = delete;
		auto topLeftCorner(Eigen::Index rows, Eigen::Index columns) &;
		auto topLeftCorner(Eigen::Index rows, Eigen::Index columns) const&;
		template <int Rows, int Columns> auto topLeftCorner() &;
		template <int Rows, int Columns> auto topLeftCorner() const&;
		auto topLeftCorner(Eigen::Index, Eigen::Index) && = delete;
		auto row(Eigen::Index index) &;
		auto row(Eigen::Index index) const&;
		auto row(Eigen::Index) && = delete;
		auto col(Eigen::Index index) &;
		auto col(Eigen::Index index) const&;
		auto col(Eigen::Index) && = delete;

		auto sum() const
		{
			using Scalar = typename Derived::Scalar;
			Scalar result(0);
			for (Eigen::Index row = 0; row < derived().rows(); ++row)
				for (Eigen::Index column = 0; column < derived().cols(); ++column)
					result += derived().coeff(row, column);
			return result;
		}
		auto trace() const
		{
			using Scalar = typename Derived::Scalar;
			Scalar result(0);
			for (Eigen::Index index = 0; index < derived().dimension(); ++index)
				result += derived().coeff(index, index);
			return result;
		}
		auto squaredNorm() const
		{
			using Scalar = typename Derived::Scalar;
			using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
			RealScalar result(0);
			for (Eigen::Index row = 0; row < derived().rows(); ++row)
				for (Eigen::Index column = 0; column < derived().cols(); ++column)
					result += Eigen::numext::abs2(derived().coeff(row, column));
			return result;
		}
		auto norm() const { return Eigen::numext::sqrt(squaredNorm()); }
		auto mean() const
		{
			eigen_assert(derived().size() != 0);
			return sum() / typename Derived::Scalar(derived().size());
		}
		template <typename D = Derived,
		          typename = std::enable_if_t<Eigen::NumTraits<typename D::Scalar>::IsComplex == 0>>
		auto maxCoeff() const
		{
			eigen_assert(derived().size() != 0);
			auto result = derived().coeff(0, 0);
			for (Eigen::Index row = 0; row < derived().rows(); ++row)
				for (Eigen::Index column = 0; column < derived().cols(); ++column)
					result = (std::max)(result, derived().coeff(row, column));
			return result;
		}
		template <typename D = Derived,
		          typename = std::enable_if_t<Eigen::NumTraits<typename D::Scalar>::IsComplex == 0>>
		auto minCoeff() const
		{
			eigen_assert(derived().size() != 0);
			auto result = derived().coeff(0, 0);
			for (Eigen::Index row = 0; row < derived().rows(); ++row)
				for (Eigen::Index column = 0; column < derived().cols(); ++column)
					result = (std::min)(result, derived().coeff(row, column));
			return result;
		}
		auto maxAbsCoeff() const
		{
			using RealScalar = typename Eigen::NumTraits<typename Derived::Scalar>::Real;
			eigen_assert(derived().size() != 0);
			RealScalar result(0);
			for (Eigen::Index row = 0; row < derived().rows(); ++row)
				for (Eigen::Index column = 0; column < derived().cols(); ++column)
					result = (std::max)(result, Eigen::numext::abs(derived().coeff(row, column)));
			return result;
		}
		bool allFinite() const
		{
			for (Eigen::Index row = 0; row < derived().rows(); ++row)
				for (Eigen::Index column = 0; column < derived().cols(); ++column)
					if (!Eigen::numext::isfinite(derived().coeff(row, column))) return false;
			return true;
		}
		bool hasNaN() const
		{
			for (Eigen::Index row = 0; row < derived().rows(); ++row)
				for (Eigen::Index column = 0; column < derived().cols(); ++column)
					if (Eigen::numext::isnan(derived().coeff(row, column))) return true;
			return false;
		}
		template <typename Other, typename Precision>
		bool isApprox(const Other& other, const Precision& precision) const
		{
			if (derived().rows() != other.rows() || derived().cols() != other.cols())
			{
				eigen_assert(false && "isApprox requires matching dimensions");
				return false;
			}
			for (Eigen::Index row = 0; row < derived().rows(); ++row)
				for (Eigen::Index column = 0; column < derived().cols(); ++column)
					if (!Eigen::internal::isApprox(derived().coeff(row, column),
					                               other.coeff(row, column), precision)) return false;
			return true;
		}
		template <typename Other>
		bool isApprox(const Other& other) const
		{
			using RealScalar = typename Eigen::NumTraits<typename Derived::Scalar>::Real;
			return isApprox(other, Eigen::NumTraits<RealScalar>::dummy_precision());
		}
	};
}
