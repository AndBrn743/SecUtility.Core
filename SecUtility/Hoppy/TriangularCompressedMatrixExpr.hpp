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
			forEachIndependent([&](const Eigen::Index row, const Eigen::Index column, const Scalar& value) {
				result += value;
				if (row != column && isTwoSidedStructure()) result += reflectedValue(value);
			});
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
			forEachIndependent([&](const Eigen::Index row, const Eigen::Index column, const Scalar& value) {
				const RealScalar magnitude = Eigen::numext::abs2(value);
				result += row != column && isTwoSidedStructure() ? RealScalar(2) * magnitude : magnitude;
			});
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
			typename Derived::Scalar result = hasStructuralZeros() ? typename Derived::Scalar(0)
			                                                    : derived().coeff(0, 0);
			forEachIndependent([&](const Eigen::Index row, const Eigen::Index column, const auto& value) {
				result = (std::max)(result, value);
				if (row != column && isTwoSidedStructure()) result = (std::max)(result, reflectedValue(value));
			});
			return result;
		}
		template <typename D = Derived,
		          typename = std::enable_if_t<Eigen::NumTraits<typename D::Scalar>::IsComplex == 0>>
		auto minCoeff() const
		{
			eigen_assert(derived().size() != 0);
			typename Derived::Scalar result = hasStructuralZeros() ? typename Derived::Scalar(0)
			                                                    : derived().coeff(0, 0);
			forEachIndependent([&](const Eigen::Index row, const Eigen::Index column, const auto& value) {
				result = std::min(result, value);
				if (row != column && isTwoSidedStructure()) result = (std::min)(result, reflectedValue(value));
			});
			return result;
		}
		auto maxAbsCoeff() const
		{
			using RealScalar = typename Eigen::NumTraits<typename Derived::Scalar>::Real;
			eigen_assert(derived().size() != 0);
			RealScalar result(0);
			forEachIndependent([&](Eigen::Index, Eigen::Index, const auto& value) {
				result = std::max(result, Eigen::numext::abs(value));
			});
			return result;
		}
		bool allFinite() const
		{
			bool result = true;
			forEachIndependent([&](Eigen::Index, Eigen::Index, const auto& value) {
				if (!Eigen::numext::isfinite(value)) result = false;
			});
			return result;
		}
		bool hasNaN() const
		{
			bool result = false;
			forEachIndependent([&](Eigen::Index, Eigen::Index, const auto& value) {
				if (Eigen::numext::isnan(value)) result = true;
			});
			return result;
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

	private:
		static constexpr bool isTwoSidedStructure()
		{
			using Tag = typename Derived::StructureTag;
			return !std::is_same_v<Tag, Detail::UpperTriangularTag>
			       && !std::is_same_v<Tag, Detail::LowerTriangularTag>;
		}
		bool hasStructuralZeros() const
		{
			using Tag = typename Derived::StructureTag;
			return derived().dimension() > 1
			       && (std::is_same_v<Tag, Detail::UpperTriangularTag>
			           || std::is_same_v<Tag, Detail::LowerTriangularTag>);
		}
		template <typename Value>
		static auto reflectedValue(const Value& value)
		{
			using Tag = typename Derived::StructureTag;
			if constexpr (std::is_same_v<Tag, Detail::AntiSymmetricTag>) return -value;
			else if constexpr (std::is_same_v<Tag, Detail::HermitianTag>) return Eigen::numext::conj(value);
			else if constexpr (std::is_same_v<Tag, Detail::AntiHermitianTag>) return -Eigen::numext::conj(value);
			else return value;
		}
		template <typename Function>
		void forEachIndependent(Function&& function) const
		{
			using Tag = typename Derived::StructureTag;
			for (Eigen::Index major = 0; major < derived().dimension(); ++major)
				for (Eigen::Index minor = 0; minor <= major; ++minor)
				{
					Eigen::Index row = Derived::PackingValue == TrianglePacking::Lower ? major : minor;
					Eigen::Index column = Derived::PackingValue == TrianglePacking::Lower ? minor : major;
					if constexpr (std::is_same_v<Tag, Detail::UpperTriangularTag>)
					{
						if (row > column) std::swap(row, column);
					}
					else if constexpr (std::is_same_v<Tag, Detail::LowerTriangularTag>)
					{
						if (row < column) std::swap(row, column);
					}
					function(row, column, derived().coeff(row, column));
				}
		}
	};
}
