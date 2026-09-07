// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/Detail/TriangularCompressedCoeffProxy.hpp>
#include <SecUtility/Hoppy/Detail/TriangularCompressedCoefficientPolicy.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <type_traits>

namespace Hoppy::Detail
{
	template <typename Derived, typename Scalar, typename StructureTag,
	          TrianglePacking Packing, int Dimension, bool Writable>
	class TriangularCompressedPlainBase
	{
	public:
		using CoeffProxy = TriangularCompressedCoeffProxy<Derived>;
		using CoefficientPolicy = TriangularCompressedCoefficientPolicy<Scalar, StructureTag, Packing>;

		Eigen::Index dimension() const noexcept { return derived().dimensionImpl(); }
		Eigen::Index rows() const noexcept { return dimension(); }
		Eigen::Index cols() const noexcept { return dimension(); }
		Eigen::Index size() const noexcept { return dimension() * dimension(); }
		Eigen::Index storedSize() const noexcept
		{
			const Eigen::Index n = dimension();
			const Eigen::Index left = n % 2 == 0 ? n / 2 : n;
			const Eigen::Index right = n % 2 == 0 ? n + 1 : n / 2 + 1;
			return left * right;
		}
		const Scalar* data() const noexcept { return derived().coeffDataImpl(); }
		template <bool Enabled = Writable, typename = std::enable_if_t<Enabled>>
		Scalar* data() noexcept { return derived().coeffDataImpl(); }

		template <bool Enabled = Writable, typename = std::enable_if_t<Enabled>>
		Derived& setZero()
		{
			std::fill_n(data(), storedSize(), Scalar(0));
			return derived();
		}

		template <bool Enabled = Writable, typename = std::enable_if_t<Enabled>>
		Derived& setConstant(const Scalar& value)
		{
			if (!CoefficientPolicy::isValidDiagonal(value))
			{
				eigen_assert(false && "constant violates the structure's diagonal invariant");
				return derived();
			}
			std::fill_n(data(), storedSize(), value);
			return derived();
		}

		template <bool Enabled = Writable,
		          typename = std::enable_if_t<Enabled
		                                      && !std::is_same_v<StructureTag, AntiSymmetricTag>
		                                      && !std::is_same_v<StructureTag, AntiHermitianTag>>>
		Derived& setOnes() { return setConstant(Scalar(1)); }

		template <bool Enabled = Writable,
		          typename = std::enable_if_t<Enabled
		                                      && !std::is_same_v<StructureTag, AntiSymmetricTag>
		                                      && !std::is_same_v<StructureTag, AntiHermitianTag>>>
		Derived& setIdentity()
		{
			for (Eigen::Index row = 0; row < dimension(); ++row)
				for (Eigen::Index column = 0; column < dimension(); ++column)
					if (CoefficientPolicy::isCanonicalSide(row, column))
						data()[CoefficientPolicy::offset(dimension(), row, column)]
						        = row == column ? Scalar(1) : Scalar(0);
			return derived();
		}

		template <bool Enabled = Writable, typename = std::enable_if_t<Enabled>>
		Derived& setRandom()
		{
			for (Eigen::Index row = 0; row < dimension(); ++row)
				for (Eigen::Index column = 0; column < dimension(); ++column)
					if (CoefficientPolicy::isCanonicalSide(row, column))
					{
						Scalar value = Eigen::internal::random<Scalar>();
						if (row == column) value = CoefficientPolicy::canonicalizeDiagonal(value);
						data()[CoefficientPolicy::offset(dimension(), row, column)] = value;
					}
			return derived();
		}

		Scalar coeff(const Eigen::Index row, const Eigen::Index column) const
		{
			if (!checkIndex(row, column)) return Scalar(0);
			if (CoefficientPolicy::isStructuralZero(row, column)) return Scalar(0);
			const Scalar& stored = data()[CoefficientPolicy::offset(dimension(), row, column)];
			return CoefficientPolicy::storedToLogical(stored, row, column);
		}

		Scalar operator()(const Eigen::Index row, const Eigen::Index column) const { return coeff(row, column); }

		template <bool Enabled = Writable, typename = std::enable_if_t<Enabled>>
		CoeffProxy coeffRef(const Eigen::Index row, const Eigen::Index column)
		{
			(void) checkIndex(row, column);
			return {derived(), row, column};
		}

		template <bool Enabled = Writable, typename = std::enable_if_t<Enabled>>
		CoeffProxy operator()(const Eigen::Index row, const Eigen::Index column) { return coeffRef(row, column); }

		template <int D = Dimension, typename = std::enable_if_t<D == 1>>
		Scalar operator()(const Eigen::Index index) const { return coeff(index, 0); }
		template <int D = Dimension, bool Enabled = Writable,
		          typename = std::enable_if_t<D == 1 && Enabled>>
		CoeffProxy operator()(const Eigen::Index index) { return coeffRef(index, 0); }
		template <int D = Dimension, typename = std::enable_if_t<D == 1>>
		Scalar operator[](const Eigen::Index index) const { return coeff(index, 0); }
		template <int D = Dimension, bool Enabled = Writable,
		          typename = std::enable_if_t<D == 1 && Enabled>>
		CoeffProxy operator[](const Eigen::Index index) { return coeffRef(index, 0); }

	protected:
		template <typename>
		friend class TriangularCompressedCoeffProxy;

		void writeLogical(const Eigen::Index row, const Eigen::Index column, const Scalar& value)
		{
			static_assert(Writable, "cannot write through a const triangular-compressed expression");
			if (!checkIndex(row, column)) return;
			if (CoefficientPolicy::isStructuralZero(row, column))
			{
				eigen_assert(false && "cannot write a structural zero");
				return;
			}
			if (row == column && !CoefficientPolicy::isValidDiagonal(value))
			{
				eigen_assert(false && "coefficient violates the structure's diagonal invariant");
				return;
			}
			derived().coeffDataImpl()[CoefficientPolicy::offset(dimension(), row, column)]
			        = CoefficientPolicy::logicalToStored(value, row, column);
		}

		void writeCanonicalized(const Eigen::Index row, const Eigen::Index column, const Scalar& value)
		{
			Scalar stored = row == column ? CoefficientPolicy::canonicalizeDiagonal(value) : value;
			derived().coeffDataImpl()[CoefficientPolicy::offset(dimension(), row, column)]
			        = CoefficientPolicy::logicalToStored(stored, row, column);
		}

		template <typename Other>
		void assignCoefficientsFrom(const Other& other)
		{
			for (Eigen::Index row = 0; row < dimension(); ++row)
				for (Eigen::Index column = 0; column < dimension(); ++column)
					if (!CoefficientPolicy::isStructuralZero(row, column))
						writeLogical(row, column, static_cast<Scalar>(other.coeff(row, column)));
		}

	private:
		const Derived& derived() const noexcept { return static_cast<const Derived&>(*this); }
		Derived& derived() noexcept { return static_cast<Derived&>(*this); }

		bool checkIndex(Eigen::Index row, Eigen::Index column) const
		{
			const bool valid = CoefficientPolicy::isValidIndex(dimension(), row, column);
			eigen_assert(valid && "triangular-compressed coefficient index is out of bounds");
			return valid;
		}
	};
}  // namespace Hoppy::Detail
