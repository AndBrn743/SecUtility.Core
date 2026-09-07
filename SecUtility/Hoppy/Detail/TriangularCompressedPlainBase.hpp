// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/Detail/TriangularCompressedCoeffProxy.hpp>
#include <SecUtility/Hoppy/Detail/TriangularCompressedCoefficientPolicy.hpp>
#include <SecUtility/Hoppy/TriangularCompressedMatrixExpr.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <numeric>
#include <type_traits>

namespace Hoppy::Detail
{
	template <typename StructureTag, unsigned int Mode>
	inline constexpr bool accepts_triangular_view_v =
	        (std::is_same_v<StructureTag, UpperTriangularTag> && (Mode & Eigen::Upper) != 0)
	        || (std::is_same_v<StructureTag, LowerTriangularTag> && (Mode & Eigen::Lower) != 0)
	        || ((std::is_same_v<StructureTag, SymmetricTag>
	             || std::is_same_v<StructureTag, HermitianTag>)
	            && ((Mode & Eigen::Upper) != 0 || (Mode & Eigen::Lower) != 0))
	        || ((std::is_same_v<StructureTag, AntiSymmetricTag>
	             || std::is_same_v<StructureTag, AntiHermitianTag>)
	            && (Mode & Eigen::UnitDiag) == 0
	            && ((Mode & Eigen::Upper) != 0 || (Mode & Eigen::Lower) != 0));

	template <typename Scalar, typename StructureTag>
	inline constexpr bool accepts_self_adjoint_view_v =
	        std::is_same_v<StructureTag, HermitianTag>
	        || (std::is_same_v<StructureTag, SymmetricTag> && Eigen::NumTraits<Scalar>::IsComplex == 0);

	template <typename Derived, typename Scalar, typename StructureTag,
	          TrianglePacking Packing, int Dimension, bool Writable>
	class TriangularCompressedPlainBase
	    : public Hoppy::TriangularCompressedMatrixExpr<Derived>
	{
		using ExpressionBase = Hoppy::TriangularCompressedMatrixExpr<Derived>;

	public:
		using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
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
		bool hasNaN() const
		{
			if (storedSize() == 0) return false;
			return std::any_of(data(), data() + storedSize(),
			                   [](const auto& element) { return Eigen::numext::isnan(element); });
		}
		bool allFinite() const
		{
			if (storedSize() == 0) return true;
			return std::all_of(data(), data() + storedSize(),
			                   [](const auto& element) { return Eigen::numext::isfinite(element); });
		}
		Scalar sum() const
		{
			Scalar result(0);
			forEachIndependentImpl([&](const Eigen::Index row, const Eigen::Index column, const Scalar& value) {
				result += value;
				if constexpr (isTwoSidedStructure())
					if (row != column)
						result += CoefficientPolicy::storedToLogical(value, column, row);
			});
			return result;
		}
		RealScalar squaredNorm() const
		{
			RealScalar result(0);
			forEachIndependentImpl([&](const Eigen::Index row, const Eigen::Index column, const Scalar& value) {
				const RealScalar magnitude = Eigen::numext::abs2(value);
				result += isTwoSidedStructure() && row != column ? RealScalar(2) * magnitude : magnitude;
			});
			return result;
		}
		RealScalar norm() const { return Eigen::numext::sqrt(squaredNorm()); }
		Scalar mean() const
		{
			if (size() == 0)
			{
				eigen_assert(false && "mean requires a nonempty expression");
				return Scalar(0);
			}
			return sum() / Scalar(size());
		}
		RealScalar maxAbsCoeff() const
		{
			if (size() == 0)
			{
				eigen_assert(false && "maxAbsCoeff requires a nonempty expression");
				return RealScalar(0);
			}
			return std::accumulate(data(), data() + storedSize(), RealScalar(0),
			                       [](const RealScalar current, const Scalar& value) {
				                       return std::max(current, Eigen::numext::abs(value));
			                       });
		}
		template <typename S = Scalar,
		          typename = std::enable_if_t<Eigen::NumTraits<S>::IsComplex == 0>>
		Scalar maxCoeff() const
		{
			if (size() == 0)
			{
				eigen_assert(false && "maxCoeff requires a nonempty expression");
				return Scalar(0);
			}
			Scalar result = hasStructuralZeros() ? Scalar(0) : data()[0];
			forEachIndependentImpl([&](const Eigen::Index row, const Eigen::Index column, const Scalar& value) {
				result = std::max(result, value);
				if constexpr (isTwoSidedStructure())
					if (row != column)
						result = std::max(result,
						                  CoefficientPolicy::storedToLogical(value, column, row));
			});
			return result;
		}
		template <typename S = Scalar,
		          typename = std::enable_if_t<Eigen::NumTraits<S>::IsComplex == 0>>
		Scalar minCoeff() const
		{
			if (size() == 0)
			{
				eigen_assert(false && "minCoeff requires a nonempty expression");
				return Scalar(0);
			}
			Scalar result = hasStructuralZeros() ? Scalar(0) : data()[0];
			forEachIndependentImpl([&](const Eigen::Index row, const Eigen::Index column, const Scalar& value) {
				result = std::min(result, value);
				if constexpr (isTwoSidedStructure())
					if (row != column)
						result = std::min(result,
						                  CoefficientPolicy::storedToLogical(value, column, row));
			});
			return result;
		}

		using DensePlainObject = Eigen::Matrix<Scalar, Dimension, Dimension>;
		DensePlainObject toDense() const
		{
			DensePlainObject result(rows(), cols());
			evalTo(result);
			return result;
		}
		template <typename Dense,
		          typename = std::enable_if_t<std::is_base_of_v<Eigen::MatrixBase<Dense>, Dense>>>
		/* IMPLICIT */ operator Dense() const
		{
			Dense result;
			evalTo(result);
			return result;
		}

		template <typename Destination>
		void evalTo(Eigen::MatrixBase<Destination>& destination) const
		{
			DensePlainObject evaluated(rows(), cols());
			for (Eigen::Index row = 0; row < rows(); ++row)
				for (Eigen::Index column = 0; column < cols(); ++column)
					evaluated.coeffRef(row, column) = coeff(row, column);
			destination.derived() = evaluated;
		}

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
		friend ExpressionBase;
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
					if (CoefficientPolicy::isAuthoritativeCoordinate(row, column))
						writeLogical(row, column, static_cast<Scalar>(other.coeff(row, column)));
		}

		template <unsigned int Mode, typename MatrixType>
		void assignFromTriangularView(const Eigen::TriangularView<MatrixType, Mode>& view)
		{
			const auto evaluated = view.nestedExpression().eval();
			constexpr bool upper = (Mode & Eigen::Upper) != 0;
			constexpr bool unit = (Mode & Eigen::UnitDiag) != 0;
			constexpr bool zero = (Mode & Eigen::ZeroDiag) != 0;
			for (Eigen::Index row = 0; row < dimension(); ++row)
				for (Eigen::Index column = 0; column < dimension(); ++column)
					if ((upper && row <= column) || (!upper && row >= column))
					{
						const Scalar value = row == column && unit ? Scalar(1)
						                     : row == column && zero ? Scalar(0)
						                     : static_cast<Scalar>(evaluated.coeff(row, column));
						writeLogical(row, column, value);
					}
		}

		template <unsigned int UpLo, typename MatrixType>
		void assignFromSelfAdjointView(const Eigen::SelfAdjointView<MatrixType, UpLo>& view)
		{
			const auto evaluated = view.nestedExpression().eval();
			constexpr bool upper = (UpLo & Eigen::Upper) != 0;
			for (Eigen::Index row = 0; row < dimension(); ++row)
				for (Eigen::Index column = 0; column < dimension(); ++column)
					if ((upper && row <= column) || (!upper && row >= column))
						writeLogical(row, column,
						             static_cast<Scalar>(evaluated.coeff(row, column)));
		}

		template <typename Function>
		void forEachIndependentImpl(Function&& function) const
		{
			const Scalar* element = data();
			for (Eigen::Index major = 0; major < dimension(); ++major)
				for (Eigen::Index minor = 0; minor <= major; ++minor, ++element)
				{
					Eigen::Index row = Packing == TrianglePacking::Lower ? major : minor;
					Eigen::Index column = Packing == TrianglePacking::Lower ? minor : major;
					if constexpr (std::is_same_v<StructureTag, UpperTriangularTag>)
					{
						if (row > column) (std::swap)(row, column);
					}
					else if constexpr (std::is_same_v<StructureTag, LowerTriangularTag>)
					{
						if (row < column) (std::swap)(row, column);
					}
					function(row, column, *element);
				}
		}

		static constexpr bool isTwoSidedStructure()
		{
			return !std::is_same_v<StructureTag, UpperTriangularTag>
			       && !std::is_same_v<StructureTag, LowerTriangularTag>;
		}
		bool hasStructuralZeros() const
		{
			return dimension() > 1
			       && (std::is_same_v<StructureTag, UpperTriangularTag>
			           || std::is_same_v<StructureTag, LowerTriangularTag>);
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
