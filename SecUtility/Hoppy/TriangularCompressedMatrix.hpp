// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/ForwardDeclarations.hpp>
#include <SecUtility/Hoppy/Detail/TriangularCompressedCheckedSize.hpp>
#include <SecUtility/Hoppy/Detail/TriangularCompressedCoeffProxy.hpp>
#include <SecUtility/Hoppy/Detail/TriangularCompressedTags.hpp>
#include <SecUtility/Hoppy/Detail/TriangularCompressedTraits.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace Hoppy::Detail
{
	template <typename TScalar, int Dimension, TrianglePacking Packing, int Options,
	          typename NormalizedStructureTag>
	class TriangularCompressedMatrix
	{
		static_assert(is_valid_triangular_dimension_v<Dimension>, "invalid compile-time dimension");
		static_assert(has_representable_logical_size_v<Dimension>, "fixed logical size is not representable");
		static_assert(is_valid_triangle_packing_v<Packing>, "invalid triangle packing");
		static_assert(is_valid_triangular_options_v<Options>, "unsupported matrix options");

	public:
		using Scalar = TScalar;
		using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
		using StorageIndex = Eigen::Index;
		using StructureTag = NormalizedStructureTag;
		using PlainObject = TriangularCompressedMatrix;
		using Nested = const TriangularCompressedMatrix&;
		using CoeffReturnType = Scalar;
		using CoeffProxy = TriangularCompressedCoeffProxy<TriangularCompressedMatrix>;
		using Allocator = std::conditional_t<Options == Eigen::DontAlign, std::allocator<Scalar>,
		                                     Eigen::aligned_allocator<Scalar>>;

		static constexpr int RowsAtCompileTime = Dimension;
		static constexpr int ColsAtCompileTime = Dimension;
		static constexpr int MaxRowsAtCompileTime = Dimension;
		static constexpr int MaxColsAtCompileTime = Dimension;
		static constexpr int SizeAtCompileTime = compileTimeLogicalSize<Dimension>();
		static constexpr int MaxSizeAtCompileTime = SizeAtCompileTime;
		static constexpr int Flags = Eigen::NestByRefBit;
		static constexpr TrianglePacking PackingValue = Packing;
		static constexpr bool IsVectorAtCompileTime = Dimension == 1;

		TriangularCompressedMatrix() : TriangularCompressedMatrix(defaultDimension()) {}
		explicit TriangularCompressedMatrix(const Eigen::Index dimension)
			: m_Dimension(validatedDimension(dimension)),
			  m_Storage(static_cast<std::size_t>(checkedStoredSize(m_Dimension)))
		{}
		TriangularCompressedMatrix(const Eigen::Index rows, const Eigen::Index columns)
			: TriangularCompressedMatrix(checkedSquareDimension(rows, columns))
		{}

		TriangularCompressedMatrix(const TriangularCompressedMatrix&) = default;
		TriangularCompressedMatrix(TriangularCompressedMatrix&&)
		        noexcept(std::is_nothrow_move_constructible_v<std::vector<Scalar, Allocator>>) = default;
		TriangularCompressedMatrix& operator=(const TriangularCompressedMatrix&) = default;
		TriangularCompressedMatrix& operator=(TriangularCompressedMatrix&&)
		        noexcept(std::is_nothrow_move_assignable_v<std::vector<Scalar, Allocator>>) = default;
		~TriangularCompressedMatrix() = default;

		static Eigen::Index requiredStoredSize(const Eigen::Index dimension)
		{
			if constexpr (Dimension != Eigen::Dynamic)
			{
				if (dimension != Dimension)
				{
					eigen_assert(false && "runtime dimension does not match fixed dimension");
					return 0;
				}
			}
			return checkedStoredSize(dimension);
		}

		Eigen::Index rows() const noexcept { return m_Dimension; }
		Eigen::Index cols() const noexcept { return m_Dimension; }
		Eigen::Index dimension() const noexcept { return m_Dimension; }
		Eigen::Index size() const { return checkedLogicalSize(m_Dimension); }
		Eigen::Index storedSize() const noexcept { return static_cast<Eigen::Index>(m_Storage.size()); }
		Scalar* data() noexcept { return m_Storage.data(); }
		const Scalar* data() const noexcept { return m_Storage.data(); }

		Scalar coeff(const Eigen::Index row, const Eigen::Index column) const
		{
			if (!isValidateIndex(row, column)) return Scalar(0);
			if (isStructuralZero(row, column)) return Scalar(0);
			const Scalar& stored = m_Storage[static_cast<std::size_t>(
			        checkedPackedOffset(row, column, m_Dimension, Packing))];
			if (row == column || isCanonicalSide(row, column)) return stored;
			if constexpr (std::is_same_v<StructureTag, AntiSymmetricTag>) return -stored;
			else if constexpr (std::is_same_v<StructureTag, HermitianTag>) return Eigen::numext::conj(stored);
			else if constexpr (std::is_same_v<StructureTag, AntiHermitianTag>) return -Eigen::numext::conj(stored);
			else return stored;
		}

		CoeffProxy coeffRef(const Eigen::Index row, const Eigen::Index column)
		{
			(void) isValidateIndex(row, column);
			return {*this, row, column};
		}
		Scalar operator()(const Eigen::Index row, const Eigen::Index column) const { return coeff(row, column); }
		CoeffProxy operator()(const Eigen::Index row, const Eigen::Index column) { return coeffRef(row, column); }

		template <int D = Dimension, typename = std::enable_if_t<D == 1>>
		Scalar operator()(const Eigen::Index index) const { return coeff(index, 0); }
		template <int D = Dimension, typename = std::enable_if_t<D == 1>>
		CoeffProxy operator()(const Eigen::Index index) { return coeffRef(index, 0); }
		template <int D = Dimension, typename = std::enable_if_t<D == 1>>
		Scalar operator[](const Eigen::Index index) const { return coeff(index, 0); }
		template <int D = Dimension, typename = std::enable_if_t<D == 1>>
		CoeffProxy operator[](const Eigen::Index index) { return coeffRef(index, 0); }

		void resize(const Eigen::Index dimension)
		{
			TriangularCompressedMatrix replacement(dimension);
			swap(replacement);
		}
		void resize(const Eigen::Index rows, const Eigen::Index columns) { resize(checkedSquareDimension(rows, columns)); }
		template <typename Other>
		void resizeLike(const Other& other) { resize(other.rows(), other.cols()); }

		void conservativeResize(Eigen::Index dimension)
		{
			TriangularCompressedMatrix replacement(dimension);
			const Eigen::Index commonDimension = (std::min)(m_Dimension, replacement.m_Dimension);
			const Eigen::Index commonStoredSize = checkedStoredSize(commonDimension);
			std::copy_n(m_Storage.begin(), commonStoredSize, replacement.m_Storage.begin());
			swap(replacement);
		}
		void conservativeResize(const Eigen::Index rows, const Eigen::Index columns)
		{
			conservativeResize(checkedSquareDimension(rows, columns));
		}

		void swap(TriangularCompressedMatrix& other) noexcept(noexcept(m_Storage.swap(other.m_Storage)))
		{
			using std::swap;
			swap(m_Dimension, other.m_Dimension);
			m_Storage.swap(other.m_Storage);
		}
		friend void swap(TriangularCompressedMatrix& left, TriangularCompressedMatrix& right)
		        noexcept(noexcept(left.swap(right)))
		{
			left.swap(right);
		}

	private:
		template <typename>
		friend class TriangularCompressedCoeffProxy;
		using Storage = std::vector<Scalar, Allocator>;

		static constexpr Eigen::Index defaultDimension() { return Dimension == Eigen::Dynamic ? 0 : Dimension; }
		static Eigen::Index validatedDimension(const Eigen::Index dimension)
		{
			if constexpr (Dimension != Eigen::Dynamic)
				if (dimension != Dimension)
				{
					eigen_assert(false && "runtime dimension does not match fixed dimension");
					return Dimension;
				}
			if (dimension < 0)
			{
				eigen_assert(false && "triangular-compressed dimension must be nonnegative");
				return defaultDimension();
			}
			constexpr Eigen::Index maximum = (std::numeric_limits<Eigen::Index>::max)();
			if (dimension != 0 && dimension > maximum / dimension)
			{
				eigen_assert(false && "triangular-compressed logical size overflows Eigen::Index");
				return defaultDimension();
			}
			const Eigen::Index storedSize = checkedStoredSize(dimension);
			if (dimension != 0 && storedSize == 0) return defaultDimension();
			constexpr std::size_t maximumBytes = (std::numeric_limits<std::size_t>::max)();
			if (static_cast<std::size_t>(storedSize) > maximumBytes / sizeof(Scalar))
			{
				eigen_assert(false && "triangular-compressed byte count overflows size_t");
				return defaultDimension();
			}
			return dimension;
		}

		bool isValidateIndex(const Eigen::Index row, const Eigen::Index column) const
		{
			if (row < 0 || column < 0 || row >= m_Dimension || column >= m_Dimension)
			{
				eigen_assert(false && "triangular-compressed coefficient index is out of bounds");
				return false;
			}
			return true;
		}
		static bool isCanonicalSide(const Eigen::Index row, const Eigen::Index column) noexcept
		{
			return Packing == TrianglePacking::Lower ? row >= column : row <= column;
		}
		static bool isStructuralZero(const Eigen::Index row, const Eigen::Index column) noexcept
		{
			if constexpr (std::is_same_v<StructureTag, UpperTriangularTag>) return row > column;
			else if constexpr (std::is_same_v<StructureTag, LowerTriangularTag>) return row < column;
			else return false;
		}
		static bool isValidDiagonal(const Scalar& value)
		{
			if constexpr (std::is_same_v<StructureTag, AntiSymmetricTag>) return value == Scalar(0);
			else if constexpr (std::is_same_v<StructureTag, HermitianTag>)
				return Eigen::numext::imag(value) == RealScalar(0);
			else if constexpr (std::is_same_v<StructureTag, AntiHermitianTag>)
				return Eigen::numext::real(value) == RealScalar(0);
			else return true;
		}

		void writeLogical(const Eigen::Index row, const Eigen::Index column, const Scalar& value)
		{
			if (!isValidateIndex(row, column)) return;
			if (isStructuralZero(row, column))
			{
				eigen_assert(false && "cannot write a structural zero");
				return;
			}
			if (row == column && !isValidDiagonal(value))
			{
				eigen_assert(false && "coefficient violates the structure's diagonal invariant");
				return;
			}
			Scalar storedValue = value;
			if (row != column && !isCanonicalSide(row, column))
			{
				if constexpr (std::is_same_v<StructureTag, AntiSymmetricTag>) storedValue = -value;
				else if constexpr (std::is_same_v<StructureTag, HermitianTag>) storedValue = Eigen::numext::conj(value);
				else if constexpr (std::is_same_v<StructureTag, AntiHermitianTag>) storedValue = -Eigen::numext::conj(value);
			}
			m_Storage[static_cast<std::size_t>(
			        checkedPackedOffset(row, column, m_Dimension, Packing))] = std::move(storedValue);
		}

		Eigen::Index m_Dimension;
		Storage m_Storage;
	};
}  // namespace Hoppy::Detail
