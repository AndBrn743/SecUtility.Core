// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/ForwardDeclarations.hpp>
#include <SecUtility/Hoppy/Detail/TriangularCompressedCheckedSize.hpp>
#include <SecUtility/Hoppy/Detail/TriangularCompressedCoeffProxy.hpp>
#include <SecUtility/Hoppy/Detail/TriangularCompressedTags.hpp>

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace Hoppy::Detail
{
	template <typename PlainObject>
	struct TriangularCompressedMappedAccess
	{
		using Scalar = typename PlainObject::Scalar;
		using RealScalar = typename PlainObject::RealScalar;
		using StructureTag = typename PlainObject::StructureTag;

		static bool isValidIndex(const Eigen::Index dimension, const Eigen::Index row, const Eigen::Index column)
		{
			if (row < 0 || column < 0 || row >= dimension || column >= dimension)
			{
				eigen_assert(false && "triangular-compressed coefficient index is out of bounds");
				return false;
			}
			return true;
		}

		static bool isCanonicalSide(const Eigen::Index row, const Eigen::Index column) noexcept
		{
			return PlainObject::PackingValue == TrianglePacking::Lower ? row >= column : row <= column;
		}

		static bool isStructuralZero(const Eigen::Index row, const Eigen::Index column) noexcept
		{
			if constexpr (std::is_same_v<StructureTag, UpperTriangularTag>) return row > column;
			else if constexpr (std::is_same_v<StructureTag, LowerTriangularTag>) return row < column;
			else return false;
		}

		static Scalar read(const Scalar* data,
		                   const Eigen::Index dimension,
		                   const Eigen::Index row,
		                   const Eigen::Index column)
		{
			if (!isValidIndex(dimension, row, column) || isStructuralZero(row, column)) return Scalar(0);
			const Scalar& stored = data[checkedPackedOffset(
			        row, column, dimension, PlainObject::PackingValue)];
			if (row == column || isCanonicalSide(row, column)) return stored;
			if constexpr (std::is_same_v<StructureTag, AntiSymmetricTag>) return -stored;
			else if constexpr (std::is_same_v<StructureTag, HermitianTag>) return Eigen::numext::conj(stored);
			else if constexpr (std::is_same_v<StructureTag, AntiHermitianTag>) return -Eigen::numext::conj(stored);
			else return stored;
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

		static void write(Scalar* data,
		                  const Eigen::Index dimension,
		                  const Eigen::Index row,
		                  const Eigen::Index column, const Scalar& value)
		{
			if (!isValidIndex(dimension, row, column)) return;
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
			Scalar stored = value;
			if (row != column && !isCanonicalSide(row, column))
			{
				if constexpr (std::is_same_v<StructureTag, AntiSymmetricTag>) stored = -value;
				else if constexpr (std::is_same_v<StructureTag, HermitianTag>) stored = Eigen::numext::conj(value);
				else if constexpr (std::is_same_v<StructureTag, AntiHermitianTag>) stored = -Eigen::numext::conj(value);
			}
			data[checkedPackedOffset(row, column, dimension, PlainObject::PackingValue)] = stored;
		}
	};

	template <typename PlainObject, int MapOptions>
	Eigen::Index validateMappedStorage(const typename PlainObject::Scalar* data, Eigen::Index dimension)
	{
		if constexpr (PlainObject::RowsAtCompileTime != Eigen::Dynamic)
			if (dimension != PlainObject::RowsAtCompileTime)
			{
				eigen_assert(false && "mapped dimension does not match fixed dimension");
				return PlainObject::RowsAtCompileTime;
			}
		if (dimension < 0)
		{
			eigen_assert(false && "mapped dimension must be nonnegative");
			return PlainObject::RowsAtCompileTime == Eigen::Dynamic ? 0 : PlainObject::RowsAtCompileTime;
		}
		const Eigen::Index storedSize = checkedStoredSize(dimension);
		if (dimension != 0 && (storedSize == 0 || checkedLogicalSize(dimension) == 0)) return 0;
		if (static_cast<std::size_t>(storedSize)
		    > (std::numeric_limits<std::size_t>::max)() / sizeof(typename PlainObject::Scalar))
		{
			eigen_assert(false && "mapped byte count overflows size_t");
			return 0;
		}
		if (storedSize != 0 && data == nullptr)
		{
			eigen_assert(false && "nonempty triangular-compressed map requires non-null storage");
			return 0;
		}
		if (data != nullptr)
		{
			eigen_assert(reinterpret_cast<std::uintptr_t>(data) % alignof(typename PlainObject::Scalar) == 0);
			if constexpr (MapOptions == Eigen::Aligned)
			{
				constexpr std::size_t alignment = Eigen::Aligned;
				const std::size_t bytes = checkedStoredByteCount<typename PlainObject::Scalar>(dimension);
				eigen_assert(reinterpret_cast<std::uintptr_t>(data) % alignment == 0 || bytes < alignment);
			}
		}
		return dimension;
	}
}  // namespace Hoppy::Detail

namespace Eigen::internal
{
	template <typename Scalar, int Dimension, Hoppy::TrianglePacking Packing, int Options,
	          typename StructureTag, int MapOptions, typename StrideType>
	struct traits<Eigen::Map<Hoppy::Detail::TriangularCompressedMatrix<
	        Scalar, Dimension, Packing, Options, StructureTag>, MapOptions, StrideType>>
	    : traits<Hoppy::Detail::TriangularCompressedMatrix<Scalar, Dimension, Packing, Options, StructureTag>>
	{
		static constexpr int Alignment = MapOptions & AlignedMask;
		static constexpr int Flags = NestByRefBit | LvalueBit;
	};

	template <typename Scalar, int Dimension, Hoppy::TrianglePacking Packing, int Options,
	          typename StructureTag, int MapOptions, typename StrideType>
	struct traits<Eigen::Map<const Hoppy::Detail::TriangularCompressedMatrix<
	        Scalar, Dimension, Packing, Options, StructureTag>, MapOptions, StrideType>>
	    : traits<Hoppy::Detail::TriangularCompressedMatrix<Scalar, Dimension, Packing, Options, StructureTag>>
	{
		static constexpr int Alignment = MapOptions & AlignedMask;
		static constexpr int Flags = NestByRefBit;
	};
}  // namespace Eigen::internal

namespace Eigen
{
	template <typename TScalar, int Dimension, Hoppy::TrianglePacking Packing, int Options,
	          typename TStructureTag, int MapOptions, typename StrideType>
	class Map<Hoppy::Detail::TriangularCompressedMatrix<
	        TScalar, Dimension, Packing, Options, TStructureTag>, MapOptions, StrideType>
	{
		using Plain = Hoppy::Detail::TriangularCompressedMatrix<
		        TScalar, Dimension, Packing, Options, TStructureTag>;
		static_assert(std::is_same_v<StrideType, Stride<0, 0>>,
		              "triangular-compressed maps support only Eigen's default contiguous stride");
		static_assert(MapOptions == Unaligned || MapOptions == Aligned,
		              "triangular-compressed maps support only Eigen::Unaligned or Eigen::Aligned");

	public:
		using Scalar = TScalar;
		using RealScalar = typename Plain::RealScalar;
		using StructureTag = typename Plain::StructureTag;
		using PlainObject = Plain;
		using StorageIndex = Eigen::Index;
		using CoeffProxy = Hoppy::Detail::TriangularCompressedCoeffProxy<Map>;
		static constexpr int RowsAtCompileTime = Dimension;
		static constexpr int ColsAtCompileTime = Dimension;
		static constexpr int Flags = Eigen::NestByRefBit | Eigen::LvalueBit;
		static constexpr Hoppy::TrianglePacking PackingValue = Packing;
		static constexpr bool IsTriangularCompressed = true;

		template <int D = Dimension, typename = std::enable_if_t<D != Dynamic>>
		explicit Map(Scalar* data) : Map(data, Dimension) {}
		Map(Scalar* data, Eigen::Index dimension)
			: m_Data(data), m_Dimension(Hoppy::Detail::validateMappedStorage<Plain, MapOptions>(data, dimension))
		{}
		Map(Scalar* data, const Eigen::Index rows, const Eigen::Index columns)
			: Map(data, Hoppy::Detail::checkedSquareDimension(rows, columns))
		{}

		Eigen::Index rows() const noexcept { return m_Dimension; }
		Eigen::Index cols() const noexcept { return m_Dimension; }
		Eigen::Index dimension() const noexcept { return m_Dimension; }
		Eigen::Index size() const { return Hoppy::Detail::checkedLogicalSize(m_Dimension); }
		Eigen::Index storedSize() const { return Hoppy::Detail::checkedStoredSize(m_Dimension); }
		Scalar* data() const noexcept { return m_Data; }
		Scalar coeff(Eigen::Index row, Eigen::Index column) const
		{
			return Hoppy::Detail::TriangularCompressedMappedAccess<Plain>::read(m_Data, m_Dimension, row, column);
		}
		CoeffProxy coeffRef(Eigen::Index row, Eigen::Index column)
		{
			(void) Hoppy::Detail::TriangularCompressedMappedAccess<Plain>::isValidIndex(
			        m_Dimension, row, column);
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
		Map& operator=(const Map& other) { return assignFrom(other); }

		template <typename Other,
		          typename = std::enable_if_t<Other::IsTriangularCompressed
		                                      && std::is_same_v<StructureTag, typename Other::StructureTag>>>
		Map& operator=(const Other& other)
		{
			return assignFrom(other);
		}

	private:
		template <typename>
		friend class Hoppy::Detail::TriangularCompressedCoeffProxy;
		template <typename Other>
		Map& assignFrom(const Other& other)
		{
			Plain temporary(other);
			if (temporary.dimension() != m_Dimension)
			{
				eigen_assert(false && "mapped assignment requires matching dimensions");
				return *this;
			}
			for (Eigen::Index row = 0; row < m_Dimension; ++row)
				for (Eigen::Index column = 0; column < m_Dimension; ++column)
					if (!Hoppy::Detail::TriangularCompressedMappedAccess<Plain>::isStructuralZero(row, column))
						writeLogical(row, column, temporary.coeff(row, column));
			return *this;
		}
		void writeLogical(Eigen::Index row, Eigen::Index column, const Scalar& value)
		{
			Hoppy::Detail::TriangularCompressedMappedAccess<Plain>::write(m_Data, m_Dimension, row, column, value);
		}
		Scalar* m_Data;
		Eigen::Index m_Dimension;
	};

	template <typename TScalar, int Dimension, Hoppy::TrianglePacking Packing, int Options,
	          typename TStructureTag, int MapOptions, typename StrideType>
	class Map<const Hoppy::Detail::TriangularCompressedMatrix<
	        TScalar, Dimension, Packing, Options, TStructureTag>, MapOptions, StrideType>
	{
		using Plain = Hoppy::Detail::TriangularCompressedMatrix<
		        TScalar, Dimension, Packing, Options, TStructureTag>;
		static_assert(std::is_same_v<StrideType, Stride<0, 0>>,
		              "triangular-compressed maps support only Eigen's default contiguous stride");
		static_assert(MapOptions == Unaligned || MapOptions == Aligned,
		              "triangular-compressed maps support only Eigen::Unaligned or Eigen::Aligned");

	public:
		using Scalar = TScalar;
		using RealScalar = typename Plain::RealScalar;
		using StructureTag = typename Plain::StructureTag;
		using PlainObject = Plain;
		using StorageIndex = Eigen::Index;
		static constexpr int RowsAtCompileTime = Dimension;
		static constexpr int ColsAtCompileTime = Dimension;
		static constexpr int Flags = Eigen::NestByRefBit;
		static constexpr Hoppy::TrianglePacking PackingValue = Packing;
		static constexpr bool IsTriangularCompressed = true;

		template <int D = Dimension, typename = std::enable_if_t<D != Dynamic>>
		explicit Map(const Scalar* data) : Map(data, Dimension) {}
		Map(const Scalar* data, Eigen::Index dimension)
			: m_Data(data), m_Dimension(Hoppy::Detail::validateMappedStorage<Plain, MapOptions>(
			                         data, dimension))
		{}
		Map(const Scalar* data, const Eigen::Index rows, const Eigen::Index columns)
			: Map(data, Hoppy::Detail::checkedSquareDimension(rows, columns))
		{}

		Eigen::Index rows() const noexcept { return m_Dimension; }
		Eigen::Index cols() const noexcept { return m_Dimension; }
		Eigen::Index dimension() const noexcept { return m_Dimension; }
		Eigen::Index size() const { return Hoppy::Detail::checkedLogicalSize(m_Dimension); }
		Eigen::Index storedSize() const { return Hoppy::Detail::checkedStoredSize(m_Dimension); }
		const Scalar* data() const noexcept { return m_Data; }
		Scalar coeff(Eigen::Index row, Eigen::Index column) const
		{
			return Hoppy::Detail::TriangularCompressedMappedAccess<Plain>::read(m_Data, m_Dimension, row, column);
		}
		Scalar operator()(const Eigen::Index row, const Eigen::Index column) const { return coeff(row, column); }
		template <int D = Dimension, typename = std::enable_if_t<D == 1>>
		Scalar operator()(const Eigen::Index index) const { return coeff(index, 0); }
		template <int D = Dimension, typename = std::enable_if_t<D == 1>>
		Scalar operator[](const Eigen::Index index) const { return coeff(index, 0); }

	private:
		const Scalar* m_Data;
		Eigen::Index m_Dimension;
	};
}  // namespace Eigen
