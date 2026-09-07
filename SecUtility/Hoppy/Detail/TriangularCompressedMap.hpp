// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/ForwardDeclarations.hpp>
#include <SecUtility/Hoppy/Detail/TriangularCompressedCheckedSize.hpp>
#include <SecUtility/Hoppy/Detail/TriangularCompressedPlainBase.hpp>
#include <SecUtility/Hoppy/Detail/TriangularCompressedTags.hpp>

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace Hoppy::Detail
{
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
	    : public Hoppy::Detail::TriangularCompressedPlainBase<
	              Map<Hoppy::Detail::TriangularCompressedMatrix<
	                          TScalar, Dimension, Packing, Options, TStructureTag>,
	                  MapOptions, StrideType>,
	              TScalar, TStructureTag, Packing, Dimension, true>
	{
		using Plain = Hoppy::Detail::TriangularCompressedMatrix<
		        TScalar, Dimension, Packing, Options, TStructureTag>;
		using PlainBase = Hoppy::Detail::TriangularCompressedPlainBase<
		        Map, TScalar, TStructureTag, Packing, Dimension, true>;
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
		using CoeffProxy = typename PlainBase::CoeffProxy;
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

		Map& operator=(const Map& other) { return assignFrom(other); }

		template <typename Other,
		          typename = std::enable_if_t<Other::IsTriangularCompressed
		                                      && std::is_same_v<StructureTag, typename Other::StructureTag>>>
		Map& operator=(const Other& other)
		{
			return assignFrom(other);
		}

		template <typename MatrixType, unsigned int Mode,
		          typename = std::enable_if_t<Hoppy::Detail::accepts_triangular_view_v<StructureTag, Mode>>>
		Map& operator=(const TriangularView<MatrixType, Mode>& view)
		{
			if (view.rows() != m_Dimension || view.cols() != m_Dimension)
			{
				eigen_assert(false && "mapped assignment requires matching dimensions");
				return *this;
			}
			Plain temporary(m_Dimension);
			temporary = view;
			this->assignCoefficientsFrom(temporary);
			return *this;
		}

		template <typename MatrixType, unsigned int UpLo,
		          bool Enabled = Hoppy::Detail::accepts_self_adjoint_view_v<Scalar, StructureTag>,
		          typename = std::enable_if_t<Enabled>>
		Map& operator=(const SelfAdjointView<MatrixType, UpLo>& view)
		{
			if (view.rows() != m_Dimension || view.cols() != m_Dimension)
			{
				eigen_assert(false && "mapped assignment requires matching dimensions");
				return *this;
			}
			Plain temporary(m_Dimension);
			temporary = view;
			this->assignCoefficientsFrom(temporary);
			return *this;
		}

	private:
		friend PlainBase;
		template <typename Other>
		Map& assignFrom(const Other& other)
		{
			Plain temporary(other);
			if (temporary.dimension() != m_Dimension)
			{
				eigen_assert(false && "mapped assignment requires matching dimensions");
				return *this;
			}
			this->assignCoefficientsFrom(temporary);
			return *this;
		}
		Eigen::Index dimensionImpl() const noexcept { return m_Dimension; }
		Scalar* coeffDataImpl() noexcept { return m_Data; }
		const Scalar* coeffDataImpl() const noexcept { return m_Data; }
		Scalar* m_Data;
		Eigen::Index m_Dimension;
	};

	template <typename TScalar, int Dimension, Hoppy::TrianglePacking Packing, int Options,
	          typename TStructureTag, int MapOptions, typename StrideType>
	class Map<const Hoppy::Detail::TriangularCompressedMatrix<
	        TScalar, Dimension, Packing, Options, TStructureTag>, MapOptions, StrideType>
	    : public Hoppy::Detail::TriangularCompressedPlainBase<
	              Map<const Hoppy::Detail::TriangularCompressedMatrix<
	                          TScalar, Dimension, Packing, Options, TStructureTag>,
	                  MapOptions, StrideType>,
	              TScalar, TStructureTag, Packing, Dimension, false>
	{
		using Plain = Hoppy::Detail::TriangularCompressedMatrix<
		        TScalar, Dimension, Packing, Options, TStructureTag>;
		using PlainBase = Hoppy::Detail::TriangularCompressedPlainBase<
		        Map, TScalar, TStructureTag, Packing, Dimension, false>;
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

	private:
		friend PlainBase;
		Eigen::Index dimensionImpl() const noexcept { return m_Dimension; }
		const Scalar* coeffDataImpl() const noexcept { return m_Data; }
		const Scalar* m_Data;
		Eigen::Index m_Dimension;
	};
}  // namespace Eigen
