// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/ForwardDeclarations.hpp>
#include <SecUtility/Hoppy/Detail/TriangularCompressedCheckedSize.hpp>
#include <SecUtility/Hoppy/Detail/TriangularCompressedPlainBase.hpp>
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
	    : public TriangularCompressedPlainBase<
	              TriangularCompressedMatrix<TScalar, Dimension, Packing, Options, NormalizedStructureTag>,
	              TScalar, NormalizedStructureTag, Packing, Dimension, true>
	{
		using PlainBase = TriangularCompressedPlainBase<
		        TriangularCompressedMatrix, TScalar, NormalizedStructureTag, Packing, Dimension, true>;
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
		using CoeffProxy = typename PlainBase::CoeffProxy;
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
		static constexpr bool IsTriangularCompressed = true;

		TriangularCompressedMatrix() : TriangularCompressedMatrix(defaultDimension()) {}
		explicit TriangularCompressedMatrix(const Eigen::Index dimension)
			: m_Dimension(validatedDimension(dimension)),
			  m_Storage(static_cast<std::size_t>(checkedStoredSize(m_Dimension)))
		{}
		TriangularCompressedMatrix(const Eigen::Index rows, const Eigen::Index columns)
			: TriangularCompressedMatrix(checkedSquareDimension(rows, columns))
		{}

		TriangularCompressedMatrix(const TriangularCompressedMatrix&) = default;
		template <typename Other,
		          typename = std::enable_if_t<Other::IsTriangularCompressed
		                                      && std::is_same_v<StructureTag, typename Other::StructureTag>>>
		explicit TriangularCompressedMatrix(const Other& other)
			: TriangularCompressedMatrix(other.dimension())
		{
			this->assignCoefficientsFrom(other);
		}
		TriangularCompressedMatrix(TriangularCompressedMatrix&&)
		        noexcept(std::is_nothrow_move_constructible_v<std::vector<Scalar, Allocator>>) = default;
		TriangularCompressedMatrix& operator=(const TriangularCompressedMatrix&) = default;
		template <typename Other,
		          typename = std::enable_if_t<Other::IsTriangularCompressed
		                                      && std::is_same_v<StructureTag, typename Other::StructureTag>>>
		TriangularCompressedMatrix& operator=(const Other& other)
		{
			TriangularCompressedMatrix replacement(other);
			swap(replacement);
			return *this;
		}
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
		friend PlainBase;
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

		Eigen::Index dimensionImpl() const noexcept { return m_Dimension; }
		Scalar* coeffDataImpl() noexcept { return m_Storage.data(); }
		const Scalar* coeffDataImpl() const noexcept { return m_Storage.data(); }

		Eigen::Index m_Dimension;
		Storage m_Storage;
	};
}  // namespace Hoppy::Detail

#include <SecUtility/Hoppy/Detail/TriangularCompressedMap.hpp>
