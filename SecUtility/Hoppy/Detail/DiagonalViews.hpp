// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/BlockDiagonalMatrixExpr.hpp>
#include <SecUtility/Hoppy/BlockVectorExpr.hpp>
#include <SecUtility/Hoppy/Detail/Traits.hpp>

#include <Eigen/Core>

#include <type_traits>
#include <utility>
#include <vector>

namespace Hoppy::Detail
{
	template <typename Source, bool Writable>
	class DiagonalReturnType;
}

template <typename Source, bool Writable>
struct Eigen::internal::traits<Hoppy::Detail::DiagonalReturnType<Source, Writable>>
{
	using Scalar = typename traits<Source>::Scalar;
	using Orientation = Hoppy::BlockVectorOrientation::Column;
	using StorageKind = Hoppy::Detail::BlockVectorStorage;
	using XprKind = MatrixXpr;
	using StorageIndex = Eigen::Index;
	static constexpr int Flags = 0;
	static constexpr int RowsAtCompileTime = Dynamic;
	static constexpr int ColsAtCompileTime = 1;
	static constexpr int MaxRowsAtCompileTime = Dynamic;
	static constexpr int MaxColsAtCompileTime = 1;
};

template <typename Derived>
struct Eigen::internal::traits<Eigen::DiagonalWrapper<const Hoppy::BlockVectorExpr<Derived>>>
{
	using Scalar = typename traits<Derived>::Scalar;
	using BlockPolicy = Hoppy::DenseBlockPolicy;
	using StorageKind = Hoppy::Detail::BlockDiagonalStorage;
	using XprKind = MatrixXpr;
	using StorageIndex = Eigen::Index;
	static constexpr int Flags = 0;
	static constexpr int RowsAtCompileTime = Dynamic;
	static constexpr int ColsAtCompileTime = Dynamic;
	static constexpr int MaxRowsAtCompileTime = Dynamic;
	static constexpr int MaxColsAtCompileTime = Dynamic;
};

namespace Hoppy::Detail
{
	template <typename Source, bool IsWritable>
	class DiagonalReturnType : public BlockVectorExpr<DiagonalReturnType<Source, IsWritable>>
	{
		using Nested = std::conditional_t<IsWritable, Source&, typename Eigen::internal::ref_selector<Source>::type>;

	public:
		using Scalar = typename Eigen::internal::traits<DiagonalReturnType>::Scalar;
		explicit DiagonalReturnType(Source& source) : m_Source(source) {}
		DiagonalReturnType(const DiagonalReturnType&) = default;
		DiagonalReturnType(DiagonalReturnType&&) = default;

		Eigen::Index blockCount() const { return m_Source.blockCount(); }
		Eigen::Index rows() const { return totalDimension(); }
		Eigen::Index cols() const { return 1; }
		Eigen::Index size() const { return totalDimension(); }
		Eigen::Index totalDimension() const { return m_Source.totalDimension(); }
		Eigen::Index storedSize() const { return totalDimension(); }
		Eigen::Index dimensionOfBlock(Eigen::Index index) const { return m_Source.dimensionOfBlock(index); }
		Eigen::Index blockOffset(Eigen::Index index) const { return m_Source.blockOffset(index); }
		Eigen::Index storageOffset(Eigen::Index index) const { return m_Source.blockOffset(index); }
		const std::vector<Eigen::Index>& blockingInfo() const& { return m_Source.blockingInfo(); }
		std::vector<Eigen::Index> blockingInfo() const&& { return m_Source.blockingInfo(); }

		auto operator[](Eigen::Index index)
		{
			if constexpr (IsWritable)
				return m_Source[index].diagonal();
			else
				return std::as_const(m_Source)[index].diagonal();
		}
		auto operator[](Eigen::Index index) const { return std::as_const(m_Source)[index].diagonal(); }
		template <typename Visitor>
		void visitStorageLeaves(Visitor&& visitor) const
		{
			Detail::visitStorageLeaves(m_Source, std::forward<Visitor>(visitor));
		}

		template <typename Derived, typename T = std::integral_constant<bool, IsWritable>,
		          typename = std::enable_if_t<T::value>>
		DiagonalReturnType& operator=(const BlockVectorExpr<Derived>& expression)
		{
			assign(expression.derived());
			return *this;
		}
		DiagonalReturnType& operator=(const DiagonalReturnType& other)
		{
			static_assert(IsWritable, "a read-only diagonal view cannot be assigned");
			assign(other);
			return *this;
		}

	private:
		template <typename Expression>
		void assign(const Expression& expression)
		{
			eigen_assert(this->hasSameBlockingAs(expression));
			for (Eigen::Index index = 0; index < blockCount(); ++index)
			{
				const Eigen::VectorX<Scalar> temporary = expression[index];
				m_Source[index].diagonal() = temporary;
			}
		}

		Nested m_Source;
	};
}  // namespace Hoppy::Detail

namespace Eigen
{
	template <typename Derived>
	class DiagonalWrapper<const Hoppy::BlockVectorExpr<Derived>>
	    : public Hoppy::BlockDiagonalMatrixExpr<DiagonalWrapper<const Hoppy::BlockVectorExpr<Derived>>>
	{
	public:
		using Scalar = typename internal::traits<Derived>::Scalar;
		explicit DiagonalWrapper(const Hoppy::BlockVectorExpr<Derived>& vector) : m_Vector(vector.derived()) {}

		Index blockCount() const { return m_Vector.blockCount(); }
		Index rows() const { return totalDimension(); }
		Index cols() const { return totalDimension(); }
		Index size() const { return totalDimension() * totalDimension(); }
		Index totalDimension() const { return m_Vector.totalDimension(); }
		Index storedSize() const
		{
			Index result = 0;
			for (Index index = 0; index < blockCount(); ++index)
				result += dimensionOfBlock(index) * dimensionOfBlock(index);
			return result;
		}
		Index dimensionOfBlock(Index index) const { return m_Vector.dimensionOfBlock(index); }
		Index blockOffset(Index index) const { return m_Vector.blockOffset(index); }
		Index storageOffset(const Index index) const
		{
			Index result = 0;
			for (Index current = 0; current < index; ++current)
				result += dimensionOfBlock(current) * dimensionOfBlock(current);
			return result;
		}
		const std::vector<Index>& blockingInfo() const& { return m_Vector.blockingInfo(); }
		std::vector<Index> blockingInfo() const&& { return m_Vector.blockingInfo(); }
		auto operator[](Index index) const
		{
			return m_Vector[index].asDiagonal();
		}
		template <typename Visitor>
		void visitStorageLeaves(Visitor&& visitor) const
		{
			Hoppy::Detail::visitStorageLeaves(m_Vector, std::forward<Visitor>(visitor));
		}

	private:
		typename internal::ref_selector<Derived>::type m_Vector;
	};
}  // namespace Eigen

namespace Hoppy
{
	template <typename Derived>
	auto BlockDiagonalMatrixExpr<Derived>::diagonal() const&
	{
		return Detail::DiagonalReturnType<const Derived, false>(derived());
	}

	template <typename Derived>
	template <typename T, typename>
	auto BlockVectorExpr<Derived>::asDiagonal() const&
	{
		return Eigen::DiagonalWrapper<const BlockVectorExpr<Derived>>(*this);
	}

}  // namespace Hoppy
