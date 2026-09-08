// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/TriangularCompressedMatrixExpr.hpp>

#include <Eigen/Core>

#include <type_traits>
#include <utility>

namespace Hoppy::Detail
{
	template <typename Parent>
	class TriangularCompressedDiagonalView
	{
	public:
		using Scalar = typename std::remove_const_t<Parent>::Scalar;
		explicit TriangularCompressedDiagonalView(Parent& parent) : m_Parent(&parent) {}
		Eigen::Index rows() const noexcept { return m_Parent->dimension(); }
		Eigen::Index cols() const noexcept { return 1; }
		Eigen::Index size() const noexcept { return rows(); }
		Scalar coeff(Eigen::Index index) const { return m_Parent->coeff(index, index); }
		Scalar operator()(const Eigen::Index index) const { return coeff(index); }
		Scalar operator[](const Eigen::Index index) const { return coeff(index); }
		auto toDense() const
		{
			Eigen::Matrix<Scalar, Eigen::Dynamic, 1> result(rows());
			for (Eigen::Index index = 0; index < rows(); ++index) result(index) = coeff(index);
			return result;
		}
		template <typename P = Parent, typename = std::enable_if_t<!std::is_const_v<P>>>
		auto operator()(Eigen::Index index) { return m_Parent->coeffRef(index, index); }
		template <typename P = Parent, typename = std::enable_if_t<!std::is_const_v<P>>>
		auto operator[](Eigen::Index index) { return m_Parent->coeffRef(index, index); }

	private:
		Parent* m_Parent;
	};

	template <typename Parent, unsigned int Mode = 0>
	class TriangularCompressedLogicalView
	{
	public:
		using Scalar = typename std::remove_const_t<Parent>::Scalar;
		TriangularCompressedLogicalView(Parent& parent,
		                                const Eigen::Index firstRow,
		                                const Eigen::Index firstColumn,
		                                const Eigen::Index rows,
		                                const Eigen::Index columns)
			: m_Parent(&parent), m_FirstRow(firstRow), m_FirstColumn(firstColumn),
			  m_Rows(rows), m_Columns(columns)
		{
			eigen_assert(firstRow >= 0 && firstColumn >= 0 && rows >= 0 && columns >= 0
			             && firstRow + rows <= parent.rows() && firstColumn + columns <= parent.cols());
		}
		Eigen::Index rows() const noexcept { return m_Rows; }
		Eigen::Index cols() const noexcept { return m_Columns; }
		Scalar coeff(const Eigen::Index row, const Eigen::Index column) const
		{
			checkIndex(row, column);
			if constexpr (Mode != 0)
			{
				if (!contains(row, column)) return Scalar(0);
				if (row == column && (Mode & Eigen::UnitDiag) != 0) return Scalar(1);
				if (row == column && (Mode & Eigen::ZeroDiag) != 0) return Scalar(0);
			}
			return m_Parent->coeff(m_FirstRow + row, m_FirstColumn + column);
		}
		Scalar operator()(const Eigen::Index row, const Eigen::Index column) const { return coeff(row, column); }
		auto toDense() const
		{
			Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> result(rows(), cols());
			for (Eigen::Index row = 0; row < rows(); ++row)
				for (Eigen::Index column = 0; column < cols(); ++column)
					result(row, column) = coeff(row, column);
			return result;
		}
		template <typename P = Parent, typename = std::enable_if_t<!std::is_const_v<P>>>
		auto operator()(const Eigen::Index row, const Eigen::Index column)
		{
			checkIndex(row, column);
			if constexpr (Mode != 0)
				eigen_assert(contains(row, column)
				             && !(row == column && ((Mode & Eigen::UnitDiag) != 0
				                                  || (Mode & Eigen::ZeroDiag) != 0)));
			return m_Parent->coeffRef(m_FirstRow + row, m_FirstColumn + column);
		}

	private:
		bool contains(const Eigen::Index row, const Eigen::Index column) const noexcept
		{
			if constexpr (Mode == 0) return true;
			const bool upper = (Mode & Eigen::Upper) != 0;
			const bool strict = (Mode & Eigen::ZeroDiag) != 0;
			return upper ? (strict ? row < column : row <= column)
			             : (strict ? row > column : row >= column);
		}
		void checkIndex(const Eigen::Index row, const Eigen::Index column) const
		{
			(void) row;
			(void) column;
			eigen_assert(row >= 0 && column >= 0 && row < m_Rows && column < m_Columns);
		}
		Parent* m_Parent;
		Eigen::Index m_FirstRow;
		Eigen::Index m_FirstColumn;
		Eigen::Index m_Rows;
		Eigen::Index m_Columns;
	};
}

namespace Hoppy
{
	template <typename Derived> auto TriangularCompressedMatrixExpr<Derived>::diagonal() &
	{
		using Parent = std::conditional_t<Derived::IsWritable, Derived, const Derived>;
		return Detail::TriangularCompressedDiagonalView<Parent>(derived());
	}
	template <typename Derived> auto TriangularCompressedMatrixExpr<Derived>::diagonal() const&
	{ return Detail::TriangularCompressedDiagonalView<const Derived>(derived()); }
	template <typename Derived> auto TriangularCompressedMatrixExpr<Derived>::diagonal() &&
	{
		using Parent = std::conditional_t<Derived::IsWritable, Derived, const Derived>;
		return Detail::TriangularCompressedDiagonalView<Parent>(derived());
	}
	template <typename Derived> template <unsigned int Mode>
	auto TriangularCompressedMatrixExpr<Derived>::triangularView() &
	{
		using Parent = std::conditional_t<Derived::IsWritable, Derived, const Derived>;
		return Detail::TriangularCompressedLogicalView<Parent, Mode>(derived(), 0, 0, derived().rows(), derived().cols());
	}
	template <typename Derived> template <unsigned int Mode>
	auto TriangularCompressedMatrixExpr<Derived>::triangularView() const&
	{ return Detail::TriangularCompressedLogicalView<const Derived, Mode>(derived(), 0, 0, derived().rows(), derived().cols()); }
	template <typename Derived> template <unsigned int Mode>
	auto TriangularCompressedMatrixExpr<Derived>::triangularView() &&
	{
		using Parent = std::conditional_t<Derived::IsWritable, Derived, const Derived>;
		return Detail::TriangularCompressedLogicalView<Parent, Mode>(derived(), 0, 0,
		                                                         derived().rows(), derived().cols());
	}
	template <typename Derived>
	auto TriangularCompressedMatrixExpr<Derived>::block(Eigen::Index row, Eigen::Index column,
	                                                    Eigen::Index rows, Eigen::Index columns) &
	{
		using Parent = std::conditional_t<Derived::IsWritable, Derived, const Derived>;
		return Detail::TriangularCompressedLogicalView<Parent>(derived(), row, column, rows, columns);
	}
	template <typename Derived> template <int Rows, int Columns>
	auto TriangularCompressedMatrixExpr<Derived>::block(const Eigen::Index row, const Eigen::Index column) &
	{ return block(row, column, Rows, Columns); }
	template <typename Derived> template <int Rows, int Columns>
	auto TriangularCompressedMatrixExpr<Derived>::block(const Eigen::Index row, const Eigen::Index column) const&
	{ return block(row, column, Rows, Columns); }
	template <typename Derived> template <int Rows, int Columns>
	auto TriangularCompressedMatrixExpr<Derived>::block(const Eigen::Index row,
	                                                    const Eigen::Index column) &&
	{ return std::move(*this).block(row, column, Rows, Columns); }
	template <typename Derived>
	auto TriangularCompressedMatrixExpr<Derived>::block(Eigen::Index row, Eigen::Index column,
	                                                    Eigen::Index rows, Eigen::Index columns) const&
	{ return Detail::TriangularCompressedLogicalView<const Derived>(derived(), row, column, rows, columns); }
	template <typename Derived>
	auto TriangularCompressedMatrixExpr<Derived>::block(Eigen::Index row, Eigen::Index column,
	                                                    Eigen::Index rows, Eigen::Index columns) &&
	{
		using Parent = std::conditional_t<Derived::IsWritable, Derived, const Derived>;
		return Detail::TriangularCompressedLogicalView<Parent>(derived(), row, column, rows, columns);
	}
	template <typename Derived> auto TriangularCompressedMatrixExpr<Derived>::topLeftCorner(const Eigen::Index rows, const Eigen::Index columns) &
	{ return block(0, 0, rows, columns); }
	template <typename Derived> auto TriangularCompressedMatrixExpr<Derived>::topLeftCorner(const Eigen::Index rows,
	                                                            const Eigen::Index columns) const&
	{ return block(0, 0, rows, columns); }
	template <typename Derived> auto TriangularCompressedMatrixExpr<Derived>::topLeftCorner(
	        const Eigen::Index rows, const Eigen::Index columns) &&
	{ return std::move(*this).block(0, 0, rows, columns); }
	template <typename Derived> template <int Rows, int Columns>
	auto TriangularCompressedMatrixExpr<Derived>::topLeftCorner() &
	{ return block(0, 0, Rows, Columns); }
	template <typename Derived> template <int Rows, int Columns>
	auto TriangularCompressedMatrixExpr<Derived>::topLeftCorner() const&
	{ return block(0, 0, Rows, Columns); }
	template <typename Derived> template <int Rows, int Columns>
	auto TriangularCompressedMatrixExpr<Derived>::topLeftCorner() &&
	{ return std::move(*this).block(0, 0, Rows, Columns); }
	template <typename Derived> auto TriangularCompressedMatrixExpr<Derived>::row(const Eigen::Index index) &
	{ return block(index, 0, 1, derived().cols()); }
	template <typename Derived> auto TriangularCompressedMatrixExpr<Derived>::row(const Eigen::Index index) const&
	{ return block(index, 0, 1, derived().cols()); }
	template <typename Derived> auto TriangularCompressedMatrixExpr<Derived>::row(const Eigen::Index index) &&
	{ return std::move(*this).block(index, 0, 1, derived().cols()); }
	template <typename Derived> auto TriangularCompressedMatrixExpr<Derived>::col(const Eigen::Index index) &
	{ return block(0, index, derived().rows(), 1); }
	template <typename Derived> auto TriangularCompressedMatrixExpr<Derived>::col(const Eigen::Index index) const&
	{ return block(0, index, derived().rows(), 1); }
	template <typename Derived> auto TriangularCompressedMatrixExpr<Derived>::col(const Eigen::Index index) &&
	{ return std::move(*this).block(0, index, derived().rows(), 1); }
}
