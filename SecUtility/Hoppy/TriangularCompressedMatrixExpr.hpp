// SPDX-License-Identifier: MIT

#pragma once

#include <Eigen/Core>

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
	};
}
