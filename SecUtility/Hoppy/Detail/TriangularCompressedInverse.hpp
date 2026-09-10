// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/Detail/TriangularCompressedExpressions.hpp>

#include <Eigen/LU>

#include <type_traits>
#include <utility>

namespace Hoppy::Detail
{
	template <typename Operand>
	class TriangularCompressedInverse
	    : public Hoppy::TriangularCompressedMatrixExpr<TriangularCompressedInverse<Operand>>
	{
		using Source = std::remove_cv_t<std::remove_reference_t<Operand>>;
	public:
		using Scalar = typename Source::Scalar;
		using DenseObject = Eigen::Matrix<Scalar, Source::RowsAtCompileTime,
		                                  Source::ColsAtCompileTime>;
		using StructureTag = typename Source::StructureTag;
		static constexpr int RowsAtCompileTime = Source::RowsAtCompileTime;
		static constexpr int ColsAtCompileTime = Source::ColsAtCompileTime;
		static constexpr TrianglePacking PackingValue = Source::PackingValue;
		static constexpr bool IsTriangularCompressed = true;
		static constexpr bool IsWritable = false;
		using PlainObject = TriangularCompressedMatrix<Scalar, RowsAtCompileTime,
		                                               PackingValue, 0, StructureTag>;

		explicit TriangularCompressedInverse(Operand operand)
			: m_Operand(std::forward<Operand>(operand)) {}
		Eigen::Index dimension() const noexcept { return m_Operand.dimension(); }
		Eigen::Index rows() const noexcept { return dimension(); }
		Eigen::Index cols() const noexcept { return dimension(); }
		Eigen::Index size() const noexcept { return dimension() * dimension(); }
		Scalar coeff(Eigen::Index row, Eigen::Index column) const
		{ return toDense().coeff(row, column); }
		DenseObject toDense() const { return m_Operand.toDense().inverse().eval(); }
		template <typename Destination>
		void evalTo(Eigen::MatrixBase<Destination>& destination) const
		{ destination.derived() = toDense(); }
		PlainObject eval() const { return PlainObject::FromUncheckedDense(toDense()); }
		template <typename Dense,
		          typename = std::enable_if_t<std::is_base_of_v<Eigen::MatrixBase<Dense>, Dense>>>
		operator Dense() const { return toDense(); }

	private:
		Operand m_Operand;
	};
}

namespace Hoppy
{
	template <typename Derived> template <typename D, typename>
	auto TriangularCompressedMatrixExpr<Derived>::inverse() const&
	{ return Detail::TriangularCompressedInverse<const Derived&>(derived()); }
	template <typename Derived> template <typename D, typename>
	auto TriangularCompressedMatrixExpr<Derived>::inverse() &&
	{ return Detail::TriangularCompressedInverse<Derived>(std::move(derived())); }
}

namespace Eigen::internal
{
	template <typename Operand>
	struct traits<Hoppy::Detail::TriangularCompressedInverse<Operand>>
	    : triangular_node_traits<Hoppy::Detail::TriangularCompressedInverse<Operand>> {};
	template <typename Operand>
	struct evaluator<Hoppy::Detail::TriangularCompressedInverse<Operand>>
	    : evaluator_base<Hoppy::Detail::TriangularCompressedInverse<Operand>>
	{
		using Expression = Hoppy::Detail::TriangularCompressedInverse<Operand>;
		using Scalar = typename Expression::Scalar;
		enum { CoeffReadCost = NumTraits<Scalar>::ReadCost, Flags = 0, Alignment = 0 };
		explicit evaluator(const Expression& expression) : m_Evaluated(expression.toDense()) {}
		Scalar coeff(Eigen::Index row, Eigen::Index column) const
		{ return m_Evaluated.coeff(row, column); }
		typename Expression::DenseObject m_Evaluated;
	};
}
