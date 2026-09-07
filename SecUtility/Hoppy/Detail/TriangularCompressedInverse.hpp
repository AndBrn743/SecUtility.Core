// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/Detail/TriangularCompressedExpressions.hpp>

#include <Eigen/LU>

#include <optional>
#include <type_traits>
#include <utility>

namespace Hoppy::Detail
{
	template <typename Operand>
	class TriangularCompressedInverse
	    : public Hoppy::TriangularCompressedMatrixExpr<TriangularCompressedInverse<Operand>>
	{
		using Source = std::remove_cv_t<std::remove_reference_t<Operand>>;
		using DenseObject = Eigen::Matrix<typename Source::Scalar,
		                                  Source::RowsAtCompileTime, Source::ColsAtCompileTime>;

	public:
		using Scalar = typename Source::Scalar;
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
		{ return evaluated().coeff(row, column); }
		const DenseObject& toDense() const { return evaluated(); }
		template <typename Destination>
		void evalTo(Eigen::MatrixBase<Destination>& destination) const
		{ destination.derived() = evaluated(); }
		PlainObject eval() const { return PlainObject::FromUncheckedDense(evaluated()); }
		template <typename Dense,
		          typename = std::enable_if_t<std::is_base_of_v<Eigen::MatrixBase<Dense>, Dense>>>
		operator Dense() const { return evaluated(); }

	private:
		const DenseObject& evaluated() const
		{
			if (!m_Evaluated) m_Evaluated.emplace(m_Operand.toDense().inverse().eval());
			return *m_Evaluated;
		}
		Operand m_Operand;
		mutable std::optional<DenseObject> m_Evaluated;
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
	    : triangular_compressed_evaluator<Hoppy::Detail::TriangularCompressedInverse<Operand>>
	{
		using Expression = Hoppy::Detail::TriangularCompressedInverse<Operand>;
		using Base = triangular_compressed_evaluator<Expression>;
		explicit evaluator(const Expression& expression) : Base(expression) {}
	};
}
