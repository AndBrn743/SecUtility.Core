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
	struct PositiveOp { template <typename T> auto operator()(T value) const { return value; } };
	struct NegativeOp { template <typename T> auto operator()(const T& value) const { return -value; } };
	struct TransposeOp { template <typename T> auto operator()(const T& value) const { return value.transpose(); } };
	struct ConjugateOp { template <typename T> auto operator()(const T& value) const { return value.conjugate(); } };
	struct AdjointOp { template <typename T> auto operator()(const T& value) const { return value.adjoint(); } };
	struct RealOp { template <typename T> auto operator()(const T& value) const { return value.real(); } };
	struct ImagOp { template <typename T> auto operator()(const T& value) const { return value.imag(); } };
	template <typename TScalar>
	struct CastOp { template <typename T> auto operator()(const T& value) const { return value.template cast<TScalar>(); } };

	template <typename Source, typename Operation, typename Orientation>
	class UnaryBlockExpression;
}

template <typename Source, typename Operation, typename TOrientation>
struct Eigen::internal::traits<Hoppy::Detail::UnaryBlockExpression<Source, Operation, TOrientation>>
{
	using SourceBlock = decltype(std::declval<const Source&>()[Eigen::Index{}]);
	using ResultBlock = decltype(std::declval<Operation>()(std::declval<SourceBlock>()));
	using Scalar = typename remove_all_t<ResultBlock>::Scalar;
	using StorageKind = typename traits<Source>::StorageKind;
	using XprKind = MatrixXpr;
	using StorageIndex = Eigen::Index;
	using BlockPolicy = Hoppy::DenseBlockPolicy;
	using Orientation = TOrientation;
	static constexpr int Flags = 0;
	static constexpr int RowsAtCompileTime = std::is_same_v<StorageKind, Hoppy::Detail::BlockVectorStorage>
	                                                 ? (std::is_same_v<Orientation, Hoppy::Row> ? 1 : Dynamic)
	                                                 : Dynamic;
	static constexpr int ColsAtCompileTime = std::is_same_v<StorageKind, Hoppy::Detail::BlockVectorStorage>
	                                                 ? (std::is_same_v<Orientation, Hoppy::Row> ? Dynamic : 1)
	                                                 : Dynamic;
	static constexpr int MaxRowsAtCompileTime = RowsAtCompileTime;
	static constexpr int MaxColsAtCompileTime = ColsAtCompileTime;
};

namespace Hoppy
{
	template <typename Derived> auto BlockDiagonalMatrixExpr<Derived>::operator+() const& { return Detail::UnaryBlockExpression<Derived, Detail::PositiveOp, Column>(derived()); }
	template <typename Derived> auto BlockDiagonalMatrixExpr<Derived>::operator-() const& { return Detail::UnaryBlockExpression<Derived, Detail::NegativeOp, Column>(derived()); }
	template <typename Derived> auto BlockDiagonalMatrixExpr<Derived>::transpose() const& { return Detail::UnaryBlockExpression<Derived, Detail::TransposeOp, Column>(derived()); }
	template <typename Derived> auto BlockDiagonalMatrixExpr<Derived>::conjugate() const& { return Detail::UnaryBlockExpression<Derived, Detail::ConjugateOp, Column>(derived()); }
	template <typename Derived> auto BlockDiagonalMatrixExpr<Derived>::adjoint() const& { return Detail::UnaryBlockExpression<Derived, Detail::AdjointOp, Column>(derived()); }
	template <typename Derived> auto BlockDiagonalMatrixExpr<Derived>::real() const& { return Detail::UnaryBlockExpression<Derived, Detail::RealOp, Column>(derived()); }
	template <typename Derived> auto BlockDiagonalMatrixExpr<Derived>::imag() const& { return Detail::UnaryBlockExpression<Derived, Detail::ImagOp, Column>(derived()); }
	template <typename Derived> template <typename NewScalar> auto BlockDiagonalMatrixExpr<Derived>::cast() const& { return Detail::UnaryBlockExpression<Derived, Detail::CastOp<NewScalar>, Column>(derived()); }

	template <typename Derived> auto BlockVectorExpr<Derived>::operator+() const& { return Detail::UnaryBlockExpression<Derived, Detail::PositiveOp, Orientation>(derived()); }
	template <typename Derived> auto BlockVectorExpr<Derived>::operator-() const& { return Detail::UnaryBlockExpression<Derived, Detail::NegativeOp, Orientation>(derived()); }
	template <typename Derived> auto BlockVectorExpr<Derived>::transpose() const& { using ResultOrientation = std::conditional_t<std::is_same_v<Orientation, Column>, Row, Column>; return Detail::UnaryBlockExpression<Derived, Detail::TransposeOp, ResultOrientation>(derived()); }
	template <typename Derived> auto BlockVectorExpr<Derived>::conjugate() const& { return Detail::UnaryBlockExpression<Derived, Detail::ConjugateOp, Orientation>(derived()); }
	template <typename Derived> auto BlockVectorExpr<Derived>::adjoint() const& { using ResultOrientation = std::conditional_t<std::is_same_v<Orientation, Column>, Row, Column>; return Detail::UnaryBlockExpression<Derived, Detail::AdjointOp, ResultOrientation>(derived()); }
	template <typename Derived> auto BlockVectorExpr<Derived>::real() const& { return Detail::UnaryBlockExpression<Derived, Detail::RealOp, Orientation>(derived()); }
	template <typename Derived> auto BlockVectorExpr<Derived>::imag() const& { return Detail::UnaryBlockExpression<Derived, Detail::ImagOp, Orientation>(derived()); }
	template <typename Derived> template <typename NewScalar> auto BlockVectorExpr<Derived>::cast() const& { return Detail::UnaryBlockExpression<Derived, Detail::CastOp<NewScalar>, Orientation>(derived()); }
}  // namespace Hoppy

namespace Hoppy::Detail
{
	template <typename Source, typename Operation, typename TOrientation>
	class UnaryBlockExpression
	    : public std::conditional_t<
	              std::is_same_v<typename Eigen::internal::traits<Source>::StorageKind, BlockDiagonalStorage>,
	              BlockDiagonalMatrixExpr<UnaryBlockExpression<Source, Operation, TOrientation>>,
	              BlockVectorExpr<UnaryBlockExpression<Source, Operation, TOrientation>>>
	{
	public:
		using Scalar = typename Eigen::internal::traits<UnaryBlockExpression>::Scalar;
		explicit UnaryBlockExpression(const Source& source) : m_Source(source) {}
		Eigen::Index blockCount() const { return m_Source.blockCount(); }
		Eigen::Index totalDimension() const { return m_Source.totalDimension(); }
		Eigen::Index storedSize() const { return m_Source.storedSize(); }
		Eigen::Index size() const { return rows() * cols(); }
		Eigen::Index rows() const { if constexpr (std::is_same_v<typename Eigen::internal::traits<Source>::StorageKind, BlockVectorStorage>) return std::is_same_v<TOrientation, Row> ? 1 : totalDimension(); else return totalDimension(); }
		Eigen::Index cols() const { if constexpr (std::is_same_v<typename Eigen::internal::traits<Source>::StorageKind, BlockVectorStorage>) return std::is_same_v<TOrientation, Row> ? totalDimension() : 1; else return totalDimension(); }
		Eigen::Index dimensionOfBlock(Eigen::Index i) const { return m_Source.dimensionOfBlock(i); }
		Eigen::Index blockOffset(Eigen::Index i) const { return m_Source.blockOffset(i); }
		Eigen::Index storageOffset(Eigen::Index i) const { return m_Source.storageOffset(i); }
		std::vector<Eigen::Index> blockingInfo() const { return m_Source.blockingInfo(); }
		auto operator[](Eigen::Index i) const { return Operation{}(m_Source[i]); }
		auto operator+() const { return UnaryBlockExpression<UnaryBlockExpression, PositiveOp, TOrientation>(*this); }
		auto operator-() const { return UnaryBlockExpression<UnaryBlockExpression, NegativeOp, TOrientation>(*this); }
		auto conjugate() const { return UnaryBlockExpression<UnaryBlockExpression, ConjugateOp, TOrientation>(*this); }
		auto real() const { return UnaryBlockExpression<UnaryBlockExpression, RealOp, TOrientation>(*this); }
		auto imag() const { return UnaryBlockExpression<UnaryBlockExpression, ImagOp, TOrientation>(*this); }
		template <typename NewScalar>
		auto cast() const { return UnaryBlockExpression<UnaryBlockExpression, CastOp<NewScalar>, TOrientation>(*this); }
		auto transpose() const
		{
			using Kind = typename Eigen::internal::traits<Source>::StorageKind;
			using ResultOrientation = std::conditional_t<
			        std::is_same_v<Kind, BlockVectorStorage>,
			        std::conditional_t<std::is_same_v<TOrientation, Column>, Row, Column>, TOrientation>;
			return UnaryBlockExpression<UnaryBlockExpression, TransposeOp, ResultOrientation>(*this);
		}
		auto adjoint() const
		{
			using Kind = typename Eigen::internal::traits<Source>::StorageKind;
			using ResultOrientation = std::conditional_t<
			        std::is_same_v<Kind, BlockVectorStorage>,
			        std::conditional_t<std::is_same_v<TOrientation, Column>, Row, Column>, TOrientation>;
			return UnaryBlockExpression<UnaryBlockExpression, AdjointOp, ResultOrientation>(*this);
		}

	private:
		typename Eigen::internal::ref_selector<Source>::type m_Source;
	};
}
