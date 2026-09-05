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
	struct InverseOp { template <typename T> auto operator()(const T& value) const { return value.inverse(); } };
	template <typename TScalar>
	struct CastOp { template <typename T> auto operator()(const T& value) const { return value.template cast<TScalar>(); } };

	template <typename Source, typename Operation, typename Orientation>
	class UnaryBlockExpression;
	template <typename Lhs, typename Rhs, typename Operation, typename Orientation>
	class BinaryBlockExpression;
	struct AddOp { template <typename L, typename R> auto operator()(const L& lhs, const R& rhs) const { return lhs + rhs; } };
	struct SubtractOp { template <typename L, typename R> auto operator()(const L& lhs, const R& rhs) const { return lhs - rhs; } };
	struct MultiplyOp { template <typename L, typename R> auto operator()(const L& lhs, const R& rhs) const { return lhs * rhs; } };
	template <typename Lhs, typename Rhs>
	using BlockwiseProduct = BinaryBlockExpression<Lhs, Rhs, MultiplyOp, Column>;
	template <typename Scalar> struct RightMultiplyOp { Scalar Value; template <typename T> auto operator()(const T& value) const { return value * Value; } };
	template <typename Scalar> struct LeftMultiplyOp { Scalar Value; template <typename T> auto operator()(const T& value) const { return Value * value; } };
	template <typename Scalar> struct DivideOp { Scalar Value; template <typename T> auto operator()(const T& value) const { return value / Value; } };
}

namespace Hoppy
{
	template <typename Derived>
	template <typename OtherDerived, typename>
	auto BlockDiagonalMatrixExpr<Derived>::operator+(const BlockDiagonalMatrixExpr<OtherDerived>& other) const&
	{
		return Detail::BinaryBlockExpression<Derived, OtherDerived, Detail::AddOp, Column>(derived(), other.derived());
	}
	template <typename Derived>
	template <typename OtherDerived, typename>
	auto BlockDiagonalMatrixExpr<Derived>::operator-(const BlockDiagonalMatrixExpr<OtherDerived>& other) const&
	{
		return Detail::BinaryBlockExpression<Derived, OtherDerived, Detail::SubtractOp, Column>(derived(), other.derived());
	}
	template <typename Derived>
	template <typename TOtherScalar, typename>
	auto BlockDiagonalMatrixExpr<Derived>::operator*(const TOtherScalar& scalar) const&
	{
		return Detail::UnaryBlockExpression<Derived, Detail::RightMultiplyOp<TOtherScalar>, Column>(derived(), {scalar});
	}
	template <typename Derived>
	template <typename TOtherScalar, typename>
	auto BlockDiagonalMatrixExpr<Derived>::operator/(const TOtherScalar& scalar) const&
	{
		return Detail::UnaryBlockExpression<Derived, Detail::DivideOp<TOtherScalar>, Column>(derived(), {scalar});
	}
	template <typename Derived>
	template <typename OtherDerived, typename>
	auto BlockDiagonalMatrixExpr<Derived>::operator*(const BlockDiagonalMatrixExpr<OtherDerived>& other) const&
	{
		return Detail::BlockwiseProduct<Derived, OtherDerived>(derived(), other.derived());
	}
	template <typename MatrixDerived, typename VectorDerived,
	          typename = std::enable_if_t<std::is_same_v<
	                  typename Eigen::internal::traits<VectorDerived>::Orientation, Column>>,
	          typename = typename Eigen::ScalarBinaryOpTraits<
	                  typename Eigen::internal::traits<MatrixDerived>::Scalar,
	                  typename Eigen::internal::traits<VectorDerived>::Scalar,
	                  Eigen::internal::scalar_product_op<
	                          typename Eigen::internal::traits<MatrixDerived>::Scalar,
	                          typename Eigen::internal::traits<VectorDerived>::Scalar>>::ReturnType>
	auto operator*(const BlockDiagonalMatrixExpr<MatrixDerived>& matrix,
	               const BlockVectorExpr<VectorDerived>& vector)
	{
		using ResultScalar = typename Eigen::ScalarBinaryOpTraits<
		        typename Eigen::internal::traits<MatrixDerived>::Scalar,
		        typename Eigen::internal::traits<VectorDerived>::Scalar,
		        Eigen::internal::scalar_product_op<
		                typename Eigen::internal::traits<MatrixDerived>::Scalar,
		                typename Eigen::internal::traits<VectorDerived>::Scalar>>::ReturnType;
		eigen_assert(matrix.hasSameBlockingAs(vector));
		BlockVector<ResultScalar, Column> result(matrix.blockingInfo());
		for (Eigen::Index index = 0; index < matrix.blockCount(); ++index)
			result[index].noalias() = matrix.derived()[index] * vector.derived()[index];
		return result;
	}
	template <typename MatrixDerived, typename DenseDerived,
	          typename = std::enable_if_t<DenseDerived::ColsAtCompileTime == 1>,
	          typename = typename Eigen::ScalarBinaryOpTraits<
	                  typename Eigen::internal::traits<MatrixDerived>::Scalar,
	                  typename DenseDerived::Scalar,
	                  Eigen::internal::scalar_product_op<
	                          typename Eigen::internal::traits<MatrixDerived>::Scalar,
	                          typename DenseDerived::Scalar>>::ReturnType>
	auto operator*(const BlockDiagonalMatrixExpr<MatrixDerived>& matrix,
	               const Eigen::MatrixBase<DenseDerived>& vector)
	{
		using ResultScalar = typename Eigen::ScalarBinaryOpTraits<
		        typename Eigen::internal::traits<MatrixDerived>::Scalar, typename DenseDerived::Scalar,
		        Eigen::internal::scalar_product_op<
		                typename Eigen::internal::traits<MatrixDerived>::Scalar,
		                typename DenseDerived::Scalar>>::ReturnType;
		eigen_assert(vector.size() == matrix.totalDimension());
		BlockVector<ResultScalar, Column> result(matrix.blockingInfo());
		for (Eigen::Index index = 0; index < matrix.blockCount(); ++index)
		{
			const auto offset = matrix.blockOffset(index);
			const auto dimension = matrix.dimensionOfBlock(index);
			result[index].noalias() = matrix.derived()[index] * vector.derived().middleRows(offset, dimension);
		}
		return result;
	}

	template <typename VectorDerived, typename MatrixDerived,
	          typename = std::enable_if_t<std::is_same_v<
	                  typename Eigen::internal::traits<VectorDerived>::Orientation, Row>>,
	          typename = typename Eigen::ScalarBinaryOpTraits<
	                  typename Eigen::internal::traits<VectorDerived>::Scalar,
	                  typename Eigen::internal::traits<MatrixDerived>::Scalar,
	                  Eigen::internal::scalar_product_op<
		                  typename Eigen::internal::traits<VectorDerived>::Scalar,
		                  typename Eigen::internal::traits<MatrixDerived>::Scalar>>::ReturnType>
	auto operator*(const BlockVectorExpr<VectorDerived>& vector,
	               const BlockDiagonalMatrixExpr<MatrixDerived>& matrix)
	{
		using ResultScalar = typename Eigen::ScalarBinaryOpTraits<
		        typename Eigen::internal::traits<VectorDerived>::Scalar,
		        typename Eigen::internal::traits<MatrixDerived>::Scalar,
		        Eigen::internal::scalar_product_op<
		                typename Eigen::internal::traits<VectorDerived>::Scalar,
		                typename Eigen::internal::traits<MatrixDerived>::Scalar>>::ReturnType;
		eigen_assert(vector.hasSameBlockingAs(matrix));
		BlockVector<ResultScalar, Row> result(matrix.blockingInfo());
		for (Eigen::Index index = 0; index < matrix.blockCount(); ++index)
			result[index].noalias() = vector.derived()[index] * matrix.derived()[index];
		return result;
	}

	template <typename DenseDerived, typename MatrixDerived,
	          typename = std::enable_if_t<DenseDerived::RowsAtCompileTime == 1>,
	          typename = typename Eigen::ScalarBinaryOpTraits<
	                  typename DenseDerived::Scalar,
	                  typename Eigen::internal::traits<MatrixDerived>::Scalar,
	                  Eigen::internal::scalar_product_op<
		                  typename DenseDerived::Scalar,
		                  typename Eigen::internal::traits<MatrixDerived>::Scalar>>::ReturnType>
	auto operator*(const Eigen::MatrixBase<DenseDerived>& vector,
	               const BlockDiagonalMatrixExpr<MatrixDerived>& matrix)
	{
		using ResultScalar = typename Eigen::ScalarBinaryOpTraits<
		        typename DenseDerived::Scalar,
		        typename Eigen::internal::traits<MatrixDerived>::Scalar,
		        Eigen::internal::scalar_product_op<
		                typename DenseDerived::Scalar,
		                typename Eigen::internal::traits<MatrixDerived>::Scalar>>::ReturnType;
		eigen_assert(vector.size() == matrix.totalDimension());
		BlockVector<ResultScalar, Row> result(matrix.blockingInfo());
		for (Eigen::Index index = 0; index < matrix.blockCount(); ++index)
		{
			const auto offset = matrix.blockOffset(index);
			const auto dimension = matrix.dimensionOfBlock(index);
			result[index].noalias() = vector.derived().middleCols(offset, dimension) * matrix.derived()[index];
		}
		return result;
	}

	template <typename Derived>
	template <typename OtherDerived, typename, typename>
	auto BlockVectorExpr<Derived>::operator+(const BlockVectorExpr<OtherDerived>& other) const&
	{
		return Detail::BinaryBlockExpression<Derived, OtherDerived, Detail::AddOp, Orientation>(derived(), other.derived());
	}
	template <typename Derived>
	template <typename OtherDerived, typename, typename>
	auto BlockVectorExpr<Derived>::operator-(const BlockVectorExpr<OtherDerived>& other) const&
	{
		return Detail::BinaryBlockExpression<Derived, OtherDerived, Detail::SubtractOp, Orientation>(derived(), other.derived());
	}
	template <typename Derived>
	template <typename TOtherScalar, typename>
	auto BlockVectorExpr<Derived>::operator*(const TOtherScalar& scalar) const&
	{
		return Detail::UnaryBlockExpression<Derived, Detail::RightMultiplyOp<TOtherScalar>, Orientation>(derived(), {scalar});
	}
	template <typename Derived>
	template <typename TOtherScalar, typename>
	auto BlockVectorExpr<Derived>::operator/(const TOtherScalar& scalar) const&
	{
		return Detail::UnaryBlockExpression<Derived, Detail::DivideOp<TOtherScalar>, Orientation>(derived(), {scalar});
	}

	template <typename Scalar, typename Derived,
	          typename = typename Eigen::ScalarBinaryOpTraits<
	                  Scalar, typename Eigen::internal::traits<Derived>::Scalar,
	                  Eigen::internal::scalar_product_op<
	                          Scalar, typename Eigen::internal::traits<Derived>::Scalar>>::ReturnType>
	auto operator*(const Scalar& scalar, const BlockDiagonalMatrixExpr<Derived>& expression)
	{
		return Detail::UnaryBlockExpression<Derived, Detail::LeftMultiplyOp<Scalar>, Column>(expression.derived(), {scalar});
	}
	template <typename Scalar, typename Derived,
	          typename = typename Eigen::ScalarBinaryOpTraits<
	                  Scalar, typename Eigen::internal::traits<Derived>::Scalar,
	                  Eigen::internal::scalar_product_op<
	                          Scalar, typename Eigen::internal::traits<Derived>::Scalar>>::ReturnType>
	auto operator*(const Scalar& scalar, const BlockVectorExpr<Derived>& expression)
	{
		using Orientation = typename Eigen::internal::traits<Derived>::Orientation;
		return Detail::UnaryBlockExpression<Derived, Detail::LeftMultiplyOp<Scalar>, Orientation>(expression.derived(), {scalar});
	}
}  // namespace Hoppy

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

template <typename Lhs, typename Rhs, typename Operation, typename TOrientation>
struct Eigen::internal::traits<Hoppy::Detail::BinaryBlockExpression<Lhs, Rhs, Operation, TOrientation>>
{
	using LhsBlock = decltype(std::declval<const Lhs&>()[Eigen::Index{}]);
	using RhsBlock = decltype(std::declval<const Rhs&>()[Eigen::Index{}]);
	using ResultBlock = decltype(std::declval<Operation>()(std::declval<LhsBlock>(), std::declval<RhsBlock>()));
	using Scalar = typename remove_all_t<ResultBlock>::Scalar;
	using StorageKind = typename traits<Lhs>::StorageKind;
	using XprKind = MatrixXpr;
	using StorageIndex = Eigen::Index;
	using BlockPolicy = Hoppy::DenseBlockPolicy;
	using Orientation = TOrientation;
	static constexpr int Flags = 0;
	static constexpr int RowsAtCompileTime = traits<Lhs>::RowsAtCompileTime;
	static constexpr int ColsAtCompileTime = traits<Lhs>::ColsAtCompileTime;
	static constexpr int MaxRowsAtCompileTime = RowsAtCompileTime;
	static constexpr int MaxColsAtCompileTime = ColsAtCompileTime;
};

template <typename Matrix, typename Transform, bool Back>
struct Eigen::internal::traits<Hoppy::Detail::CongruenceExpression<Matrix, Transform, Back>>
{
	using Scalar = typename Hoppy::Detail::congruence_result_scalar<
	        typename traits<Matrix>::Scalar, typename traits<Transform>::Scalar>::type;
	using StorageKind = Hoppy::Detail::BlockDiagonalStorage;
	using XprKind = MatrixXpr;
	using StorageIndex = Eigen::Index;
	using BlockPolicy = Hoppy::DenseBlockPolicy;
	using Orientation = Hoppy::Column;
	static constexpr int Flags = 0;
	static constexpr int RowsAtCompileTime = Dynamic;
	static constexpr int ColsAtCompileTime = Dynamic;
	static constexpr int MaxRowsAtCompileTime = Dynamic;
	static constexpr int MaxColsAtCompileTime = Dynamic;
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
	template <typename Derived> template <typename S, typename> auto BlockDiagonalMatrixExpr<Derived>::inverse() const& { return Detail::UnaryBlockExpression<Derived, Detail::InverseOp, Column>(derived()); }
	template <typename Derived> template <typename TransformDerived, typename, typename> auto BlockDiagonalMatrixExpr<Derived>::transformedBy(const BlockDiagonalMatrixExpr<TransformDerived>& transform) const& { return Detail::CongruenceExpression<Derived, TransformDerived, false>(derived(), transform.derived()); }
	template <typename Derived> template <typename TransformDerived, typename, typename> auto BlockDiagonalMatrixExpr<Derived>::backTransformedBy(const BlockDiagonalMatrixExpr<TransformDerived>& transform) const& { return Detail::CongruenceExpression<Derived, TransformDerived, true>(derived(), transform.derived()); }
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
		explicit UnaryBlockExpression(const Source& source, Operation operation = {})
			: m_Source(source), m_Operation(std::move(operation))
		{}
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
		auto operator[](Eigen::Index i) const { return m_Operation(m_Source[i]); }

	private:
		typename Eigen::internal::ref_selector<Source>::type m_Source;
		Operation m_Operation;
	};

	template <typename Lhs, typename Rhs, typename Operation, typename TOrientation>
	class BinaryBlockExpression
	    : public std::conditional_t<
	              std::is_same_v<typename Eigen::internal::traits<Lhs>::StorageKind, BlockDiagonalStorage>,
	              BlockDiagonalMatrixExpr<BinaryBlockExpression<Lhs, Rhs, Operation, TOrientation>>,
	              BlockVectorExpr<BinaryBlockExpression<Lhs, Rhs, Operation, TOrientation>>>
	{
	public:
		using Scalar = typename Eigen::internal::traits<BinaryBlockExpression>::Scalar;
		BinaryBlockExpression(const Lhs& lhs, const Rhs& rhs) : m_Lhs(lhs), m_Rhs(rhs)
		{
			eigen_assert(m_Lhs.blockingInfo() == m_Rhs.blockingInfo());
		}
		Eigen::Index blockCount() const { return m_Lhs.blockCount(); }
		Eigen::Index rows() const { return m_Lhs.rows(); }
		Eigen::Index cols() const { return m_Lhs.cols(); }
		Eigen::Index size() const { return m_Lhs.size(); }
		Eigen::Index totalDimension() const { return m_Lhs.totalDimension(); }
		Eigen::Index storedSize() const { return m_Lhs.storedSize(); }
		Eigen::Index dimensionOfBlock(Eigen::Index i) const { return m_Lhs.dimensionOfBlock(i); }
		Eigen::Index blockOffset(Eigen::Index i) const { return m_Lhs.blockOffset(i); }
		Eigen::Index storageOffset(Eigen::Index i) const { return m_Lhs.storageOffset(i); }
		std::vector<Eigen::Index> blockingInfo() const { return m_Lhs.blockingInfo(); }
		auto operator[](Eigen::Index i) const { return Operation{}(m_Lhs[i], m_Rhs[i]); }

	private:
		typename Eigen::internal::ref_selector<Lhs>::type m_Lhs;
		typename Eigen::internal::ref_selector<Rhs>::type m_Rhs;
	};

	template <typename Matrix, typename Transform, bool Back>
	class CongruenceExpression : public BlockDiagonalMatrixExpr<CongruenceExpression<Matrix, Transform, Back>>
	{
	public:
		using Scalar = typename Eigen::internal::traits<CongruenceExpression>::Scalar;
		CongruenceExpression(const Matrix& matrix, const Transform& transform)
			: m_Matrix(matrix), m_Transform(transform)
		{
			eigen_assert(m_Matrix.blockingInfo() == m_Transform.blockingInfo());
		}
		Eigen::Index blockCount() const { return m_Matrix.blockCount(); }
		Eigen::Index rows() const { return m_Matrix.rows(); }
		Eigen::Index cols() const { return m_Matrix.cols(); }
		Eigen::Index size() const { return m_Matrix.size(); }
		Eigen::Index totalDimension() const { return m_Matrix.totalDimension(); }
		Eigen::Index storedSize() const { return m_Matrix.storedSize(); }
		Eigen::Index dimensionOfBlock(Eigen::Index index) const { return m_Matrix.dimensionOfBlock(index); }
		Eigen::Index blockOffset(Eigen::Index index) const { return m_Matrix.blockOffset(index); }
		Eigen::Index storageOffset(Eigen::Index index) const { return m_Matrix.storageOffset(index); }
		std::vector<Eigen::Index> blockingInfo() const { return m_Matrix.blockingInfo(); }
		auto operator[](Eigen::Index index) const
		{
			if constexpr (Back)
				return m_Transform[index].adjoint() * m_Matrix[index] * m_Transform[index];
			else
				return m_Transform[index] * m_Matrix[index] * m_Transform[index].adjoint();
		}

	private:
		typename Eigen::internal::ref_selector<Matrix>::type m_Matrix;
		typename Eigen::internal::ref_selector<Transform>::type m_Transform;
	};
}
