// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Andy Brown

#pragma once

#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>


/// \file
/// C++17 Eigen-compatible representations of exact ordered low-rank updates.
///
/// A general matrix represents `U * diag(c) * V.adjoint()`. Symmetric matrices represent
/// `U * diag(c) * U.transpose()` (including for complex scalars), while self-adjoint matrices represent
/// `U * diag(r) * U.adjoint()` with real coefficients. For real scalars the symmetric and self-adjoint aliases are
/// the same type.
///
/// Each nonzero inserted coefficient remains a distinct term in insertion order. Exactly zero coefficients are
/// ignored. No tolerance-based pruning, cancellation, compression, dependency detection, term removal, bounded
/// history, or recompression is performed. Those facilities are deferred to a future version.
///
/// Owning matrices use O((rows + cols) * termCount) scalar storage (O(rows * termCount) for single-factor
/// structures). Applying an update to a block with `p` columns costs O((rows + cols) * termCount * p). `toDense()`
/// materializes the full matrix. Products, transforms, scalar operations, and dense/low-rank sums are the supported
/// Eigen interoperability boundary; these types deliberately provide neither general coefficient access nor sparse
/// iterators/compressed storage.
///
/// `LowRankMatrixBase` exposes representation-independent term access. Expressions that can expose their complete
/// coefficient and factor blocks without allocation additionally derive from `BulkLowRankMatrixBase`; returned bulk
/// objects are read-only Eigen expressions and do not necessarily provide contiguous or direct memory access.
///
/// For owning matrices, appending terms invalidates existing views only if a backing buffer reallocates; `reserve`
/// invalidates them only when it grows capacity, and `clear` invalidates term views and references while preserving
/// dimensions and capacity. Automatic growth is geometric in the number of complete terms that all backing buffers
/// can accept. Views returned by a lazy expression inherit the lifetime and invalidation rules of its nested
/// expression. Arguments to `addTerm` and `addTerms` must not alias storage owned by the destination; callers that
/// need to reinsert exposed views must evaluate them first. Invalid dimensions, shapes, indices, and capacities are
/// programming errors checked with `eigen_assert`; violating them when Eigen assertions are disabled is undefined
/// behavior.
///
/// This Eigen-extension header is distributed under MPL-2.0, Eigen's primary license. The surrounding
/// SecUtility.Core headers remain independently licensed as marked in their respective files.
namespace Hoppy
{
	namespace Detail
	{
		struct IdentityLowRankOperation;
		struct TransposeLowRankOperation;
		struct ConjugateLowRankOperation;
		struct AdjointLowRankOperation;
		struct MultiplyLowRankOperation;
		struct DivideLowRankOperation;

		struct SymmetricLowRankStructure;
		struct SelfAdjointLowRankStructure;

		template <typename Scalar_, typename StructurePolicy_, int DimensionAtCompileTime_>
		class SingleFactorLowRankMatrix;

		template <typename Nested_, typename Operation_>
		class LowRankUnaryExpr;

		template <typename Nested_, typename Factor_, typename Operation_, bool FactorOnLeft_>
		class LowRankScalarExpr;

		template <typename DenseNested_, typename LowRankNested_, int DenseSign_, int LowRankSign_>
		class DenseLowRankSumExpr;

		template <typename Matrix>
		struct UnaryExpressionTraits;

		template <typename LeftScalar, typename RightScalar>
		struct IsScalarProductCompatible;

		template <typename LeftScalar, typename RightScalar>
		struct IsScalarQuotientCompatible;

		template <typename Matrix>
		struct IsLowRankExpression;

		template <typename Matrix>
		struct IsDenseExpression;

		template <typename Matrix>
		[[nodiscard]] auto MakeTranspose(Matrix&& matrix);

		template <typename Matrix>
		[[nodiscard]] auto MakeConjugate(Matrix&& matrix);

		template <typename Matrix>
		[[nodiscard]] auto MakeAdjoint(Matrix&& matrix);

		template <bool FactorOnLeft, typename Matrix, typename Factor>
		[[nodiscard]] auto MakeProduct(Matrix&& matrix, Factor&& factor);

		template <typename Matrix, typename Factor>
		[[nodiscard]] auto MakeQuotient(Matrix&& matrix, Factor&& factor);

		template <bool Subtract, typename Left, typename Right>
		[[nodiscard]] auto AddLowRank(const Left& left, const Right& right);

		template <int DenseSign, int LowRankSign, typename Dense, typename LowRank>
		[[nodiscard]] auto MakeDenseLowRankSum(Dense&& dense, LowRank&& lowRank);
	}

	template <typename Derived>
	class LowRankMatrixBase;

	template <typename Derived>
	class BulkLowRankMatrixBase;

	template <typename Scalar_, int RowsAtCompileTime_, int ColsAtCompileTime_>
	class LowRankMatrix;

	struct LowRankStorage
	{
	};

	struct DenseLowRankSumStorage
	{
	};

	struct DenseLowRankSumShape
	{
	};

	template <typename Scalar>
	using LowRankMatrixX = LowRankMatrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

	template <typename Scalar>
	class LowRankDiagonalPreconditioner;

	template <typename Scalar, int DimensionAtCompileTime>
	using LowRankSymmetricMatrix =
	        Detail::SingleFactorLowRankMatrix<Scalar, Detail::SymmetricLowRankStructure, DimensionAtCompileTime>;

	template <typename Scalar>
	using LowRankSymmetricMatrixX = LowRankSymmetricMatrix<Scalar, Eigen::Dynamic>;

	template <typename Scalar, int DimensionAtCompileTime>
	using LowRankSelfAdjointMatrix =
	        Detail::SingleFactorLowRankMatrix<Scalar,
	                                          std::conditional_t<Eigen::NumTraits<Scalar>::IsComplex,
	                                                             Detail::SelfAdjointLowRankStructure,
	                                                             Detail::SymmetricLowRankStructure>,
	                                          DimensionAtCompileTime>;

	template <typename Scalar>
	using LowRankSelfAdjointMatrixX = LowRankSelfAdjointMatrix<Scalar, Eigen::Dynamic>;
}


template <typename Scalar_, int RowsAtCompileTime_, int ColsAtCompileTime_>
struct Eigen::internal::traits<Hoppy::LowRankMatrix<Scalar_, RowsAtCompileTime_, ColsAtCompileTime_>>
{
	using Scalar = Scalar_;
	using StorageKind = Hoppy::LowRankStorage;
	using StorageIndex = Eigen::Index;
	using XprKind = Eigen::MatrixXpr;

	static constexpr int RowsAtCompileTime = RowsAtCompileTime_;
	static constexpr int ColsAtCompileTime = ColsAtCompileTime_;
	static constexpr int MaxRowsAtCompileTime = RowsAtCompileTime_;
	static constexpr int MaxColsAtCompileTime = ColsAtCompileTime_;
	static constexpr int Flags = Eigen::NestByRefBit;
};


template <typename Scalar_, typename StructurePolicy_, int DimensionAtCompileTime_>
struct Eigen::internal::traits<
        Hoppy::Detail::SingleFactorLowRankMatrix<Scalar_, StructurePolicy_, DimensionAtCompileTime_>>
{
	using Scalar = Scalar_;
	using StorageKind = Hoppy::LowRankStorage;
	using StorageIndex = Eigen::Index;
	using XprKind = Eigen::MatrixXpr;

	static constexpr int RowsAtCompileTime = DimensionAtCompileTime_;
	static constexpr int ColsAtCompileTime = DimensionAtCompileTime_;
	static constexpr int MaxRowsAtCompileTime = DimensionAtCompileTime_;
	static constexpr int MaxColsAtCompileTime = DimensionAtCompileTime_;
	static constexpr int Flags = Eigen::NestByRefBit;
};


template <typename Nested_, typename Operation_>
struct Eigen::internal::traits<Hoppy::Detail::LowRankUnaryExpr<Nested_, Operation_>>
{
	using Nested = std::remove_reference_t<Nested_>;
	using Scalar = typename traits<Nested>::Scalar;
	using StorageKind = Hoppy::LowRankStorage;
	using StorageIndex = Eigen::Index;
	using XprKind = Eigen::MatrixXpr;

	static constexpr bool SwapsDimensions = Operation_::SwapsDimensions;
	static constexpr int RowsAtCompileTime =
	        SwapsDimensions ? traits<Nested>::ColsAtCompileTime : traits<Nested>::RowsAtCompileTime;
	static constexpr int ColsAtCompileTime =
	        SwapsDimensions ? traits<Nested>::RowsAtCompileTime : traits<Nested>::ColsAtCompileTime;
	static constexpr int MaxRowsAtCompileTime =
	        SwapsDimensions ? traits<Nested>::MaxColsAtCompileTime : traits<Nested>::MaxRowsAtCompileTime;
	static constexpr int MaxColsAtCompileTime =
	        SwapsDimensions ? traits<Nested>::MaxRowsAtCompileTime : traits<Nested>::MaxColsAtCompileTime;
	static constexpr int Flags = 0;
};


template <typename Nested_, typename Factor_, typename Operation_, bool FactorOnLeft_>
struct Eigen::internal::traits<Hoppy::Detail::LowRankScalarExpr<Nested_, Factor_, Operation_, FactorOnLeft_>>
{
	using Nested = std::remove_reference_t<Nested_>;
	using NestedScalar = typename traits<Nested>::Scalar;
	using Scalar = typename Operation_::template ResultScalar<NestedScalar, Factor_, FactorOnLeft_>;
	using StorageKind = Hoppy::LowRankStorage;
	using StorageIndex = Eigen::Index;
	using XprKind = Eigen::MatrixXpr;

	static constexpr int RowsAtCompileTime = traits<Nested>::RowsAtCompileTime;
	static constexpr int ColsAtCompileTime = traits<Nested>::ColsAtCompileTime;
	static constexpr int MaxRowsAtCompileTime = traits<Nested>::MaxRowsAtCompileTime;
	static constexpr int MaxColsAtCompileTime = traits<Nested>::MaxColsAtCompileTime;
	static constexpr int Flags = 0;
};


template <typename DenseNested_, typename LowRankNested_, int DenseSign_, int LowRankSign_>
struct Eigen::internal::traits<
        Hoppy::Detail::DenseLowRankSumExpr<DenseNested_, LowRankNested_, DenseSign_, LowRankSign_>>
{
	using DenseNested = std::remove_reference_t<DenseNested_>;
	using LowRankNested = std::remove_reference_t<LowRankNested_>;
	using DenseScalar = typename traits<DenseNested>::Scalar;
	using LowRankScalar = typename traits<LowRankNested>::Scalar;
	using Scalar = typename Eigen::ScalarBinaryOpTraits<
	        DenseScalar,
	        LowRankScalar,
	        Eigen::internal::scalar_sum_op<DenseScalar, LowRankScalar>>::ReturnType;
	using StorageKind = Hoppy::DenseLowRankSumStorage;
	using StorageIndex = Eigen::Index;
	using XprKind = Eigen::MatrixXpr;

	static constexpr int RowsAtCompileTime = Eigen::internal::min_size_prefer_fixed(
	        traits<DenseNested>::RowsAtCompileTime, traits<LowRankNested>::RowsAtCompileTime);
	static constexpr int ColsAtCompileTime = Eigen::internal::min_size_prefer_fixed(
	        traits<DenseNested>::ColsAtCompileTime, traits<LowRankNested>::ColsAtCompileTime);
	static constexpr int MaxRowsAtCompileTime = Eigen::internal::min_size_prefer_fixed(
	        traits<DenseNested>::MaxRowsAtCompileTime, traits<LowRankNested>::MaxRowsAtCompileTime);
	static constexpr int MaxColsAtCompileTime = Eigen::internal::min_size_prefer_fixed(
	        traits<DenseNested>::MaxColsAtCompileTime, traits<LowRankNested>::MaxColsAtCompileTime);
	static constexpr int Flags = 0;
};


template <typename Derived>
struct Eigen::internal::traits<Hoppy::LowRankMatrixBase<Derived>> : traits<Derived>
{
	static constexpr int Flags = traits<Derived>::Flags | Eigen::NestByRefBit;
};


template <>
struct Eigen::internal::storage_kind_to_shape<Hoppy::LowRankStorage>
{
	using Shape = Eigen::SparseShape;
};


template <>
struct Eigen::internal::storage_kind_to_shape<Hoppy::DenseLowRankSumStorage>
{
	using Shape = Hoppy::DenseLowRankSumShape;
};


template <>
struct Eigen::internal::AssignmentKind<Eigen::DenseShape, Hoppy::DenseLowRankSumShape>
{
	using Kind = Eigen::internal::EigenBase2EigenBase;
};


template <typename Derived, typename Rhs, int ProductType>
struct Eigen::internal::
        generic_product_impl<Hoppy::LowRankMatrixBase<Derived>, Rhs, Eigen::SparseShape, Eigen::DenseShape, ProductType>
    : generic_product_impl_base<Hoppy::LowRankMatrixBase<Derived>,
                                Rhs,
                                generic_product_impl<Hoppy::LowRankMatrixBase<Derived>, Rhs>>
{
	template <typename Dest>
	static void scaleAndAddTo(
	        Dest& destination,
	        const Hoppy::LowRankMatrixBase<Derived>& lowRank,
	        const Rhs& rhs,
	        const std::common_type_t<typename traits<Derived>::Scalar, typename traits<Rhs>::Scalar>& alpha)
	{
		lowRank.derived().scaleAndAddToDense(destination, rhs, alpha);
	}
};


template <typename DenseNested, typename LowRankNested, int DenseSign, int LowRankSign, typename Rhs, int ProductType>
struct Eigen::internal::generic_product_impl<
        Hoppy::Detail::DenseLowRankSumExpr<DenseNested, LowRankNested, DenseSign, LowRankSign>,
        Rhs,
        Hoppy::DenseLowRankSumShape,
        Eigen::DenseShape,
        ProductType>
    : generic_product_impl_base<
              Hoppy::Detail::DenseLowRankSumExpr<DenseNested, LowRankNested, DenseSign, LowRankSign>,
              Rhs,
              generic_product_impl<
                      Hoppy::Detail::DenseLowRankSumExpr<DenseNested, LowRankNested, DenseSign, LowRankSign>,
                      Rhs>>
{
	using Expression = Hoppy::Detail::DenseLowRankSumExpr<DenseNested, LowRankNested, DenseSign, LowRankSign>;

	template <typename Dest>
	static void scaleAndAddTo(
	        Dest& destination,
	        const Expression& expression,
	        const Rhs& rhs,
	        const std::common_type_t<typename traits<Expression>::Scalar, typename traits<Rhs>::Scalar>& alpha)
	{
		using ProductScalar = std::common_type_t<typename traits<Expression>::Scalar, typename traits<Rhs>::Scalar>;
		destination.noalias() += ProductScalar{DenseSign} * alpha * expression.dense().template cast<ProductScalar>()
		                         * rhs.template cast<ProductScalar>();
		expression.lowRank().scaleAndAddToDense(destination, rhs, ProductScalar{LowRankSign} * alpha);
	}
};


template <typename Derived>
class Hoppy::LowRankMatrixBase : public Eigen::EigenBase<Derived>
{
public:
	using Scalar = typename Eigen::internal::traits<Derived>::Scalar;
	using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
	using StorageIndex = typename Eigen::internal::traits<Derived>::StorageIndex;

	static constexpr int RowsAtCompileTime = Eigen::internal::traits<Derived>::RowsAtCompileTime;
	static constexpr int ColsAtCompileTime = Eigen::internal::traits<Derived>::ColsAtCompileTime;
	static constexpr int MaxRowsAtCompileTime = Eigen::internal::traits<Derived>::MaxRowsAtCompileTime;
	static constexpr int MaxColsAtCompileTime = Eigen::internal::traits<Derived>::MaxColsAtCompileTime;
	static constexpr int IsRowMajor = false;
	static constexpr int Flags = Eigen::internal::traits<Derived>::Flags;

	[[nodiscard]] constexpr Eigen::Index rows() const noexcept
	{
		return asDerived().rowsImpl();
	}

	[[nodiscard]] constexpr Eigen::Index cols() const noexcept
	{
		return asDerived().colsImpl();
	}

	[[nodiscard]] Eigen::Index termCount() const noexcept
	{
		return asDerived().termCountImpl();
	}

	[[nodiscard]] decltype(auto) coefficientOfTerm(const Eigen::Index index) const
	{
		return asDerived().coefficientOfTermImpl(index);
	}

	[[nodiscard]] decltype(auto) leftVectorOfTerm(const Eigen::Index index) const
	{
		return asDerived().leftVectorOfTermImpl(index);
	}

	[[nodiscard]] decltype(auto) rightVectorOfTerm(const Eigen::Index index) const
	{
		return asDerived().rightVectorOfTermImpl(index);
	}

	[[nodiscard]] auto term(const Eigen::Index index) const
	{
		struct Term
		{
			decltype(std::declval<const LowRankMatrixBase&>().leftVectorOfTerm(index)) leftVector;
			decltype(std::declval<const LowRankMatrixBase&>().coefficientOfTerm(index)) coefficient;
			decltype(std::declval<const LowRankMatrixBase&>().rightVectorOfTerm(index)) rightVector;
		};

		return Term{leftVectorOfTerm(index), coefficientOfTerm(index), rightVectorOfTerm(index)};
	}

	template <typename Rhs>
	[[nodiscard]] auto operator*(const Eigen::MatrixBase<Rhs>& rhs) const
	{
		return Eigen::Product<LowRankMatrixBase, Rhs, Eigen::DefaultProduct>{*this, rhs.derived()};
	}

	template <typename Destination, typename Rhs, typename Alpha>
	void scaleAndAddToDense(Destination& destination, const Rhs& rhs, const Alpha& alpha) const
	{
		using ProductScalar = std::common_type_t<Scalar, typename Rhs::Scalar, Alpha>;
		for (Eigen::Index index = 0; index < termCount(); index++)
		{
			const Eigen::Matrix<ProductScalar, 1, Rhs::ColsAtCompileTime> projected =
			        rightVectorOfTerm(index).template cast<ProductScalar>().adjoint()
			        * rhs.template cast<ProductScalar>();
			destination.noalias() += alpha * static_cast<ProductScalar>(coefficientOfTerm(index))
			                         * leftVectorOfTerm(index).template cast<ProductScalar>() * projected;
		}
	}

	template <typename Destination, typename Alpha>
	void addScaledToDense(Destination& destination, const Alpha& alpha) const
	{
		using DestinationScalar = typename Destination::Scalar;
		for (Eigen::Index index = 0; index < termCount(); index++)
		{
			destination.noalias() += static_cast<DestinationScalar>(alpha)
			                         * static_cast<DestinationScalar>(coefficientOfTerm(index))
			                         * leftVectorOfTerm(index).template cast<DestinationScalar>()
			                         * rightVectorOfTerm(index).template cast<DestinationScalar>().adjoint();
		}
	}

	template <typename Factor,
	          std::enable_if_t<Detail::IsScalarProductCompatible<Scalar, std::decay_t<Factor>>::value, int> = 0>
	[[nodiscard]] auto operator*(Factor&& factor) const&
	{
		return Detail::MakeProduct<false>(asDerived(), std::forward<Factor>(factor));
	}

	template <typename Factor,
	          std::enable_if_t<Detail::IsScalarProductCompatible<Scalar, std::decay_t<Factor>>::value, int> = 0>
	[[nodiscard]] auto operator*(Factor&& factor) &&
	{
		return Detail::MakeProduct<false>(std::move(asDerived()), std::forward<Factor>(factor));
	}

	template <typename Factor,
	          std::enable_if_t<Detail::IsScalarProductCompatible<Scalar, std::decay_t<Factor>>::value, int> = 0>
	[[nodiscard]] auto operator*(Factor&& factor) const&&
	{
		return Detail::MakeProduct<false>(Derived{asDerived()}, std::forward<Factor>(factor));
	}

	template <typename Factor,
	          std::enable_if_t<Detail::IsScalarQuotientCompatible<Scalar, std::decay_t<Factor>>::value, int> = 0>
	/// Division deliberately follows the underlying scalar's division-by-zero behavior.
	[[nodiscard]] auto operator/(Factor&& factor) const&
	{
		return Detail::MakeQuotient(asDerived(), std::forward<Factor>(factor));
	}

	template <typename Factor,
	          std::enable_if_t<Detail::IsScalarQuotientCompatible<Scalar, std::decay_t<Factor>>::value, int> = 0>
	[[nodiscard]] auto operator/(Factor&& factor) &&
	{
		return Detail::MakeQuotient(std::move(asDerived()), std::forward<Factor>(factor));
	}

	template <typename Factor,
	          std::enable_if_t<Detail::IsScalarQuotientCompatible<Scalar, std::decay_t<Factor>>::value, int> = 0>
	[[nodiscard]] auto operator/(Factor&& factor) const&&
	{
		return Detail::MakeQuotient(Derived{asDerived()}, std::forward<Factor>(factor));
	}

	template <typename Factor,
	          std::enable_if_t<Detail::IsScalarProductCompatible<std::decay_t<Factor>, Scalar>::value, int> = 0>
	friend auto operator*(Factor&& factor, const LowRankMatrixBase& matrix)
	{
		return Detail::MakeProduct<true>(matrix.asDerived(), std::forward<Factor>(factor));
	}

	template <typename Factor,
	          std::enable_if_t<Detail::IsScalarProductCompatible<std::decay_t<Factor>, Scalar>::value, int> = 0>
	friend auto operator*(Factor&& factor, LowRankMatrixBase&& matrix)
	{
		return Detail::MakeProduct<true>(std::move(matrix.asDerived()), std::forward<Factor>(factor));
	}

	template <typename Other,
	          std::enable_if_t<Detail::IsLowRankExpression<std::decay_t<Other>>::value
	                                   && std::is_same_v<Scalar, typename std::decay_t<Other>::Scalar>,
	                           int> = 0>
	/// Eagerly creates an owning low-rank matrix containing both expansions in order.
	/// A lazy low-rank sum expression may replace this result strategy in a future version.
	[[nodiscard]] auto operator+(const Other& other) const
	{
		return Detail::AddLowRank<false>(asDerived(), other);
	}

	template <typename Other,
	          std::enable_if_t<Detail::IsLowRankExpression<std::decay_t<Other>>::value
	                                   && std::is_same_v<Scalar, typename std::decay_t<Other>::Scalar>,
	                           int> = 0>
	/// Eagerly creates an owning low-rank matrix, negating and appending the right expansion.
	/// A lazy low-rank difference expression may replace this result strategy in a future version.
	[[nodiscard]] auto operator-(const Other& other) const
	{
		return Detail::AddLowRank<true>(asDerived(), other);
	}

	template <typename Dense, std::enable_if_t<Detail::IsDenseExpression<std::decay_t<Dense>>::value, int> = 0>
	[[nodiscard]] auto operator+(Dense&& dense) const&
	{
		return Detail::MakeDenseLowRankSum<1, 1>(std::forward<Dense>(dense), asDerived());
	}

	template <typename Dense, std::enable_if_t<Detail::IsDenseExpression<std::decay_t<Dense>>::value, int> = 0>
	[[nodiscard]] auto operator+(Dense&& dense) &&
	{
		return Detail::MakeDenseLowRankSum<1, 1>(std::forward<Dense>(dense), std::move(asDerived()));
	}

	template <typename Dense, std::enable_if_t<Detail::IsDenseExpression<std::decay_t<Dense>>::value, int> = 0>
	[[nodiscard]] auto operator-(Dense&& dense) const&
	{
		return Detail::MakeDenseLowRankSum<-1, 1>(std::forward<Dense>(dense), asDerived());
	}

	template <typename Dense, std::enable_if_t<Detail::IsDenseExpression<std::decay_t<Dense>>::value, int> = 0>
	[[nodiscard]] auto operator-(Dense&& dense) &&
	{
		return Detail::MakeDenseLowRankSum<-1, 1>(std::forward<Dense>(dense), std::move(asDerived()));
	}

	template <typename Dense, std::enable_if_t<Detail::IsDenseExpression<std::decay_t<Dense>>::value, int> = 0>
	friend auto operator+(Dense&& dense, const LowRankMatrixBase& lowRank)
	{
		return Detail::MakeDenseLowRankSum<1, 1>(std::forward<Dense>(dense), lowRank.asDerived());
	}

	template <typename Dense, std::enable_if_t<Detail::IsDenseExpression<std::decay_t<Dense>>::value, int> = 0>
	friend auto operator+(Dense&& dense, LowRankMatrixBase&& lowRank)
	{
		return Detail::MakeDenseLowRankSum<1, 1>(std::forward<Dense>(dense), std::move(lowRank.asDerived()));
	}

	template <typename Dense, std::enable_if_t<Detail::IsDenseExpression<std::decay_t<Dense>>::value, int> = 0>
	friend auto operator-(Dense&& dense, const LowRankMatrixBase& lowRank)
	{
		return Detail::MakeDenseLowRankSum<1, -1>(std::forward<Dense>(dense), lowRank.asDerived());
	}

	template <typename Dense, std::enable_if_t<Detail::IsDenseExpression<std::decay_t<Dense>>::value, int> = 0>
	friend auto operator-(Dense&& dense, LowRankMatrixBase&& lowRank)
	{
		return Detail::MakeDenseLowRankSum<1, -1>(std::forward<Dense>(dense), std::move(lowRank.asDerived()));
	}

	[[nodiscard]] Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime> toDense() const
	{
		using DenseMatrix = Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>;
		DenseMatrix result = DenseMatrix::Zero(rows(), cols());
		asDerived().addScaledToDense(result, Scalar{1});
		return result;
	}

	[[nodiscard]] Eigen::Matrix<Scalar, 1, ColsAtCompileTime> row(const Eigen::Index index) const
	{
		eigen_assert(index >= 0 && index < rows() && "Low-rank matrix row index is out of range");
		using Row = Eigen::Matrix<Scalar, 1, ColsAtCompileTime>;
		Row result = Row::Zero(1, cols());
		for (Eigen::Index termIndex = 0; termIndex < termCount(); termIndex++)
		{
			result.noalias() += static_cast<Scalar>(coefficientOfTerm(termIndex)) * leftVectorOfTerm(termIndex)[index]
			                    * rightVectorOfTerm(termIndex).adjoint();
		}
		return result;
	}

	[[nodiscard]] Eigen::Matrix<Scalar, RowsAtCompileTime, 1> col(const Eigen::Index index) const
	{
		eigen_assert(index >= 0 && index < cols() && "Low-rank matrix column index is out of range");
		using Column = Eigen::Matrix<Scalar, RowsAtCompileTime, 1>;
		Column result = Column::Zero(rows());
		for (Eigen::Index termIndex = 0; termIndex < termCount(); termIndex++)
		{
			result.noalias() += static_cast<Scalar>(coefficientOfTerm(termIndex))
			                    * Eigen::numext::conj(rightVectorOfTerm(termIndex)[index])
			                    * leftVectorOfTerm(termIndex);
		}
		return result;
	}

	[[nodiscard]] Eigen::Vector<Scalar, Eigen::internal::min_size_prefer_dynamic(RowsAtCompileTime, ColsAtCompileTime)>
	diagonal() const
	{
		constexpr int DiagonalSize = Eigen::internal::min_size_prefer_dynamic(RowsAtCompileTime, ColsAtCompileTime);
		using DiagonalVector = Eigen::Vector<Scalar, DiagonalSize>;
		const Eigen::Index size = (std::min)(rows(), cols());
		DiagonalVector result = DiagonalVector::Zero(size);
		for (Eigen::Index termIndex = 0; termIndex < termCount(); termIndex++)
		{
			result.array() += static_cast<Scalar>(coefficientOfTerm(termIndex))
			                  * leftVectorOfTerm(termIndex).head(size).array()
			                  * rightVectorOfTerm(termIndex).head(size).conjugate().array();
		}
		return result;
	}

	[[nodiscard]] RealScalar squaredNorm() const
	{
		if (termCount() == 0)
		{
			return RealScalar{};
		}

		Scalar sum{};
		for (Eigen::Index leftIndex = 0; leftIndex < termCount(); leftIndex++)
		{
			for (Eigen::Index rightIndex = 0; rightIndex < termCount(); rightIndex++)
			{
				sum += Eigen::numext::conj(static_cast<Scalar>(coefficientOfTerm(leftIndex)))
				       * static_cast<Scalar>(coefficientOfTerm(rightIndex))
				       * leftVectorOfTerm(leftIndex).dot(leftVectorOfTerm(rightIndex))
				       * Eigen::numext::conj(rightVectorOfTerm(leftIndex).dot(rightVectorOfTerm(rightIndex)));
			}
		}
		const RealScalar result = Eigen::numext::real(sum);
		return (std::max)(RealScalar{0}, result);
	}

	[[nodiscard]] RealScalar norm() const
	{
		return std::sqrt(squaredNorm());
	}

	[[nodiscard]] auto transpose() const&
	{
		return Detail::MakeTranspose(asDerived());
	}

	[[nodiscard]] auto transpose() &&
	{
		return Detail::MakeTranspose(std::move(asDerived()));
	}

	[[nodiscard]] auto transpose() const&&
	{
		return Detail::MakeTranspose(Derived{asDerived()});
	}

	[[nodiscard]] auto conjugate() const&
	{
		return Detail::MakeConjugate(asDerived());
	}

	[[nodiscard]] auto conjugate() &&
	{
		return Detail::MakeConjugate(std::move(asDerived()));
	}

	[[nodiscard]] auto conjugate() const&&
	{
		return Detail::MakeConjugate(Derived{asDerived()});
	}

	[[nodiscard]] auto adjoint() const&
	{
		return Detail::MakeAdjoint(asDerived());
	}

	[[nodiscard]] auto adjoint() &&
	{
		return Detail::MakeAdjoint(std::move(asDerived()));
	}

	[[nodiscard]] auto adjoint() const&&
	{
		return Detail::MakeAdjoint(Derived{asDerived()});
	}

protected:
	constexpr LowRankMatrixBase() noexcept = default;
	LowRankMatrixBase(const LowRankMatrixBase&) = default;
	LowRankMatrixBase(LowRankMatrixBase&&) noexcept = default;
	LowRankMatrixBase& operator=(const LowRankMatrixBase&) = default;
	LowRankMatrixBase& operator=(LowRankMatrixBase&&) noexcept = default;
	~LowRankMatrixBase() = default;

private:
	[[nodiscard]] constexpr const Derived& asDerived() const noexcept
	{
		return static_cast<const Derived&>(*this);
	}

	[[nodiscard]] constexpr Derived& asDerived() noexcept
	{
		return static_cast<Derived&>(*this);
	}
};


/// Base for low-rank expressions that expose their complete coefficient and factor blocks without allocation.
/// `coefficients()`, `leftVectors()`, and `rightVectors()` return read-only Eigen objects representing all terms in
/// insertion order. The returned objects may be maps or lazy expressions; this interface promises neither ownership
/// nor contiguous or direct memory access. Their lifetimes and invalidation rules follow the derived expression.
template <typename Derived>
class Hoppy::BulkLowRankMatrixBase : public LowRankMatrixBase<Derived>
{
	using Base = LowRankMatrixBase<Derived>;

public:
	using Base::cols;
	using Base::rows;
	using Base::termCount;
	using typename Base::RealScalar;
	using typename Base::Scalar;

	[[nodiscard]] decltype(auto) coefficients() const noexcept
	{
		return asDerived().coefficientsImpl();
	}

	[[nodiscard]] decltype(auto) leftVectors() const noexcept
	{
		return asDerived().leftVectorsImpl();
	}

	[[nodiscard]] decltype(auto) rightVectors() const noexcept
	{
		return asDerived().rightVectorsImpl();
	}

	template <typename Destination, typename Rhs, typename Alpha>
	void scaleAndAddToDense(Destination& destination, const Rhs& rhs, const Alpha& alpha) const
	{
		using ProductScalar = std::common_type_t<Scalar, typename Rhs::Scalar, Alpha>;
		using Intermediate = Eigen::Matrix<ProductScalar, Eigen::Dynamic, Rhs::ColsAtCompileTime>;
		const Intermediate projected =
		        rightVectors().template cast<ProductScalar>().adjoint() * rhs.template cast<ProductScalar>();
		const Intermediate weighted = coefficients().template cast<ProductScalar>().asDiagonal() * projected;
		destination.noalias() += alpha * leftVectors().template cast<ProductScalar>() * weighted;
	}

	template <typename Destination, typename Alpha>
	void addScaledToDense(Destination& destination, const Alpha& alpha) const
	{
		using DestinationScalar = typename Destination::Scalar;
		destination.noalias() += static_cast<DestinationScalar>(alpha)
		                         * leftVectors().template cast<DestinationScalar>()
		                         * coefficients().template cast<DestinationScalar>().asDiagonal()
		                         * rightVectors().template cast<DestinationScalar>().adjoint();
	}

protected:
	constexpr BulkLowRankMatrixBase() noexcept = default;
	BulkLowRankMatrixBase(const BulkLowRankMatrixBase&) = default;
	BulkLowRankMatrixBase(BulkLowRankMatrixBase&&) noexcept = default;
	BulkLowRankMatrixBase& operator=(const BulkLowRankMatrixBase&) = default;
	BulkLowRankMatrixBase& operator=(BulkLowRankMatrixBase&&) noexcept = default;
	~BulkLowRankMatrixBase() = default;

private:
	[[nodiscard]] constexpr const Derived& asDerived() const noexcept
	{
		return static_cast<const Derived&>(*this);
	}
};


namespace Hoppy::Detail
{
	struct IdentityLowRankOperation
	{
		static constexpr bool SwapsDimensions = false;

		template <typename Matrix>
		[[nodiscard]] static decltype(auto) Coefficients(const Matrix& matrix)
		{
			return matrix.coefficients();
		}

		template <typename Matrix>
		[[nodiscard]] static decltype(auto) LeftVectors(const Matrix& matrix)
		{
			return matrix.leftVectors();
		}

		template <typename Matrix>
		[[nodiscard]] static decltype(auto) RightVectors(const Matrix& matrix)
		{
			return matrix.rightVectors();
		}

		template <typename Matrix>
		[[nodiscard]] static decltype(auto) CoefficientOfTerm(const Matrix& matrix, const Eigen::Index index)
		{
			return matrix.coefficientOfTerm(index);
		}

		template <typename Matrix>
		[[nodiscard]] static decltype(auto) LeftVectorOfTerm(const Matrix& matrix, const Eigen::Index index)
		{
			return matrix.leftVectorOfTerm(index);
		}

		template <typename Matrix>
		[[nodiscard]] static decltype(auto) RightVectorOfTerm(const Matrix& matrix, const Eigen::Index index)
		{
			return matrix.rightVectorOfTerm(index);
		}
	};

	struct TransposeLowRankOperation
	{
		static constexpr bool SwapsDimensions = true;

		template <typename Matrix>
		[[nodiscard]] static decltype(auto) Coefficients(const Matrix& matrix)
		{
			return matrix.coefficients();
		}

		template <typename Matrix>
		[[nodiscard]] static auto LeftVectors(const Matrix& matrix)
		{
			return matrix.rightVectors().conjugate();
		}

		template <typename Matrix>
		[[nodiscard]] static auto RightVectors(const Matrix& matrix)
		{
			return matrix.leftVectors().conjugate();
		}

		template <typename Matrix>
		[[nodiscard]] static decltype(auto) CoefficientOfTerm(const Matrix& matrix, const Eigen::Index index)
		{
			return matrix.coefficientOfTerm(index);
		}

		template <typename Matrix>
		[[nodiscard]] static auto LeftVectorOfTerm(const Matrix& matrix, const Eigen::Index index)
		{
			return matrix.rightVectorOfTerm(index).conjugate();
		}

		template <typename Matrix>
		[[nodiscard]] static auto RightVectorOfTerm(const Matrix& matrix, const Eigen::Index index)
		{
			return matrix.leftVectorOfTerm(index).conjugate();
		}
	};

	struct ConjugateLowRankOperation
	{
		static constexpr bool SwapsDimensions = false;

		template <typename Matrix>
		[[nodiscard]] static auto Coefficients(const Matrix& matrix)
		{
			return matrix.coefficients().conjugate();
		}

		template <typename Matrix>
		[[nodiscard]] static auto LeftVectors(const Matrix& matrix)
		{
			return matrix.leftVectors().conjugate();
		}

		template <typename Matrix>
		[[nodiscard]] static auto RightVectors(const Matrix& matrix)
		{
			return matrix.rightVectors().conjugate();
		}

		template <typename Matrix>
		[[nodiscard]] static auto CoefficientOfTerm(const Matrix& matrix, const Eigen::Index index)
		{
			return Eigen::numext::conj(matrix.coefficientOfTerm(index));
		}

		template <typename Matrix>
		[[nodiscard]] static auto LeftVectorOfTerm(const Matrix& matrix, const Eigen::Index index)
		{
			return matrix.leftVectorOfTerm(index).conjugate();
		}

		template <typename Matrix>
		[[nodiscard]] static auto RightVectorOfTerm(const Matrix& matrix, const Eigen::Index index)
		{
			return matrix.rightVectorOfTerm(index).conjugate();
		}
	};

	struct AdjointLowRankOperation
	{
		static constexpr bool SwapsDimensions = true;

		template <typename Matrix>
		[[nodiscard]] static auto Coefficients(const Matrix& matrix)
		{
			return matrix.coefficients().conjugate();
		}

		template <typename Matrix>
		[[nodiscard]] static decltype(auto) LeftVectors(const Matrix& matrix)
		{
			return matrix.rightVectors();
		}

		template <typename Matrix>
		[[nodiscard]] static decltype(auto) RightVectors(const Matrix& matrix)
		{
			return matrix.leftVectors();
		}

		template <typename Matrix>
		[[nodiscard]] static auto CoefficientOfTerm(const Matrix& matrix, const Eigen::Index index)
		{
			return Eigen::numext::conj(matrix.coefficientOfTerm(index));
		}

		template <typename Matrix>
		[[nodiscard]] static decltype(auto) LeftVectorOfTerm(const Matrix& matrix, const Eigen::Index index)
		{
			return matrix.rightVectorOfTerm(index);
		}

		template <typename Matrix>
		[[nodiscard]] static decltype(auto) RightVectorOfTerm(const Matrix& matrix, const Eigen::Index index)
		{
			return matrix.leftVectorOfTerm(index);
		}
	};

	template <typename LeftScalar, typename RightScalar>
	struct IsScalarProductCompatible
	    : Eigen::internal::has_ReturnType<
	              Eigen::ScalarBinaryOpTraits<LeftScalar,
	                                          RightScalar,
	                                          Eigen::internal::scalar_product_op<LeftScalar, RightScalar>>>
	{
	};

	template <typename LeftScalar, typename RightScalar>
	struct IsScalarQuotientCompatible
	    : Eigen::internal::has_ReturnType<
	              Eigen::ScalarBinaryOpTraits<LeftScalar,
	                                          RightScalar,
	                                          Eigen::internal::scalar_quotient_op<LeftScalar, RightScalar>>>
	{
	};

	struct MultiplyLowRankOperation
	{
		template <typename MatrixScalar, typename Factor, bool FactorOnLeft>
		using ResultScalar =
		        std::conditional_t<FactorOnLeft,
		                           typename Eigen::ScalarBinaryOpTraits<
		                                   Factor,
		                                   MatrixScalar,
		                                   Eigen::internal::scalar_product_op<Factor, MatrixScalar>>::ReturnType,
		                           typename Eigen::ScalarBinaryOpTraits<
		                                   MatrixScalar,
		                                   Factor,
		                                   Eigen::internal::scalar_product_op<MatrixScalar, Factor>>::ReturnType>;

		template <typename Result, bool IsFactorOnLeft, typename Coefficients, typename Factor>
		[[nodiscard]] static auto CoefficientsResult(const Coefficients& coefficients, const Factor& factor)
		{
			if constexpr (IsFactorOnLeft)
			{
				return Result{factor} * coefficients.template cast<Result>();
			}
			else
			{
				return coefficients.template cast<Result>() * Result{factor};
			}
		}

		template <typename Result, bool IsFactorOnLeft, typename Coefficient, typename Factor>
		[[nodiscard]] static Result CoefficientResult(const Coefficient& coefficient, const Factor& factor)
		{
			if constexpr (IsFactorOnLeft)
			{
				return Result{factor} * Result{coefficient};
			}
			else
			{
				return Result{coefficient} * Result{factor};
			}
		}
	};

	struct DivideLowRankOperation
	{
		template <typename MatrixScalar, typename Factor, bool>
		using ResultScalar = typename Eigen::ScalarBinaryOpTraits<
		        MatrixScalar,
		        Factor,
		        Eigen::internal::scalar_quotient_op<MatrixScalar, Factor>>::ReturnType;

		template <typename Result, bool, typename Coefficients, typename Factor>
		[[nodiscard]] static auto CoefficientsResult(const Coefficients& coefficients, const Factor& factor)
		{
			return coefficients.template cast<Result>() / Result{factor};
		}

		template <typename Result, bool, typename Coefficient, typename Factor>
		[[nodiscard]] static Result CoefficientResult(const Coefficient& coefficient, const Factor& factor)
		{
			return Result{coefficient} / Result{factor};
		}
	};

	struct SymmetricLowRankStructure
	{
		template <typename Scalar>
		using CoefficientScalar = Scalar;
		using TransposeOperation = IdentityLowRankOperation;
		using ConjugateOperation = ConjugateLowRankOperation;
		using AdjointOperation = ConjugateLowRankOperation;
		template <typename>
		using ScaledResultStructure = SymmetricLowRankStructure;

		template <typename Vectors>
		[[nodiscard]] static auto RightFactor(Vectors&& vectors)
		{
			return std::forward<Vectors>(vectors).conjugate();
		}
	};

	struct SelfAdjointLowRankStructure
	{
		template <typename Scalar>
		using CoefficientScalar = typename Eigen::NumTraits<Scalar>::Real;
		using TransposeOperation = ConjugateLowRankOperation;
		using ConjugateOperation = ConjugateLowRankOperation;
		using AdjointOperation = IdentityLowRankOperation;
		template <typename Factor>
		using ScaledResultStructure =
		        std::conditional_t<Eigen::NumTraits<Factor>::IsComplex, void, SelfAdjointLowRankStructure>;

		template <typename Vectors>
		[[nodiscard]] static auto RightFactor(Vectors&& vectors)
		{
			return std::forward<Vectors>(vectors);
		}
	};

	inline std::size_t CheckedBufferSize(const Eigen::Index dimension, const Eigen::Index termCount)
	{
		eigen_assert(dimension >= 0 && termCount >= 0 && "A matrix dimension or term count cannot be negative");

		const auto unsignedDimension = static_cast<std::size_t>(dimension);
		const auto unsignedTermCount = static_cast<std::size_t>(termCount);
		if (unsignedDimension != 0 && unsignedTermCount > std::numeric_limits<std::size_t>::max() / unsignedDimension)
		{
			throw std::length_error("Low-rank matrix storage size overflow");
		}
		return unsignedDimension * unsignedTermCount;
	}

	inline Eigen::Index BufferTermCapacity(const std::size_t scalarCapacity, const Eigen::Index scalarsPerTerm) noexcept
	{
		if (scalarsPerTerm == 0)
		{
			return (std::numeric_limits<Eigen::Index>::max)();
		}

		const auto termCapacity = scalarCapacity / static_cast<std::size_t>(scalarsPerTerm);
		const auto maximumCapacity = static_cast<std::size_t>((std::numeric_limits<Eigen::Index>::max)());
		return static_cast<Eigen::Index>((std::min)(termCapacity, maximumCapacity));
	}

	inline Eigen::Index GeometricTermCapacity(const Eigen::Index currentCapacity,
	                                          const Eigen::Index requiredCapacity) noexcept
	{
		if (requiredCapacity <= currentCapacity)
		{
			return currentCapacity;
		}

		const auto maximumCapacity = (std::numeric_limits<Eigen::Index>::max)();
		const auto doubledCapacity = currentCapacity > maximumCapacity / 2 ? maximumCapacity : 2 * currentCapacity;
		return (std::max)(requiredCapacity, (std::max)(Eigen::Index{1}, doubledCapacity));
	}
}


template <typename Scalar_, int RowsAtCompileTime_, int ColsAtCompileTime_>
class Hoppy::LowRankMatrix
    : public BulkLowRankMatrixBase<LowRankMatrix<Scalar_, RowsAtCompileTime_, ColsAtCompileTime_>>
{
	static_assert(RowsAtCompileTime_ == Eigen::Dynamic || RowsAtCompileTime_ >= 0);
	static_assert(ColsAtCompileTime_ == Eigen::Dynamic || ColsAtCompileTime_ >= 0);

public:
	using Base = BulkLowRankMatrixBase<LowRankMatrix>;
	using SemanticBase = LowRankMatrixBase<LowRankMatrix>;
	friend Base;
	friend SemanticBase;
	using Base::cols;
	using Base::rows;
	using Base::termCount;

	using Scalar = Scalar_;
	using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
	using StorageIndex = Eigen::Index;
	using CoefficientVector = Eigen::VectorX<Scalar>;
	using LeftVector = Eigen::Vector<Scalar, RowsAtCompileTime_>;
	using RightVector = Eigen::Vector<Scalar, ColsAtCompileTime_>;
	using LeftVectors = Eigen::Matrix<Scalar, RowsAtCompileTime_, Eigen::Dynamic>;
	using RightVectors = Eigen::Matrix<Scalar, ColsAtCompileTime_, Eigen::Dynamic>;

	static constexpr int RowsAtCompileTime = RowsAtCompileTime_;
	static constexpr int ColsAtCompileTime = ColsAtCompileTime_;
	static constexpr int MaxRowsAtCompileTime = RowsAtCompileTime_;
	static constexpr int MaxColsAtCompileTime = ColsAtCompileTime_;
	static constexpr int IsRowMajor = false;
	static constexpr int Flags = Eigen::NestByRefBit;

	constexpr LowRankMatrix() noexcept = default;
	LowRankMatrix(const LowRankMatrix&) = default;
	LowRankMatrix(LowRankMatrix&&) noexcept = default;
	LowRankMatrix& operator=(const LowRankMatrix&) = default;
	LowRankMatrix& operator=(LowRankMatrix&&) noexcept = default;
	~LowRankMatrix() = default;

	explicit LowRankMatrix(const Eigen::Index rows, const Eigen::Index cols) : m_Rows(rows), m_Cols(cols)
	{
		eigen_assert(rows >= 0 && cols >= 0);
	}

	template <typename OtherDerived, std::enable_if_t<std::is_same_v<Scalar, typename OtherDerived::Scalar>, int> = 0>
	explicit LowRankMatrix(const LowRankMatrixBase<OtherDerived>& other) : LowRankMatrix(other.rows(), other.cols())
	{
		reserve(other.termCount());
		for (Eigen::Index index = 0; index < other.termCount(); index++)
		{
			addTerm(other.coefficientOfTerm(index), other.leftVectorOfTerm(index), other.rightVectorOfTerm(index));
		}
	}

	template <typename OtherDerived, std::enable_if_t<std::is_same_v<Scalar, typename OtherDerived::Scalar>, int> = 0>
	explicit LowRankMatrix(const BulkLowRankMatrixBase<OtherDerived>& other) : LowRankMatrix(other.rows(), other.cols())
	{
		reserve(other.termCount());
		addTerms(other.coefficients(), other.leftVectors(), other.rightVectors());
	}

	template <typename LeftDerived, typename RightDerived>
	LowRankMatrix(const Scalar& coefficient,
	              const Eigen::MatrixBase<LeftDerived>& leftVector,
	              const Eigen::MatrixBase<RightDerived>& rightVector)
	    : m_Rows(leftVector.size()), m_Cols(rightVector.size())
	{
		addTerm(coefficient, leftVector, rightVector);
	}

	template <typename CoefficientsDerived, typename LeftDerived, typename RightDerived>
	LowRankMatrix(const Eigen::MatrixBase<CoefficientsDerived>& coefficients,
	              const Eigen::MatrixBase<LeftDerived>& leftVectors,
	              const Eigen::MatrixBase<RightDerived>& rightVectors)
	    : m_Rows(leftVectors.rows()), m_Cols(rightVectors.rows())
	{
		addTerms(coefficients, leftVectors, rightVectors);
	}

	[[nodiscard]] Eigen::Index capacity() const noexcept
	{
		return (std::min)({Detail::BufferTermCapacity(m_Coefficients.capacity(), 1),
		                   Detail::BufferTermCapacity(m_LeftVectorBuffer.capacity(), rows()),
		                   Detail::BufferTermCapacity(m_RightVectorBuffer.capacity(), cols())});
	}

	void reserve(const Eigen::Index termCapacity)
	{
		eigen_assert(termCapacity >= 0 && "A term capacity cannot be negative");
		m_Coefficients.reserve(static_cast<std::size_t>(termCapacity));
		m_LeftVectorBuffer.reserve(Detail::CheckedBufferSize(rows(), termCapacity));
		m_RightVectorBuffer.reserve(Detail::CheckedBufferSize(cols(), termCapacity));
	}

	void clear() noexcept
	{
		m_Coefficients.clear();
		m_LeftVectorBuffer.clear();
		m_RightVectorBuffer.clear();
	}

	LowRankMatrix& operator+=(const LowRankMatrix& other)
	{
		eigen_assert(rows() == other.rows() && cols() == other.cols() && "Low-rank matrix dimensions do not agree");
		if (this == &other)
		{
			for (Scalar& coefficient : m_Coefficients)
			{
				coefficient *= Scalar{2};
			}
			return *this;
		}
		return addTerms(other.coefficients(), other.leftVectors(), other.rightVectors());
	}

	LowRankMatrix& operator-=(const LowRankMatrix& other)
	{
		eigen_assert(rows() == other.rows() && cols() == other.cols() && "Low-rank matrix dimensions do not agree");
		if (this == &other)
		{
			clear();
			return *this;
		}
		return addTerms(-other.coefficients(), other.leftVectors(), other.rightVectors());
	}

	template <typename LeftDerived, typename RightDerived>
	/// The vector arguments must not alias storage owned by `*this`; evaluate exposed views before reinserting them.
	LowRankMatrix& addTerm(Scalar coefficient,
	                       const Eigen::MatrixBase<LeftDerived>& leftVector,
	                       const Eigen::MatrixBase<RightDerived>& rightVector)
	{
		if (coefficient == Scalar{})
		{
			return *this;
		}

		validateVector(leftVector, rows());
		validateVector(rightVector, cols());
		ensureCapacity(termCount() + 1);
		m_Coefficients.push_back(std::move(coefficient));
		appendVector<RowsAtCompileTime>(m_LeftVectorBuffer, rows(), leftVector);
		appendVector<ColsAtCompileTime>(m_RightVectorBuffer, cols(), rightVector);
		return *this;
	}

	/// The input expressions must not alias storage owned by `*this`; evaluate exposed views before reinserting them.
	template <typename CoefficientsDerived, typename LeftDerived, typename RightDerived>
	LowRankMatrix& addTerms(const Eigen::MatrixBase<CoefficientsDerived>& coefficients,
	                        const Eigen::MatrixBase<LeftDerived>& leftVectors,
	                        const Eigen::MatrixBase<RightDerived>& rightVectors)
	{
		eigen_assert((coefficients.rows() == 1 || coefficients.cols() == 1)
		             && "Low-rank coefficients must be a vector");
		eigen_assert(coefficients.size() == leftVectors.cols() && coefficients.size() == rightVectors.cols()
		             && "Low-rank coefficient and vector term counts do not agree");
		eigen_assert(leftVectors.rows() == rows() && rightVectors.rows() == cols()
		             && "Low-rank term vectors do not match the matrix dimensions");

		const CoefficientVector evaluatedCoefficients = coefficients.reshaped();
		Eigen::Index nonzeroCount = 0;
		for (Eigen::Index index = 0; index < evaluatedCoefficients.size(); index++)
		{
			nonzeroCount += evaluatedCoefficients[index] == Scalar{} ? 0 : 1;
		}
		if (nonzeroCount == 0)
		{
			return *this;
		}

		ensureCapacity(termCount() + nonzeroCount);
		const auto oldLeftSize = m_LeftVectorBuffer.size();
		const auto oldRightSize = m_RightVectorBuffer.size();
		m_LeftVectorBuffer.resize(oldLeftSize + Detail::CheckedBufferSize(rows(), nonzeroCount));
		m_RightVectorBuffer.resize(oldRightSize + Detail::CheckedBufferSize(cols(), nonzeroCount));
		Eigen::Index destinationIndex = 0;
		for (Eigen::Index sourceIndex = 0; sourceIndex < evaluatedCoefficients.size(); sourceIndex++)
		{
			if (evaluatedCoefficients[sourceIndex] != Scalar{})
			{
				m_Coefficients.push_back(evaluatedCoefficients[sourceIndex]);
				assignVector<RowsAtCompileTime>(
				        m_LeftVectorBuffer,
				        oldLeftSize + static_cast<std::size_t>(rows()) * static_cast<std::size_t>(destinationIndex),
				        rows(),
				        leftVectors.col(sourceIndex));
				assignVector<ColsAtCompileTime>(
				        m_RightVectorBuffer,
				        oldRightSize + static_cast<std::size_t>(cols()) * static_cast<std::size_t>(destinationIndex),
				        cols(),
				        rightVectors.col(sourceIndex));
				destinationIndex++;
			}
		}
		return *this;
	}

private:
	[[nodiscard]] constexpr Eigen::Index rowsImpl() const noexcept
	{
		return m_Rows.value();
	}

	[[nodiscard]] constexpr Eigen::Index colsImpl() const noexcept
	{
		return m_Cols.value();
	}

	[[nodiscard]] Eigen::Index termCountImpl() const noexcept
	{
		return static_cast<Eigen::Index>(m_Coefficients.size());
	}

	/// Read-only views into owned storage. Any operation that increases capacity invalidates all existing views.
	/// `clear` invalidates term views and references but retains the buffers and their capacity.
	[[nodiscard]] auto coefficientsImpl() const noexcept
	{
		return Eigen::Map<const CoefficientVector>{m_Coefficients.data(), termCount()};
	}

	[[nodiscard]] auto leftVectorsImpl() const noexcept
	{
		return Eigen::Map<const LeftVectors>{m_LeftVectorBuffer.data(), rows(), termCount()};
	}

	[[nodiscard]] auto rightVectorsImpl() const noexcept
	{
		return Eigen::Map<const RightVectors>{m_RightVectorBuffer.data(), cols(), termCount()};
	}

	[[nodiscard]] const Scalar& coefficientOfTermImpl(const Eigen::Index index) const
	{
		validateTermIndex(index);
		return m_Coefficients[static_cast<std::size_t>(index)];
	}

	[[nodiscard]] auto leftVectorOfTermImpl(const Eigen::Index index) const
	{
		validateTermIndex(index);
		return Eigen::Map<const LeftVector>{m_LeftVectorBuffer.data() + rows() * index, rows()};
	}

	[[nodiscard]] auto rightVectorOfTermImpl(const Eigen::Index index) const
	{
		validateTermIndex(index);
		return Eigen::Map<const RightVector>{m_RightVectorBuffer.data() + cols() * index, cols()};
	}

	template <typename Derived>
	static void validateVector(const Eigen::MatrixBase<Derived>& vector, const Eigen::Index expectedSize)
	{
		eigen_assert((vector.rows() == 1 || vector.cols() == 1) && vector.size() == expectedSize
		             && "A low-rank term vector has the wrong shape or dimension");
		(void)vector;
		(void)expectedSize;
	}

	void validateTermIndex(const Eigen::Index index) const
	{
		eigen_assert(index >= 0 && index < termCount() && "Low-rank term index is out of range");
		(void)index;
	}

	void ensureCapacity(const Eigen::Index requiredCapacity)
	{
		if (requiredCapacity > capacity())
		{
			reserve(Detail::GeometricTermCapacity(capacity(), requiredCapacity));
		}
	}

	template <int SizeAtCompileTime, typename Derived>
	static void appendVector(std::vector<Scalar>& destination,
	                         const Eigen::Index size,
	                         const Eigen::MatrixBase<Derived>& expression)
	{
		const auto oldSize = destination.size();
		destination.resize(oldSize + static_cast<std::size_t>(size));
		assignVector<SizeAtCompileTime>(destination, oldSize, size, expression);
	}

	template <int SizeAtCompileTime, typename Derived>
	static void assignVector(std::vector<Scalar>& destination,
	                         const std::size_t offset,
	                         const Eigen::Index size,
	                         const Eigen::MatrixBase<Derived>& expression)
	{
		if (size != 0)
		{
			Eigen::Map<Eigen::Vector<Scalar, SizeAtCompileTime>>{destination.data() + offset, size} =
			        expression.reshaped();
		}
	}

private:
	Eigen::internal::variable_if_dynamic<Eigen::Index, RowsAtCompileTime> m_Rows{};
	Eigen::internal::variable_if_dynamic<Eigen::Index, ColsAtCompileTime> m_Cols{};
	std::vector<Scalar> m_Coefficients{};
	std::vector<Scalar> m_LeftVectorBuffer{};
	std::vector<Scalar> m_RightVectorBuffer{};
};


template <typename Scalar_, typename StructurePolicy_, int DimensionAtCompileTime_>
class Hoppy::Detail::SingleFactorLowRankMatrix
    : public BulkLowRankMatrixBase<SingleFactorLowRankMatrix<Scalar_, StructurePolicy_, DimensionAtCompileTime_>>
{
	static_assert(DimensionAtCompileTime_ == Eigen::Dynamic || DimensionAtCompileTime_ >= 0);

public:
	using Base = BulkLowRankMatrixBase<SingleFactorLowRankMatrix>;
	using SemanticBase = LowRankMatrixBase<SingleFactorLowRankMatrix>;
	friend Base;
	friend SemanticBase;
	using Base::cols;
	using Base::rows;
	using Base::termCount;

	using Scalar = Scalar_;
	using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
	using StructurePolicy = StructurePolicy_;
	using CoefficientScalar = typename StructurePolicy::template CoefficientScalar<Scalar>;
	using StorageIndex = Eigen::Index;
	using CoefficientVector = Eigen::VectorX<CoefficientScalar>;
	using Vector = Eigen::Vector<Scalar, DimensionAtCompileTime_>;
	using Vectors = Eigen::Matrix<Scalar, DimensionAtCompileTime_, Eigen::Dynamic>;

	static constexpr int RowsAtCompileTime = DimensionAtCompileTime_;
	static constexpr int ColsAtCompileTime = DimensionAtCompileTime_;
	static constexpr int MaxRowsAtCompileTime = DimensionAtCompileTime_;
	static constexpr int MaxColsAtCompileTime = DimensionAtCompileTime_;
	static constexpr int IsRowMajor = false;
	static constexpr int Flags = Eigen::NestByRefBit;

	constexpr SingleFactorLowRankMatrix() noexcept = default;
	SingleFactorLowRankMatrix(const SingleFactorLowRankMatrix&) = default;
	SingleFactorLowRankMatrix(SingleFactorLowRankMatrix&&) noexcept = default;
	SingleFactorLowRankMatrix& operator=(const SingleFactorLowRankMatrix&) = default;
	SingleFactorLowRankMatrix& operator=(SingleFactorLowRankMatrix&&) noexcept = default;
	~SingleFactorLowRankMatrix() = default;

	explicit SingleFactorLowRankMatrix(const Eigen::Index dimension) : m_Dimension(dimension)
	{
		eigen_assert(dimension >= 0);
	}

	template <typename OtherDerived,
	          std::enable_if_t<std::is_same_v<Scalar, typename OtherDerived::Scalar>
	                                   && std::is_same_v<StructurePolicy, typename OtherDerived::StructurePolicy>,
	                           int> = 0>
	explicit SingleFactorLowRankMatrix(const LowRankMatrixBase<OtherDerived>& other)
	    : SingleFactorLowRankMatrix(other.rows())
	{
		reserve(other.termCount());
		for (Eigen::Index index = 0; index < other.termCount(); index++)
		{
			addTerm(other.coefficientOfTerm(index), other.leftVectorOfTerm(index));
		}
	}

	template <typename OtherDerived,
	          std::enable_if_t<std::is_same_v<Scalar, typename OtherDerived::Scalar>
	                                   && std::is_same_v<StructurePolicy, typename OtherDerived::StructurePolicy>,
	                           int> = 0>
	explicit SingleFactorLowRankMatrix(const BulkLowRankMatrixBase<OtherDerived>& other)
	    : SingleFactorLowRankMatrix(other.rows())
	{
		reserve(other.termCount());
		addTerms(other.coefficients(), other.leftVectors());
	}

	template <typename VectorDerived>
	SingleFactorLowRankMatrix(const CoefficientScalar& coefficient, const Eigen::MatrixBase<VectorDerived>& vector)
	    : m_Dimension(vector.size())
	{
		addTerm(coefficient, vector);
	}

	template <typename CoefficientsDerived,
	          typename VectorsDerived,
	          std::enable_if_t<std::is_convertible_v<typename CoefficientsDerived::Scalar, CoefficientScalar>, int> = 0>
	SingleFactorLowRankMatrix(const Eigen::MatrixBase<CoefficientsDerived>& coefficients,
	                          const Eigen::MatrixBase<VectorsDerived>& vectors)
	    : m_Dimension(vectors.rows())
	{
		addTerms(coefficients, vectors);
	}

	[[nodiscard]] Eigen::Index capacity() const noexcept
	{
		return (std::min)(Detail::BufferTermCapacity(m_Coefficients.capacity(), 1),
		                  Detail::BufferTermCapacity(m_VectorBuffer.capacity(), rows()));
	}

	void reserve(const Eigen::Index termCapacity)
	{
		eigen_assert(termCapacity >= 0 && "A term capacity cannot be negative");
		m_Coefficients.reserve(static_cast<std::size_t>(termCapacity));
		m_VectorBuffer.reserve(Detail::CheckedBufferSize(rows(), termCapacity));
	}

	void clear() noexcept
	{
		m_Coefficients.clear();
		m_VectorBuffer.clear();
	}

	SingleFactorLowRankMatrix& operator+=(const SingleFactorLowRankMatrix& other)
	{
		eigen_assert(rows() == other.rows() && "Structured low-rank matrix dimensions do not agree");
		if (this == &other)
		{
			for (CoefficientScalar& coefficient : m_Coefficients)
			{
				coefficient *= CoefficientScalar{2};
			}
			return *this;
		}
		return addTerms(other.coefficients(), other.leftVectors());
	}

	SingleFactorLowRankMatrix& operator-=(const SingleFactorLowRankMatrix& other)
	{
		eigen_assert(rows() == other.rows() && "Structured low-rank matrix dimensions do not agree");
		if (this == &other)
		{
			clear();
			return *this;
		}
		return addTerms(-other.coefficients(), other.leftVectors());
	}

	template <typename VectorDerived>
	/// The vector argument must not alias storage owned by `*this`; evaluate exposed views before reinserting it.
	SingleFactorLowRankMatrix& addTerm(CoefficientScalar coefficient, const Eigen::MatrixBase<VectorDerived>& vector)
	{
		if (coefficient == CoefficientScalar{})
		{
			return *this;
		}

		validateVector(vector);
		ensureCapacity(termCount() + 1);
		m_Coefficients.push_back(std::move(coefficient));
		appendVector(m_VectorBuffer, vector);
		return *this;
	}

	template <typename CoefficientsDerived,
	          typename VectorsDerived,
	          std::enable_if_t<std::is_convertible_v<typename CoefficientsDerived::Scalar, CoefficientScalar>, int> = 0>
	/// The input expressions must not alias storage owned by `*this`; evaluate exposed views before reinserting them.
	SingleFactorLowRankMatrix& addTerms(const Eigen::MatrixBase<CoefficientsDerived>& coefficients,
	                                    const Eigen::MatrixBase<VectorsDerived>& vectors)
	{
		eigen_assert((coefficients.rows() == 1 || coefficients.cols() == 1)
		             && "Low-rank coefficients must be a vector");
		eigen_assert(coefficients.size() == vectors.cols()
		             && "Low-rank coefficient and vector term counts do not agree");
		eigen_assert(vectors.rows() == rows() && "Low-rank term vectors do not match the matrix dimension");

		const CoefficientVector evaluatedCoefficients = coefficients.reshaped();
		Eigen::Index nonzeroCount = 0;
		for (Eigen::Index index = 0; index < evaluatedCoefficients.size(); index++)
		{
			nonzeroCount += evaluatedCoefficients[index] == CoefficientScalar{} ? 0 : 1;
		}
		if (nonzeroCount == 0)
		{
			return *this;
		}

		ensureCapacity(termCount() + nonzeroCount);
		const auto oldVectorSize = m_VectorBuffer.size();
		m_VectorBuffer.resize(oldVectorSize + Detail::CheckedBufferSize(rows(), nonzeroCount));
		Eigen::Index destinationIndex = 0;
		for (Eigen::Index sourceIndex = 0; sourceIndex < evaluatedCoefficients.size(); sourceIndex++)
		{
			if (evaluatedCoefficients[sourceIndex] != CoefficientScalar{})
			{
				m_Coefficients.push_back(evaluatedCoefficients[sourceIndex]);
				assignVector(m_VectorBuffer,
				             oldVectorSize
				                     + static_cast<std::size_t>(rows()) * static_cast<std::size_t>(destinationIndex),
				             vectors.col(sourceIndex));
				destinationIndex++;
			}
		}
		return *this;
	}

	[[nodiscard]] LowRankMatrix<Scalar, DimensionAtCompileTime_, DimensionAtCompileTime_> toGeneral() const
	{
		return {this->coefficients(), this->leftVectors(), StructurePolicy::RightFactor(this->leftVectors())};
	}

private:
	[[nodiscard]] constexpr Eigen::Index rowsImpl() const noexcept
	{
		return m_Dimension.value();
	}

	[[nodiscard]] constexpr Eigen::Index colsImpl() const noexcept
	{
		return m_Dimension.value();
	}

	[[nodiscard]] Eigen::Index termCountImpl() const noexcept
	{
		return static_cast<Eigen::Index>(m_Coefficients.size());
	}

	/// Read-only views into owned storage. Any operation that increases capacity invalidates all existing views.
	/// `clear` invalidates term views and references but retains the buffers and their capacity.
	[[nodiscard]] auto coefficientsImpl() const noexcept
	{
		return Eigen::Map<const CoefficientVector>{m_Coefficients.data(), termCount()};
	}

	[[nodiscard]] auto leftVectorsImpl() const noexcept
	{
		return Eigen::Map<const Vectors>{m_VectorBuffer.data(), rows(), termCount()};
	}

	[[nodiscard]] auto rightVectorsImpl() const noexcept
	{
		return StructurePolicy::RightFactor(leftVectorsImpl());
	}

	[[nodiscard]] const CoefficientScalar& coefficientOfTermImpl(const Eigen::Index index) const
	{
		validateTermIndex(index);
		return m_Coefficients[static_cast<std::size_t>(index)];
	}

	[[nodiscard]] auto leftVectorOfTermImpl(const Eigen::Index index) const
	{
		validateTermIndex(index);
		return Eigen::Map<const Vector>{m_VectorBuffer.data() + rows() * index, rows()};
	}

	[[nodiscard]] auto rightVectorOfTermImpl(const Eigen::Index index) const
	{
		return StructurePolicy::RightFactor(leftVectorOfTermImpl(index));
	}

	template <typename Derived>
	void validateVector(const Eigen::MatrixBase<Derived>& vector) const
	{
		eigen_assert((vector.rows() == 1 || vector.cols() == 1) && vector.size() == rows()
		             && "A low-rank term vector has the wrong shape or dimension");
		(void)vector;
	}

	void validateTermIndex(const Eigen::Index index) const
	{
		eigen_assert(index >= 0 && index < termCount() && "Low-rank term index is out of range");
		(void)index;
	}

	void ensureCapacity(const Eigen::Index requiredCapacity)
	{
		if (requiredCapacity > capacity())
		{
			reserve(Detail::GeometricTermCapacity(capacity(), requiredCapacity));
		}
	}

	template <typename Derived>
	static void appendVector(std::vector<Scalar>& destination, const Eigen::MatrixBase<Derived>& expression)
	{
		const auto oldSize = destination.size();
		destination.resize(oldSize + static_cast<std::size_t>(expression.size()));
		assignVector(destination, oldSize, expression);
	}

	template <typename Derived>
	static void assignVector(std::vector<Scalar>& destination,
	                         const std::size_t offset,
	                         const Eigen::MatrixBase<Derived>& expression)
	{
		if (expression.size() != 0)
		{
			Eigen::Map<Vector>{destination.data() + offset, expression.size()} = expression.reshaped();
		}
	}

private:
	Eigen::internal::variable_if_dynamic<Eigen::Index, DimensionAtCompileTime_> m_Dimension{};
	std::vector<CoefficientScalar> m_Coefficients{};
	std::vector<Scalar> m_VectorBuffer{};
};


namespace Hoppy::Detail
{
	template <typename Matrix>
	struct IsLowRankExpression : std::is_base_of<LowRankMatrixBase<Matrix>, Matrix>
	{
	};

	template <typename Matrix>
	struct IsDenseExpression : std::is_base_of<Eigen::MatrixBase<Matrix>, Matrix>
	{
	};

	template <typename StructurePolicy>
	struct UnaryOperationsForStructure
	{
		using TransposeOperation = typename StructurePolicy::TransposeOperation;
		using ConjugateOperation = typename StructurePolicy::ConjugateOperation;
		using AdjointOperation = typename StructurePolicy::AdjointOperation;
		using ResultStructurePolicy = StructurePolicy;
	};

	template <>
	struct UnaryOperationsForStructure<void>
	{
		using TransposeOperation = TransposeLowRankOperation;
		using ConjugateOperation = ConjugateLowRankOperation;
		using AdjointOperation = AdjointLowRankOperation;
		using ResultStructurePolicy = void;
	};

	template <typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>
	struct UnaryExpressionTraits<LowRankMatrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>>
	    : UnaryOperationsForStructure<void>
	{
	};

	template <typename Scalar, typename StructurePolicy, int DimensionAtCompileTime>
	struct UnaryExpressionTraits<SingleFactorLowRankMatrix<Scalar, StructurePolicy, DimensionAtCompileTime>>
	    : UnaryOperationsForStructure<StructurePolicy>
	{
	};

	template <typename Nested, typename Operation>
	struct UnaryExpressionTraits<LowRankUnaryExpr<Nested, Operation>>
	    : UnaryExpressionTraits<std::remove_cv_t<std::remove_reference_t<Nested>>>
	{
	};
}


template <typename Nested_, typename Operation_>
class Hoppy::Detail::LowRankUnaryExpr
    : public std::conditional_t<std::is_base_of_v<BulkLowRankMatrixBase<std::decay_t<Nested_>>, std::decay_t<Nested_>>,
                                BulkLowRankMatrixBase<LowRankUnaryExpr<Nested_, Operation_>>,
                                LowRankMatrixBase<LowRankUnaryExpr<Nested_, Operation_>>>
{
public:
	using Base = LowRankMatrixBase<LowRankUnaryExpr>;
	using SemanticBase =
	        std::conditional_t<std::is_base_of_v<BulkLowRankMatrixBase<std::decay_t<Nested_>>, std::decay_t<Nested_>>,
	                           BulkLowRankMatrixBase<LowRankUnaryExpr<Nested_, Operation_>>,
	                           LowRankMatrixBase<LowRankUnaryExpr<Nested_, Operation_>>>;
	friend Base;
	friend SemanticBase;
	using Base::cols;
	using Base::rows;
	using Base::termCount;

	using Nested = Nested_;
	using Operation = Operation_;
	using Scalar = typename Eigen::internal::traits<LowRankUnaryExpr>::Scalar;
	using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
	using StorageIndex = Eigen::Index;
	using StructurePolicy = typename UnaryExpressionTraits<LowRankUnaryExpr>::ResultStructurePolicy;

	static constexpr int RowsAtCompileTime = Eigen::internal::traits<LowRankUnaryExpr>::RowsAtCompileTime;
	static constexpr int ColsAtCompileTime = Eigen::internal::traits<LowRankUnaryExpr>::ColsAtCompileTime;
	static constexpr int MaxRowsAtCompileTime = Eigen::internal::traits<LowRankUnaryExpr>::MaxRowsAtCompileTime;
	static constexpr int MaxColsAtCompileTime = Eigen::internal::traits<LowRankUnaryExpr>::MaxColsAtCompileTime;
	static constexpr int IsRowMajor = false;
	static constexpr int Flags = 0;

	explicit LowRankUnaryExpr(Nested nested) : m_Nested(std::forward<Nested>(nested))
	{
	}

private:
	[[nodiscard]] Eigen::Index rowsImpl() const noexcept
	{
		return Operation::SwapsDimensions ? m_Nested.cols() : m_Nested.rows();
	}

	[[nodiscard]] Eigen::Index colsImpl() const noexcept
	{
		return Operation::SwapsDimensions ? m_Nested.rows() : m_Nested.cols();
	}

	[[nodiscard]] Eigen::Index termCountImpl() const noexcept
	{
		return m_Nested.termCount();
	}

	[[nodiscard]] decltype(auto) coefficientsImpl() const
	{
		return Operation::Coefficients(m_Nested);
	}

	[[nodiscard]] decltype(auto) leftVectorsImpl() const
	{
		return Operation::LeftVectors(m_Nested);
	}

	[[nodiscard]] decltype(auto) rightVectorsImpl() const
	{
		return Operation::RightVectors(m_Nested);
	}

	[[nodiscard]] decltype(auto) coefficientOfTermImpl(const Eigen::Index index) const
	{
		return Operation::CoefficientOfTerm(m_Nested, index);
	}

	[[nodiscard]] decltype(auto) leftVectorOfTermImpl(const Eigen::Index index) const
	{
		return Operation::LeftVectorOfTerm(m_Nested, index);
	}

	[[nodiscard]] decltype(auto) rightVectorOfTermImpl(const Eigen::Index index) const
	{
		return Operation::RightVectorOfTerm(m_Nested, index);
	}

private:
	Nested m_Nested;
};


namespace Hoppy::Detail
{
	template <typename StructurePolicy, typename Factor>
	struct ScaledResultStructure
	{
		using Type = typename StructurePolicy::template ScaledResultStructure<Factor>;
	};

	template <typename Factor>
	struct ScaledResultStructure<void, Factor>
	{
		using Type = void;
	};
}


template <typename Nested_, typename Factor_, typename Operation_, bool FactorOnLeft_>
class Hoppy::Detail::LowRankScalarExpr
    : public std::conditional_t<std::is_base_of_v<BulkLowRankMatrixBase<std::decay_t<Nested_>>, std::decay_t<Nested_>>,
                                BulkLowRankMatrixBase<LowRankScalarExpr<Nested_, Factor_, Operation_, FactorOnLeft_>>,
                                LowRankMatrixBase<LowRankScalarExpr<Nested_, Factor_, Operation_, FactorOnLeft_>>>
{
public:
	using Base = LowRankMatrixBase<LowRankScalarExpr>;
	using SemanticBase =
	        std::conditional_t<std::is_base_of_v<BulkLowRankMatrixBase<std::decay_t<Nested_>>, std::decay_t<Nested_>>,
	                           BulkLowRankMatrixBase<LowRankScalarExpr<Nested_, Factor_, Operation_, FactorOnLeft_>>,
	                           LowRankMatrixBase<LowRankScalarExpr<Nested_, Factor_, Operation_, FactorOnLeft_>>>;
	friend Base;
	friend SemanticBase;
	using Base::cols;
	using Base::rows;
	using Base::termCount;

	using Nested = Nested_;
	using Factor = Factor_;
	using Operation = Operation_;
	using Scalar = typename Eigen::internal::traits<LowRankScalarExpr>::Scalar;
	using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
	using StorageIndex = Eigen::Index;
	using NestedMatrix = std::remove_cv_t<std::remove_reference_t<Nested>>;
	using NestedCoefficientScalar =
	        std::decay_t<decltype(std::declval<const NestedMatrix&>().coefficientOfTerm(Eigen::Index{}))>;
	using CoefficientScalar = typename Operation::template ResultScalar<NestedCoefficientScalar, Factor, FactorOnLeft_>;
	using NestedStructurePolicy = typename UnaryExpressionTraits<NestedMatrix>::ResultStructurePolicy;
	using StructurePolicy = typename ScaledResultStructure<NestedStructurePolicy, Factor>::Type;

	static constexpr int RowsAtCompileTime = Eigen::internal::traits<LowRankScalarExpr>::RowsAtCompileTime;
	static constexpr int ColsAtCompileTime = Eigen::internal::traits<LowRankScalarExpr>::ColsAtCompileTime;
	static constexpr int MaxRowsAtCompileTime = Eigen::internal::traits<LowRankScalarExpr>::MaxRowsAtCompileTime;
	static constexpr int MaxColsAtCompileTime = Eigen::internal::traits<LowRankScalarExpr>::MaxColsAtCompileTime;
	static constexpr int IsRowMajor = false;
	static constexpr int Flags = 0;

	LowRankScalarExpr(Nested nested, Factor factor)
	    : m_Nested(std::forward<Nested>(nested)), m_Factor(std::move(factor))
	{
	}

private:
	[[nodiscard]] Eigen::Index rowsImpl() const noexcept
	{
		return m_Nested.rows();
	}

	[[nodiscard]] Eigen::Index colsImpl() const noexcept
	{
		return m_Nested.cols();
	}

	[[nodiscard]] Eigen::Index termCountImpl() const noexcept
	{
		return m_Nested.termCount();
	}

	[[nodiscard]] auto coefficientsImpl() const
	{
		return Operation::template CoefficientsResult<CoefficientScalar, FactorOnLeft_>(m_Nested.coefficients(),
		                                                                                m_Factor);
	}

	[[nodiscard]] auto leftVectorsImpl() const
	{
		return m_Nested.leftVectors().template cast<Scalar>();
	}

	[[nodiscard]] auto rightVectorsImpl() const
	{
		return m_Nested.rightVectors().template cast<Scalar>();
	}

	[[nodiscard]] CoefficientScalar coefficientOfTermImpl(const Eigen::Index index) const
	{
		return Operation::template CoefficientResult<CoefficientScalar, FactorOnLeft_>(
		        m_Nested.coefficientOfTerm(index), m_Factor);
	}

	[[nodiscard]] auto leftVectorOfTermImpl(const Eigen::Index index) const
	{
		return m_Nested.leftVectorOfTerm(index).template cast<Scalar>();
	}

	[[nodiscard]] auto rightVectorOfTermImpl(const Eigen::Index index) const
	{
		return m_Nested.rightVectorOfTerm(index).template cast<Scalar>();
	}

private:
	Nested m_Nested;
	Factor m_Factor;
};


template <typename DenseNested_, typename LowRankNested_, int DenseSign_, int LowRankSign_>
class Hoppy::Detail::DenseLowRankSumExpr
    : public Eigen::EigenBase<DenseLowRankSumExpr<DenseNested_, LowRankNested_, DenseSign_, LowRankSign_>>
{
	static_assert(DenseSign_ == 1 || DenseSign_ == -1);
	static_assert(LowRankSign_ == 1 || LowRankSign_ == -1);

public:
	using DenseNested = DenseNested_;
	using LowRankNested = LowRankNested_;
	using Scalar = typename Eigen::internal::traits<DenseLowRankSumExpr>::Scalar;
	using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
	using StorageIndex = Eigen::Index;

	static constexpr int RowsAtCompileTime = Eigen::internal::traits<DenseLowRankSumExpr>::RowsAtCompileTime;
	static constexpr int ColsAtCompileTime = Eigen::internal::traits<DenseLowRankSumExpr>::ColsAtCompileTime;
	static constexpr int MaxRowsAtCompileTime = Eigen::internal::traits<DenseLowRankSumExpr>::MaxRowsAtCompileTime;
	static constexpr int MaxColsAtCompileTime = Eigen::internal::traits<DenseLowRankSumExpr>::MaxColsAtCompileTime;
	static constexpr int IsRowMajor = false;
	static constexpr int Flags = 0;
	static constexpr int DenseSign = DenseSign_;
	static constexpr int LowRankSign = LowRankSign_;

	DenseLowRankSumExpr(DenseNested dense, LowRankNested lowRank)
	    : m_Dense(std::forward<DenseNested>(dense)), m_LowRank(std::forward<LowRankNested>(lowRank))
	{
		eigen_assert(m_Dense.rows() == m_LowRank.rows() && m_Dense.cols() == m_LowRank.cols()
		             && "Dense and low-rank matrix dimensions do not agree");
	}

	[[nodiscard]] Eigen::Index rows() const noexcept
	{
		return m_Dense.rows();
	}

	[[nodiscard]] Eigen::Index cols() const noexcept
	{
		return m_Dense.cols();
	}

	[[nodiscard]] const auto& dense() const noexcept
	{
		return m_Dense;
	}

	[[nodiscard]] const auto& lowRank() const noexcept
	{
		return m_LowRank;
	}

	template <typename Rhs>
	[[nodiscard]] auto operator*(const Eigen::MatrixBase<Rhs>& rhs) const
	{
		return Eigen::Product<DenseLowRankSumExpr, Rhs, Eigen::DefaultProduct>{*this, rhs.derived()};
	}

	template <typename Destination>
	void evalTo(Destination& destination) const
	{
		using DestinationScalar = typename Destination::Scalar;
		destination = DestinationScalar{DenseSign} * m_Dense.template cast<DestinationScalar>();
		m_LowRank.addScaledToDense(destination, DestinationScalar{LowRankSign});
	}

	template <typename Destination>
	void addTo(Destination& destination) const
	{
		using DestinationScalar = typename Destination::Scalar;
		destination += DestinationScalar{DenseSign} * m_Dense.template cast<DestinationScalar>();
		m_LowRank.addScaledToDense(destination, DestinationScalar{LowRankSign});
	}

	template <typename Destination>
	void subTo(Destination& destination) const
	{
		using DestinationScalar = typename Destination::Scalar;
		destination -= DestinationScalar{DenseSign} * m_Dense.template cast<DestinationScalar>();
		m_LowRank.addScaledToDense(destination, DestinationScalar{-LowRankSign});
	}

	[[nodiscard]] Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime> toDense() const
	{
		Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime> result(rows(), cols());
		evalTo(result);
		return result;
	}

	[[nodiscard]] Eigen::Vector<Scalar, Eigen::internal::min_size_prefer_dynamic(RowsAtCompileTime, ColsAtCompileTime)>
	diagonal() const
	{
		return Scalar{DenseSign} * m_Dense.diagonal().template cast<Scalar>()
		       + Scalar{LowRankSign} * m_LowRank.diagonal().template cast<Scalar>();
	}

private:
	DenseNested m_Dense;
	LowRankNested m_LowRank;
};


template <typename Scalar>
class Hoppy::LowRankDiagonalPreconditioner : public Eigen::DiagonalPreconditioner<Scalar>
{
	using Base = Eigen::DiagonalPreconditioner<Scalar>;

public:
	LowRankDiagonalPreconditioner() = default;

	template <typename Matrix>
	explicit LowRankDiagonalPreconditioner(const Matrix& matrix)
	{
		compute(matrix);
	}

	template <typename Matrix>
	LowRankDiagonalPreconditioner& analyzePattern(const Matrix&)
	{
		return *this;
	}

	template <typename Matrix>
	LowRankDiagonalPreconditioner& factorize(const Matrix& matrix)
	{
		return compute(matrix);
	}

	template <typename Matrix>
	LowRankDiagonalPreconditioner& compute(const Matrix& matrix)
	{
		const Eigen::VectorX<Scalar> diagonal = matrix.diagonal();
		this->m_invdiag.resize(diagonal.size());
		for (Eigen::Index index = 0; index < diagonal.size(); index++)
		{
			this->m_invdiag[index] = diagonal[index] == Scalar{0} ? Scalar{1} : Scalar{1} / diagonal[index];
		}
		this->m_isInitialized = true;
		return *this;
	}
};


namespace Hoppy::Detail
{
	template <typename Nested, typename Factor, typename Operation, bool FactorOnLeft>
	struct UnaryExpressionTraits<LowRankScalarExpr<Nested, Factor, Operation, FactorOnLeft>>
	    : UnaryOperationsForStructure<
	              typename LowRankScalarExpr<Nested, Factor, Operation, FactorOnLeft>::StructurePolicy>
	{
	};
}


namespace Hoppy::Detail
{
	template <typename Matrix>
	using UnaryNested = std::conditional_t<std::is_lvalue_reference_v<Matrix>, Matrix, std::decay_t<Matrix>>;

	template <typename Matrix>
	[[nodiscard]] auto MakeTranspose(Matrix&& matrix)
	{
		using CleanMatrix = std::remove_cv_t<std::remove_reference_t<Matrix>>;
		using Operation = typename UnaryExpressionTraits<CleanMatrix>::TransposeOperation;
		return LowRankUnaryExpr<UnaryNested<Matrix&&>, Operation>{std::forward<Matrix>(matrix)};
	}

	template <typename Matrix>
	[[nodiscard]] auto MakeConjugate(Matrix&& matrix)
	{
		using CleanMatrix = std::remove_cv_t<std::remove_reference_t<Matrix>>;
		using Operation = typename UnaryExpressionTraits<CleanMatrix>::ConjugateOperation;
		return LowRankUnaryExpr<UnaryNested<Matrix&&>, Operation>{std::forward<Matrix>(matrix)};
	}

	template <typename Matrix>
	[[nodiscard]] auto MakeAdjoint(Matrix&& matrix)
	{
		using CleanMatrix = std::remove_cv_t<std::remove_reference_t<Matrix>>;
		using Operation = typename UnaryExpressionTraits<CleanMatrix>::AdjointOperation;
		return LowRankUnaryExpr<UnaryNested<Matrix&&>, Operation>{std::forward<Matrix>(matrix)};
	}

	template <bool FactorOnLeft, typename Matrix, typename Factor>
	[[nodiscard]] auto MakeProduct(Matrix&& matrix, Factor&& factor)
	{
		using StoredFactor = std::decay_t<Factor>;
		return LowRankScalarExpr<UnaryNested<Matrix&&>, StoredFactor, MultiplyLowRankOperation, FactorOnLeft>{
		        std::forward<Matrix>(matrix), std::forward<Factor>(factor)};
	}

	template <typename Matrix, typename Factor>
	[[nodiscard]] auto MakeQuotient(Matrix&& matrix, Factor&& factor)
	{
		using StoredFactor = std::decay_t<Factor>;
		return LowRankScalarExpr<UnaryNested<Matrix&&>, StoredFactor, DivideLowRankOperation, false>{
		        std::forward<Matrix>(matrix), std::forward<Factor>(factor)};
	}

	template <int LeftSize, int RightSize>
	inline constexpr int MergedCompileTimeSize = LeftSize == RightSize         ? LeftSize
	                                             : LeftSize == Eigen::Dynamic  ? RightSize
	                                             : RightSize == Eigen::Dynamic ? LeftSize
	                                                                           : Eigen::Dynamic;

	template <typename Left, typename Right>
	struct LowRankSumTraits
	{
		using Scalar = typename Left::Scalar;
		using LeftStructure = typename UnaryExpressionTraits<Left>::ResultStructurePolicy;
		using RightStructure = typename UnaryExpressionTraits<Right>::ResultStructurePolicy;
		static constexpr bool PreservesStructure =
		        !std::is_void_v<LeftStructure> && std::is_same_v<LeftStructure, RightStructure>;
		using StructurePolicy = std::conditional_t<PreservesStructure, LeftStructure, void>;
		static constexpr int RowsAtCompileTime =
		        MergedCompileTimeSize<Left::RowsAtCompileTime, Right::RowsAtCompileTime>;
		static constexpr int ColsAtCompileTime =
		        MergedCompileTimeSize<Left::ColsAtCompileTime, Right::ColsAtCompileTime>;
		using Result = std::conditional_t<PreservesStructure,
		                                  SingleFactorLowRankMatrix<Scalar, StructurePolicy, RowsAtCompileTime>,
		                                  LowRankMatrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>>;
	};

	template <bool Subtract, typename Left, typename Right>
	[[nodiscard]] auto AddLowRank(const Left& left, const Right& right)
	{
		eigen_assert(left.rows() == right.rows() && left.cols() == right.cols()
		             && "Low-rank matrix dimensions do not agree");
		using Traits = LowRankSumTraits<Left, Right>;
		using Result = typename Traits::Result;

		if constexpr (Traits::PreservesStructure)
		{
			Result result(left.rows());
			result.reserve(left.termCount() + right.termCount());
			for (Eigen::Index index = 0; index < left.termCount(); index++)
			{
				result.addTerm(left.coefficientOfTerm(index), left.leftVectorOfTerm(index));
			}
			for (Eigen::Index index = 0; index < right.termCount(); index++)
			{
				const auto coefficient = Subtract ? -right.coefficientOfTerm(index) : right.coefficientOfTerm(index);
				result.addTerm(coefficient, right.leftVectorOfTerm(index));
			}
			return result;
		}
		else
		{
			Result result(left.rows(), left.cols());
			result.reserve(left.termCount() + right.termCount());
			for (Eigen::Index index = 0; index < left.termCount(); index++)
			{
				result.addTerm(
				        left.coefficientOfTerm(index), left.leftVectorOfTerm(index), left.rightVectorOfTerm(index));
			}
			for (Eigen::Index index = 0; index < right.termCount(); index++)
			{
				const auto coefficient = Subtract ? -right.coefficientOfTerm(index) : right.coefficientOfTerm(index);
				result.addTerm(coefficient, right.leftVectorOfTerm(index), right.rightVectorOfTerm(index));
			}
			return result;
		}
	}

	template <typename Dense>
	[[nodiscard]] decltype(auto) NestDenseExpression(Dense&& dense)
	{
		using Expression = std::decay_t<Dense>;
		if constexpr (std::is_lvalue_reference_v<Dense&&>)
		{
			return std::forward<Dense>(dense);
		}
		else if constexpr (std::is_same_v<Expression, typename Expression::PlainObject>)
		{
			return Expression{std::forward<Dense>(dense)};
		}
		else
		{
			return dense.eval();
		}
	}

	template <int DenseSign, int LowRankSign, typename Dense, typename LowRank>
	[[nodiscard]] auto MakeDenseLowRankSum(Dense&& dense, LowRank&& lowRank)
	{
		using NestedDense = decltype(NestDenseExpression(std::forward<Dense>(dense)));
		using NestedLowRank = UnaryNested<LowRank&&>;
		return DenseLowRankSumExpr<NestedDense, NestedLowRank, DenseSign, LowRankSign>{
		        NestDenseExpression(std::forward<Dense>(dense)), std::forward<LowRank>(lowRank)};
	}
}
