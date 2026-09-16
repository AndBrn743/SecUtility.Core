// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Andy Brown

#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>


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

		template <typename Matrix>
		struct UnaryExpressionTraits;

		template <typename LeftScalar, typename RightScalar>
		struct IsScalarProductCompatible;

		template <typename LeftScalar, typename RightScalar>
		struct IsScalarQuotientCompatible;

		template <typename Matrix>
		struct IsLowRankExpression;

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
	}

	template <typename Derived>
	class LowRankMatrixBase;

	template <typename Scalar_, int RowsAtCompileTime_, int ColsAtCompileTime_>
	class LowRankMatrix;

	struct LowRankStorage
	{
	};

	template <typename Scalar>
	using LowRankMatrixX = LowRankMatrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

	template <typename Scalar, int DimensionAtCompileTime>
	using LowRankSymmetricMatrix =
	        Detail::SingleFactorLowRankMatrix<Scalar, Detail::SymmetricLowRankStructure, DimensionAtCompileTime>;

	template <typename Scalar>
	using LowRankSymmetricMatrixX = LowRankSymmetricMatrix<Scalar, Eigen::Dynamic>;

	template <typename Scalar, int DimensionAtCompileTime>
	using LowRankSelfAdjointMatrix = Detail::SingleFactorLowRankMatrix<
	        Scalar,
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
	static constexpr int Flags = 0;
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
	static constexpr int Flags = 0;
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
struct Eigen::internal::traits<
        Hoppy::Detail::LowRankScalarExpr<Nested_, Factor_, Operation_, FactorOnLeft_>>
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


template <typename Derived, typename Rhs, int ProductType>
struct Eigen::internal::generic_product_impl<Hoppy::LowRankMatrixBase<Derived>,
                                             Rhs,
                                             Eigen::SparseShape,
                                             Eigen::DenseShape,
                                             ProductType>
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
		using ProductScalar = std::common_type_t<typename traits<Derived>::Scalar, typename traits<Rhs>::Scalar>;
		using Intermediate = Eigen::Matrix<ProductScalar, Eigen::Dynamic, traits<Rhs>::ColsAtCompileTime>;

		const Intermediate projected = lowRank.rightVectors().adjoint() * rhs;
		const Intermediate weighted = lowRank.coefficients().template cast<ProductScalar>().asDiagonal() * projected;
		destination.noalias() += alpha * lowRank.leftVectors() * weighted;
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

	[[nodiscard]] constexpr Eigen::Index rows() const noexcept { return asDerived().rowsImpl(); }
	[[nodiscard]] constexpr Eigen::Index cols() const noexcept { return asDerived().colsImpl(); }
	[[nodiscard]] Eigen::Index termCount() const noexcept { return asDerived().termCountImpl(); }

	[[nodiscard]] decltype(auto) coefficients() const noexcept { return asDerived().coefficientsImpl(); }
	[[nodiscard]] decltype(auto) leftVectors() const noexcept { return asDerived().leftVectorsImpl(); }
	[[nodiscard]] decltype(auto) rightVectors() const noexcept { return asDerived().rightVectorsImpl(); }

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

	template <typename Factor,
	          std::enable_if_t<Detail::IsScalarProductCompatible<Scalar, std::decay_t<Factor>>::value, int> = 0>
	[[nodiscard]] auto operator*(Factor&& factor) const &
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
	[[nodiscard]] auto operator*(Factor&& factor) const &&
	{
		return Detail::MakeProduct<false>(Derived{asDerived()}, std::forward<Factor>(factor));
	}

	template <typename Factor,
	          std::enable_if_t<Detail::IsScalarQuotientCompatible<Scalar, std::decay_t<Factor>>::value, int> = 0>
	/// Division deliberately follows the underlying scalar's division-by-zero behavior.
	[[nodiscard]] auto operator/(Factor&& factor) const &
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
	[[nodiscard]] auto operator/(Factor&& factor) const &&
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
	[[nodiscard]] auto operator+(const Other& other) const
	{
		return Detail::AddLowRank<false>(asDerived(), other);
	}

	template <typename Other,
	          std::enable_if_t<Detail::IsLowRankExpression<std::decay_t<Other>>::value
	                                   && std::is_same_v<Scalar, typename std::decay_t<Other>::Scalar>,
	                           int> = 0>
	[[nodiscard]] auto operator-(const Other& other) const
	{
		return Detail::AddLowRank<true>(asDerived(), other);
	}

	[[nodiscard]] Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime> toDense() const
	{
		using DenseMatrix = Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>;
		if (termCount() == 0)
		{
			return DenseMatrix::Zero(rows(), cols());
		}
		return leftVectors() * coefficients().template cast<Scalar>().asDiagonal() * rightVectors().adjoint();
	}

	[[nodiscard]] Eigen::Matrix<Scalar, 1, ColsAtCompileTime> row(const Eigen::Index index) const
	{
		eigen_assert(index >= 0 && index < rows() && "Low-rank matrix row index is out of range");
		if (termCount() == 0)
		{
			return Eigen::Matrix<Scalar, 1, ColsAtCompileTime>::Zero(1, cols());
		}
		return leftVectors().row(index) * coefficients().template cast<Scalar>().asDiagonal()
		       * rightVectors().adjoint();
	}

	[[nodiscard]] Eigen::Matrix<Scalar, RowsAtCompileTime, 1> col(const Eigen::Index index) const
	{
		eigen_assert(index >= 0 && index < cols() && "Low-rank matrix column index is out of range");
		if (termCount() == 0)
		{
			return Eigen::Matrix<Scalar, RowsAtCompileTime, 1>::Zero(rows());
		}
		return leftVectors() * (coefficients().template cast<Scalar>().asDiagonal()
		       * rightVectors().row(index).conjugate().transpose());
	}

	[[nodiscard]] Eigen::Vector<Scalar,
	                            Eigen::internal::min_size_prefer_dynamic(RowsAtCompileTime, ColsAtCompileTime)>
	diagonal() const
	{
		constexpr int DiagonalSize =
		        Eigen::internal::min_size_prefer_dynamic(RowsAtCompileTime, ColsAtCompileTime);
		using DiagonalVector = Eigen::Vector<Scalar, DiagonalSize>;
		const Eigen::Index size = (std::min)(rows(), cols());
		if (termCount() == 0)
		{
			return DiagonalVector::Zero(size);
		}
		return (leftVectors().topRows(size).array() * rightVectors().topRows(size).conjugate().array()).matrix()
		       * coefficients().template cast<Scalar>();
	}

	[[nodiscard]] RealScalar squaredNorm() const
	{
		if (termCount() == 0)
		{
			return RealScalar{};
		}

		const Eigen::MatrixX<Scalar> coefficients = this->coefficients().template cast<Scalar>();
		const Eigen::MatrixX<Scalar> coefficientProducts = coefficients.conjugate() * coefficients.transpose();
		const Eigen::MatrixX<Scalar> leftGram = leftVectors().adjoint() * leftVectors();
		const Eigen::MatrixX<Scalar> rightGram = rightVectors().adjoint() * rightVectors();
		const RealScalar result = Eigen::numext::real(
		        coefficientProducts.cwiseProduct(leftGram).cwiseProduct(rightGram.conjugate()).sum());
		return (std::max)(RealScalar{0}, result);
	}

	[[nodiscard]] RealScalar norm() const { return std::sqrt(squaredNorm()); }

	[[nodiscard]] auto transpose() const & { return Detail::MakeTranspose(asDerived()); }
	[[nodiscard]] auto transpose() && { return Detail::MakeTranspose(std::move(asDerived())); }
	[[nodiscard]] auto transpose() const && { return Detail::MakeTranspose(Derived{asDerived()}); }

	[[nodiscard]] auto conjugate() const & { return Detail::MakeConjugate(asDerived()); }
	[[nodiscard]] auto conjugate() && { return Detail::MakeConjugate(std::move(asDerived())); }
	[[nodiscard]] auto conjugate() const && { return Detail::MakeConjugate(Derived{asDerived()}); }

	[[nodiscard]] auto adjoint() const & { return Detail::MakeAdjoint(asDerived()); }
	[[nodiscard]] auto adjoint() && { return Detail::MakeAdjoint(std::move(asDerived())); }
	[[nodiscard]] auto adjoint() const && { return Detail::MakeAdjoint(Derived{asDerived()}); }

protected:
	constexpr LowRankMatrixBase() noexcept = default;
	LowRankMatrixBase(const LowRankMatrixBase&) = default;
	LowRankMatrixBase(LowRankMatrixBase&&) noexcept = default;
	LowRankMatrixBase& operator=(const LowRankMatrixBase&) = default;
	LowRankMatrixBase& operator=(LowRankMatrixBase&&) noexcept = default;
	~LowRankMatrixBase() = default;

private:
	[[nodiscard]] constexpr const Derived& asDerived() const noexcept { return static_cast<const Derived&>(*this); }
	[[nodiscard]] constexpr Derived& asDerived() noexcept { return static_cast<Derived&>(*this); }
};


namespace Hoppy::Detail
{
	struct IdentityLowRankOperation
	{
		static constexpr bool SwapsDimensions = false;

		template <typename Matrix>
		[[nodiscard]] static decltype(auto) Coefficients(const Matrix& matrix) { return matrix.coefficients(); }
		template <typename Matrix>
		[[nodiscard]] static decltype(auto) LeftVectors(const Matrix& matrix) { return matrix.leftVectors(); }
		template <typename Matrix>
		[[nodiscard]] static decltype(auto) RightVectors(const Matrix& matrix) { return matrix.rightVectors(); }
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
		[[nodiscard]] static decltype(auto) Coefficients(const Matrix& matrix) { return matrix.coefficients(); }
		template <typename Matrix>
		[[nodiscard]] static auto LeftVectors(const Matrix& matrix) { return matrix.rightVectors().conjugate(); }
		template <typename Matrix>
		[[nodiscard]] static auto RightVectors(const Matrix& matrix) { return matrix.leftVectors().conjugate(); }
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
		[[nodiscard]] static auto Coefficients(const Matrix& matrix) { return matrix.coefficients().conjugate(); }
		template <typename Matrix>
		[[nodiscard]] static auto LeftVectors(const Matrix& matrix) { return matrix.leftVectors().conjugate(); }
		template <typename Matrix>
		[[nodiscard]] static auto RightVectors(const Matrix& matrix) { return matrix.rightVectors().conjugate(); }
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
		[[nodiscard]] static auto Coefficients(const Matrix& matrix) { return matrix.coefficients().conjugate(); }
		template <typename Matrix>
		[[nodiscard]] static decltype(auto) LeftVectors(const Matrix& matrix) { return matrix.rightVectors(); }
		template <typename Matrix>
		[[nodiscard]] static decltype(auto) RightVectors(const Matrix& matrix) { return matrix.leftVectors(); }
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
	    : Eigen::internal::has_ReturnType<Eigen::ScalarBinaryOpTraits<
	              LeftScalar,
	              RightScalar,
	              Eigen::internal::scalar_product_op<LeftScalar, RightScalar>>>
	{
	};

	template <typename LeftScalar, typename RightScalar>
	struct IsScalarQuotientCompatible
	    : Eigen::internal::has_ReturnType<Eigen::ScalarBinaryOpTraits<
	              LeftScalar,
	              RightScalar,
	              Eigen::internal::scalar_quotient_op<LeftScalar, RightScalar>>>
	{
	};

	struct MultiplyLowRankOperation
	{
		template <typename MatrixScalar, typename Factor, bool FactorOnLeft>
		using ResultScalar = std::conditional_t<
		        FactorOnLeft,
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

	inline std::size_t CheckedBufferSize(const Eigen::Index dimension, const std::size_t termCount)
	{
		eigen_assert(dimension >= 0 && "A matrix dimension cannot be negative");

		const auto unsignedDimension = static_cast<std::size_t>(dimension);
		if (unsignedDimension != 0 && termCount > std::numeric_limits<std::size_t>::max() / unsignedDimension)
		{
			throw std::length_error("Low-rank matrix storage size overflow");
		}
		return unsignedDimension * termCount;
	}
}


template <typename Scalar_, int RowsAtCompileTime_, int ColsAtCompileTime_>
class Hoppy::LowRankMatrix
    : public LowRankMatrixBase<LowRankMatrix<Scalar_, RowsAtCompileTime_, ColsAtCompileTime_>>
{
	static_assert(RowsAtCompileTime_ == Eigen::Dynamic || RowsAtCompileTime_ >= 0);
	static_assert(ColsAtCompileTime_ == Eigen::Dynamic || ColsAtCompileTime_ >= 0);

public:
	using Base = LowRankMatrixBase<LowRankMatrix>;
	friend Base;
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
	static constexpr int Flags = 0;

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

	template <typename OtherDerived,
	          std::enable_if_t<std::is_same_v<Scalar, typename OtherDerived::Scalar>, int> = 0>
	explicit LowRankMatrix(const LowRankMatrixBase<OtherDerived>& other) : LowRankMatrix(other.rows(), other.cols())
	{
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

	[[nodiscard]] std::size_t capacity() const noexcept { return m_Coefficients.capacity(); }

	void reserve(const Eigen::Index termCapacity)
	{
		eigen_assert(termCapacity >= 0 && "A term capacity cannot be negative");
		const auto capacity = static_cast<std::size_t>(termCapacity);
		m_Coefficients.reserve(capacity);
		m_LeftVectorBuffer.reserve(Detail::CheckedBufferSize(rows(), capacity));
		m_RightVectorBuffer.reserve(Detail::CheckedBufferSize(cols(), capacity));
	}

	void clear() noexcept
	{
		m_Coefficients.clear();
		m_LeftVectorBuffer.clear();
		m_RightVectorBuffer.clear();
	}

	LowRankMatrix& operator+=(const LowRankMatrix& other)
	{
		eigen_assert(rows() == other.rows() && cols() == other.cols()
		             && "Low-rank matrix dimensions do not agree");
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
		eigen_assert(rows() == other.rows() && cols() == other.cols()
		             && "Low-rank matrix dimensions do not agree");
		if (this == &other)
		{
			clear();
			return *this;
		}
		return addTerms(-other.coefficients(), other.leftVectors(), other.rightVectors());
	}

	template <typename LeftDerived, typename RightDerived>
	LowRankMatrix& addTerm(const Scalar& coefficient,
	                       const Eigen::MatrixBase<LeftDerived>& leftVector,
	                       const Eigen::MatrixBase<RightDerived>& rightVector)
	{
		if (coefficient == Scalar{})
		{
			return *this;
		}

		validateVector(leftVector, rows());
		validateVector(rightVector, cols());
		const Scalar evaluatedCoefficient = coefficient;
		const Eigen::VectorX<Scalar> evaluatedLeft = leftVector.reshaped();
		const Eigen::VectorX<Scalar> evaluatedRight = rightVector.reshaped();
		appendEvaluatedTerm(evaluatedCoefficient, evaluatedLeft, evaluatedRight);
		return *this;
	}

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

		CoefficientVector filteredCoefficients(nonzeroCount);
		Eigen::MatrixX<Scalar> filteredLeft(rows(), nonzeroCount);
		Eigen::MatrixX<Scalar> filteredRight(cols(), nonzeroCount);
		Eigen::Index destinationIndex = 0;
		for (Eigen::Index sourceIndex = 0; sourceIndex < evaluatedCoefficients.size(); sourceIndex++)
		{
			if (evaluatedCoefficients[sourceIndex] != Scalar{})
			{
				filteredCoefficients[destinationIndex] = evaluatedCoefficients[sourceIndex];
				filteredLeft.col(destinationIndex) = leftVectors.col(sourceIndex);
				filteredRight.col(destinationIndex) = rightVectors.col(sourceIndex);
				destinationIndex++;
			}
		}

		const auto finalCount = static_cast<std::size_t>(termCount() + nonzeroCount);
		reserve(static_cast<Eigen::Index>(finalCount));
		m_Coefficients.insert(m_Coefficients.end(), filteredCoefficients.data(),
		                      filteredCoefficients.data() + filteredCoefficients.size());
		appendBuffer(m_LeftVectorBuffer, filteredLeft.data(), filteredLeft.size());
		appendBuffer(m_RightVectorBuffer, filteredRight.data(), filteredRight.size());
		return *this;
	}

private:
	[[nodiscard]] constexpr Eigen::Index rowsImpl() const noexcept { return m_Rows.value(); }
	[[nodiscard]] constexpr Eigen::Index colsImpl() const noexcept { return m_Cols.value(); }
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
	}

	void validateTermIndex(const Eigen::Index index) const
	{
		eigen_assert(index >= 0 && index < termCount() && "Low-rank term index is out of range");
	}

	void appendEvaluatedTerm(const Scalar& coefficient,
	                         const Eigen::VectorX<Scalar>& leftVector,
	                         const Eigen::VectorX<Scalar>& rightVector)
	{
		reserve(termCount() + 1);
		m_Coefficients.push_back(coefficient);
		appendBuffer(m_LeftVectorBuffer, leftVector.data(), leftVector.size());
		appendBuffer(m_RightVectorBuffer, rightVector.data(), rightVector.size());
	}

	static void appendBuffer(std::vector<Scalar>& destination, const Scalar* const source, const Eigen::Index size)
	{
		if (size != 0)
		{
			destination.insert(destination.end(), source, source + size);
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
    : public LowRankMatrixBase<SingleFactorLowRankMatrix<Scalar_, StructurePolicy_, DimensionAtCompileTime_>>
{
	static_assert(DimensionAtCompileTime_ == Eigen::Dynamic || DimensionAtCompileTime_ >= 0);

public:
	using Base = LowRankMatrixBase<SingleFactorLowRankMatrix>;
	friend Base;
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
	static constexpr int Flags = 0;

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

	[[nodiscard]] std::size_t capacity() const noexcept { return m_Coefficients.capacity(); }

	void reserve(const Eigen::Index termCapacity)
	{
		eigen_assert(termCapacity >= 0 && "A term capacity cannot be negative");
		const auto capacity = static_cast<std::size_t>(termCapacity);
		m_Coefficients.reserve(capacity);
		m_VectorBuffer.reserve(Detail::CheckedBufferSize(rows(), capacity));
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
	SingleFactorLowRankMatrix& addTerm(const CoefficientScalar& coefficient,
	                                   const Eigen::MatrixBase<VectorDerived>& vector)
	{
		if (coefficient == CoefficientScalar{})
		{
			return *this;
		}

		validateVector(vector);
		const CoefficientScalar evaluatedCoefficient = coefficient;
		const Eigen::VectorX<Scalar> evaluatedVector = vector.reshaped();
		reserve(termCount() + 1);
		m_Coefficients.push_back(evaluatedCoefficient);
		appendBuffer(m_VectorBuffer, evaluatedVector.data(), evaluatedVector.size());
		return *this;
	}

	template <typename CoefficientsDerived,
	          typename VectorsDerived,
	          std::enable_if_t<std::is_convertible_v<typename CoefficientsDerived::Scalar, CoefficientScalar>, int> = 0>
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

		CoefficientVector filteredCoefficients(nonzeroCount);
		Eigen::MatrixX<Scalar> filteredVectors(rows(), nonzeroCount);
		Eigen::Index destinationIndex = 0;
		for (Eigen::Index sourceIndex = 0; sourceIndex < evaluatedCoefficients.size(); sourceIndex++)
		{
			if (evaluatedCoefficients[sourceIndex] != CoefficientScalar{})
			{
				filteredCoefficients[destinationIndex] = evaluatedCoefficients[sourceIndex];
				filteredVectors.col(destinationIndex) = vectors.col(sourceIndex);
				destinationIndex++;
			}
		}

		const auto finalCount = static_cast<std::size_t>(termCount() + nonzeroCount);
		reserve(static_cast<Eigen::Index>(finalCount));
		m_Coefficients.insert(m_Coefficients.end(), filteredCoefficients.data(),
		                      filteredCoefficients.data() + filteredCoefficients.size());
		appendBuffer(m_VectorBuffer, filteredVectors.data(), filteredVectors.size());
		return *this;
	}

	[[nodiscard]] LowRankMatrix<Scalar, DimensionAtCompileTime_, DimensionAtCompileTime_> toGeneral() const
	{
		return {this->coefficients(), this->leftVectors(), StructurePolicy::RightFactor(this->leftVectors())};
	}

private:
	[[nodiscard]] constexpr Eigen::Index rowsImpl() const noexcept { return m_Dimension.value(); }
	[[nodiscard]] constexpr Eigen::Index colsImpl() const noexcept { return m_Dimension.value(); }
	[[nodiscard]] Eigen::Index termCountImpl() const noexcept
	{
		return static_cast<Eigen::Index>(m_Coefficients.size());
	}

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
	}

	void validateTermIndex(const Eigen::Index index) const
	{
		eigen_assert(index >= 0 && index < termCount() && "Low-rank term index is out of range");
	}

	static void appendBuffer(std::vector<Scalar>& destination, const Scalar* const source, const Eigen::Index size)
	{
		if (size != 0)
		{
			destination.insert(destination.end(), source, source + size);
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
    : public LowRankMatrixBase<LowRankUnaryExpr<Nested_, Operation_>>
{
public:
	using Base = LowRankMatrixBase<LowRankUnaryExpr>;
	friend Base;
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

	explicit LowRankUnaryExpr(Nested nested) : m_Nested(std::forward<Nested>(nested)) {}

private:
	[[nodiscard]] Eigen::Index rowsImpl() const noexcept
	{
		return Operation::SwapsDimensions ? m_Nested.cols() : m_Nested.rows();
	}
	[[nodiscard]] Eigen::Index colsImpl() const noexcept
	{
		return Operation::SwapsDimensions ? m_Nested.rows() : m_Nested.cols();
	}
	[[nodiscard]] Eigen::Index termCountImpl() const noexcept { return m_Nested.termCount(); }
	[[nodiscard]] decltype(auto) coefficientsImpl() const { return Operation::Coefficients(m_Nested); }
	[[nodiscard]] decltype(auto) leftVectorsImpl() const { return Operation::LeftVectors(m_Nested); }
	[[nodiscard]] decltype(auto) rightVectorsImpl() const { return Operation::RightVectors(m_Nested); }
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
    : public LowRankMatrixBase<LowRankScalarExpr<Nested_, Factor_, Operation_, FactorOnLeft_>>
{
public:
	using Base = LowRankMatrixBase<LowRankScalarExpr>;
	friend Base;
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
	using CoefficientScalar =
	        typename Operation::template ResultScalar<NestedCoefficientScalar, Factor, FactorOnLeft_>;
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
	[[nodiscard]] Eigen::Index rowsImpl() const noexcept { return m_Nested.rows(); }
	[[nodiscard]] Eigen::Index colsImpl() const noexcept { return m_Nested.cols(); }
	[[nodiscard]] Eigen::Index termCountImpl() const noexcept { return m_Nested.termCount(); }
	[[nodiscard]] auto coefficientsImpl() const
	{
		return Operation::template CoefficientsResult<CoefficientScalar, FactorOnLeft_>(m_Nested.coefficients(),
		                                                                                 m_Factor);
	}
	[[nodiscard]] auto leftVectorsImpl() const { return m_Nested.leftVectors().template cast<Scalar>(); }
	[[nodiscard]] auto rightVectorsImpl() const { return m_Nested.rightVectors().template cast<Scalar>(); }
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


namespace Hoppy::Detail
{
	template <typename Nested, typename Factor, typename Operation, bool FactorOnLeft>
	struct UnaryExpressionTraits<LowRankScalarExpr<Nested, Factor, Operation, FactorOnLeft>>
	    : UnaryOperationsForStructure<typename LowRankScalarExpr<Nested, Factor, Operation, FactorOnLeft>::StructurePolicy>
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
	inline constexpr int MergedCompileTimeSize = LeftSize == RightSize   ? LeftSize
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
		using Result = std::conditional_t<
		        PreservesStructure,
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
			result.addTerms(left.coefficients(), left.leftVectors());
			if constexpr (Subtract)
			{
				result.addTerms(-right.coefficients(), right.leftVectors());
			}
			else
			{
				result.addTerms(right.coefficients(), right.leftVectors());
			}
			return result;
		}
		else
		{
			Result result(left.rows(), left.cols());
			result.reserve(left.termCount() + right.termCount());
			result.addTerms(left.coefficients(), left.leftVectors(), left.rightVectors());
			if constexpr (Subtract)
			{
				result.addTerms(-right.coefficients(), right.leftVectors(), right.rightVectors());
			}
			else
			{
				result.addTerms(right.coefficients(), right.leftVectors(), right.rightVectors());
			}
			return result;
		}
	}
}
