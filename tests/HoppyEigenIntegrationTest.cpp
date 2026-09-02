// SPDX-License-Identifier: MIT

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/ForwardDeclarations.hpp>

#include <Eigen/Dense>

#include <complex>
#include <type_traits>
#include <utility>

namespace Hoppy::Test::Spike
{
	struct BlockStorage;
	struct VectorStorage;
	struct BlockShape;
	struct VectorShape;

	template <typename TStorage>
	class Expression;
}  // namespace Hoppy::Test::Spike

namespace Eigen::internal
{
	template <typename TStorage>
	struct traits<Hoppy::Test::Spike::Expression<TStorage>>
	{
		using Scalar = double;
		using StorageKind = TStorage;
		using XprKind = MatrixXpr;
		using StorageIndex = Eigen::Index;
		static constexpr int Flags = NestByRefBit;
		static constexpr int RowsAtCompileTime = Dynamic;
		static constexpr int ColsAtCompileTime = Dynamic;
		static constexpr int MaxRowsAtCompileTime = Dynamic;
		static constexpr int MaxColsAtCompileTime = Dynamic;
	};
}  // namespace Eigen::internal

namespace Hoppy::Test::Spike
{

	template <typename Derived>
	class ExpressionBase : public Eigen::EigenBase<Derived>
	{
	public:
		using Scalar = typename Eigen::internal::traits<Derived>::Scalar;
		using CoeffReturnType = Scalar;
		static constexpr int SizeAtCompileTime = Eigen::Dynamic;
		static constexpr int MaxSizeAtCompileTime = Eigen::Dynamic;
		static constexpr bool IsVectorAtCompileTime = false;

		using Eigen::EigenBase<Derived>::const_cast_derived;
		using Eigen::EigenBase<Derived>::derived;

		template <typename UnaryOp>
		auto unaryExpr(const UnaryOp& operation) const
		{
			return Eigen::CwiseUnaryOp<UnaryOp, const Derived>(derived(), operation);
		}

		template <typename OtherDerived, typename BinaryOp>
		auto binaryExpr(const ExpressionBase<OtherDerived>& other, const BinaryOp& operation) const
		{
			return Eigen::CwiseBinaryOp<BinaryOp, const Derived, const OtherDerived>(
			        derived(), other.derived(), operation);
		}

		auto transpose() const
		{
			return Eigen::Transpose<const Derived>(derived());
		}
	};

	template <typename TStorage>
	class Expression : public ExpressionBase<Expression<TStorage>>
	{
	public:
		using Scalar = double;
		using StorageIndex = Eigen::Index;
		using Nested = const Expression&;
		static constexpr int RowsAtCompileTime = Eigen::Dynamic;
		static constexpr int ColsAtCompileTime = Eigen::Dynamic;
		static constexpr int MaxRowsAtCompileTime = Eigen::Dynamic;
		static constexpr int MaxColsAtCompileTime = Eigen::Dynamic;
		static constexpr int SizeAtCompileTime = Eigen::Dynamic;
		static constexpr int MaxSizeAtCompileTime = Eigen::Dynamic;
		static constexpr int Flags = Eigen::NestByRefBit;

		explicit Expression(Eigen::MatrixXd values) : values_(std::move(values)) {}

		Eigen::Index rows() const { return values_.rows(); }
		Eigen::Index cols() const { return values_.cols(); }
		double coeff(Eigen::Index row, Eigen::Index column) const { return values_.coeff(row, column); }

	private:
		Eigen::MatrixXd values_;
	};

	using BlockExpression = Expression<BlockStorage>;
	using VectorExpression = Expression<VectorStorage>;

	template <typename ExpressionType>
	Eigen::MatrixXd evaluateWithEigen(const ExpressionType& expression)
	{
		Eigen::internal::evaluator<ExpressionType> evaluator(expression);
		Eigen::MatrixXd result(expression.rows(), expression.cols());
		for (Eigen::Index column = 0; column < expression.cols(); ++column)
			for (Eigen::Index row = 0; row < expression.rows(); ++row)
				result(row, column) = evaluator.coeff(row, column);
		return result;
	}
}  // namespace Hoppy::Test::Spike

namespace Eigen::internal
{
	template <>
	struct storage_kind_to_shape<Hoppy::Test::Spike::BlockStorage>
	{
		using Shape = Hoppy::Test::Spike::BlockShape;
	};

	template <>
	struct storage_kind_to_shape<Hoppy::Test::Spike::VectorStorage>
	{
		using Shape = Hoppy::Test::Spike::VectorShape;
	};

	template <>
	struct promote_storage_type<Hoppy::Test::Spike::BlockStorage, Hoppy::Test::Spike::BlockStorage>
	{
		using ret = Hoppy::Test::Spike::BlockStorage;
	};

	template <>
	struct promote_storage_type<Hoppy::Test::Spike::VectorStorage, Hoppy::Test::Spike::VectorStorage>
	{
		using ret = Hoppy::Test::Spike::VectorStorage;
	};

	template <typename BinaryOp>
	struct cwise_promote_storage_type<
	        Hoppy::Test::Spike::BlockStorage,
	        Hoppy::Test::Spike::BlockStorage,
	        BinaryOp>
	{
		using ret = Hoppy::Test::Spike::BlockStorage;
	};

	template <typename BinaryOp>
	struct cwise_promote_storage_type<
	        Hoppy::Test::Spike::VectorStorage,
	        Hoppy::Test::Spike::VectorStorage,
	        BinaryOp>
	{
		using ret = Hoppy::Test::Spike::VectorStorage;
	};

	template <typename TStorage>
	struct evaluator<Hoppy::Test::Spike::Expression<TStorage>>
	    : evaluator_base<Hoppy::Test::Spike::Expression<TStorage>>
	{
		using ExpressionType = Hoppy::Test::Spike::Expression<TStorage>;
		static constexpr int CoeffReadCost = NumTraits<double>::ReadCost;
		static constexpr int Flags = 0;
		static constexpr int Alignment = 0;

		explicit evaluator(const ExpressionType& expression) : expression_(expression) {}
		double coeff(Eigen::Index row, Eigen::Index column) const { return expression_.coeff(row, column); }

	private:
		const ExpressionType& expression_;
	};
}  // namespace Eigen::internal

namespace Eigen
{
	template <typename UnaryOp, typename XprType>
	class CwiseUnaryOpImpl<UnaryOp, XprType, Hoppy::Test::Spike::BlockStorage>
	    : public Hoppy::Test::Spike::ExpressionBase<CwiseUnaryOp<UnaryOp, XprType>>
	{
	public:
		using Base = Hoppy::Test::Spike::ExpressionBase<CwiseUnaryOp<UnaryOp, XprType>>;
	};

	template <typename UnaryOp, typename XprType>
	class CwiseUnaryOpImpl<UnaryOp, XprType, Hoppy::Test::Spike::VectorStorage>
	    : public Hoppy::Test::Spike::ExpressionBase<CwiseUnaryOp<UnaryOp, XprType>>
	{
	public:
		using Base = Hoppy::Test::Spike::ExpressionBase<CwiseUnaryOp<UnaryOp, XprType>>;
	};

	template <typename BinaryOp, typename Lhs, typename Rhs>
	class CwiseBinaryOpImpl<BinaryOp, Lhs, Rhs, Hoppy::Test::Spike::BlockStorage>
	    : public Hoppy::Test::Spike::ExpressionBase<CwiseBinaryOp<BinaryOp, Lhs, Rhs>>
	{
	public:
		using Base = Hoppy::Test::Spike::ExpressionBase<CwiseBinaryOp<BinaryOp, Lhs, Rhs>>;
	};

	template <typename BinaryOp, typename Lhs, typename Rhs>
	class CwiseBinaryOpImpl<BinaryOp, Lhs, Rhs, Hoppy::Test::Spike::VectorStorage>
	    : public Hoppy::Test::Spike::ExpressionBase<CwiseBinaryOp<BinaryOp, Lhs, Rhs>>
	{
	public:
		using Base = Hoppy::Test::Spike::ExpressionBase<CwiseBinaryOp<BinaryOp, Lhs, Rhs>>;
	};

	template <typename XprType>
	class TransposeImpl<XprType, Hoppy::Test::Spike::BlockStorage>
	    : public Hoppy::Test::Spike::ExpressionBase<Transpose<XprType>>
	{
	public:
		using Base = Hoppy::Test::Spike::ExpressionBase<Transpose<XprType>>;
	};

	template <typename XprType>
	class TransposeImpl<XprType, Hoppy::Test::Spike::VectorStorage>
	    : public Hoppy::Test::Spike::ExpressionBase<Transpose<XprType>>
	{
	public:
		using Base = Hoppy::Test::Spike::ExpressionBase<Transpose<XprType>>;
	};
}  // namespace Eigen

namespace Hoppy::Test::Spike
{
	template <typename T, typename = void>
	struct HasReturnType : std::false_type
	{};

	template <typename T>
	struct HasReturnType<T, std::void_t<typename T::ReturnType>> : std::true_type
	{};

	using SumOp = Eigen::internal::scalar_sum_op<double, double>;
	using NegateOp = Eigen::internal::scalar_opposite_op<double>;
	using Unary = Eigen::CwiseUnaryOp<NegateOp, const BlockExpression>;
	using Binary = Eigen::CwiseBinaryOp<SumOp, const BlockExpression, const BlockExpression>;
	using Transposed = Eigen::Transpose<const BlockExpression>;

	static_assert(std::is_same_v<typename Eigen::internal::traits<BlockExpression>::StorageKind, BlockStorage>);
	static_assert(std::is_same_v<typename Eigen::internal::storage_kind_to_shape<BlockStorage>::Shape, BlockShape>);
	static_assert(std::is_same_v<typename Eigen::internal::promote_storage_type<BlockStorage, BlockStorage>::ret,
	                             BlockStorage>);
	static_assert(std::is_base_of_v<ExpressionBase<Unary>, Eigen::CwiseUnaryOpImpl<NegateOp, const BlockExpression,
	                                                                             BlockStorage>>);
	static_assert(std::is_base_of_v<ExpressionBase<Binary>, Eigen::CwiseBinaryOpImpl<SumOp, const BlockExpression,
	                                                                                const BlockExpression,
	                                                                                BlockStorage>>);
	static_assert(std::is_base_of_v<ExpressionBase<Transposed>,
	                                Eigen::TransposeImpl<const BlockExpression, BlockStorage>>);
	static_assert(std::is_same_v<typename Eigen::internal::ref_selector<BlockExpression>::type,
	                             const BlockExpression&>);
	static_assert(std::is_same_v<typename Eigen::internal::evaluator_traits<BlockExpression>::Shape, BlockShape>);
	static_assert(std::is_same_v<typename Eigen::ScalarBinaryOpTraits<double, double, SumOp>::ReturnType, double>);
	static_assert(!HasReturnType<Eigen::ScalarBinaryOpTraits<int, std::complex<double>, SumOp>>::value);
	static_assert(std::is_same_v<Eigen::MatrixX<double>, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>);
	static_assert(std::is_same_v<Eigen::VectorX<double>, Eigen::Vector<double, Eigen::Dynamic>>);
	static_assert(std::is_constructible_v<Eigen::DiagonalWrapper<const Eigen::VectorXd>, const Eigen::VectorXd&>);
	static_assert(std::is_class_v<Eigen::internal::generic_product_impl<Eigen::MatrixXd, Eigen::MatrixXd>>);
}  // namespace Hoppy::Test::Spike

TEST_CASE("Eigen 5 custom storage dispatch evaluates unary, binary, and transpose expressions")
{
	using namespace Hoppy::Test::Spike;
	Eigen::MatrixXd values(2, 2);
	values << 1.0, 2.0, 3.0, 4.0;
	const BlockExpression left(values);
	const BlockExpression right(Eigen::MatrixXd::Constant(2, 2, 0.5));

	Hoppy::Test::requireApprox(evaluateWithEigen(left.unaryExpr(NegateOp{})), -values);
	Hoppy::Test::requireApprox(evaluateWithEigen(left.binaryExpr(right, SumOp{})),
	                           values + Eigen::MatrixXd::Constant(2, 2, 0.5));
	Hoppy::Test::requireApprox(evaluateWithEigen(left.transpose()), values.transpose());
}

TEST_CASE("Eigen 5 diagonal wrapper and dense product dispatch evaluate numerically")
{
	Eigen::VectorXd diagonal(3);
	diagonal << 2.0, 3.0, 5.0;
	const Eigen::MatrixXd dense = diagonal.asDiagonal();
	Hoppy::Test::requireApprox(dense, diagonal.asDiagonal().toDenseMatrix());

	Eigen::MatrixXd lhs(2, 3);
	lhs << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
	Eigen::MatrixXd rhs(3, 1);
	rhs << 7.0, 8.0, 9.0;
	const Eigen::MatrixXd product = lhs * rhs;
	Hoppy::Test::requireApprox(product, lhs.lazyProduct(rhs));
}

TEST_CASE("Eigen ref_selector safely owns temporary expression nodes")
{
	using namespace Hoppy::Test::Spike;
	Eigen::MatrixXd values(2, 2);
	values << 1.0, 2.0, 3.0, 4.0;
	const BlockExpression source(values);
	const auto nested = source.unaryExpr(NegateOp{}).unaryExpr(NegateOp{});
	Hoppy::Test::requireApprox(evaluateWithEigen(nested), values);
}

#ifndef EIGEN_NO_DEBUG
TEST_CASE("forced Eigen assertion hook throws at a non-noexcept assertion site")
{
	Eigen::MatrixXd lhs(2, 2);
	Eigen::MatrixXd rhs(3, 1);
	REQUIRE_THROWS_AS(lhs.lazyProduct(rhs), Hoppy::Test::EigenAssertionFailure);
}
#endif
