// SPDX-License-Identifier: MIT

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/ForwardDeclarations.hpp>

#include <Eigen/Dense>

#include <complex>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace Hoppy::Test::Spike
{
	enum class Packing
	{
		Lower,
		Upper
	};

	struct SymmetricTag
	{};
	struct AntiSymmetricTag
	{};
	struct HermitianTag
	{};
	struct AntiHermitianTag
	{};

	template <typename Scalar, typename Tag>
	using normalized_tag_t = std::conditional_t<
	        Eigen::NumTraits<Scalar>::IsComplex == 0,
	        std::conditional_t<std::is_same_v<Tag, HermitianTag>, SymmetricTag,
	                           std::conditional_t<std::is_same_v<Tag, AntiHermitianTag>, AntiSymmetricTag, Tag>>,
	        Tag>;

	template <typename TScalar, typename Tag, Packing TPacking>
	class MappedMatrix
	{
	public:
		using Scalar = TScalar;
		using Structure = normalized_tag_t<std::remove_const_t<Scalar>, Tag>;
		static constexpr Packing PackingValue = TPacking;
	};

	template <typename Scalar, Packing TPacking>
	using Symmetric = MappedMatrix<Scalar, normalized_tag_t<Scalar, SymmetricTag>, TPacking>;
	template <typename Scalar, Packing TPacking>
	using AntiSymmetric = MappedMatrix<Scalar, normalized_tag_t<Scalar, AntiSymmetricTag>, TPacking>;
	template <typename Scalar, Packing TPacking>
	using Hermitian = MappedMatrix<Scalar, normalized_tag_t<Scalar, HermitianTag>, TPacking>;
	template <typename Scalar, Packing TPacking>
	using AntiHermitian = MappedMatrix<Scalar, normalized_tag_t<Scalar, AntiHermitianTag>, TPacking>;

	struct BlockStorage;
	struct VectorStorage;
	struct BlockShape;
	struct VectorShape;

	template <typename TStorage>
	class Expression;
}  // namespace Hoppy::Test::Spike

namespace Eigen
{
	template <typename Scalar, typename Tag, Hoppy::Test::Spike::Packing PackingValue, int MapOptions>
	class Map<Hoppy::Test::Spike::MappedMatrix<Scalar, Tag, PackingValue>, MapOptions, Stride<0, 0>>
	{
	public:
		using Index = Eigen::Index;
		using Value = std::remove_const_t<Scalar>;
		using StrideType = Stride<0, 0>;
		static constexpr bool IsAligned = (MapOptions & Eigen::Aligned) == Eigen::Aligned;

		Map(Scalar* data, Index dimension) : m_Data(data), m_Dimension(dimension)
		{
			eigen_assert(dimension >= 0);
			if constexpr (IsAligned)
				eigen_assert(reinterpret_cast<std::uintptr_t>(data) % EIGEN_MAX_ALIGN_BYTES == 0);
		}

		Index rows() const { return m_Dimension; }
		Index cols() const { return m_Dimension; }
		Scalar& coeffRef(Index row, Index column) const { return m_Data[offset(row, column)]; }
		const Value& coeff(Index row, Index column) const { return m_Data[offset(row, column)]; }

	private:
		Index offset(Index row, Index column) const
		{
			eigen_assert(row >= 0 && row < m_Dimension && column >= 0 && column < m_Dimension);
			if constexpr (PackingValue == Hoppy::Test::Spike::Packing::Lower)
			{
				if (row < column) std::swap(row, column);
				return row * (row + 1) / 2 + column;
			}
			if (column < row) std::swap(row, column);
			return column * (column + 1) / 2 + row;
		}

		Scalar* m_Data;
		Index m_Dimension;
	};

	template <typename Scalar, typename Tag, Hoppy::Test::Spike::Packing PackingValue, int MapOptions>
	class Map<const Hoppy::Test::Spike::MappedMatrix<Scalar, Tag, PackingValue>, MapOptions, Stride<0, 0>>
	{
	public:
		using Index = Eigen::Index;
		using StrideType = Stride<0, 0>;
		static constexpr bool IsAligned = (MapOptions & Eigen::Aligned) == Eigen::Aligned;

		Map(const Scalar* data, Index dimension) : m_Data(data), m_Dimension(dimension)
		{
			eigen_assert(dimension >= 0);
			if constexpr (IsAligned)
				eigen_assert(reinterpret_cast<std::uintptr_t>(data) % EIGEN_MAX_ALIGN_BYTES == 0);
		}

		Index rows() const { return m_Dimension; }
		Index cols() const { return m_Dimension; }
		const Scalar& coeff(Index row, Index column) const
		{
			if constexpr (PackingValue == Hoppy::Test::Spike::Packing::Lower)
			{
				if (row < column) std::swap(row, column);
				return m_Data[row * (row + 1) / 2 + column];
			}
			if (column < row) std::swap(row, column);
			return m_Data[column * (column + 1) / 2 + row];
		}

	private:
		const Scalar* m_Data;
		Index m_Dimension;
	};
}  // namespace Eigen

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

		explicit Expression(Eigen::MatrixXd values) : m_Values(std::move(values)) {}

		Eigen::Index rows() const { return m_Values.rows(); }
		Eigen::Index cols() const { return m_Values.cols(); }
		double coeff(Eigen::Index row, Eigen::Index column) const { return m_Values.coeff(row, column); }

	private:
		Eigen::MatrixXd m_Values;
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
	template <typename Derived>
	struct generic_xpr_base<Derived, MatrixXpr, Hoppy::Test::Spike::BlockStorage>
	{
		using type = Hoppy::Test::Spike::ExpressionBase<Derived>;
	};

	template <typename Derived>
	struct generic_xpr_base<Derived, MatrixXpr, Hoppy::Test::Spike::VectorStorage>
	{
		using type = Hoppy::Test::Spike::ExpressionBase<Derived>;
	};

	template <typename TStorage>
	struct plain_object_eval<Hoppy::Test::Spike::Expression<TStorage>, TStorage>
	{
		using type = Eigen::MatrixXd;
	};

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

		explicit evaluator(const ExpressionType& expression) : m_Expression(expression) {}
		double coeff(Eigen::Index row, Eigen::Index column) const { return m_Expression.coeff(row, column); }

	private:
		const ExpressionType& m_Expression;
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
	struct has_return_type : std::false_type
	{};

	template <typename T>
	struct has_return_type<T, std::void_t<typename T::ReturnType>> : std::true_type
	{};

	template <typename MapType>
	struct is_supported_map : std::false_type
	{};

	template <typename Scalar, typename Tag, Packing PackingValue, int MapOptions>
	struct is_supported_map<
	        Eigen::Map<MappedMatrix<Scalar, Tag, PackingValue>, MapOptions, Eigen::Stride<0, 0>>>
	    : std::true_type
	{};

	template <typename Scalar, typename Tag, Packing PackingValue, int MapOptions>
	struct is_supported_map<
	        Eigen::Map<const MappedMatrix<Scalar, Tag, PackingValue>, MapOptions, Eigen::Stride<0, 0>>>
	    : std::true_type
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
	static_assert(std::is_same_v<typename Eigen::internal::generic_xpr_base<
	                                     BlockExpression, Eigen::MatrixXpr, BlockStorage>::type,
	                             ExpressionBase<BlockExpression>>);
	static_assert(std::is_same_v<typename Eigen::internal::plain_object_eval<
	                                     BlockExpression, BlockStorage>::type,
	                             Eigen::MatrixXd>);
	static_assert(std::is_same_v<typename Eigen::ScalarBinaryOpTraits<double, double, SumOp>::ReturnType, double>);
	static_assert(!has_return_type<Eigen::ScalarBinaryOpTraits<int, std::complex<double>, SumOp>>::value);
	static_assert(std::is_same_v<Eigen::MatrixX<double>, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>);
	static_assert(std::is_same_v<Eigen::VectorX<double>, Eigen::Vector<double, Eigen::Dynamic>>);
	static_assert(std::is_constructible_v<Eigen::DiagonalWrapper<const Eigen::VectorXd>, const Eigen::VectorXd&>);
	static_assert(std::is_class_v<Eigen::internal::generic_product_impl<Eigen::MatrixXd, Eigen::MatrixXd>>);

	static_assert(std::is_same_v<Hermitian<double, Packing::Lower>, Symmetric<double, Packing::Lower>>);
	static_assert(std::is_same_v<AntiHermitian<double, Packing::Upper>,
	                             AntiSymmetric<double, Packing::Upper>>);
	static_assert(!std::is_same_v<Hermitian<std::complex<double>, Packing::Lower>,
	                              Symmetric<std::complex<double>, Packing::Lower>>);
	static_assert(!std::is_same_v<AntiHermitian<std::complex<double>, Packing::Upper>,
	                              AntiSymmetric<std::complex<double>, Packing::Upper>>);

	using LowerMap = Eigen::Map<Symmetric<double, Packing::Lower>>;
	using UpperMap = Eigen::Map<Symmetric<double, Packing::Upper>>;
	using ConstLowerMap = Eigen::Map<const Symmetric<double, Packing::Lower>>;
	using ConstUpperMap = Eigen::Map<const Symmetric<double, Packing::Upper>>;
	using AlignedLowerMap = Eigen::Map<Symmetric<double, Packing::Lower>, Eigen::Aligned>;
	using AlignedConstUpperMap = Eigen::Map<const Symmetric<double, Packing::Upper>, Eigen::Aligned>;
	using CustomStrideMap = Eigen::Map<Symmetric<double, Packing::Lower>, Eigen::Unaligned,
	                                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
	static_assert(is_supported_map<LowerMap>::value);
	static_assert(is_supported_map<UpperMap>::value);
	static_assert(is_supported_map<ConstLowerMap>::value);
	static_assert(is_supported_map<ConstUpperMap>::value);
	static_assert(is_supported_map<AlignedLowerMap>::value);
	static_assert(is_supported_map<AlignedConstUpperMap>::value);
	static_assert(!is_supported_map<CustomStrideMap>::value);
	static_assert(std::is_same_v<typename LowerMap::StrideType, Eigen::Stride<0, 0>>);
	static_assert(!LowerMap::IsAligned);
	static_assert(AlignedLowerMap::IsAligned);
}  // namespace Hoppy::Test::Spike

TEST_CASE("triangular spike maps mutable and const packed buffers in both orders")
{
	using namespace Hoppy::Test::Spike;
	alignas(EIGEN_MAX_ALIGN_BYTES) double lowerBuffer[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	LowerMap lower(lowerBuffer, 3);
	lower.coeffRef(0, 2) = 7.0;
	REQUIRE(lowerBuffer[3] == 7.0);
	REQUIRE(lower.coeff(2, 0) == 7.0);
	ConstLowerMap constLower(lowerBuffer, 3);
	REQUIRE(constLower.coeff(0, 2) == 7.0);

	alignas(EIGEN_MAX_ALIGN_BYTES) double upperBuffer[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	UpperMap mutableUpper(upperBuffer, 3);
	mutableUpper.coeffRef(2, 0) = 8.0;
	REQUIRE(upperBuffer[3] == 8.0);
	ConstUpperMap upper(upperBuffer, 3);
	REQUIRE(upper.coeff(0, 2) == 8.0);
	REQUIRE(upper.coeff(2, 0) == 8.0);

	AlignedLowerMap aligned(lowerBuffer, 3);
	REQUIRE(aligned.coeff(1, 1) == 3.0);
	AlignedConstUpperMap alignedConst(upperBuffer, 3);
	REQUIRE(alignedConst.coeff(1, 1) == 3.0);
}

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

#ifdef HOPPY_TEST_EIGEN_ASSERT_THROWS
TEST_CASE("forced Eigen assertion hook throws at a non-noexcept assertion site")
{
	Eigen::MatrixXd lhs(2, 2);
	Eigen::MatrixXd rhs(3, 1);
	REQUIRE_THROWS_AS(lhs.lazyProduct(rhs), Hoppy::Test::EigenAssertionFailure);
}
#endif
