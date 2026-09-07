// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/ForwardDeclarations.hpp>
#include <SecUtility/Hoppy/Detail/TriangularCompressedCheckedSize.hpp>
#include <SecUtility/Hoppy/Detail/TriangularCompressedPlainBase.hpp>
#include <SecUtility/Hoppy/Detail/TriangularCompressedTags.hpp>
#include <SecUtility/Hoppy/Detail/TriangularCompressedTraits.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace Hoppy::Detail
{
	template <typename TScalar, int Dimension, TrianglePacking Packing, int Options,
	          typename NormalizedStructureTag>
	class TriangularCompressedMatrix
	    : public TriangularCompressedPlainBase<
	              TriangularCompressedMatrix<TScalar, Dimension, Packing, Options, NormalizedStructureTag>,
	              TScalar, NormalizedStructureTag, Packing, Dimension, true>
	{
		using PlainBase = TriangularCompressedPlainBase<
		        TriangularCompressedMatrix, TScalar, NormalizedStructureTag, Packing, Dimension, true>;
		static_assert(is_valid_triangular_dimension_v<Dimension>, "invalid compile-time dimension");
		static_assert(has_representable_logical_size_v<Dimension>, "fixed logical size is not representable");
		static_assert(is_valid_triangle_packing_v<Packing>, "invalid triangle packing");
		static_assert(is_valid_triangular_options_v<Options>, "unsupported matrix options");

	public:
		using Scalar = TScalar;
		using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
		using StorageIndex = Eigen::Index;
		using StructureTag = NormalizedStructureTag;
		using PlainObject = TriangularCompressedMatrix;
		using Nested = const TriangularCompressedMatrix&;
		using CoeffReturnType = Scalar;
		using CoeffProxy = typename PlainBase::CoeffProxy;
		using Allocator = std::conditional_t<Options == Eigen::DontAlign, std::allocator<Scalar>,
		                                     Eigen::aligned_allocator<Scalar>>;

		static constexpr int RowsAtCompileTime = Dimension;
		static constexpr int ColsAtCompileTime = Dimension;
		static constexpr int MaxRowsAtCompileTime = Dimension;
		static constexpr int MaxColsAtCompileTime = Dimension;
		static constexpr int SizeAtCompileTime = compileTimeLogicalSize<Dimension>();
		static constexpr int MaxSizeAtCompileTime = SizeAtCompileTime;
		static constexpr int Flags = Eigen::NestByRefBit;
		static constexpr TrianglePacking PackingValue = Packing;
		static constexpr bool IsVectorAtCompileTime = Dimension == 1;
		static constexpr bool IsTriangularCompressed = true;

		TriangularCompressedMatrix() : TriangularCompressedMatrix(defaultDimension()) {}
		explicit TriangularCompressedMatrix(const Eigen::Index dimension)
			: m_Dimension(validatedDimension(dimension)),
			  m_Storage(static_cast<std::size_t>(checkedStoredSize(m_Dimension)))
		{}
		TriangularCompressedMatrix(const Eigen::Index rows, const Eigen::Index columns)
			: TriangularCompressedMatrix(checkedSquareDimension(rows, columns))
		{}

		TriangularCompressedMatrix(const TriangularCompressedMatrix&) = default;
		template <typename Other,
		          typename = std::enable_if_t<Other::IsTriangularCompressed
		                                      && std::is_same_v<StructureTag, typename Other::StructureTag>>>
		explicit TriangularCompressedMatrix(const Other& other)
			: TriangularCompressedMatrix(other.dimension())
		{
			this->assignCoefficientsFrom(other);
		}
		TriangularCompressedMatrix(TriangularCompressedMatrix&&)
		        noexcept(std::is_nothrow_move_constructible_v<std::vector<Scalar, Allocator>>) = default;
		TriangularCompressedMatrix& operator=(const TriangularCompressedMatrix&) = default;
		template <typename Other,
		          typename = std::enable_if_t<Other::IsTriangularCompressed
		                                      && std::is_same_v<StructureTag, typename Other::StructureTag>>>
		TriangularCompressedMatrix& operator=(const Other& other)
		{
			TriangularCompressedMatrix replacement(other);
			swap(replacement);
			return *this;
		}
		TriangularCompressedMatrix& operator=(TriangularCompressedMatrix&&)
		        noexcept(std::is_nothrow_move_assignable_v<std::vector<Scalar, Allocator>>) = default;
		template <typename MatrixType, unsigned int Mode,
		          typename = std::enable_if_t<accepts_triangular_view_v<StructureTag, Mode>>>
		TriangularCompressedMatrix& operator=(const Eigen::TriangularView<MatrixType, Mode>& view)
		{
			TriangularCompressedMatrix replacement(view.rows(), view.cols());
			replacement.template assignFromTriangularView<Mode>(view);
			swap(replacement);
			return *this;
		}
		template <typename MatrixType, unsigned int UpLo,
		          bool Enabled = accepts_self_adjoint_view_v<Scalar, StructureTag>,
		          typename = std::enable_if_t<Enabled>>
		TriangularCompressedMatrix& operator=(const Eigen::SelfAdjointView<MatrixType, UpLo>& view)
		{
			TriangularCompressedMatrix replacement(view.rows(), view.cols());
			replacement.template assignFromSelfAdjointView<UpLo>(view);
			swap(replacement);
			return *this;
		}
		~TriangularCompressedMatrix() = default;

		static Eigen::Index requiredStoredSize(const Eigen::Index dimension)
		{
			if constexpr (Dimension != Eigen::Dynamic)
			{
				if (dimension != Dimension)
				{
					eigen_assert(false && "runtime dimension does not match fixed dimension");
					return 0;
				}
			}
			return checkedStoredSize(dimension);
		}

		static TriangularCompressedMatrix Zero() { return makeZero(defaultDimension()); }
		static TriangularCompressedMatrix Zero(const Eigen::Index dimension)
		{
			return makeZero(dimension);
		}
		static TriangularCompressedMatrix Zero(const Eigen::Index rows, const Eigen::Index columns)
		{
			return Zero(checkedSquareDimension(rows, columns));
		}

		template <typename Tag = StructureTag,
		          typename = std::enable_if_t<!std::is_same_v<Tag, AntiSymmetricTag>
		                                      && !std::is_same_v<Tag, AntiHermitianTag>>>
		static TriangularCompressedMatrix Ones() { return makeOnes(defaultDimension()); }
		template <typename Tag = StructureTag,
		          typename = std::enable_if_t<!std::is_same_v<Tag, AntiSymmetricTag>
		                                      && !std::is_same_v<Tag, AntiHermitianTag>>>
		static TriangularCompressedMatrix Ones(const Eigen::Index dimension)
		{
			return makeOnes(dimension);
		}
		template <typename Tag = StructureTag,
		          typename = std::enable_if_t<!std::is_same_v<Tag, AntiSymmetricTag>
		                                      && !std::is_same_v<Tag, AntiHermitianTag>>>
		static TriangularCompressedMatrix Ones(const Eigen::Index rows, const Eigen::Index columns)
		{
			return Ones(checkedSquareDimension(rows, columns));
		}

		static TriangularCompressedMatrix Constant(const Scalar& value)
		{
			return makeConstant(defaultDimension(), value);
		}
		static TriangularCompressedMatrix Constant(const Eigen::Index dimension, const Scalar& value)
		{
			return makeConstant(dimension, value);
		}
		static TriangularCompressedMatrix Constant(const Eigen::Index rows, const Eigen::Index columns,
		                                           const Scalar& value)
		{
			return makeConstant(checkedSquareDimension(rows, columns), value);
		}

		static TriangularCompressedMatrix Random() { return makeRandom(defaultDimension()); }
		static TriangularCompressedMatrix Random(const Eigen::Index dimension)
		{
			return makeRandom(dimension);
		}
		static TriangularCompressedMatrix Random(const Eigen::Index rows, const Eigen::Index columns)
		{
			return Random(checkedSquareDimension(rows, columns));
		}

		template <typename Tag = StructureTag,
		          typename = std::enable_if_t<!std::is_same_v<Tag, AntiSymmetricTag>
		                                      && !std::is_same_v<Tag, AntiHermitianTag>>>
		static TriangularCompressedMatrix Identity()
		{
			return makeIdentity(defaultDimension());
		}
		template <typename Tag = StructureTag,
		          typename = std::enable_if_t<!std::is_same_v<Tag, AntiSymmetricTag>
		                                      && !std::is_same_v<Tag, AntiHermitianTag>>>
		static TriangularCompressedMatrix Identity(const Eigen::Index dimension)
		{
			return makeIdentity(dimension);
		}
		template <typename Tag = StructureTag,
		          typename = std::enable_if_t<!std::is_same_v<Tag, AntiSymmetricTag>
		                                      && !std::is_same_v<Tag, AntiHermitianTag>>>
		static TriangularCompressedMatrix Identity(const Eigen::Index rows, const Eigen::Index columns)
		{
			return Identity(checkedSquareDimension(rows, columns));
		}

		template <typename Expression, typename Tag = StructureTag,
		          typename = std::enable_if_t<!std::is_same_v<Tag, LowerTriangularTag>>>
		static TriangularCompressedMatrix FromUpper(const Expression& expression)
		{
			return fromSelectedTriangle(expression, true);
		}

		template <typename Expression, typename Tag = StructureTag,
		          typename = std::enable_if_t<!std::is_same_v<Tag, UpperTriangularTag>>>
		static TriangularCompressedMatrix FromLower(const Expression& expression)
		{
			return fromSelectedTriangle(expression, false);
		}

		template <typename Expression>
		static TriangularCompressedMatrix FromUncheckedDense(const Expression& expression)
		{
			TriangularCompressedMatrix result(checkedSquareDimension(expression.rows(), expression.cols()));
			const bool upper = std::is_same_v<StructureTag, UpperTriangularTag>
			                   || (!std::is_same_v<StructureTag, LowerTriangularTag>
			                       && Packing == TrianglePacking::Upper);
			forSelectedTriangle(result, expression, upper, true);
			return result;
		}

		template <typename Expression>
		static TriangularCompressedMatrix FromModifiedDense(const Expression& expression)
		{
			const Eigen::Index dimension = checkedSquareDimension(expression.rows(), expression.cols());
			const auto evaluated = expression.eval();
			TriangularCompressedMatrix result(dimension);
			const bool upper = std::is_same_v<StructureTag, UpperTriangularTag>
			                   || (!std::is_same_v<StructureTag, LowerTriangularTag>
			                       && Packing == TrianglePacking::Upper);
			for (Eigen::Index row = 0; row < dimension; ++row)
				for (Eigen::Index column = 0; column < dimension; ++column)
					if ((upper && row <= column) || (!upper && row >= column))
					{
						const Scalar direct = static_cast<Scalar>(evaluated.coeff(row, column));
						Scalar value;
						if constexpr (std::is_same_v<StructureTag, UpperTriangularTag>
						              || std::is_same_v<StructureTag, LowerTriangularTag>) value = direct;
						else
						{
							const Scalar reflected = static_cast<Scalar>(evaluated.coeff(column, row));
							if constexpr (std::is_same_v<StructureTag, SymmetricTag>)
								value = (direct + reflected) / Scalar(2);
							else if constexpr (std::is_same_v<StructureTag, AntiSymmetricTag>)
								value = (direct - reflected) / Scalar(2);
							else if constexpr (std::is_same_v<StructureTag, HermitianTag>)
								value = (direct + Eigen::numext::conj(reflected)) / Scalar(2);
							else
								value = (direct - Eigen::numext::conj(reflected)) / Scalar(2);
						}
						result.writeCanonicalized(row, column, value);
					}
			return result;
		}

		void resize(const Eigen::Index dimension)
		{
			TriangularCompressedMatrix replacement(dimension);
			swap(replacement);
		}
		void resize(const Eigen::Index rows, const Eigen::Index columns) { resize(checkedSquareDimension(rows, columns)); }
		template <typename Other>
		void resizeLike(const Other& other) { resize(other.rows(), other.cols()); }

		void conservativeResize(Eigen::Index dimension)
		{
			TriangularCompressedMatrix replacement(dimension);
			const Eigen::Index commonDimension = (std::min)(m_Dimension, replacement.m_Dimension);
			const Eigen::Index commonStoredSize = checkedStoredSize(commonDimension);
			std::copy_n(m_Storage.begin(), commonStoredSize, replacement.m_Storage.begin());
			swap(replacement);
		}
		void conservativeResize(const Eigen::Index rows, const Eigen::Index columns)
		{
			conservativeResize(checkedSquareDimension(rows, columns));
		}

		void swap(TriangularCompressedMatrix& other) noexcept(noexcept(m_Storage.swap(other.m_Storage)))
		{
			using std::swap;
			swap(m_Dimension, other.m_Dimension);
			m_Storage.swap(other.m_Storage);
		}
		friend void swap(TriangularCompressedMatrix& left, TriangularCompressedMatrix& right)
		        noexcept(noexcept(left.swap(right)))
		{
			left.swap(right);
		}

	private:
		friend PlainBase;
		using Storage = std::vector<Scalar, Allocator>;
		static TriangularCompressedMatrix makeZero(const Eigen::Index dimension)
		{
			TriangularCompressedMatrix result(dimension);
			result.setZero();
			return result;
		}
		static TriangularCompressedMatrix makeOnes(const Eigen::Index dimension)
		{
			TriangularCompressedMatrix result(dimension);
			result.setOnes();
			return result;
		}
		static TriangularCompressedMatrix makeRandom(const Eigen::Index dimension)
		{
			TriangularCompressedMatrix result(dimension);
			result.setRandom();
			return result;
		}
		static TriangularCompressedMatrix makeIdentity(const Eigen::Index dimension)
		{
			TriangularCompressedMatrix result(dimension);
			result.setIdentity();
			return result;
		}
		static TriangularCompressedMatrix makeConstant(const Eigen::Index dimension, const Scalar& value)
		{
			TriangularCompressedMatrix result(dimension);
			result.setConstant(value);
			return result;
		}

		template <typename Expression>
		static void forSelectedTriangle(TriangularCompressedMatrix& result, const Expression& expression,
		                                const bool upper, const bool canonicalizeDiagonal)
		{
			for (Eigen::Index row = 0; row < result.dimension(); ++row)
				for (Eigen::Index column = 0; column < result.dimension(); ++column)
					if ((upper && row <= column) || (!upper && row >= column))
					{
						const Scalar value = static_cast<Scalar>(expression.coeff(row, column));
						if (canonicalizeDiagonal) result.writeCanonicalized(row, column, value);
						else result.writeLogical(row, column, value);
					}
		}

		template <typename Expression>
		static TriangularCompressedMatrix fromSelectedTriangle(const Expression& expression, const bool upper)
		{
			TriangularCompressedMatrix result(checkedSquareDimension(expression.rows(), expression.cols()));
			forSelectedTriangle(result, expression, upper, false);
			return result;
		}

		static constexpr Eigen::Index defaultDimension() { return Dimension == Eigen::Dynamic ? 0 : Dimension; }
		static Eigen::Index validatedDimension(const Eigen::Index dimension)
		{
			if constexpr (Dimension != Eigen::Dynamic)
				if (dimension != Dimension)
				{
					eigen_assert(false && "runtime dimension does not match fixed dimension");
					return Dimension;
				}
			if (dimension < 0)
			{
				eigen_assert(false && "triangular-compressed dimension must be nonnegative");
				return defaultDimension();
			}
			constexpr Eigen::Index maximum = (std::numeric_limits<Eigen::Index>::max)();
			if (dimension != 0 && dimension > maximum / dimension)
			{
				eigen_assert(false && "triangular-compressed logical size overflows Eigen::Index");
				return defaultDimension();
			}
			const Eigen::Index storedSize = checkedStoredSize(dimension);
			if (dimension != 0 && storedSize == 0) return defaultDimension();
			constexpr std::size_t maximumBytes = (std::numeric_limits<std::size_t>::max)();
			if (static_cast<std::size_t>(storedSize) > maximumBytes / sizeof(Scalar))
			{
				eigen_assert(false && "triangular-compressed byte count overflows size_t");
				return defaultDimension();
			}
			return dimension;
		}

		Eigen::Index dimensionImpl() const noexcept { return m_Dimension; }
		Scalar* coeffDataImpl() noexcept { return m_Storage.data(); }
		const Scalar* coeffDataImpl() const noexcept { return m_Storage.data(); }

		Eigen::Index m_Dimension;
		Storage m_Storage;
	};
}  // namespace Hoppy::Detail

#include <SecUtility/Hoppy/Detail/TriangularCompressedMap.hpp>
