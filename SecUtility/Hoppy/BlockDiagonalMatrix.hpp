// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/BlockPolicy.hpp>
#include <SecUtility/Hoppy/BlockDiagonalMatrixExpr.hpp>
#include <SecUtility/Hoppy/Detail/CheckedDimensions.hpp>
#include <SecUtility/Hoppy/Detail/Storage.hpp>
#include <SecUtility/Hoppy/Detail/Traits.hpp>
#include <SecUtility/Hoppy/ForwardDeclarations.hpp>

#include <Eigen/Core>

#include <initializer_list>
#include <utility>
#include <vector>

namespace Hoppy
{
	template <typename TScalar, typename TBlockPolicy>
	class BlockDiagonalMatrix : public BlockDiagonalMatrixExpr<BlockDiagonalMatrix<TScalar, TBlockPolicy>>
	{
		static_assert(IsBlockPolicy<TBlockPolicy>, "BlockDiagonalMatrix requires a Hoppy-provided block policy");

	public:
		using Scalar = TScalar;
		using BlockPolicy = TBlockPolicy;
		using Block = typename BlockPolicy::template View<Scalar>;
		using ConstBlock = typename BlockPolicy::template ConstView<Scalar>;
		using StorageIndex = Eigen::Index;
		static constexpr int RowsAtCompileTime = Eigen::Dynamic;
		static constexpr int ColsAtCompileTime = Eigen::Dynamic;
		static constexpr int SizeAtCompileTime = Eigen::Dynamic;
		static constexpr int MaxSizeAtCompileTime = Eigen::Dynamic;
		static constexpr int Flags = Eigen::NestByRefBit;

		BlockDiagonalMatrix() = default;

		explicit BlockDiagonalMatrix(const std::vector<Eigen::Index>& dimensions)
			: BlockDiagonalMatrix(dimensions.begin(), dimensions.end())
		{}

		explicit BlockDiagonalMatrix(std::initializer_list<Eigen::Index> dimensions)
			: BlockDiagonalMatrix(dimensions.begin(), dimensions.end())
		{}

		template <typename Iterator,
		          typename = std::enable_if_t<Detail::CheckedDimensionsDetail::is_supported_iterator<Iterator>::value>>
		BlockDiagonalMatrix(Iterator first, Iterator last)
			: m_Storage(Detail::BuildCheckedDimensions(first, last))
		{}

		template <typename Derived>
		BlockDiagonalMatrix(const BlockDiagonalMatrixExpr<Derived>& expression)
			: BlockDiagonalMatrix(expression.blockingInfo())
		{
			assignSameBlocking(expression.derived());
		}

		template <typename Derived>
		BlockDiagonalMatrix& operator=(const BlockDiagonalMatrixExpr<Derived>& expression)
		{
			if (this->hasSameBlockingAs(expression))
				assignSameBlocking(expression.derived());
			else
			{
				BlockDiagonalMatrix replacement(expression);
				swap(replacement);
			}
			return *this;
		}

		void swap(BlockDiagonalMatrix& other) noexcept { m_Storage.swap(other.m_Storage); }

		Eigen::Index blockCount() const noexcept { return m_Storage.blockCount(); }
		Eigen::Index rows() const noexcept { return totalDimension(); }
		Eigen::Index cols() const noexcept { return totalDimension(); }
		Eigen::Index size() const noexcept { return totalDimension() * totalDimension(); }
		Eigen::Index totalDimension() const noexcept { return m_Storage.totalDimension(); }
		Eigen::Index storedSize() const noexcept { return m_Storage.storedSize(); }
		Scalar* data() noexcept { return m_Storage.data(); }
		const Scalar* data() const noexcept { return m_Storage.data(); }
		std::vector<Eigen::Index> blockingInfo() const { return m_Storage.blockingInfo(); }
		Eigen::Index dimensionOfBlock(Eigen::Index index) const { return m_Storage.dimensionOfBlock(index); }
		Eigen::Index blockOffset(Eigen::Index index) const { return m_Storage.blockOffset(index); }
		Eigen::Index storageOffset(Eigen::Index index) const { return m_Storage.storageOffset(index); }

		Block operator[](const Eigen::Index index)
		{
			const auto dimension = dimensionOfBlock(index);
			return Block(data() + storageOffset(index), dimension, dimension);
		}

		ConstBlock operator[](const Eigen::Index index) const
		{
			const auto dimension = dimensionOfBlock(index);
			return ConstBlock(data() + storageOffset(index), dimension, dimension);
		}

		template <typename Derived> BlockDiagonalMatrix& operator+=(const BlockDiagonalMatrixExpr<Derived>& other)
		{
			eigen_assert(this->hasSameBlockingAs(other));
			for (Eigen::Index index = 0; index < blockCount(); ++index)
				(*this)[index] += other.derived()[index];
			return *this;
		}
		template <typename Derived> BlockDiagonalMatrix& operator-=(const BlockDiagonalMatrixExpr<Derived>& other)
		{
			eigen_assert(this->hasSameBlockingAs(other));
			for (Eigen::Index index = 0; index < blockCount(); ++index)
				(*this)[index] -= other.derived()[index];
			return *this;
		}
		template <typename TOtherScalar>
		BlockDiagonalMatrix& operator*=(const TOtherScalar& scalar)
		{
			for (Eigen::Index index = 0; index < blockCount(); ++index)
				(*this)[index] *= scalar;
			return *this;
		}
		template <typename TOtherScalar>
		BlockDiagonalMatrix& operator/=(const TOtherScalar& scalar)
		{
			for (Eigen::Index index = 0; index < blockCount(); ++index)
				(*this)[index] /= scalar;
			return *this;
		}

		void resize(const std::vector<Eigen::Index>& dimensions) { resize(dimensions.begin(), dimensions.end()); }
		void resize(const std::initializer_list<Eigen::Index> dimensions) { resize(dimensions.begin(), dimensions.end()); }

		template <typename Iterator,
		          typename = std::enable_if_t<Detail::CheckedDimensionsDetail::is_supported_iterator<Iterator>::value>>
		void resize(Iterator first, Iterator last)
		{
			BlockDiagonalMatrix replacement(first, last);
			swap(replacement);
		}

		template <typename TOther>
		void resizeLike(const TOther& other)
		{
			resize(other.blockingInfo());
		}

		BlockDiagonalMatrix& setZero() { return setConstant(Scalar{}); }
		BlockDiagonalMatrix& setOnes() { return setConstant(Scalar{1}); }

		BlockDiagonalMatrix& setConstant(const Scalar& value)
		{
			for (Eigen::Index index = 0; index < blockCount(); ++index)
				(*this)[index].setConstant(value);
			return *this;
		}

		BlockDiagonalMatrix& setRandom()
		{
			for (Eigen::Index index = 0; index < blockCount(); ++index)
				(*this)[index].setRandom();
			return *this;
		}

		BlockDiagonalMatrix& setIdentity()
		{
			for (Eigen::Index index = 0; index < blockCount(); ++index)
				(*this)[index].setIdentity();
			return *this;
		}

		BlockDiagonalMatrix& setZero(const std::vector<Eigen::Index>& dimensions)
		{
			return setZero(dimensions.begin(), dimensions.end());
		}
		BlockDiagonalMatrix& setZero(const std::initializer_list<Eigen::Index> dimensions)
		{
			return setZero(dimensions.begin(), dimensions.end());
		}
		template <typename Iterator>
		BlockDiagonalMatrix& setZero(Iterator first, Iterator last)
		{
			return reblockAndFill(first, last, [](BlockDiagonalMatrix& value) { value.setZero(); });
		}

		BlockDiagonalMatrix& setOnes(const std::vector<Eigen::Index>& dimensions)
		{
			return setOnes(dimensions.begin(), dimensions.end());
		}
		BlockDiagonalMatrix& setOnes(const std::initializer_list<Eigen::Index> dimensions)
		{
			return setOnes(dimensions.begin(), dimensions.end());
		}
		template <typename Iterator>
		BlockDiagonalMatrix& setOnes(Iterator first, Iterator last)
		{
			return reblockAndFill(first, last, [](BlockDiagonalMatrix& value) { value.setOnes(); });
		}

		BlockDiagonalMatrix& setRandom(const std::vector<Eigen::Index>& dimensions)
		{
			return setRandom(dimensions.begin(), dimensions.end());
		}
		BlockDiagonalMatrix& setRandom(const std::initializer_list<Eigen::Index> dimensions)
		{
			return setRandom(dimensions.begin(), dimensions.end());
		}
		template <typename Iterator>
		BlockDiagonalMatrix& setRandom(Iterator first, Iterator last)
		{
			return reblockAndFill(first, last, [](BlockDiagonalMatrix& value) { value.setRandom(); });
		}

		BlockDiagonalMatrix& setIdentity(const std::vector<Eigen::Index>& dimensions)
		{
			return setIdentity(dimensions.begin(), dimensions.end());
		}
		BlockDiagonalMatrix& setIdentity(const std::initializer_list<Eigen::Index> dimensions)
		{
			return setIdentity(dimensions.begin(), dimensions.end());
		}
		template <typename Iterator>
		BlockDiagonalMatrix& setIdentity(Iterator first, Iterator last)
		{
			return reblockAndFill(first, last, [](BlockDiagonalMatrix& value) { value.setIdentity(); });
		}

		BlockDiagonalMatrix& setConstant(const std::vector<Eigen::Index>& dimensions, const Scalar& value)
		{
			return setConstant(dimensions.begin(), dimensions.end(), value);
		}

		BlockDiagonalMatrix& setConstant(const std::initializer_list<Eigen::Index> dimensions, const Scalar& value)
		{
			return setConstant(dimensions.begin(), dimensions.end(), value);
		}

		template <typename Iterator>
		BlockDiagonalMatrix& setConstant(Iterator first, Iterator last, const Scalar& value)
		{
			BlockDiagonalMatrix replacement(first, last);
			replacement.setConstant(value);
			swap(replacement);
			return *this;
		}

		static BlockDiagonalMatrix WithBlocking(const std::vector<Eigen::Index>& dimensions)
		{
			return BlockDiagonalMatrix(dimensions);
		}
		static BlockDiagonalMatrix WithBlocking(std::initializer_list<Eigen::Index> dimensions)
		{
			return BlockDiagonalMatrix(dimensions);
		}
		template <typename Iterator>
		static BlockDiagonalMatrix WithBlocking(Iterator first, Iterator last)
		{
			return BlockDiagonalMatrix(first, last);
		}

		template <typename TOther>
		static BlockDiagonalMatrix WithBlockingOf(const TOther& other)
		{
			return WithBlocking(other.blockingInfo());
		}

		static BlockDiagonalMatrix Zero(const std::vector<Eigen::Index>& dimensions) { return Zero(dimensions.begin(), dimensions.end()); }
		static BlockDiagonalMatrix Zero(const std::initializer_list<Eigen::Index> dimensions) { return Zero(dimensions.begin(), dimensions.end()); }
		template <typename Iterator>
		static BlockDiagonalMatrix Zero(Iterator first, Iterator last) { BlockDiagonalMatrix result(first, last); result.setZero(); return result; }

		static BlockDiagonalMatrix Ones(const std::vector<Eigen::Index>& dimensions) { return Ones(dimensions.begin(), dimensions.end()); }
		static BlockDiagonalMatrix Ones(const std::initializer_list<Eigen::Index> dimensions) { return Ones(dimensions.begin(), dimensions.end()); }
		template <typename Iterator>
		static BlockDiagonalMatrix Ones(Iterator first, Iterator last) { BlockDiagonalMatrix result(first, last); result.setOnes(); return result; }

		static BlockDiagonalMatrix Random(const std::vector<Eigen::Index>& dimensions) { return Random(dimensions.begin(), dimensions.end()); }
		static BlockDiagonalMatrix Random(const std::initializer_list<Eigen::Index> dimensions) { return Random(dimensions.begin(), dimensions.end()); }
		template <typename Iterator>
		static BlockDiagonalMatrix Random(Iterator first, Iterator last) { BlockDiagonalMatrix result(first, last); result.setRandom(); return result; }

		static BlockDiagonalMatrix Identity(const std::vector<Eigen::Index>& dimensions) { return Identity(dimensions.begin(), dimensions.end()); }
		static BlockDiagonalMatrix Identity(const std::initializer_list<Eigen::Index> dimensions) { return Identity(dimensions.begin(), dimensions.end()); }
		template <typename Iterator>
		static BlockDiagonalMatrix Identity(Iterator first, Iterator last) { BlockDiagonalMatrix result(first, last); result.setIdentity(); return result; }

		static BlockDiagonalMatrix Constant(const std::vector<Eigen::Index>& dimensions, const Scalar& value)
		{
			return Constant(dimensions.begin(), dimensions.end(), value);
		}
		static BlockDiagonalMatrix Constant(const std::initializer_list<Eigen::Index> dimensions, const Scalar& value)
		{
			return Constant(dimensions.begin(), dimensions.end(), value);
		}
		template <typename Iterator>
		static BlockDiagonalMatrix Constant(Iterator first, Iterator last, const Scalar& value)
		{
			BlockDiagonalMatrix result(first, last);
			result.setConstant(value);
			return result;
		}

		template <typename DenseDerived>
		static BlockDiagonalMatrix FromDense(
		        const Eigen::MatrixBase<DenseDerived>& dense, const std::vector<Eigen::Index>& dimensions)
		{
			return FromDense(dense, dimensions.begin(), dimensions.end());
		}
		template <typename DenseDerived>
		static BlockDiagonalMatrix FromDense(
		        const Eigen::MatrixBase<DenseDerived>& dense, std::initializer_list<Eigen::Index> dimensions)
		{
			return FromDense(dense, dimensions.begin(), dimensions.end());
		}
		template <typename DenseDerived, typename Iterator>
		static BlockDiagonalMatrix FromDense(
		        const Eigen::MatrixBase<DenseDerived>& dense, Iterator first, Iterator last)
		{
			BlockDiagonalMatrix result(first, last);
			if (dense.rows() != dense.cols() || dense.rows() != result.totalDimension())
			{
				eigen_assert(false && "FromDense requires a square matrix matching the blocking");
				return result;
			}
			for (Eigen::Index index = 0; index < result.blockCount(); ++index)
			{
				const auto offset = result.blockOffset(index);
				const auto dimension = result.dimensionOfBlock(index);
				result[index] = dense.derived().block(offset, offset, dimension, dimension).template cast<Scalar>();
			}
			return result;
		}

		template <typename DenseDerived, typename TOther>
		static BlockDiagonalMatrix FromDenseLike(
		        const Eigen::MatrixBase<DenseDerived>& dense, const TOther& other)
		{
			return FromDense(dense, other.blockingInfo());
		}

		template <typename BlockIterator>
		static BlockDiagonalMatrix FromBlocks(BlockIterator first, BlockIterator last)
		{
			using Category = typename std::iterator_traits<BlockIterator>::iterator_category;
			static_assert(std::is_base_of_v<std::forward_iterator_tag, Category>,
			              "FromBlocks requires forward iterators");
			std::vector<Eigen::Index> dimensions;
			for (auto iterator = first; iterator != last; ++iterator)
			{
				const auto& block = *iterator;
				if (block.rows() != block.cols())
				{
					eigen_assert(false && "FromBlocks requires square blocks");
					return {};
				}
				dimensions.push_back(block.rows());
			}
			BlockDiagonalMatrix result(dimensions);
			Eigen::Index index = 0;
			for (; first != last; ++first, ++index)
			{
				const auto& block = *first;
				result[index] = block.template cast<Scalar>();
			}
			return result;
		}

		template <typename TBlock>
		static BlockDiagonalMatrix FromBlocks(std::initializer_list<TBlock> blocks)
		{
			return FromBlocks(blocks.begin(), blocks.end());
		}

		template <typename DenseDerived>
		static BlockDiagonalMatrix SingleBlock(const Eigen::MatrixBase<DenseDerived>& dense)
		{
			if (dense.rows() != dense.cols())
			{
				eigen_assert(false && "SingleBlock requires a square matrix");
				return {};
			}
			BlockDiagonalMatrix result({dense.rows()});
			result[0] = dense.template cast<Scalar>();
			return result;
		}

	private:
		template <typename Derived>
		void assignSameBlocking(const Derived& expression)
		{
			for (Eigen::Index index = 0; index < blockCount(); ++index)
				(*this)[index] = expression[index];
		}

		template <typename Iterator, typename Fill>
		BlockDiagonalMatrix& reblockAndFill(Iterator first, Iterator last, Fill fill)
		{
			BlockDiagonalMatrix replacement(first, last);
			fill(replacement);
			swap(replacement);
			return *this;
		}

		Detail::PackedStorage<Scalar> m_Storage;
	};

	template <typename TScalar, typename TBlockPolicy>
	void swap(BlockDiagonalMatrix<TScalar, TBlockPolicy>& lhs,
	          BlockDiagonalMatrix<TScalar, TBlockPolicy>& rhs) noexcept
	{
		lhs.swap(rhs);
	}
}  // namespace Hoppy

#include <SecUtility/Hoppy/Detail/BlockExpressions.hpp>
