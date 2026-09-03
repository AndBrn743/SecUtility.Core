// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/Detail/CheckedDimensions.hpp>
#include <SecUtility/Hoppy/BlockVectorExpr.hpp>
#include <SecUtility/Hoppy/Detail/Storage.hpp>
#include <SecUtility/Hoppy/Detail/Traits.hpp>
#include <SecUtility/Hoppy/ForwardDeclarations.hpp>

#include <Eigen/Core>

#include <initializer_list>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace Hoppy
{
	template <typename TScalar, typename TOrientation>
	class BlockVector : public BlockVectorExpr<BlockVector<TScalar, TOrientation>>
	{
		static_assert(std::is_same_v<TOrientation, Column> || std::is_same_v<TOrientation, Row>,
		              "BlockVector orientation must be Hoppy::Column or Hoppy::Row");

		using ColumnPlain = Eigen::VectorX<TScalar>;
		using RowPlain = Eigen::Matrix<TScalar, 1, Eigen::Dynamic>;
		using Plain = std::conditional_t<std::is_same_v<TOrientation, Column>, ColumnPlain, RowPlain>;

	public:
		using Scalar = TScalar;
		using Orientation = TOrientation;
		using Block = Eigen::Map<Plain, Eigen::Unaligned>;
		using ConstBlock = Eigen::Map<const Plain, Eigen::Unaligned>;
		using StorageIndex = Eigen::Index;
		static constexpr int RowsAtCompileTime = std::is_same_v<Orientation, Row> ? 1 : Eigen::Dynamic;
		static constexpr int ColsAtCompileTime = std::is_same_v<Orientation, Row> ? Eigen::Dynamic : 1;
		static constexpr int SizeAtCompileTime = Eigen::Dynamic;
		static constexpr int MaxSizeAtCompileTime = Eigen::Dynamic;
		static constexpr int Flags = Eigen::NestByRefBit;

		BlockVector() = default;

		explicit BlockVector(const std::vector<Eigen::Index>& dimensions)
			: BlockVector(dimensions.begin(), dimensions.end())
		{}

		explicit BlockVector(std::initializer_list<Eigen::Index> dimensions)
			: BlockVector(dimensions.begin(), dimensions.end())
		{}

		template <typename Iterator,
		          typename = std::enable_if_t<Detail::CheckedDimensionsDetail::is_supported_iterator<Iterator>::value>>
		BlockVector(Iterator first, Iterator last)
			: m_Storage(Detail::BuildCheckedVectorDimensions(first, last))
		{}

		template <typename Derived>
		BlockVector(const BlockVectorExpr<Derived>& expression)
			: BlockVector(expression.blockingInfo())
		{
			assignSameBlocking(expression.derived());
		}

		template <typename Derived>
		BlockVector& operator=(const BlockVectorExpr<Derived>& expression)
		{
			if (this->hasSameBlockingAs(expression))
				assignSameBlocking(expression.derived());
			else
			{
				BlockVector replacement(expression);
				swap(replacement);
			}
			return *this;
		}

		void swap(BlockVector& other) noexcept { m_Storage.swap(other.m_Storage); }

		Eigen::Index blockCount() const noexcept { return m_Storage.blockCount(); }
		Eigen::Index rows() const noexcept { return std::is_same_v<Orientation, Row> ? 1 : totalDimension(); }
		Eigen::Index cols() const noexcept { return std::is_same_v<Orientation, Row> ? totalDimension() : 1; }
		Eigen::Index size() const noexcept { return totalDimension(); }
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
			return Block(data() + storageOffset(index), dimension);
		}

		ConstBlock operator[](const Eigen::Index index) const
		{
			const auto dimension = dimensionOfBlock(index);
			return ConstBlock(data() + storageOffset(index), dimension);
		}

		Block asDense() { return Block(data(), totalDimension()); }
		ConstBlock asDense() const { return ConstBlock(data(), totalDimension()); }

		void resize(const std::vector<Eigen::Index>& dimensions) { resize(dimensions.begin(), dimensions.end()); }
		void resize(const std::initializer_list<Eigen::Index> dimensions) { resize(dimensions.begin(), dimensions.end()); }
		template <typename Iterator,
		          typename = std::enable_if_t<Detail::CheckedDimensionsDetail::is_supported_iterator<Iterator>::value>>
		void resize(Iterator first, Iterator last) { BlockVector replacement(first, last); swap(replacement); }
		template <typename TOther>
		void resizeLike(const TOther& other) { resize(other.blockingInfo()); }

		BlockVector& setZero() { return setConstant(Scalar{}); }
		BlockVector& setOnes() { return setConstant(Scalar{1}); }
		BlockVector& setConstant(const Scalar& value) { asDense().setConstant(value); return *this; }
		BlockVector& setRandom() { asDense().setRandom(); return *this; }

	private:
		template <typename Derived>
		void assignSameBlocking(const Derived& expression)
		{
			for (Eigen::Index index = 0; index < blockCount(); ++index)
				(*this)[index] = expression[index];
		}

		template <typename Iterator, typename Fill>
		BlockVector& reblockAndFill(Iterator first, Iterator last, Fill fill)
		{
			BlockVector replacement(first, last); fill(replacement); swap(replacement); return *this;
		}
	public:

		BlockVector& setZero(const std::vector<Eigen::Index>& dimensions) { return setZero(dimensions.begin(), dimensions.end()); }
		BlockVector& setZero(const std::initializer_list<Eigen::Index> dimensions) { return setZero(dimensions.begin(), dimensions.end()); }
		template <typename Iterator> BlockVector& setZero(Iterator first, Iterator last) { return reblockAndFill(first, last, [](auto& value) { value.setZero(); }); }
		BlockVector& setOnes(const std::vector<Eigen::Index>& dimensions) { return setOnes(dimensions.begin(), dimensions.end()); }
		BlockVector& setOnes(const std::initializer_list<Eigen::Index> dimensions) { return setOnes(dimensions.begin(), dimensions.end()); }
		template <typename Iterator> BlockVector& setOnes(Iterator first, Iterator last) { return reblockAndFill(first, last, [](auto& value) { value.setOnes(); }); }
		BlockVector& setRandom(const std::vector<Eigen::Index>& dimensions) { return setRandom(dimensions.begin(), dimensions.end()); }
		BlockVector& setRandom(const std::initializer_list<Eigen::Index> dimensions) { return setRandom(dimensions.begin(), dimensions.end()); }
		template <typename Iterator> BlockVector& setRandom(Iterator first, Iterator last) { return reblockAndFill(first, last, [](auto& value) { value.setRandom(); }); }
		BlockVector& setConstant(const std::vector<Eigen::Index>& dimensions, const Scalar& value) { return setConstant(dimensions.begin(), dimensions.end(), value); }
		BlockVector& setConstant(const std::initializer_list<Eigen::Index> dimensions, const Scalar& value) { return setConstant(dimensions.begin(), dimensions.end(), value); }
		template <typename Iterator> BlockVector& setConstant(Iterator first, Iterator last, const Scalar& value) { return reblockAndFill(first, last, [&](auto& result) { result.setConstant(value); }); }

		static BlockVector WithBlocking(const std::vector<Eigen::Index>& dimensions) { return BlockVector(dimensions); }
		static BlockVector WithBlocking(std::initializer_list<Eigen::Index> dimensions) { return BlockVector(dimensions); }
		template <typename Iterator> static BlockVector WithBlocking(Iterator first, Iterator last) { return BlockVector(first, last); }
		template <typename TOther> static BlockVector WithBlockingOf(const TOther& other) { return WithBlocking(other.blockingInfo()); }

	private:
		template <typename Iterator, typename Fill>
		static BlockVector makeFilled(Iterator first, Iterator last, Fill fill) { BlockVector result(first, last); fill(result); return result; }
	public:
		static BlockVector Zero(const std::vector<Eigen::Index>& d) { return Zero(d.begin(), d.end()); }
		static BlockVector Zero(const std::initializer_list<Eigen::Index> d) { return Zero(d.begin(), d.end()); }
		template <typename Iterator> static BlockVector Zero(Iterator f, Iterator l) { return makeFilled(f, l, [](auto& v) { v.setZero(); }); }
		static BlockVector Ones(const std::vector<Eigen::Index>& d) { return Ones(d.begin(), d.end()); }
		static BlockVector Ones(const std::initializer_list<Eigen::Index> d) { return Ones(d.begin(), d.end()); }
		template <typename Iterator> static BlockVector Ones(Iterator f, Iterator l) { return makeFilled(f, l, [](auto& v) { v.setOnes(); }); }
		static BlockVector Random(const std::vector<Eigen::Index>& d) { return Random(d.begin(), d.end()); }
		static BlockVector Random(const std::initializer_list<Eigen::Index> d) { return Random(d.begin(), d.end()); }
		template <typename Iterator> static BlockVector Random(Iterator f, Iterator l) { return makeFilled(f, l, [](auto& v) { v.setRandom(); }); }
		static BlockVector Constant(const std::vector<Eigen::Index>& d, const Scalar& value) { return Constant(d.begin(), d.end(), value); }
		static BlockVector Constant(const std::initializer_list<Eigen::Index> d, const Scalar& value) { return Constant(d.begin(), d.end(), value); }
		template <typename Iterator> static BlockVector Constant(Iterator f, Iterator l, const Scalar& value) { return makeFilled(f, l, [&](auto& v) { v.setConstant(value); }); }

		template <typename DenseDerived>
		BlockVector& extractFromDense(const Eigen::MatrixBase<DenseDerived>& dense)
		{
			if ((dense.rows() != 1 && dense.cols() != 1) || dense.size() != size()) { eigen_assert(false && "dense vector shape mismatch"); return *this; }
			for (Eigen::Index index = 0; index < size(); ++index) data()[index] = static_cast<Scalar>(dense.derived().coeff(index));
			return *this;
		}
		template <typename DenseDerived, typename Iterator>
		static BlockVector FromDense(const Eigen::MatrixBase<DenseDerived>& dense, Iterator first, Iterator last) { BlockVector result(first, last); result.extractFromDense(dense); return result; }
		template <typename DenseDerived> static BlockVector FromDense(const Eigen::MatrixBase<DenseDerived>& dense, const std::vector<Eigen::Index>& d) { return FromDense(dense, d.begin(), d.end()); }
		template <typename DenseDerived> static BlockVector FromDense(const Eigen::MatrixBase<DenseDerived>& dense, std::initializer_list<Eigen::Index> d) { return FromDense(dense, d.begin(), d.end()); }
		template <typename DenseDerived, typename TOther> static BlockVector FromDenseLike(const Eigen::MatrixBase<DenseDerived>& dense, const TOther& other) { return FromDense(dense, other.blockingInfo()); }
		template <typename DenseDerived> static BlockVector SingleBlock(const Eigen::MatrixBase<DenseDerived>& dense) { eigen_assert(dense.rows() == 1 || dense.cols() == 1); return FromDense(dense, {dense.size()}); }
		template <typename BlockIterator>
		static BlockVector FromBlocks(BlockIterator first, BlockIterator last)
		{
			using Category = typename std::iterator_traits<BlockIterator>::iterator_category;
			static_assert(std::is_base_of_v<std::forward_iterator_tag, Category>, "FromBlocks requires forward iterators");
			std::vector<Eigen::Index> dimensions;
			for (auto iterator = first; iterator != last; ++iterator)
			{
				const auto& block = *iterator;
				if (block.rows() != 1 && block.cols() != 1) { eigen_assert(false && "FromBlocks requires vectors"); return {}; }
				dimensions.push_back(block.size());
			}
			BlockVector result(dimensions);
			Eigen::Index offset = 0;
			for (; first != last; ++first)
			{
				const auto& block = *first;
				for (Eigen::Index index = 0; index < block.size(); ++index) result.data()[offset++] = static_cast<Scalar>(block.coeff(index));
			}
			return result;
		}
		template <typename TBlock>
		static BlockVector FromBlocks(std::initializer_list<TBlock> blocks) { return FromBlocks(blocks.begin(), blocks.end()); }

	private:
		Detail::PackedStorage<Scalar> m_Storage;
	};

	template <typename TScalar, typename TOrientation>
	void swap(BlockVector<TScalar, TOrientation>& lhs, BlockVector<TScalar, TOrientation>& rhs) noexcept
	{
		lhs.swap(rhs);
	}
}  // namespace Hoppy
