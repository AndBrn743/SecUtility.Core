// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/BlockPolicy.hpp>
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
	class BlockDiagonalMatrix
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

		template <typename Iterator>
		BlockDiagonalMatrix(Iterator first, Iterator last)
			: m_Storage(Detail::BuildCheckedDimensions(first, last))
		{}

		BlockDiagonalMatrix(const BlockDiagonalMatrix&) = default;
		BlockDiagonalMatrix(BlockDiagonalMatrix&&) noexcept = default;
		BlockDiagonalMatrix& operator=(const BlockDiagonalMatrix&) = default;
		BlockDiagonalMatrix& operator=(BlockDiagonalMatrix&&) noexcept = default;
		~BlockDiagonalMatrix() = default;

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

		Block operator[](Eigen::Index index)
		{
			const auto dimension = dimensionOfBlock(index);
			return Block(data() + storageOffset(index), dimension, dimension);
		}

		ConstBlock operator[](Eigen::Index index) const
		{
			const auto dimension = dimensionOfBlock(index);
			return ConstBlock(data() + storageOffset(index), dimension, dimension);
		}

	private:
		Detail::PackedStorage<Scalar> m_Storage;
	};

	template <typename TScalar, typename TBlockPolicy>
	void swap(BlockDiagonalMatrix<TScalar, TBlockPolicy>& lhs,
	          BlockDiagonalMatrix<TScalar, TBlockPolicy>& rhs) noexcept
	{
		lhs.swap(rhs);
	}
}  // namespace Hoppy
