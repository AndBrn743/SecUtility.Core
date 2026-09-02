// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/Detail/CheckedDimensions.hpp>
#include <SecUtility/Hoppy/Detail/Storage.hpp>
#include <SecUtility/Hoppy/Detail/Traits.hpp>
#include <SecUtility/Hoppy/ForwardDeclarations.hpp>

#include <Eigen/Core>

#include <initializer_list>
#include <type_traits>
#include <utility>
#include <vector>

namespace Hoppy
{
	template <typename TScalar, typename TOrientation>
	class BlockVector
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

		template <typename Iterator>
		BlockVector(Iterator first, Iterator last)
			: m_Storage(Detail::BuildCheckedVectorDimensions(first, last))
		{}

		BlockVector(const BlockVector&) = default;
		BlockVector(BlockVector&&) noexcept = default;
		BlockVector& operator=(const BlockVector&) = default;
		BlockVector& operator=(BlockVector&&) noexcept = default;
		~BlockVector() = default;

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

		Block operator[](Eigen::Index index)
		{
			const auto dimension = dimensionOfBlock(index);
			return Block(data() + storageOffset(index), dimension);
		}

		ConstBlock operator[](Eigen::Index index) const
		{
			const auto dimension = dimensionOfBlock(index);
			return ConstBlock(data() + storageOffset(index), dimension);
		}

		Block asDense() { return Block(data(), totalDimension()); }
		ConstBlock asDense() const { return ConstBlock(data(), totalDimension()); }

	private:
		Detail::PackedStorage<Scalar> m_Storage;
	};

	template <typename TScalar, typename TOrientation>
	void swap(BlockVector<TScalar, TOrientation>& lhs, BlockVector<TScalar, TOrientation>& rhs) noexcept
	{
		lhs.swap(rhs);
	}
}  // namespace Hoppy
