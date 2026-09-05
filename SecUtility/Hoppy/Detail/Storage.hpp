// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/Detail/CheckedDimensions.hpp>

#include <Eigen/Core>

#include <utility>
#include <vector>

namespace Hoppy::Detail
{
	template <typename TScalar>
	class PackedStorage
	{
	public:
		PackedStorage() = default;

		explicit PackedStorage(CheckedDimensions dimensions)
			: m_Data(static_cast<std::size_t>(dimensions.StoredSize)),
			  m_Dimensions(std::move(dimensions.Dimensions)),
			  m_BlockOffsets(std::move(dimensions.BlockOffsets)),
			  m_StorageOffsets(std::move(dimensions.StorageOffsets)),
			  m_TotalDimension(dimensions.TotalDimension),
			  m_StoredSize(dimensions.StoredSize)
		{}

		void swap(PackedStorage& other) noexcept
		{
			using std::swap;
			m_Data.swap(other.m_Data);
			m_Dimensions.swap(other.m_Dimensions);
			m_BlockOffsets.swap(other.m_BlockOffsets);
			m_StorageOffsets.swap(other.m_StorageOffsets);
			swap(m_TotalDimension, other.m_TotalDimension);
			swap(m_StoredSize, other.m_StoredSize);
		}

		Eigen::Index blockCount() const noexcept { return static_cast<Eigen::Index>(m_Dimensions.size()); }
		Eigen::Index totalDimension() const noexcept { return m_TotalDimension; }
		Eigen::Index storedSize() const noexcept { return m_StoredSize; }
		TScalar* data() noexcept { return m_Data.data(); }
		const TScalar* data() const noexcept { return m_Data.data(); }

		const std::vector<Eigen::Index>& blockingInfo() const noexcept { return m_Dimensions; }

		Eigen::Index dimensionOfBlock(const Eigen::Index index) const
		{
			assertBlockIndex(index);
			return m_Dimensions[static_cast<std::size_t>(index)];
		}

		Eigen::Index blockOffset(const Eigen::Index index) const
		{
			assertBlockIndex(index);
			return m_BlockOffsets[static_cast<std::size_t>(index)];
		}

		Eigen::Index storageOffset(const Eigen::Index index) const
		{
			assertBlockIndex(index);
			return m_StorageOffsets[static_cast<std::size_t>(index)];
		}

	private:
		void assertBlockIndex(const Eigen::Index index [[maybe_unused]]) const
		{
			eigen_assert(index >= 0 && index < blockCount());
		}

		std::vector<TScalar> m_Data;
		std::vector<Eigen::Index> m_Dimensions;
		std::vector<Eigen::Index> m_BlockOffsets{0};
		std::vector<Eigen::Index> m_StorageOffsets{0};
		Eigen::Index m_TotalDimension = 0;
		Eigen::Index m_StoredSize = 0;
	};

	template <typename TScalar>
	void swap(PackedStorage<TScalar>& lhs, PackedStorage<TScalar>& rhs) noexcept
	{
		lhs.swap(rhs);
	}
}  // namespace Hoppy::Detail
