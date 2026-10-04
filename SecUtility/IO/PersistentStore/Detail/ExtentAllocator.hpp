// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/IO/PersistentStore/Detail/CheckedArithmetic.hpp>
#include <SecUtility/IO/PersistentStore/Detail/Directory.hpp>
#include <SecUtility/IO/PersistentStore/Detail/Format.hpp>

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <string_view>
#include <utility>
#include <vector>


namespace SecUtility::IO::PersistentStoreDetail
{
	inline constexpr UInt64 MinimumFreeExtentBytes = 32;

	struct ExtentAllocatorMetrics
	{
		UInt64 PhysicalBytes = 0;
		UInt64 ProtectedBytes = 0;
		UInt64 ReusableBytes = 0;
		UInt64 PendingBytes = 0;
	};

	class ExtentAllocator final
	{
	public:
		void Rebuild(const UInt64 physicalBytes,
		             std::vector<Extent> protectedExtents,
		             const std::vector<Extent>& pendingExtents = {})
		{
			m_FreeByOffset.clear();
			m_FreeBySize.clear();
			m_ReservedExtents.clear();
			m_PhysicalBytes = physicalBytes;
			m_PendingExtents = pendingExtents;
			protectedExtents.push_back({0, SuperblockBytes});
			NormalizeProtected(physicalBytes, protectedExtents);
			m_ProtectedExtents = std::move(protectedExtents);
			NormalizeProtected(physicalBytes, m_PendingExtents);
			std::vector<Extent> unavailableExtents = m_ProtectedExtents;
			unavailableExtents.insert(unavailableExtents.end(), m_PendingExtents.begin(), m_PendingExtents.end());
			NormalizeProtected(physicalBytes, unavailableExtents);

			UInt64 cursor = 0;
			for (const Extent& extent : unavailableExtents)
			{
				if (cursor < extent.Offset)
				{
					InsertFree({cursor, extent.Offset - cursor});
				}
				cursor = std::max(cursor, CheckedAdd(extent.Offset, extent.Capacity, "protected extent end"));
			}
			if (cursor < physicalBytes)
			{
				InsertFree({cursor, physicalBytes - cursor});
			}
		}

		Allocation Reserve(const UInt64 payloadBytes, const UInt64 alignment, const bool absorbSmallFragments = true)
		{
			if (alignment == 0 || alignment > MaximumPayloadAlignment || (alignment & (alignment - 1)) != 0)
			{
				throw FormatException("allocator alignment must be a power of two from 1 through 4096");
			}
			const UInt64 storedBytes = std::max<UInt64>(payloadBytes, 1);
			auto selected = m_FreeByOffset.end();
			for (auto sizeIterator = m_FreeBySize.lower_bound({storedBytes, 0}); sizeIterator != m_FreeBySize.end();
			     ++sizeIterator)
			{
				auto iterator = m_FreeByOffset.find(sizeIterator->second);
				const UInt64 payloadOffset = CheckedAlignUp(iterator->first, alignment, "allocator payload alignment");
				const UInt64 frontBytes = payloadOffset - iterator->first;
				if (frontBytes <= iterator->second && storedBytes <= iterator->second - frontBytes)
				{
					selected = iterator;
					break;
				}
			}

			if (selected == m_FreeByOffset.end())
			{
				const UInt64 allocationOffset = m_PhysicalBytes;
				const UInt64 payloadOffset = CheckedAlignUp(allocationOffset, alignment, "appended payload alignment");
				const UInt64 capacity =
				        CheckedAdd(payloadOffset - allocationOffset, storedBytes, "appended allocation capacity");
				const UInt64 replacementPhysicalBytes =
				        CheckedAdd(allocationOffset, capacity, "appended allocation end");
				m_ReservedExtents.push_back({allocationOffset, capacity});
				m_PhysicalBytes = replacementPhysicalBytes;
				return {{allocationOffset, capacity}, payloadOffset};
			}

			const Extent source{selected->first, selected->second};
			const UInt64 alignedOffset = CheckedAlignUp(source.Offset, alignment, "reused payload alignment");
			const UInt64 frontBytes = alignedOffset - source.Offset;
			const bool hasFrontSplit =
			        frontBytes != 0 && (!absorbSmallFragments || frontBytes >= MinimumFreeExtentBytes);
			const UInt64 allocationOffset = hasFrontSplit ? alignedOffset : source.Offset;
			const UInt64 allocationPrefix = alignedOffset - allocationOffset;
			const UInt64 requiredCapacity = CheckedAdd(allocationPrefix, storedBytes, "reused allocation capacity");
			const UInt64 availableCapacity = source.Capacity - (allocationOffset - source.Offset);
			const UInt64 backBytes = availableCapacity - requiredCapacity;
			const bool hasBackSplit = backBytes != 0 && (!absorbSmallFragments || backBytes >= MinimumFreeExtentBytes);
			const UInt64 allocationCapacity = hasBackSplit ? requiredCapacity : availableCapacity;
			auto replacementByOffset = m_FreeByOffset;
			auto replacementBySize = m_FreeBySize;
			replacementBySize.erase({source.Capacity, source.Offset});
			replacementByOffset.erase(source.Offset);
			if (hasFrontSplit)
			{
				InsertFreeUncoalesced(replacementByOffset, replacementBySize, {source.Offset, frontBytes});
			}
			if (hasBackSplit)
			{
				InsertFreeUncoalesced(replacementByOffset,
				                      replacementBySize,
				                      {CheckedAdd(allocationOffset, requiredCapacity, "back split offset"), backBytes});
			}
			m_ReservedExtents.push_back({allocationOffset, allocationCapacity});
			m_FreeByOffset.swap(replacementByOffset);
			m_FreeBySize.swap(replacementBySize);
			return {{allocationOffset, allocationCapacity}, alignedOffset};
		}

		void ReleasePending(const Extent extent)
		{
			if (extent.Capacity != 0)
			{
				m_PendingExtents.push_back(extent);
			}
		}

		void RetryPendingReleases()
		{
			auto replacementByOffset = m_FreeByOffset;
			auto replacementBySize = m_FreeBySize;
			for (const Extent extent : m_PendingExtents)
			{
				InsertFreeCoalesced(replacementByOffset, replacementBySize, extent);
			}
			m_FreeByOffset.swap(replacementByOffset);
			m_FreeBySize.swap(replacementBySize);
			m_PendingExtents.clear();
		}
		const std::map<UInt64, UInt64>& GetFreeByOffset() const noexcept
		{
			return m_FreeByOffset;
		}
		UInt64 GetPhysicalBytes() const noexcept
		{
			return m_PhysicalBytes;
		}

		ExtentAllocatorMetrics GetMetrics() const
		{
			ExtentAllocatorMetrics metrics;
			metrics.PhysicalBytes = m_PhysicalBytes;
			for (const Extent& extent : m_ProtectedExtents)
			{
				metrics.ProtectedBytes += extent.Capacity;
			}
			for (const Extent& extent : m_ReservedExtents)
			{
				metrics.ProtectedBytes += extent.Capacity;
			}
			for (const auto& [offset, capacity] : m_FreeByOffset)
			{
				metrics.ReusableBytes += capacity;
			}
			for (const Extent& extent : m_PendingExtents)
			{
				metrics.PendingBytes += extent.Capacity;
			}
			return metrics;
		}

		ExtentAllocatorMetrics Audit() const
		{
			if (m_FreeByOffset.size() != m_FreeBySize.size())
			{
				throw InvalidOperationException("PersistentStore allocator indexes have different sizes");
			}
			UInt64 previousEnd = 0;
			bool hasPrevious = false;
			for (const auto& [offset, capacity] : m_FreeByOffset)
			{
				if (capacity == 0 || !IsRangeContained(0, m_PhysicalBytes, offset, capacity))
				{
					throw InvalidOperationException("PersistentStore allocator contains an invalid free extent");
				}
				if (hasPrevious && offset <= previousEnd)
				{
					throw InvalidOperationException(
					        "PersistentStore allocator free extents overlap or are uncoalesced");
				}
				if (m_FreeBySize.count({capacity, offset}) != 1)
				{
					throw InvalidOperationException("PersistentStore allocator indexes disagree");
				}
				previousEnd = CheckedAdd(offset, capacity, "audited free extent end");
				hasPrevious = true;
			}
			std::vector<Extent> coverage = m_ProtectedExtents;
			coverage.insert(coverage.end(), m_ReservedExtents.begin(), m_ReservedExtents.end());
			coverage.insert(coverage.end(), m_PendingExtents.begin(), m_PendingExtents.end());
			for (const auto& [offset, capacity] : m_FreeByOffset)
			{
				coverage.push_back({offset, capacity});
			}
			std::sort(coverage.begin(),
			          coverage.end(),
			          [](const Extent& left, const Extent& right) { return left.Offset < right.Offset; });
			UInt64 cursor = 0;
			for (const Extent extent : coverage)
			{
				if (extent.Offset != cursor)
				{
					throw InvalidOperationException("PersistentStore allocator coverage has a gap or overlap");
				}
				cursor = CheckedAdd(cursor, extent.Capacity, "audited allocator coverage");
			}
			if (cursor != m_PhysicalBytes)
			{
				throw InvalidOperationException("PersistentStore allocator coverage does not match physical length");
			}
			return GetMetrics();
		}

	private:
		static void NormalizeProtected(const UInt64 physicalBytes, std::vector<Extent>& ref_extents)
		{
			for (const auto& [offset, capacity] : std::as_const(ref_extents))
			{
				if (capacity == 0 || !IsRangeContained(0, physicalBytes, offset, capacity))
				{
					throw FormatException("protected extent lies outside the physical file");
				}
			}
			std::sort(ref_extents.begin(),
			          ref_extents.end(),
			          [](const Extent& left, const Extent& right) { return left.Offset < right.Offset; });
			std::vector<Extent> merged;
			for (const Extent& extent : ref_extents)
			{
				if (merged.empty()
				    || CheckedAdd(merged.back().Offset, merged.back().Capacity, "protected extent end") < extent.Offset)
				{
					merged.push_back(extent);
				}
				else
				{
					const UInt64 end =
					        std::max(CheckedAdd(merged.back().Offset, merged.back().Capacity, "protected extent end"),
					                 CheckedAdd(extent.Offset, extent.Capacity, "protected extent end"));
					merged.back().Capacity = end - merged.back().Offset;
				}
			}
			ref_extents = std::move(merged);
		}

		void InsertFree(const Extent extent)
		{
			if (extent.Capacity == 0)
			{
				return;
			}
			m_FreeByOffset.emplace(extent.Offset, extent.Capacity);
			m_FreeBySize.emplace(extent.Capacity, extent.Offset);
		}

		static void InsertFreeUncoalesced(std::map<UInt64, UInt64>& ref_byOffset,
		                                  std::set<std::pair<UInt64, UInt64>>& ref_bySize,
		                                  const Extent extent)
		{
			if (extent.Capacity == 0)
			{
				return;
			}
			ref_byOffset.emplace(extent.Offset, extent.Capacity);
			ref_bySize.emplace(extent.Capacity, extent.Offset);
		}

		static void InsertFreeCoalesced(std::map<UInt64, UInt64>& ref_byOffset,
		                                std::set<std::pair<UInt64, UInt64>>& ref_bySize,
		                                Extent extent)
		{
			if (extent.Capacity == 0)
			{
				return;
			}
			auto next = ref_byOffset.lower_bound(extent.Offset);
			if (next != ref_byOffset.begin())
			{
				auto previous = std::prev(next);
				const UInt64 previousEnd = CheckedAdd(previous->first, previous->second, "free extent end");
				if (previousEnd > extent.Offset)
				{
					throw InvalidOperationException("released PersistentStore extents overlap");
				}
				if (previousEnd == extent.Offset)
				{
					extent.Offset = previous->first;
					extent.Capacity = CheckedAdd(previous->second, extent.Capacity, "coalesced extent capacity");
					ref_bySize.erase({previous->second, previous->first});
					ref_byOffset.erase(previous);
				}
			}
			next = ref_byOffset.lower_bound(extent.Offset);
			if (next != ref_byOffset.end())
			{
				const UInt64 extentEnd = CheckedAdd(extent.Offset, extent.Capacity, "free extent end");
				if (extentEnd > next->first)
				{
					throw InvalidOperationException("released PersistentStore extents overlap");
				}
				if (extentEnd == next->first)
				{
					extent.Capacity = CheckedAdd(extent.Capacity, next->second, "coalesced extent capacity");
					ref_bySize.erase({next->second, next->first});
					ref_byOffset.erase(next);
				}
			}
			const auto [iterator, inserted] = ref_byOffset.emplace(extent.Offset, extent.Capacity);
			if (!inserted)
			{
				throw InvalidOperationException("released PersistentStore extent has a duplicate offset");
			}
			try
			{
				ref_bySize.emplace(extent.Capacity, extent.Offset);
			}
			catch (...)
			{
				ref_byOffset.erase(iterator);
				throw;
			}
		}

		UInt64 m_PhysicalBytes = 0;
		std::map<UInt64, UInt64> m_FreeByOffset;
		std::set<std::pair<UInt64, UInt64>> m_FreeBySize;
		std::vector<Extent> m_ProtectedExtents;
		std::vector<Extent> m_ReservedExtents;
		std::vector<Extent> m_PendingExtents;
	};
}
