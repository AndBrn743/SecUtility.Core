// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

// ReSharper disable CppUseDesignatedInitializers
#pragma once

#include <SecUtility/IO/PersistentStore/Detail/CheckedArithmetic.hpp>
#include <SecUtility/IO/PersistentStore/Detail/Directory.hpp>
#include <SecUtility/IO/PersistentStore/Detail/Format.hpp>

#include <algorithm>
#include <map>
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
	};

	class ExtentAllocator final
	{
	public:
		void Rebuild(const UInt64 physicalBytes, std::vector<Extent> protectedExtents)
		{
			protectedExtents.push_back({0, SuperblockBytes});
			NormalizeProtected(physicalBytes, protectedExtents);
			std::map<UInt64, UInt64> replacementFreeExtents;

			UInt64 cursor = 0;
			for (const Extent& extent : protectedExtents)
			{
				if (cursor < extent.Offset)
				{
					replacementFreeExtents.emplace(cursor, extent.Offset - cursor);
				}
				cursor = std::max(cursor, CheckedAdd(extent.Offset, extent.Capacity, "protected extent end"));
			}
			if (cursor < physicalBytes)
			{
				replacementFreeExtents.emplace(cursor, physicalBytes - cursor);
			}
			m_FreeByOffset.swap(replacementFreeExtents);
			m_ProtectedExtents = std::move(protectedExtents);
			m_ReservedExtents.clear();
			m_PhysicalBytes = physicalBytes;
		}

		Allocation Reserve(const UInt64 payloadBytes, const UInt64 alignment, const bool absorbSmallFragments = true)
		{
			if (alignment == 0 || alignment > MaximumPayloadAlignment || (alignment & (alignment - 1)) != 0)
			{
				throw FormatException("allocator alignment must be a power of two from 1 through 4096");
			}
			const UInt64 storedBytes = std::max<UInt64>(payloadBytes, 1);
			auto selected = m_FreeByOffset.end();
			for (auto iterator = m_FreeByOffset.begin(); iterator != m_FreeByOffset.end(); ++iterator)
			{
				const UInt64 payloadOffset = CheckedAlignUp(iterator->first, alignment, "allocator payload alignment");
				const UInt64 frontBytes = payloadOffset - iterator->first;
				if (frontBytes <= iterator->second && storedBytes <= iterator->second - frontBytes
				    && (selected == m_FreeByOffset.end() || iterator->second < selected->second))
				{
					selected = iterator;
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
			std::map<UInt64, UInt64> preparedSplits;
			if (hasFrontSplit)
			{
				preparedSplits.emplace(source.Offset, frontBytes);
			}
			if (hasBackSplit)
			{
				preparedSplits.emplace(CheckedAdd(allocationOffset, requiredCapacity, "back split offset"), backBytes);
			}
			m_ReservedExtents.push_back({allocationOffset, allocationCapacity});
			m_FreeByOffset.erase(selected);
			while (!preparedSplits.empty())
			{
				m_FreeByOffset.insert(preparedSplits.extract(preparedSplits.begin()));
			}
			return {{allocationOffset, allocationCapacity}, alignedOffset};
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
			return metrics;
		}

		ExtentAllocatorMetrics Audit() const
		{
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
				previousEnd = CheckedAdd(offset, capacity, "audited free extent end");
				hasPrevious = true;
			}
			std::vector<Extent> coverage = m_ProtectedExtents;
			coverage.insert(coverage.end(), m_ReservedExtents.begin(), m_ReservedExtents.end());
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

		UInt64 m_PhysicalBytes = 0;
		std::map<UInt64, UInt64> m_FreeByOffset;
		std::vector<Extent> m_ProtectedExtents;
		std::vector<Extent> m_ReservedExtents;
	};
}
