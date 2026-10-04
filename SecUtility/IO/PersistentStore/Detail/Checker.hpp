// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/IO/PersistentStoreChecker.hpp>
#include <SecUtility/IO/PersistentStore/Detail/FileBackend.hpp>
#include <SecUtility/IO/PersistentStore/Detail/FormatCodec.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <exception>
#include <string>
#include <utility>
#include <vector>


namespace SecUtility::IO::PersistentStoreDetail
{
	struct CheckedRoot
	{
		RootCandidate Candidate;
	};

	inline CheckedRoot CheckOneRoot(const FileBackend& backend,
	                                const UInt64 slotOffset,
	                                const UInt64 physicalFileBytes)
	{
		CheckedRoot checked;
		checked.Candidate.SlotOffset = slotOffset;
		if (!IsRangeContained(0, physicalFileBytes, slotOffset, HeaderSlotBytes))
		{
			checked.Candidate.RejectionReason = "header slot lies outside the physical file";
			return checked;
		}
		try
		{
			std::array<Byte, HeaderSlotBytes> headerBytes{};
			backend.ReadExact(slotOffset, headerBytes.data(), headerBytes.size());
			checked.Candidate.Header = ParseHeader(ConstByteView(headerBytes), physicalFileBytes);
			std::vector<Byte> directoryBytes(CheckedNarrow<std::size_t>(
			        checked.Candidate.Header.DirectoryBytes, "checker directory buffer size"));
			backend.ReadExact(checked.Candidate.Header.DirectoryOffset,
			                  directoryBytes.data(), directoryBytes.size());
			checked.Candidate.ParsedDirectory = ParseDirectory(
			        checked.Candidate.Header, ConstByteView(directoryBytes), physicalFileBytes);
			checked.Candidate.IsValid = true;
		}
		catch (const std::exception& exception)
		{
			checked.Candidate.RejectionReason = exception.what();
		}
		return checked;
	}

	inline std::vector<Extent> CollectRootRanges(const RootCandidate& root, const bool includesSuperblock)
	{
		std::vector<Extent> ranges;
		if (includesSuperblock) ranges.push_back({0, SuperblockBytes});
		if (!root.IsValid) return ranges;
		ranges.push_back({root.Header.DirectoryOffset, root.Header.DirectoryBytes});
		for (const DirectoryEntry& entry : root.ParsedDirectory.Entries)
			ranges.push_back(entry.AllocatedExtent);
		return ranges;
	}

	inline std::vector<Extent> MergeCheckerRanges(std::vector<Extent> ranges)
	{
		std::sort(ranges.begin(), ranges.end(), [](const Extent& left, const Extent& right)
		{
			return left.Offset < right.Offset
			       || (left.Offset == right.Offset && left.Capacity < right.Capacity);
		});
		std::vector<Extent> merged;
		for (const Extent range : ranges)
		{
			if (range.Capacity == 0) continue;
			if (merged.empty())
			{
				merged.push_back(range);
				continue;
			}
			Extent& previous = merged.back();
			const UInt64 previousEnd = CheckedAdd(previous.Offset, previous.Capacity, "checker range end");
			const UInt64 rangeEnd = CheckedAdd(range.Offset, range.Capacity, "checker range end");
			if (range.Offset > previousEnd)
			{
				merged.push_back(range);
			}
			else if (rangeEnd > previousEnd)
			{
				previous.Capacity = rangeEnd - previous.Offset;
			}
		}
		return merged;
	}

	inline UInt64 MeasureRanges(const std::vector<Extent>& ranges, const UInt64 limit)
	{
		UInt64 bytes = 0;
		for (const Extent range : ranges)
		{
			if (range.Offset >= limit) continue;
			const UInt64 end = std::min(CheckedAdd(range.Offset, range.Capacity, "checker range end"), limit);
			bytes = CheckedAdd(bytes, end - range.Offset, "checker byte total");
		}
		return bytes;
	}

	inline UInt64 MeasureDifference(const std::vector<Extent>& left, const std::vector<Extent>& right)
	{
		UInt64 bytes = 0;
		for (const Extent leftRange : left)
		{
			UInt64 cursor = leftRange.Offset;
			const UInt64 end = CheckedAdd(leftRange.Offset, leftRange.Capacity, "checker range end");
			for (const Extent rightRange : right)
			{
				const UInt64 rightEnd = CheckedAdd(rightRange.Offset, rightRange.Capacity, "checker range end");
				if (rightEnd <= cursor) continue;
				if (rightRange.Offset >= end) break;
				if (rightRange.Offset > cursor) bytes = CheckedAdd(bytes, rightRange.Offset - cursor, "checker difference");
				cursor = std::max(cursor, rightEnd);
				if (cursor >= end) break;
			}
			if (cursor < end) bytes = CheckedAdd(bytes, end - cursor, "checker difference");
		}
		return bytes;
	}

	inline bool ReportCrossRootConflicts(const RootCandidate& left,
	                                     const RootCandidate& right,
	                                     std::vector<std::string>& ref_findings)
	{
		if (!left.IsValid || !right.IsValid) return false;
		struct ClassifiedRange
		{
			Extent Range;
			const DirectoryEntry* EntryPtr = nullptr;
		};
		auto Classify = [](const RootCandidate& root)
		{
			std::vector<ClassifiedRange> ranges{{{root.Header.DirectoryOffset, root.Header.DirectoryBytes}, nullptr}};
			for (const DirectoryEntry& entry : root.ParsedDirectory.Entries)
				ranges.push_back({entry.AllocatedExtent, &entry});
			return ranges;
		};
		const std::vector<ClassifiedRange> leftRanges = Classify(left);
		const std::vector<ClassifiedRange> rightRanges = Classify(right);
		for (const ClassifiedRange& leftRange : leftRanges)
		{
			for (const ClassifiedRange& rightRange : rightRanges)
			{
				if (!DoRangesOverlap(leftRange.Range.Offset, leftRange.Range.Capacity,
				                     rightRange.Range.Offset, rightRange.Range.Capacity)) continue;
				const bool hasSameRange = leftRange.Range.Offset == rightRange.Range.Offset
				                          && leftRange.Range.Capacity == rightRange.Range.Capacity;
				const bool isSharedDirectory = hasSameRange && leftRange.EntryPtr == nullptr
				                               && rightRange.EntryPtr == nullptr
				                               && left.Header.Generation == right.Header.Generation
				                               && left.Header.DirectoryChecksum == right.Header.DirectoryChecksum;
				const bool isSharedPayload = hasSameRange && leftRange.EntryPtr != nullptr
				                             && rightRange.EntryPtr != nullptr
				                             && leftRange.EntryPtr->Key == rightRange.EntryPtr->Key
				                             && leftRange.EntryPtr->PayloadOffset == rightRange.EntryPtr->PayloadOffset
				                             && leftRange.EntryPtr->PayloadBytes == rightRange.EntryPtr->PayloadBytes;
				if (isSharedDirectory || isSharedPayload) continue;
				ref_findings.emplace_back("recoverable roots contain inconsistent overlapping extents");
				return true;
			}
		}
		return false;
	}

	inline PersistentStoreSlotReport MakeSlotReport(const RootCandidate& candidate)
	{
		return {candidate.IsValid, candidate.IsValid ? candidate.Header.Generation : 0,
		        candidate.IsValid ? std::string{} : candidate.RejectionReason};
	}

	inline PersistentStoreCheckReport CheckPersistentStoreFile(const std::filesystem::path& path)
	{
		FileBackend backend = FileBackend::Open(path, FileAccess::ReadOnly);
		PersistentStoreCheckReport report;
		report.PhysicalFileBytes = backend.GetPhysicalFileBytes();
		const CheckedRoot slotA = CheckOneRoot(backend, HeaderSlotAOffset, report.PhysicalFileBytes);
		const CheckedRoot slotB = CheckOneRoot(backend, HeaderSlotBOffset, report.PhysicalFileBytes);
		report.SlotA = MakeSlotReport(slotA.Candidate);
		report.SlotB = MakeSlotReport(slotB.Candidate);
		report.HasCrossRootConflict = ReportCrossRootConflicts(
		        slotA.Candidate, slotB.Candidate, report.Findings);

		const RootCandidate* selectedPtr = nullptr;
		const RootCandidate* otherPtr = nullptr;
		if (slotA.Candidate.IsValid && (!slotB.Candidate.IsValid
		                                  || slotA.Candidate.Header.Generation >= slotB.Candidate.Header.Generation))
		{
			selectedPtr = &slotA.Candidate;
			otherPtr = slotB.Candidate.IsValid ? &slotB.Candidate : nullptr;
		}
		else if (slotB.Candidate.IsValid)
		{
			selectedPtr = &slotB.Candidate;
			otherPtr = slotA.Candidate.IsValid ? &slotA.Candidate : nullptr;
		}
		if (selectedPtr == nullptr) return report;

		report.HasSelectedRoot = true;
		report.SelectedGeneration = selectedPtr->Header.Generation;
		report.CommittedFileBytes = selectedPtr->Header.CommittedLogicalFileBytes;
		report.LiveRecordCount = selectedPtr->ParsedDirectory.Entries.size();
		for (const DirectoryEntry& entry : selectedPtr->ParsedDirectory.Entries)
			report.LivePayloadBytes = CheckedAdd(report.LivePayloadBytes, entry.PayloadBytes, "checker live bytes");

		const std::vector<Extent> selectedRanges = MergeCheckerRanges(CollectRootRanges(*selectedPtr, true));
		std::vector<Extent> allRanges = selectedRanges;
		std::vector<Extent> otherRanges;
		if (otherPtr != nullptr)
		{
			otherRanges = MergeCheckerRanges(CollectRootRanges(*otherPtr, false));
			allRanges.insert(allRanges.end(), otherRanges.begin(), otherRanges.end());
		}
		allRanges = MergeCheckerRanges(std::move(allRanges));
		report.OtherRootOnlyBytes = MeasureDifference(otherRanges, selectedRanges);
		report.MetadataBytes = SuperblockBytes + selectedPtr->Header.DirectoryBytes
		                       + (otherPtr == nullptr ? 0 : MeasureDifference(
		                               MergeCheckerRanges({{otherPtr->Header.DirectoryOffset,
		                                                    otherPtr->Header.DirectoryBytes}}), selectedRanges));
		const UInt64 protectedCommittedBytes = MeasureRanges(allRanges, report.CommittedFileBytes);
		report.ReclaimableBytes = report.CommittedFileBytes - protectedCommittedBytes;
		report.PhysicalTailBytes = report.PhysicalFileBytes - report.CommittedFileBytes;
		return report;
	}
}
