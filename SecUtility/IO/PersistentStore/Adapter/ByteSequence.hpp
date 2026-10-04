// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/IO/PersistentStore.hpp>
#include <SecUtility/IO/PersistentStore/Detail/Adapter.hpp>
#include <SecUtility/IO/PersistentStore/Detail/Mapping.hpp>

#include <algorithm>
#include <array>
#include <memory>
#include <vector>


namespace SecUtility::IO
{
	class LeasedByteView final
	{
	public:
		LeasedByteView() noexcept = default;

		LeasedByteView(std::shared_ptr<PersistentStoreDetail::MappingLease> leasePtr,
		               const Byte* dataPtr,
		               const std::size_t size) noexcept
		    : m_LeasePtr(std::move(leasePtr)), m_DataPtr(dataPtr), m_Size(size)
		{
			/* NO CODE */
		}

		const Byte* data() const noexcept { return m_DataPtr; }
		std::size_t size() const noexcept { return m_Size; }
		const Byte& operator[](const std::size_t index) const noexcept { return m_DataPtr[index]; }
		const Byte* begin() const noexcept { return m_DataPtr; }
		const Byte* end() const noexcept { return m_DataPtr + m_Size; }

	private:
		std::shared_ptr<PersistentStoreDetail::MappingLease> m_LeasePtr;
		const Byte* m_DataPtr = nullptr;
		std::size_t m_Size = 0;
	};

	namespace PersistentStoreDetail
	{
		inline constexpr auto ByteSequenceEncodingId =
		        MakePersistentEncodingId("edb7849b-6697-554d-adc4-7f2965c2e025");
	}

	template <>
	struct PersistentTraits<std::vector<Byte>>
	{
		using LeasedType = LeasedByteView;

		static PersistentStoreDetail::PayloadLayout Measure(const std::vector<Byte>& value)
		{
			return {PersistentStoreDetail::CheckedAdd(PersistentStoreDetail::AdapterHeaderBytes,
			                                          value.size(), "byte-sequence payload bytes"), 1};
		}

		static void Encode(const std::vector<Byte>& value, const PersistentStoreDetail::MutableByteView bytes)
		{
			using namespace PersistentStoreDetail;
			WriteAdapterHeader(bytes, ByteSequenceEncodingId, AdapterHeaderBytes, bytes.size(), value.size());
			std::copy(value.begin(), value.end(), bytes.data() + AdapterHeaderBytes);
		}

		static std::vector<Byte> Decode(const PersistentStoreDetail::ConstByteView bytes)
		{
			using namespace PersistentStoreDetail;
			if (bytes.size() < AdapterHeaderBytes)
			{
				throw FormatException("truncated PersistentStore byte-sequence payload");
			}
			const UInt64 dataBytes = bytes.size() - AdapterHeaderBytes;
			ValidateAdapterHeader(bytes, ByteSequenceEncodingId, AdapterHeaderBytes, dataBytes);
			return std::vector<Byte>(bytes.data() + AdapterHeaderBytes, bytes.data() + bytes.size());
		}

		static LeasedType DecodeLeased(std::shared_ptr<PersistentStoreDetail::MappingLease> leasePtr)
		{
			using namespace PersistentStoreDetail;
			const ConstByteView bytes(leasePtr->Data(), leasePtr->Size());
			if (bytes.size() < AdapterHeaderBytes)
			{
				throw FormatException("truncated PersistentStore byte-sequence payload");
			}
			const UInt64 dataBytes = bytes.size() - AdapterHeaderBytes;
			ValidateAdapterHeader(bytes, ByteSequenceEncodingId, AdapterHeaderBytes, dataBytes);
			return LeasedType(std::move(leasePtr), bytes.data() + AdapterHeaderBytes,
			                  CheckedNarrow<std::size_t>(dataBytes, "byte-sequence leased size"));
		}
	};
}
