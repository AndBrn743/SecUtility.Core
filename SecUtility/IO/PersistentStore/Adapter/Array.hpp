// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/IO/PersistentStore/Adapter/NativeObject.hpp>

#include <array>
#include <utility>
#include <vector>


namespace SecUtility::IO::PersistentStoreDetail
{
	inline constexpr auto ArrayEncodingId =
	        MakePersistentEncodingId("61a55e49-bf70-51bd-aaab-950a981a8332");
	inline constexpr UInt64 ArrayHeaderBytes = 64;

	template <typename T>
	PayloadLayout MeasureArray(const UInt64 count)
	{
		static_assert(IsPersistentScalar<T>);
		const UInt64 dataOffset = CheckedAlignUp(ArrayHeaderBytes, alignof(T), "array data offset");
		const UInt64 dataBytes = CheckedMultiply(count, sizeof(T), "array data bytes");
		return {CheckedAdd(dataOffset, dataBytes, "array payload bytes"), alignof(T)};
	}

	template <typename T, typename ElementAt>
	void EncodeArray(const UInt64 count, ElementAt&& elementAt, const MutableByteView bytes)
	{
		const PayloadLayout layout = MeasureArray<T>(count);
		if (bytes.size() != layout.Bytes)
		{
			throw FormatException("array payload size changed after measurement");
		}
		const UInt64 dataOffset = CheckedAlignUp(ArrayHeaderBytes, alignof(T), "array data offset");
		const UInt64 dataBytes = CheckedMultiply(count, sizeof(T), "array data bytes");
		WriteAdapterHeader(bytes, ArrayEncodingId, ArrayHeaderBytes, layout.Bytes, dataBytes);
		FormatCodecDetail::WriteLittleEndian<UInt32>(
		        bytes, 40, scalar_encoding<T>::Code, "array scalar code");
		FormatCodecDetail::WriteLittleEndian<UInt32>(bytes, 44, 0, "array reserved");
		FormatCodecDetail::WriteLittleEndian<UInt64>(bytes, 48, count, "array element count");
		FormatCodecDetail::WriteLittleEndian<UInt64>(bytes, 56, dataOffset, "array data offset");
		for (UInt64 index = 0; index < count; ++index)
		{
			const UInt64 elementOffset = CheckedAdd(
			        dataOffset, CheckedMultiply(index, sizeof(T), "array element offset"), "array element offset");
			WriteScalar<T>(bytes, CheckedNarrow<std::size_t>(elementOffset, "array index"),
			               elementAt(CheckedNarrow<std::size_t>(index, "array index")));
		}
	}

	template <typename T>
	std::pair<UInt64, UInt64> ValidateArray(const ConstByteView bytes)
	{
		if (bytes.size() < ArrayHeaderBytes
		    || !std::equal(ArrayEncodingId.data(), ArrayEncodingId.data() + ArrayEncodingId.size(), bytes.data()))
		{
			throw FormatException("PersistentStore array encoding type mismatch");
		}
		const UInt64 count = FormatCodecDetail::ReadLittleEndian<UInt64>(bytes, 48, "array element count");
		const UInt64 dataOffset = FormatCodecDetail::ReadLittleEndian<UInt64>(bytes, 56, "array data offset");
		const UInt64 dataBytes = CheckedMultiply(count, sizeof(T), "array data bytes");
		ValidateAdapterHeader(bytes, ArrayEncodingId, ArrayHeaderBytes, dataBytes);
		if (FormatCodecDetail::ReadLittleEndian<UInt32>(bytes, 40, "array scalar code")
		            != scalar_encoding<T>::Code
		    || FormatCodecDetail::ReadLittleEndian<UInt32>(bytes, 44, "array reserved") != 0
		    || dataOffset != CheckedAlignUp(ArrayHeaderBytes, alignof(T), "array data offset")
		    || dataOffset > bytes.size() || dataBytes != bytes.size() - dataOffset)
		{
			throw FormatException("malformed or incompatible PersistentStore array payload");
		}
		return {count, dataOffset};
	}
}


namespace SecUtility::IO
{
	template <typename T, std::size_t Size>
	struct PersistentTraits<std::array<T, Size>, std::enable_if_t<PersistentStoreDetail::IsPersistentScalar<T>>>
	{
		static PersistentStoreDetail::PayloadLayout Measure(const std::array<T, Size>&)
		{
			return PersistentStoreDetail::MeasureArray<T>(Size);
		}

		static void Encode(const std::array<T, Size>& value, const PersistentStoreDetail::MutableByteView bytes)
		{
			PersistentStoreDetail::EncodeArray<T>(Size, [&value](const std::size_t index) { return value[index]; }, bytes);
		}

		static std::array<T, Size> Decode(const PersistentStoreDetail::ConstByteView bytes)
		{
			const auto [count, dataOffset] = PersistentStoreDetail::ValidateArray<T>(bytes);
			if (count != Size)
			{
				throw FormatException("PersistentStore fixed array element count mismatch");
			}
			std::array<T, Size> value{};
			for (std::size_t index = 0; index < Size; ++index)
			{
				value[index] = PersistentStoreDetail::ReadScalar<T>(
				        bytes, static_cast<std::size_t>(dataOffset) + index * sizeof(T));
			}
			return value;
		}
	};

	template <typename T>
	struct PersistentTraits<std::vector<T>,
	                        std::enable_if_t<PersistentStoreDetail::IsPersistentScalar<T>
	                                         && !std::is_same_v<T, Byte>>>
	{
		static PersistentStoreDetail::PayloadLayout Measure(const std::vector<T>& value)
		{
			return PersistentStoreDetail::MeasureArray<T>(value.size());
		}

		static void Encode(const std::vector<T>& value, const PersistentStoreDetail::MutableByteView bytes)
		{
			PersistentStoreDetail::EncodeArray<T>(
			        value.size(), [&value](const std::size_t index) { return value[index]; }, bytes);
		}

		static std::vector<T> Decode(const PersistentStoreDetail::ConstByteView bytes)
		{
			const auto [count, dataOffset] = PersistentStoreDetail::ValidateArray<T>(bytes);
			std::vector<T> value(PersistentStoreDetail::CheckedNarrow<std::size_t>(count, "array element count"));
			for (std::size_t index = 0; index < value.size(); ++index)
			{
				value[index] = PersistentStoreDetail::ReadScalar<T>(
				        bytes, static_cast<std::size_t>(dataOffset) + index * sizeof(T));
			}
			return value;
		}
	};
}
