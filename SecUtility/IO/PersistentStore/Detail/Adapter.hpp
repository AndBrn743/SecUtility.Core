// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/IO/PersistentStore/EncodingId.hpp>
#include <SecUtility/IO/PersistentStore/Detail/FormatCodec.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>


namespace SecUtility::IO::PersistentStoreDetail
{
	struct PayloadLayout
	{
		UInt64 Bytes = 0;
		UInt64 Alignment = 1;
	};

	inline constexpr UInt32 AdapterEncodingVersion = 1;
	inline constexpr UInt64 AdapterHeaderBytes = 40;

	inline void WriteAdapterHeader(const MutableByteView bytes,
	                               const PersistentEncodingId& encodingId,
	                               const UInt32 headerBytes,
	                               const UInt64 totalBytes,
	                               const UInt64 dataBytes)
	{
		if (bytes.size() != totalBytes || headerBytes < AdapterHeaderBytes || headerBytes > totalBytes
		    || dataBytes > totalBytes - headerBytes)
		{
			throw FormatException("invalid adapter payload layout");
		}
		std::copy(encodingId.data(), encodingId.data() + encodingId.size(), bytes.data());
		FormatCodecDetail::WriteLittleEndian<UInt32>(bytes, 16, AdapterEncodingVersion, "adapter version");
		FormatCodecDetail::WriteLittleEndian<UInt32>(bytes, 20, headerBytes, "adapter header bytes");
		FormatCodecDetail::WriteLittleEndian<UInt64>(bytes, 24, totalBytes, "adapter total bytes");
		FormatCodecDetail::WriteLittleEndian<UInt64>(bytes, 32, dataBytes, "adapter data bytes");
	}

	inline void ValidateAdapterHeader(const ConstByteView bytes,
	                                  const PersistentEncodingId& encodingId,
	                                  const UInt32 expectedHeaderBytes,
	                                  const UInt64 expectedDataBytes)
	{
		if (bytes.size() < AdapterHeaderBytes
		    || !std::equal(encodingId.data(), encodingId.data() + encodingId.size(), bytes.data()))
		{
			throw FormatException("PersistentStore payload encoding type mismatch");
		}
		if (FormatCodecDetail::ReadLittleEndian<UInt32>(bytes, 16, "adapter version") != AdapterEncodingVersion)
		{
			throw FormatException("unsupported PersistentStore payload encoding version");
		}
		const UInt32 headerBytes = FormatCodecDetail::ReadLittleEndian<UInt32>(bytes, 20, "adapter header bytes");
		const UInt64 totalBytes = FormatCodecDetail::ReadLittleEndian<UInt64>(bytes, 24, "adapter total bytes");
		const UInt64 dataBytes = FormatCodecDetail::ReadLittleEndian<UInt64>(bytes, 32, "adapter data bytes");
		if (headerBytes != expectedHeaderBytes || totalBytes != bytes.size() || dataBytes != expectedDataBytes
		    || dataBytes > totalBytes - headerBytes)
		{
			throw FormatException("malformed PersistentStore adapter payload lengths");
		}
	}

	template <typename T>
	struct scalar_encoding;

	// Persistent arithmetic support is an explicit wire-format whitelist, not a consequence of
	// std::is_arithmetic_v<T>. The latter also admits bool, character types, implementation-dependent
	// long double representations, and newer extended floating-point types for which R1 assigns no
	// stable scalar code or byte encoding. Supporting a newer C++ language mode does not implicitly
	// add a persistent representation; each additional type requires a format decision, golden byte
	// vectors, and cross-platform representation tests.
#define SECUTILITY_DEFINE_PERSISTENT_SCALAR(type, code) \
	template <> struct scalar_encoding<type> { static constexpr UInt32 Code = code; }
	SECUTILITY_DEFINE_PERSISTENT_SCALAR(std::int8_t, 1);
	SECUTILITY_DEFINE_PERSISTENT_SCALAR(std::uint8_t, 2);
	SECUTILITY_DEFINE_PERSISTENT_SCALAR(std::int16_t, 3);
	SECUTILITY_DEFINE_PERSISTENT_SCALAR(std::uint16_t, 4);
	SECUTILITY_DEFINE_PERSISTENT_SCALAR(std::int32_t, 5);
	SECUTILITY_DEFINE_PERSISTENT_SCALAR(std::uint32_t, 6);
	SECUTILITY_DEFINE_PERSISTENT_SCALAR(std::int64_t, 7);
	SECUTILITY_DEFINE_PERSISTENT_SCALAR(std::uint64_t, 8);
	SECUTILITY_DEFINE_PERSISTENT_SCALAR(float, 9);
	SECUTILITY_DEFINE_PERSISTENT_SCALAR(double, 10);
#undef SECUTILITY_DEFINE_PERSISTENT_SCALAR

	template <typename T, typename = void>
	struct is_persistent_scalar : std::false_type {};

	template <typename T>
	struct is_persistent_scalar<T, std::void_t<decltype(scalar_encoding<T>::Code)>> : std::true_type {};

	template <typename T>
	inline constexpr bool IsPersistentScalar = is_persistent_scalar<T>::value;

	// Encodes one explicitly supported arithmetic value through an unsigned integer holding the
	// same representation bits. This is deliberately unavailable to arbitrary trivially copyable
	// objects: treating a complete object as one integer would mishandle field boundaries, padding,
	// and byte order. Registered native objects instead use descriptor-validated representation
	// copying in Adapter/NativeObject.hpp.
	template <typename T>
	void WriteScalar(const MutableByteView bytes, const std::size_t offset, const T value)
	{
		static_assert(IsPersistentScalar<T>);
		using Bits = std::conditional_t<sizeof(T) == 1, UInt8,
		             std::conditional_t<sizeof(T) == 2, UInt16,
		             std::conditional_t<sizeof(T) == 4, UInt32, UInt64>>>;
		Bits bits{};
		std::memcpy(&bits, &value, sizeof(T));
		FormatCodecDetail::WriteLittleEndian<Bits>(bytes, offset, bits, "scalar data");
	}

	// Decodes the inverse of WriteScalar for a type with an assigned R1 scalar encoding code. Extended
	// arithmetic types such as std::float128_t, when provided by the selected language/library,
	// remain unsupported until the persistent format defines their code, exact representation
	// requirements, and portable byte conversion.
	template <typename T>
	T ReadScalar(const ConstByteView bytes, const std::size_t offset)
	{
		static_assert(IsPersistentScalar<T>);
		using Bits = std::conditional_t<sizeof(T) == 1, UInt8,
		             std::conditional_t<sizeof(T) == 2, UInt16,
		             std::conditional_t<sizeof(T) == 4, UInt32, UInt64>>>;
		const Bits bits = FormatCodecDetail::ReadLittleEndian<Bits>(bytes, offset, "scalar data");
		T value{};
		std::memcpy(&value, &bits, sizeof(T));
		return value;
	}
}
