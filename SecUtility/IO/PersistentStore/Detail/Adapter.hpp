// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/IO/PersistentStore/Detail/FormatCodec.hpp>
#include <SecUtility/IO/PersistentStore/EncodingId.hpp>
#include <SecUtility/Raw/Int.hpp>

#include <algorithm>
#include <climits>
#include <complex>
#include <cstddef>
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

	template <typename T, typename = void>
	struct scalar_encoding;

	template <typename T>
	inline constexpr bool HasFixedWidthIntegerRepresentation =
	        std::is_integral_v<T> && !std::is_same_v<T, bool> && !std::is_same_v<T, char> && !std::is_same_v<T, wchar_t>
	        && !std::is_same_v<T, char16_t> && !std::is_same_v<T, char32_t>
	        && (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8)
	        && std::numeric_limits<T>::digits == static_cast<int>(sizeof(T) * CHAR_BIT - (std::is_signed_v<T> ? 1 : 0))
	        && (!std::is_signed_v<T> || std::numeric_limits<T>::lowest() == -std::numeric_limits<T>::max() - 1);

	template <typename T>
	struct scalar_encoding<T, std::enable_if_t<HasFixedWidthIntegerRepresentation<T>>>
	{
		static constexpr UInt32 Code = sizeof(T) == 1   ? (std::is_signed_v<T> ? 1 : 2)
		                               : sizeof(T) == 2 ? (std::is_signed_v<T> ? 3 : 4)
		                               : sizeof(T) == 4 ? (std::is_signed_v<T> ? 5 : 6)
		                                                : (std::is_signed_v<T> ? 7 : 8);
	};

	// Scalar codes are shared by structured adapters. Fundamental integers with an exact supported
	// representation map to the corresponding fixed-width wire code, allowing Eigen matrices of short,
	// int, long, and long long when the local representation qualifies. Native-object and array support
	// remains the narrower explicit fixed-width/float/double whitelist below. Supporting a newer C++
	// language mode does not add a representation without a format decision and cross-platform tests.
#define SECUTILITY_DEFINE_PERSISTENT_SCALAR(type, code)                                                                \
	template <>                                                                                                        \
	struct scalar_encoding<type>                                                                                       \
	{                                                                                                                  \
		static constexpr UInt32 Code = code;                                                                           \
	}
	SECUTILITY_DEFINE_PERSISTENT_SCALAR(float, 9);
	SECUTILITY_DEFINE_PERSISTENT_SCALAR(double, 10);
#undef SECUTILITY_DEFINE_PERSISTENT_SCALAR
	template <>
	struct scalar_encoding<std::complex<float>>
	{
		static constexpr UInt32 Code = 11;
	};
	template <>
	struct scalar_encoding<std::complex<double>>
	{
		static constexpr UInt32 Code = 12;
	};

	template <typename T, typename = void>
	struct has_scalar_encoding : std::false_type
	{
	};

	template <typename T>
	struct has_scalar_encoding<T, std::void_t<decltype(scalar_encoding<T>::Code)>> : std::true_type
	{
	};

	template <typename T>
	inline constexpr bool HasScalarEncoding = has_scalar_encoding<T>::value;

	template <typename T>
	struct is_persistent_scalar
	    : std::bool_constant<std::is_same_v<T, Int8> || std::is_same_v<T, UInt8> || std::is_same_v<T, Int16>
	                         || std::is_same_v<T, UInt16> || std::is_same_v<T, Int32> || std::is_same_v<T, UInt32>
	                         || std::is_same_v<T, Int64> || std::is_same_v<T, UInt64> || std::is_same_v<T, float>
	                         || std::is_same_v<T, double>>
	{
	};

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
		using Bits = std::conditional_t<
		        sizeof(T) == 1,
		        UInt8,
		        std::conditional_t<sizeof(T) == 2, UInt16, std::conditional_t<sizeof(T) == 4, UInt32, UInt64>>>;
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
		using Bits = std::conditional_t<
		        sizeof(T) == 1,
		        UInt8,
		        std::conditional_t<sizeof(T) == 2, UInt16, std::conditional_t<sizeof(T) == 4, UInt32, UInt64>>>;
		const Bits bits = FormatCodecDetail::ReadLittleEndian<Bits>(bytes, offset, "scalar data");
		T value{};
		std::memcpy(&value, &bits, sizeof(T));
		return value;
	}
}
