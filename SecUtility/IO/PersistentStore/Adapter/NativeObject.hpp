// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/IO/PersistentStore.hpp>
#include <SecUtility/IO/PersistentStore/Detail/Adapter.hpp>
#include <SecUtility/Misc/Endian.hpp>

#include <array>
#include <cstring>
#include <limits>
#include <type_traits>


namespace SecUtility::IO
{
	// A downstream specialization must assign one UUID permanently to the encoding family, normally
	// with MakePersistentEncodingId("xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"), and increment its
	// SchemaVersion whenever the persisted object representation changes. Generate the UUID once;
	// never derive it from a compiler type name or generate it at build time or runtime.
	//
	// Specializing this descriptor explicitly opts into persisting the complete native object
	// representation. That representation is ABI-specific and includes padding bytes. The caller must
	// ensure every representation byte is initialized before insertion; otherwise padding can make files
	// nondeterministic or disclose unrelated memory. Types that cannot meet that contract must use a
	// field-wise adapter instead of NativeObjectDescriptor.
	template <typename T>
	struct NativeObjectDescriptor;

	namespace PersistentStoreDetail
	{
		inline constexpr auto NativeScalarEncodingId =
		        MakePersistentEncodingId("3a8d7c52-8f30-5d5d-8b44-2b28ac88d8c1");
		inline constexpr auto NativeObjectEncodingId =
		        MakePersistentEncodingId("16e3d2b6-e4d9-58eb-99a8-d8d8662b8a64");

		template <typename T, typename = void>
		struct has_native_object_descriptor : std::false_type {};

		template <typename T>
		struct has_native_object_descriptor<T,
		                                    std::void_t<decltype(NativeObjectDescriptor<T>::EncodingId),
		                                                decltype(NativeObjectDescriptor<T>::SchemaVersion)>>
		    : std::true_type {};

		template <typename T>
		inline constexpr bool HasNativeObjectDescriptor = has_native_object_descriptor<T>::value;
	}

	template <typename T>
	struct PersistentTraits<T, std::enable_if_t<PersistentStoreDetail::IsPersistentScalar<T>>>
	{
		static_assert(!std::is_floating_point_v<T> || std::numeric_limits<T>::is_iec559,
		              "PersistentStore floating scalars require IEC 60559 representation");

		static PersistentStoreDetail::PayloadLayout Measure(const T&)
		{
			return {48 + sizeof(T), alignof(T)};
		}

		static void Encode(const T value, const PersistentStoreDetail::MutableByteView bytes)
		{
			using namespace PersistentStoreDetail;
			WriteAdapterHeader(bytes, NativeScalarEncodingId, 48, bytes.size(), sizeof(T));
			FormatCodecDetail::WriteLittleEndian<UInt32>(
			        bytes, 40, scalar_encoding<T>::Code, "native scalar code");
			FormatCodecDetail::WriteLittleEndian<UInt32>(bytes, 44, 0, "native scalar reserved");
			WriteScalar<T>(bytes, 48, value);
		}

		static T Decode(const PersistentStoreDetail::ConstByteView bytes)
		{
			using namespace PersistentStoreDetail;
			ValidateAdapterHeader(bytes, NativeScalarEncodingId, 48, sizeof(T));
			if (FormatCodecDetail::ReadLittleEndian<UInt32>(bytes, 40, "native scalar code")
			            != scalar_encoding<T>::Code
			    || FormatCodecDetail::ReadLittleEndian<UInt32>(bytes, 44, "native scalar reserved") != 0)
			{
				throw FormatException("PersistentStore native scalar type mismatch");
			}
			return ReadScalar<T>(bytes, 48);
		}
	};

	template <typename T>
	struct PersistentTraits<T,
	                        std::enable_if_t<PersistentStoreDetail::HasNativeObjectDescriptor<T>
	                                         && !PersistentStoreDetail::IsPersistentScalar<T>>>
	{
		static_assert(std::is_class_v<T> || std::is_union_v<T>);
		static_assert(std::is_trivially_copyable_v<T>);
		static_assert(std::is_nothrow_default_constructible_v<T>);
		static_assert(!std::is_pointer_v<T> && !std::is_member_pointer_v<T> && !std::is_volatile_v<T>);

		static PersistentStoreDetail::PayloadLayout Measure(const T&)
		{
			return {88 + sizeof(T), alignof(T)};
		}

		static void Encode(const T& value, const PersistentStoreDetail::MutableByteView bytes)
		{
			using namespace PersistentStoreDetail;
			static_assert(std::is_same_v<std::remove_cv_t<decltype(NativeObjectDescriptor<T>::EncodingId)>,
			                             PersistentEncodingId>);
			WriteAdapterHeader(bytes, NativeObjectEncodingId, 88, bytes.size(), sizeof(T));
			std::copy(NativeObjectDescriptor<T>::EncodingId.data(),
			          NativeObjectDescriptor<T>::EncodingId.data() + NativeObjectDescriptor<T>::EncodingId.size(),
			          bytes.data() + 40);
			FormatCodecDetail::WriteLittleEndian<UInt32>(
			        bytes, 56, NativeObjectDescriptor<T>::SchemaVersion, "native object schema version");
			FormatCodecDetail::WriteLittleEndian<UInt32>(bytes, 60, sizeof(T), "native object size");
			FormatCodecDetail::WriteLittleEndian<UInt32>(bytes, 64, alignof(T), "native object alignment");
			FormatCodecDetail::WriteLittleEndian<UInt32>(bytes, 68, ByteOrderMarker, "native object byte order");
			FormatCodecDetail::WriteLittleEndian<UInt64>(bytes, 72, sizeof(T), "native object representation bytes");
			FormatCodecDetail::WriteLittleEndian<UInt64>(bytes, 80, 0, "native object reserved");
			// This copies padding deliberately under NativeObjectDescriptor's explicit representation contract.
			std::memcpy(bytes.data() + 88, &value, sizeof(T));
		}

		static T Decode(const PersistentStoreDetail::ConstByteView bytes)
		{
			using namespace PersistentStoreDetail;
			ValidateAdapterHeader(bytes, NativeObjectEncodingId, 88, sizeof(T));
			if (!std::equal(NativeObjectDescriptor<T>::EncodingId.data(),
			                NativeObjectDescriptor<T>::EncodingId.data() + NativeObjectDescriptor<T>::EncodingId.size(),
			                bytes.data() + 40)
			    || FormatCodecDetail::ReadLittleEndian<UInt32>(bytes, 56, "native object schema version")
			               != NativeObjectDescriptor<T>::SchemaVersion
			    || FormatCodecDetail::ReadLittleEndian<UInt32>(bytes, 60, "native object size") != sizeof(T)
			    || FormatCodecDetail::ReadLittleEndian<UInt32>(bytes, 64, "native object alignment") != alignof(T)
			    || FormatCodecDetail::ReadLittleEndian<UInt32>(bytes, 68, "native object byte order") != ByteOrderMarker
			    || FormatCodecDetail::ReadLittleEndian<UInt64>(bytes, 72, "native object representation bytes")
			               != sizeof(T)
			    || FormatCodecDetail::ReadLittleEndian<UInt64>(bytes, 80, "native object reserved") != 0)
			{
				throw FormatException("PersistentStore native object schema or representation mismatch");
			}
			T value{};
			std::memcpy(&value, bytes.data() + 88, sizeof(T));
			return value;
		}
	};
}
