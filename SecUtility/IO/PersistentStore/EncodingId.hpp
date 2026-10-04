// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/Diagnostic/Exception.hpp>
#include <SecUtility/Raw/Int.hpp>

#include <array>
#include <cstddef>


namespace SecUtility::IO
{
	class PersistentEncodingId final
	{
	public:
		constexpr PersistentEncodingId() noexcept = default;
		explicit constexpr PersistentEncodingId(const std::array<Byte, 16>& bytes) noexcept : m_Bytes(bytes)
		{
		}

		constexpr const Byte* data() const noexcept
		{
			return m_Bytes.data();
		}
		static constexpr std::size_t size() noexcept
		{
			return 16;
		}

		friend constexpr bool operator==(const PersistentEncodingId& left, const PersistentEncodingId& right) noexcept
		{
			for (std::size_t index = 0; index < size(); ++index)
			{
				if (left.m_Bytes[index] != right.m_Bytes[index])
				{
					return false;
				}
			}
			return true;
		}

		friend constexpr bool operator!=(const PersistentEncodingId& left, const PersistentEncodingId& right) noexcept
		{
			return !(left == right);
		}

	private:
		std::array<Byte, 16> m_Bytes{};
	};

	namespace PersistentEncodingIdDetail
	{
		constexpr int DecodeHexDigit(const char character) noexcept
		{
			return character >= '0' && character <= '9'   ? character - '0'
			       : character >= 'a' && character <= 'f' ? character - 'a' + 10
			       : character >= 'A' && character <= 'F' ? character - 'A' + 10
			                                              : -1;
		}
	}

	// Parses canonical UUID text into its 16 left-to-right bytes. The result never depends on the
	// mixed-endian field layout of a platform GUID/UUID structure. Invalid text fails constant
	// evaluation when this function is used to initialize a constexpr encoding identifier.
	template <std::size_t Size>
	constexpr PersistentEncodingId MakePersistentEncodingId(const char (&uuid)[Size])
	{
		static_assert(Size == 37, "Persistent encoding UUID must contain 36 characters");
		if (uuid[8] != '-' || uuid[13] != '-' || uuid[18] != '-' || uuid[23] != '-' || uuid[36] != '\0')
		{
			throw FormatException("Persistent encoding UUID is not in canonical form");
		}

		std::array<Byte, 16> bytes{};
		std::size_t textIndex = 0;
		for (std::size_t byteIndex = 0; byteIndex < bytes.size(); ++byteIndex)
		{
			if (textIndex == 8 || textIndex == 13 || textIndex == 18 || textIndex == 23)
			{
				++textIndex;
			}
			const int high = PersistentEncodingIdDetail::DecodeHexDigit(uuid[textIndex]);
			const int low = PersistentEncodingIdDetail::DecodeHexDigit(uuid[textIndex + 1]);
			if (high < 0 || low < 0)
			{
				throw FormatException("Persistent encoding UUID contains a non-hexadecimal digit");
			}
			bytes[byteIndex] = Byte{static_cast<unsigned char>((high << 4) | low)};
			textIndex += 2;
		}
		return PersistentEncodingId(bytes);
	}
}
