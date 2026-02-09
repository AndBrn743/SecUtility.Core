// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <random>
#include <string>

#include <SecUtility/Meta/TypeTrait.hpp>
#include <SecUtility/Raw/Int.hpp>
#include <SecUtility/Raw/Float.hpp>


namespace SecUtility
{
	/// <summary>
	/// Convenience random number generator for development, testing, and prototyping.
	///
	/// This class provides a quick and easy way to generate random values when:
	/// - Writing unit tests that need random data
	/// - Prototyping or experimenting with algorithms
	/// - Generating placeholder/fake data during development
	///
	/// <b>Intentional limitations:</b>
	/// - Uses std::mt19937_64 (non-cryptographic, but fast and good quality)
	/// - Singleton pattern - not suitable for scenarios requiring independent generators
	/// - Not thread-safe for concurrent SetSeed() calls
	///
	/// For cryptographic security, advanced distribution customization, or
	/// independent generator state, please use STL directly.
	/// </summary>
	/* Singleton */ class Random
	{
	public:
		Random(const Random&) = delete;
		Random& operator=(const Random&) = delete;
		Random(Random&&) = delete;
		Random& operator=(Random&&) = delete;

		static Random& Get()
		{
			static Random Instance;
			return Instance;
		}

		// not thread-safe
		static void SetSeed(const unsigned int seed)
		{
			Get().m_Generator.seed(seed);
		}

#define SECSCF_RANDOM_INTERNAL_DefineNextFunctionForType(TYPE)                                                         \
	static TYPE Next##TYPE(const TYPE min = 0, const TYPE max = std::numeric_limits<TYPE>::max())                      \
	{                                                                                                                  \
		static_assert(std::is_integral<TYPE>::value, "This macro is for integer type only");                           \
		std::uniform_int_distribution<TYPE> distribution(min, max);                                                    \
		return distribution(Get().m_Generator);                                                                        \
	}

		// SECSCF_RANDOM_INTERNAL_DefineNextFunctionForType(Int8);
		SECSCF_RANDOM_INTERNAL_DefineNextFunctionForType(Int16);
		SECSCF_RANDOM_INTERNAL_DefineNextFunctionForType(Int32);
		SECSCF_RANDOM_INTERNAL_DefineNextFunctionForType(Int64);
		// SECSCF_RANDOM_INTERNAL_DefineNextFunctionForType(UInt8);
		SECSCF_RANDOM_INTERNAL_DefineNextFunctionForType(UInt16);
		SECSCF_RANDOM_INTERNAL_DefineNextFunctionForType(UInt32);
		SECSCF_RANDOM_INTERNAL_DefineNextFunctionForType(UInt64);

#undef SECSCF_RANDOM_INTERNAL_DefineNextFunctionForType

#define SECSCF_RANDOM_INTERNAL_DefineNextFunctionForType(TYPE)                                                         \
	static TYPE Next##TYPE(const TYPE min = 0, const TYPE max = 1)                                                     \
	{                                                                                                                  \
		static_assert(std::is_floating_point<TYPE>::value, "This macro is for floating point type only");              \
		std::uniform_real_distribution<TYPE> distribution(min, max);                                                   \
		return distribution(Get().m_Generator);                                                                        \
	}

		SECSCF_RANDOM_INTERNAL_DefineNextFunctionForType(Double);
		SECSCF_RANDOM_INTERNAL_DefineNextFunctionForType(Single);

#undef SECSCF_RANDOM_INTERNAL_DefineNextFunctionForType

		static int Next(const int min = 0, const int max = std::numeric_limits<int>::max())
		{
			std::uniform_int_distribution distribution(min, max);
			return distribution(Get().m_Generator);
		}

		static std::string NextString(
		        const Int32 length = 16,
		        const std::string& chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890")
		{
			std::string result;
			result.resize(length);

			for (Int32 i = 0; i < length; i++)
			{
				result[i] = chars[Next() % chars.length()];
			}

			return result;
		}


	private:
		Random() : m_Generator(std::random_device{}())
		{
			/* NO CODE */
		}

		std::mt19937_64 m_Generator;
	};
}
