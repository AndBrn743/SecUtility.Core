// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <cinttypes>
#include <complex>
#include <type_traits>
#include <SecUtility/Raw/Int.hpp>
#if __has_include(<stdfloat>)
#include <stdfloat>
#endif

namespace SecUtility::Math
{
	namespace Detail::IsScalar
	{
		template <typename T>
		struct is_scalar
		{
			static constexpr bool value = std::is_same_v<T, char>                   //
			                              || std::is_same_v<T, signed char>         //
			                              || std::is_same_v<T, short>               //
			                              || std::is_same_v<T, int>                 //
			                              || std::is_same_v<T, long>                //
			                              || std::is_same_v<T, long long>           //
			                              || std::is_same_v<T, unsigned char>       //
			                              || std::is_same_v<T, unsigned short>      //
			                              || std::is_same_v<T, unsigned int>        //
			                              || std::is_same_v<T, unsigned long>       //
			                              || std::is_same_v<T, unsigned long long>  //
			                              || std::is_same_v<T, std::size_t>         //
			                              || std::is_same_v<T, std::ptrdiff_t>      //
			                              || std::is_same_v<T, Int8>         //
			                              || std::is_same_v<T, Int16>        //
			                              || std::is_same_v<T, Int32>        //
			                              || std::is_same_v<T, Int64>        //
			                              || std::is_same_v<T, UInt8>        //
			                              || std::is_same_v<T, UInt16>       //
			                              || std::is_same_v<T, UInt32>       //
			                              || std::is_same_v<T, UInt64>       //
			                              || std::is_same_v<T, float>               //
			                              || std::is_same_v<T, double>              //
			                              || std::is_same_v<T, long double>         //
#if __STDCPP_FLOAT16_T__
			                              || std::is_same_v<T, std::float16_t>
#endif
#if __STDCPP_FLOAT32_T__
			                              || std::is_same_v<T, std::float32_t>
#endif
#if __STDCPP_FLOAT64_T__
			                              || std::is_same_v<T, std::float64_t>
#endif
#if __STDCPP_FLOAT128_T__
			                              || std::is_same_v<T, std::float128_t>
#endif
#if __STDCPP_BFLOAT16_T__
			                              || std::is_same_v<T, std::bfloat16_t>
#endif
			        ;
		};

		template <typename T>
		struct is_scalar<std::complex<T>> : std::true_type
		{
			/* NO CODE */
		};
	}

	template <typename T>
	inline constexpr bool IsScalar = Detail::IsScalar::is_scalar<T>::value;
}
