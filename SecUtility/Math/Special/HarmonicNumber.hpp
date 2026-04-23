// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown
#pragma once

#include <SecUtility/Macro/ForceInline.hpp>
#include <SecUtility/Math/Constant.hpp>
#include <SecUtility/Math/Special/Digamma.hpp>


#if defined(SEC_IF_CONSTEVAL) && __has_include(<gcem.hpp>)
#define SEC_MATH_CONDITIONAL_CONSTEXPR constexpr
#else
#define SEC_MATH_CONDITIONAL_CONSTEXPR
#endif

namespace SecUtility::Math
{
	template <typename T = double, typename Int>
	SEC_MATH_CONDITIONAL_CONSTEXPR SEC_FORCE_INLINE std::enable_if_t<std::is_integral_v<Int>, T>  //
	HarmonicNumber(const Int n) noexcept
	{
		return Digamma(static_cast<T>(n + 1)) + Constant::EulerGamma<T>;
	}
}

#undef SEC_MATH_CONDITIONAL_CONSTEXPR
