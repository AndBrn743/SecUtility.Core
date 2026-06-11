// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once


namespace SecUtility
{
	template <typename T, T V>
	struct Constant
	{
		static constexpr T Value = V;
		using ValueType = T;
		using Type = Constant;

		/* IMPLICIT */ constexpr operator ValueType() const noexcept
		{
			return Value;
		}

		constexpr ValueType operator()() const noexcept
		{
			return Value;
		}
	};
}
