// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2026 Andy Brown

#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

#if defined(__has_builtin)
#if __has_builtin(__builtin_add_overflow) && __has_builtin(__builtin_sub_overflow)                                     \
        && __has_builtin(__builtin_mul_overflow)
#define SEC_UTILITY_HAS_CHECKED_ARITHMETIC_BUILTINS true
#endif
#elif defined(__GNUC__)
#define SEC_UTILITY_HAS_CHECKED_ARITHMETIC_BUILTINS true
#endif

namespace SecUtility::Math
{
	namespace Detail
	{
		template <typename T>
		constexpr std::optional<T> CheckedAddFallback(T lhs, T rhs) noexcept
		{
			constexpr auto min = std::numeric_limits<T>::min();
			constexpr auto max = std::numeric_limits<T>::max();

			if constexpr (std::is_unsigned_v<T>)
			{
				if (lhs > static_cast<T>(max - rhs))
				{
					return std::nullopt;
				}
			}
			else if ((rhs > 0 && lhs > static_cast<T>(max - rhs)) || (rhs < 0 && lhs < static_cast<T>(min - rhs)))
			{
				return std::nullopt;
			}

			return static_cast<T>(lhs + rhs);
		}

		template <typename T>
		constexpr std::optional<T> CheckedSubtractFallback(T lhs, T rhs) noexcept
		{
			constexpr auto min = std::numeric_limits<T>::min();
			constexpr auto max = std::numeric_limits<T>::max();

			if constexpr (std::is_unsigned_v<T>)
			{
				if (lhs < rhs)
				{
					return std::nullopt;
				}
			}
			else if ((rhs > 0 && lhs < static_cast<T>(min + rhs)) || (rhs < 0 && lhs > static_cast<T>(max + rhs)))
			{
				return std::nullopt;
			}

			return static_cast<T>(lhs - rhs);
		}

		template <typename T>
		constexpr std::optional<T> CheckedMultiplyFallback(T lhs, T rhs) noexcept
		{
			constexpr auto min = std::numeric_limits<T>::min();
			constexpr auto max = std::numeric_limits<T>::max();

			if constexpr (std::is_unsigned_v<T>)
			{
				if (rhs != 0 && lhs > static_cast<T>(max / rhs))
				{
					return std::nullopt;
				}
			}
			else if (lhs > 0)
			{
				if ((rhs > 0 && lhs > static_cast<T>(max / rhs)) || (rhs < 0 && rhs < static_cast<T>(min / lhs)))
				{
					return std::nullopt;
				}
			}
			else if (lhs < 0)
			{
				if ((rhs > 0 && lhs < static_cast<T>(min / rhs)) || (rhs < 0 && lhs < static_cast<T>(max / rhs)))
				{
					return std::nullopt;
				}
			}

			return static_cast<T>(lhs * rhs);
		}

		template <typename T>
		constexpr std::optional<T> CheckedAddSame(T lhs, T rhs) noexcept
		{
#if defined(SEC_UTILITY_HAS_CHECKED_ARITHMETIC_BUILTINS)
			if constexpr (!std::is_same_v<T, bool>)
			{
				T result{};
				if (__builtin_add_overflow(lhs, rhs, &result))
				{
					return std::nullopt;
				}
				return result;
			}
#endif
			return CheckedAddFallback(lhs, rhs);
		}

		template <typename T>
		constexpr std::optional<T> CheckedSubtractSame(T lhs, T rhs) noexcept
		{
#if defined(SEC_UTILITY_HAS_CHECKED_ARITHMETIC_BUILTINS)
			if constexpr (!std::is_same_v<T, bool>)
			{
				T result{};
				if (__builtin_sub_overflow(lhs, rhs, &result))
				{
					return std::nullopt;
				}
				return result;
			}
#endif
			return CheckedSubtractFallback(lhs, rhs);
		}

		template <typename T>
		constexpr std::optional<T> CheckedMultiplySame(T lhs, T rhs) noexcept
		{
#if defined(SEC_UTILITY_HAS_CHECKED_ARITHMETIC_BUILTINS)
			if constexpr (!std::is_same_v<T, bool>)
			{
				T result{};
				if (__builtin_mul_overflow(lhs, rhs, &result))
				{
					return std::nullopt;
				}
				return result;
			}
#endif
			return CheckedMultiplyFallback(lhs, rhs);
		}
	}

	template <typename T, typename U, std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<U>, int> = 0>
	constexpr std::optional<T> CheckedCast(U value) noexcept
	{
		if constexpr (std::is_signed_v<U>)
		{
			const auto wideValue = static_cast<std::intmax_t>(value);
			if constexpr (std::is_signed_v<T>)
			{
				if (wideValue < static_cast<std::intmax_t>(std::numeric_limits<T>::min())
				    || wideValue > static_cast<std::intmax_t>(std::numeric_limits<T>::max()))
				{
					return std::nullopt;
				}
			}
			else
			{
				if (wideValue < 0
				    || static_cast<std::uintmax_t>(wideValue)
				               > static_cast<std::uintmax_t>(std::numeric_limits<T>::max()))
				{
					return std::nullopt;
				}
			}
		}
		else
		{
			if (const auto wideValue = static_cast<std::uintmax_t>(value);
			    wideValue > static_cast<std::uintmax_t>(std::numeric_limits<T>::max()))
			{
				return std::nullopt;
			}
		}

		return static_cast<T>(value);
	}

	template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
	constexpr std::optional<T> CheckedAdd(T lhs, T rhs) noexcept
	{
		return Detail::CheckedAddSame(lhs, rhs);
	}

	template <typename T,
	          typename Lhs,
	          typename Rhs,
	          std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<Lhs> && std::is_integral_v<Rhs>, int> = 0>
	constexpr std::optional<T> CheckedAdd(Lhs lhs, Rhs rhs) noexcept
	{
		const auto convertedLhs = CheckedCast<T>(lhs);
		const auto convertedRhs = CheckedCast<T>(rhs);
		if (!convertedLhs || !convertedRhs)
		{
			return std::nullopt;
		}
		return Detail::CheckedAddSame(*convertedLhs, *convertedRhs);
	}

	template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
	constexpr std::optional<T> CheckedSubtract(T lhs, T rhs) noexcept
	{
		return Detail::CheckedSubtractSame(lhs, rhs);
	}

	template <typename T,
	          typename Lhs,
	          typename Rhs,
	          std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<Lhs> && std::is_integral_v<Rhs>, int> = 0>
	constexpr std::optional<T> CheckedSubtract(Lhs lhs, Rhs rhs) noexcept
	{
		const auto convertedLhs = CheckedCast<T>(lhs);
		const auto convertedRhs = CheckedCast<T>(rhs);
		if (!convertedLhs || !convertedRhs)
		{
			return std::nullopt;
		}
		return Detail::CheckedSubtractSame(*convertedLhs, *convertedRhs);
	}

	template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
	constexpr std::optional<T> CheckedMultiply(T lhs, T rhs) noexcept
	{
		return Detail::CheckedMultiplySame(lhs, rhs);
	}

	template <typename T,
	          typename Lhs,
	          typename Rhs,
	          std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<Lhs> && std::is_integral_v<Rhs>, int> = 0>
	constexpr std::optional<T> CheckedMultiply(Lhs lhs, Rhs rhs) noexcept
	{
		const auto convertedLhs = CheckedCast<T>(lhs);
		const auto convertedRhs = CheckedCast<T>(rhs);
		if (!convertedLhs || !convertedRhs)
		{
			return std::nullopt;
		}
		return Detail::CheckedMultiplySame(*convertedLhs, *convertedRhs);
	}

	template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
	constexpr std::optional<T> CheckedDivide(T lhs, T rhs) noexcept
	{
		if (rhs == 0)
		{
			return std::nullopt;
		}
		if constexpr (std::is_signed_v<T>)
		{
			if (lhs == std::numeric_limits<T>::min() && rhs == T{-1})
			{
				return std::nullopt;
			}
		}
		return static_cast<T>(lhs / rhs);
	}

	template <typename T,
	          typename Lhs,
	          typename Rhs,
	          std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<Lhs> && std::is_integral_v<Rhs>, int> = 0>
	constexpr std::optional<T> CheckedDivide(Lhs lhs, Rhs rhs) noexcept
	{
		const auto convertedLhs = CheckedCast<T>(lhs);
		const auto convertedRhs = CheckedCast<T>(rhs);
		if (!convertedLhs || !convertedRhs)
		{
			return std::nullopt;
		}
		return CheckedDivide(*convertedLhs, *convertedRhs);
	}

	template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
	constexpr std::optional<T> CheckedNegate(T value) noexcept
	{
		if constexpr (std::is_unsigned_v<T>)
		{
			if (value != 0)
			{
				return std::nullopt;
			}
			return T{0};
		}
		else
		{
			if (value == std::numeric_limits<T>::min())
			{
				return std::nullopt;
			}
			return static_cast<T>(-value);
		}
	}
}

#undef SEC_UTILITY_HAS_CHECKED_ARITHMETIC_BUILTINS
