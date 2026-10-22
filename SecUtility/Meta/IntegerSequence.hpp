// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>


namespace SecUtility
{
	namespace Detail::IntegerSequence
	{
		template <typename T>
		constexpr std::uintmax_t Magnitude(T value) noexcept
		{
			// Unsigned subtraction also handles the minimum signed value.
			return value < 0 ? -static_cast<std::uintmax_t>(value) : static_cast<std::uintmax_t>(value);
		}

		template <typename T, typename Step>
		constexpr std::uintmax_t RangeSize(T start, T end, Step step) noexcept
		{
			const auto distance = step < 0 ? static_cast<std::uintmax_t>(start) - static_cast<std::uintmax_t>(end)
			                               : static_cast<std::uintmax_t>(end) - static_cast<std::uintmax_t>(start);
			const auto magnitude = Magnitude(step);
			return distance / magnitude + (distance % magnitude != 0 ? 1 : 0);
		}

		template <typename T, T Start, T End, auto Step>
		struct MakeIntegerRangeHelper
		{
			static_assert(std::is_integral_v<T>, "Index type must be integral");
			static_assert(std::is_integral_v<decltype(Step)>, "Step must be integral");
			static_assert(Step != 0, "Step must be nonzero");
			static_assert(Step < 0 ? Start >= End : Start <= End, "Step must agree with the range direction");

			static constexpr auto Count = RangeSize(Start, End, Step);

			static constexpr T Value(const std::size_t i) noexcept
			{
				// Each offset is smaller than the range distance, so multiplication fits.
				const auto offset = Magnitude(Step) * i;
				const auto value = Step < 0 ? static_cast<std::uintmax_t>(Start) - offset
				                            : static_cast<std::uintmax_t>(Start) + offset;
				if constexpr (std::is_signed_v<T>)
				{
					if (value > static_cast<std::uintmax_t>(std::numeric_limits<T>::max()))
					{
						// Recover a negative result without an out-of-range unsigned-to-signed cast.
						return static_cast<T>(-1 - static_cast<T>(~value));
					}
				}
				return static_cast<T>(value);
			}

			template <std::size_t... Is>
			static auto Make(std::index_sequence<Is...>) -> std::integer_sequence<T, Value(Is)...>;

			using Type = decltype(Make(std::make_index_sequence<Count>{}));
		};

		template <typename T, auto... Args>
		struct MakeIntegerSequenceHelper
		{
			static_assert(sizeof...(Args) >= 1 && sizeof...(Args) <= 3,
			              "Expected End; Start, End; or Start, End, Step");
		};

		template <typename T, auto Start, auto End, auto Step>
		struct MakeIntegerSequenceHelper<T, Start, End, Step> : MakeIntegerRangeHelper<T, Start, End, Step>
		{
			/* NO CODE */
		};

		template <typename T, auto Start, auto End>
		struct MakeIntegerSequenceHelper<T, Start, End> : MakeIntegerRangeHelper<T, Start, End, 1>
		{
			/* NO CODE */
		};

		template <typename T, auto End>
		struct MakeIntegerSequenceHelper<T, End> : MakeIntegerRangeHelper<T, 0, End, 1>
		{
			/* NO CODE */
		};

		template <typename Sequence>
		struct MakeReversedIntegerSequenceHelper;

		template <typename Int, auto... Is>
		struct MakeReversedIntegerSequenceHelper<std::integer_sequence<Int, Is...>>
		{
			using Type = std::integer_sequence<Int, static_cast<Int>(sizeof...(Is)) - Int{1} - Is...>;
		};
	}

	// End; Start, End; or Start, End, Step. End is exclusive and Step defaults to +1.
	// Bounds must fit T; a nonzero integral step must agree with the range direction.
	// Invalid arguments produce hard compilation errors.
	template <typename T, auto... Args>
	using MakeIntegerSequence = typename Detail::IntegerSequence::MakeIntegerSequenceHelper<T, Args...>::Type;

	template <auto... Args>
	using MakeIndexSequence = MakeIntegerSequence<std::size_t, Args...>;

	template <typename T, std::size_t N>
	using MakeReversedIntegerSequence =
	        typename Detail::IntegerSequence::MakeReversedIntegerSequenceHelper<std::make_integer_sequence<T, N>>::Type;

	template <std::size_t N>
	using MakeReversedIndexSequence = MakeReversedIntegerSequence<std::size_t, N>;
}
