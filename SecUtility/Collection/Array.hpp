//
// Created by Andy on 6/12/2026.
//

#pragma once

#include <SecUtility/Meta/TypeTrait.hpp>
#include <array>

namespace SecUtility
{
	template <typename... StdArrays>
	constexpr std::enable_if_t<(is_std_array<std::remove_cvref_t<StdArrays>>::value && ...),
	                           std::array<std::common_type_t<typename std::remove_cvref_t<StdArrays>::value_type...>,
	                                      (std::tuple_size_v<std::remove_cvref_t<StdArrays>> + ...)>>
	Concat(StdArrays&&... arrays)
	{
		using value_type = std::common_type_t<typename std::remove_cvref_t<StdArrays>::value_type...>;
		static_assert((std::is_same_v<value_type, typename std::remove_cvref_t<StdArrays>::value_type> && ...));

		std::array<value_type, (std::tuple_size_v<std::remove_cvref_t<StdArrays>> + ...)> result{};

		auto copyOrMoveToResult = [&result, offset = std::size_t{0}](auto&& source) mutable
		{
			using Source = decltype(source);
			for (std::size_t i = 0; i < std::tuple_size_v<std::remove_cvref_t<Source>>; ++i, ++offset)
			{
				if constexpr (std::is_lvalue_reference_v<Source>)
				{
					result[offset] = source[i];
				}
				else
				{
					result[offset] = std::move(source[i]);
				}
			}
		};

		(copyOrMoveToResult(std::forward<StdArrays>(arrays)), ...);

		return result;
	}
}