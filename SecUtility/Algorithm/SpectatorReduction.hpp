// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Andy Brown

#pragma once

#include <functional>
#include <iterator>


namespace SecUtility::Algorithm
{
	template <typename BiDirectionIterator,
	          typename BiDirectionOutputIterator,
	          typename BinaryOp,
	          typename InputProjector,
	          typename OutputProjector>
	constexpr void SpectatorReduction(const BiDirectionIterator first,
	                                  const BiDirectionIterator last,
	                                  const BiDirectionOutputIterator out,
	                                  BinaryOp binaryOp,
	                                  InputProjector inputProjector,
	                                  OutputProjector outputProjector)
	{
		const auto n = std::distance(first, last);
		if (n < 2)
		{
			return;
		}

		{
			auto inputHead = first;
			auto outputHead = out;
			std::invoke(outputProjector, *++outputHead) = std::invoke(inputProjector, *inputHead);

			const auto back = std::prev(last);

			while (++inputHead != back)
			{
				const auto oldOutputHead = outputHead;
				std::invoke(outputProjector, *++outputHead) = std::invoke(binaryOp,
				                                                          std::invoke(outputProjector, *oldOutputHead),
				                                                          std::invoke(inputProjector, *inputHead));
			}
		}

		{
			auto input = last;
			auto suffixReduction = std::invoke(inputProjector, *--input);

			for (auto output = std::next(out, n - 2); output != out; --output)
			{
				std::invoke(outputProjector, *output) =
				        std::invoke(binaryOp, std::invoke(outputProjector, *output), suffixReduction);
				suffixReduction = std::invoke(binaryOp, std::invoke(inputProjector, *--input), suffixReduction);
			}

			std::invoke(outputProjector, *out) = suffixReduction;
		}
	}

	template <typename BiDirectionIterator,
	          typename BiDirectionOutputIterator,
	          typename BinaryOp,
	          typename InputProjector>
	constexpr void SpectatorReduction(const BiDirectionIterator first,
	                                  const BiDirectionIterator last,
	                                  const BiDirectionOutputIterator out,
	                                  BinaryOp binaryOp,
	                                  InputProjector inputProjector)
	{
		SpectatorReduction(first,
		                   last,
		                   out,
		                   std::move(binaryOp),
		                   std::move(inputProjector),
		                   [](auto&& o) -> decltype(auto) { return std::forward<decltype(o)>(o); });
	}

	template <typename BiDirectionIterator, typename BiDirectionOutputIterator, typename BinaryOp>
	constexpr void SpectatorReduction(const BiDirectionIterator first,
	                                  const BiDirectionIterator last,
	                                  const BiDirectionOutputIterator out,
	                                  BinaryOp binaryOp)
	{
		SpectatorReduction(
		        first,
		        last,
		        out,
		        std::move(binaryOp),
		        [](auto&& o) -> decltype(auto) { return std::forward<decltype(o)>(o); },
		        [](auto&& o) -> decltype(auto) { return std::forward<decltype(o)>(o); });
	}
}
