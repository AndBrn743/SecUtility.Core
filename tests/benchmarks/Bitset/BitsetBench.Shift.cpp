// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#include "BitsetBench.Common.hpp"

namespace
{
	constexpr std::size_t RuntimeShift = std::numeric_limits<std::size_t>::max();

	template <typename T, std::size_t N, bool Left, std::size_t Amount = RuntimeShift>
	void BM_CopyShift(benchmark::State& state)
	{
		auto original = RandomBitset<T, N>();
		const auto shift = [&]
		{
			if constexpr (Amount == RuntimeShift)
			{
				return static_cast<std::size_t>(state.range(0));
			}
			else
			{
				return Amount;
			}
		}();
		for (auto _ : state)
		{
			benchmark::DoNotOptimize(original);
			auto copy = original;
			Shift<Left>(copy, shift);
			benchmark::DoNotOptimize(copy);
		}
	}

	template <typename T, std::size_t N, bool Left>
	void BM_ShiftOnly(benchmark::State& state)
	{
		auto original = RandomBitset<T, N>();
		std::vector<T> work(BatchSize, original);
		const auto shift = static_cast<std::size_t>(state.range(0));
		for (auto _ : state)
		{
			ResetBatch(state, work, original);
			for (auto& bs : work)
			{
				Shift<Left>(bs, shift);
				benchmark::DoNotOptimize(bs);
			}
		}
		state.SetItemsProcessed(state.iterations() * BatchSize);
	}

	template <typename T, std::size_t N>
	void RegisterOwner(const std::string& type)
	{
		const auto suffix = "/" + type + "/" + std::to_string(N);
		AddShifts(benchmark::RegisterBenchmark("ShiftLeft/Copy" + suffix, &BM_CopyShift<T, N, true>), N);
		AddShifts(benchmark::RegisterBenchmark("ShiftRight/Copy" + suffix, &BM_CopyShift<T, N, false>), N);
		AddShifts(benchmark::RegisterBenchmark("ShiftLeft/Batch32" + suffix, &BM_ShiftOnly<T, N, true>), N);
		AddShifts(benchmark::RegisterBenchmark("ShiftRight/Batch32" + suffix, &BM_ShiftOnly<T, N, false>), N);
		if constexpr (N == 64 || N == 65 || N == 129 || N == 5409)
		{
			benchmark::RegisterBenchmark("ShiftLeft/Constant7" + suffix, &BM_CopyShift<T, N, true, 7>);
			benchmark::RegisterBenchmark("ShiftRight/Constant7" + suffix, &BM_CopyShift<T, N, false, 7>);
			benchmark::RegisterBenchmark("ShiftLeft/Constant64" + suffix, &BM_CopyShift<T, N, true, 64>);
			benchmark::RegisterBenchmark("ShiftRight/Constant64" + suffix, &BM_CopyShift<T, N, false, 64>);
		}
	}

	template <std::size_t N>
	void RegisterSize()
	{
		RegisterOwner<Bitset<N>, N>("Static");
		RegisterOwner<DynamicBitset, N>("Dynamic");
		RegisterOwner<std::bitset<N>, N>("Std");
	}

	template <std::size_t... Sizes>
	void RegisterSizes(std::index_sequence<Sizes...>)
	{
		(RegisterSize<Sizes>(), ...);
	}
}

void BitsetBenchmark::RegisterShifts()
{
	RegisterSizes(OwnerSizes{});
}
