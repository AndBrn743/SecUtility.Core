// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#include "BitsetBench.Common.hpp"

namespace
{
	template <typename T, std::size_t N>
	void BM_Equal(benchmark::State& state, const int mismatch)
	{
		auto lhs = RandomBitset<T, N>();
		auto rhs = lhs;
		if constexpr (N != 0)
		{
			if (mismatch != 0)
			{
				const std::size_t index = mismatch == 1 ? 0 : N - 1;
				rhs[index] = !static_cast<bool>(rhs[index]);
			}
		}
		for (auto _ : state)
		{
			benchmark::DoNotOptimize(lhs);
			benchmark::DoNotOptimize(rhs);
			auto equal = lhs == rhs;
			benchmark::DoNotOptimize(equal);
		}
	}

	template <typename T, std::size_t N>
	void RegisterOwner(const std::string& type)
	{
		const auto suffix = "/" + type + "/" + std::to_string(N);
		benchmark::RegisterBenchmark("Equal" + suffix + "/Same", &BM_Equal<T, N>, 0);
		if constexpr (N != 0)
		{
			benchmark::RegisterBenchmark("Equal" + suffix + "/FirstMismatch", &BM_Equal<T, N>, 1);
			benchmark::RegisterBenchmark("Equal" + suffix + "/LastMismatch", &BM_Equal<T, N>, 2);
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

void BitsetBenchmark::RegisterEquality()
{
	RegisterSizes(OperationSizes{});
}
