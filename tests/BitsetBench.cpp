// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

// Build and run:
//   cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release -DSECUTILITY_BUILD_BENCHMARKS=ON
//   cmake --build build-bench --target BitsetBench
//   ./build-bench/tests/BitsetBench --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
//
// Shift cases measure copying plus shifting. DynamicBitset copies include allocation.
// Setup and random input generation are outside the timed loops.

#include <SecUtility/Collection/Bitset.hpp>
#include <benchmark/benchmark.h>

#include <bitset>
#include <cstddef>
#include <cstdint>
#include <random>
#include <type_traits>

using namespace SecUtility;

namespace
{
	template <typename BitsetType, std::size_t N>
	BitsetType RandomBitset()
	{
		auto bs = []
		{
			if constexpr (std::is_same_v<BitsetType, DynamicBitset>)
			{
				return BitsetType(N);
			}
			else
			{
				return BitsetType{};
			}
		}();
		std::mt19937_64 rng(42);
		for (std::size_t i = 0; i < N; ++i)
		{
			bs[i] = static_cast<bool>(rng() & 1u);
		}
		return bs;
	}

	template <typename BitsetType, std::size_t N, bool Left>
	void BM_CopyShift(benchmark::State& state)
	{
		auto bs = RandomBitset<BitsetType, N>();
		const auto shift = static_cast<std::size_t>(state.range(0));
		for (auto _ : state)
		{
			benchmark::DoNotOptimize(bs);
			auto copy = bs;
			if constexpr (Left)
			{
				copy <<= shift;
			}
			else
			{
				copy >>= shift;
			}
			benchmark::DoNotOptimize(copy);
		}
	}

	template <std::size_t N, std::size_t Start>
	void BM_CopyShiftLeftSegment(benchmark::State& state)
	{
		auto bs = RandomBitset<DynamicBitset, N>();
		const auto shift = static_cast<std::size_t>(state.range(0));
		for (auto _ : state)
		{
			benchmark::DoNotOptimize(bs);
			auto copy = bs;
			copy.Segment(Start, copy.Size() - Start) <<= shift;
			benchmark::DoNotOptimize(copy);
		}
	}

	template <typename Derived>
	std::size_t OneCount(const BitsetBase<Derived>& bs)
	{
		return bs.OneCount();
	}

	template <std::size_t N>
	std::size_t OneCount(const std::bitset<N>& bs)
	{
		return bs.count();
	}

	template <typename BitsetType, std::size_t N>
	void BM_OneCount(benchmark::State& state)
	{
		auto bs = RandomBitset<BitsetType, N>();
		for (auto _ : state)
		{
			// Make the input opaque each iteration so counting cannot be hoisted.
			benchmark::DoNotOptimize(bs);
			auto count = OneCount(bs);
			benchmark::DoNotOptimize(count);
		}
	}
}

// Keep existing names for comparison with earlier benchmark output.
// 42-bit storage: segment starts at bit 11 and contains 31 bits in one block.
BENCHMARK_TEMPLATE(BM_CopyShiftLeftSegment, 42, 11)->Name("BM_ShiftLeft_Single")->Arg(7)->Arg(31)->Arg(41);

// 100-bit storage: segment starts at bit 11 and contains 89 bits across two blocks.
BENCHMARK_TEMPLATE(BM_CopyShiftLeftSegment, 100, 11)->Name("BM_ShiftLeft_TwoBlock")->Arg(7)->Arg(31)->Arg(63)->Arg(99);

// 5,409 bits span 85 blocks, with 31 tail-padding bits.
BENCHMARK_TEMPLATE(BM_CopyShift, DynamicBitset, 5409, true)
        ->Name("BM_ShiftLeft_ManyBlocks_ZeroHead")
        ->Arg(7)
        ->Arg(63)
        ->Arg(127)
        ->Arg(256);

BENCHMARK_TEMPLATE(BM_CopyShiftLeftSegment, 5409, 0)
        ->Name("BM_ShiftLeft_ManyBlocks_ZeroHeadSeg")
        ->Arg(7)
        ->Arg(63)
        ->Arg(127)
        ->Arg(256);

// The segment starts at bit 19 and contains 5,390 bits.
BENCHMARK_TEMPLATE(BM_CopyShiftLeftSegment, 5409, 19)
        ->Name("BM_ShiftLeft_ManyBlocks_NonZeroHead")
        ->Arg(7)
        ->Arg(63)
        ->Arg(127)
        ->Arg(256);

BENCHMARK_TEMPLATE(BM_CopyShift, Bitset<5409>, 5409, true)
        ->Name("BM_ShiftLeft_ManyBlocks_ZeroHead_5409")
        ->Arg(7)
        ->Arg(63)
        ->Arg(127)
        ->Arg(256);
BENCHMARK_TEMPLATE(BM_CopyShift, std::bitset<5409>, 5409, true)
        ->Name("BM_ShiftLeft_ManyBlocks_ZeroHead_STD5409")
        ->Arg(7)
        ->Arg(63)
        ->Arg(127)
        ->Arg(256);

BENCHMARK_TEMPLATE(BM_CopyShift, Bitset<5409>, 5409, false)
        ->Name("BM_ShiftRight_ManyBlocks_ZeroHead_5409")
        ->Arg(7)
        ->Arg(63)
        ->Arg(127)
        ->Arg(256);
BENCHMARK_TEMPLATE(BM_CopyShift, std::bitset<5409>, 5409, false)
        ->Name("BM_ShiftRight_ManyBlocks_ZeroHead_STD5409")
        ->Arg(7)
        ->Arg(63)
        ->Arg(127)
        ->Arg(256);

BENCHMARK_TEMPLATE(BM_OneCount, Bitset<5409>, 5409)->Name("BM_PC_ManyBlocks_ZeroHead_5409");
BENCHMARK_TEMPLATE(BM_OneCount, std::bitset<5409>, 5409)->Name("BM_PC_ManyBlocks_ZeroHead_STD5409");

BENCHMARK_MAIN();
