// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

// Helpers have internal linkage so each family is compiled independently.
#include <SecUtility/Collection/Bitset.hpp>
#include <benchmark/benchmark.h>

#include <algorithm>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace SecUtility;

namespace
{
	constexpr std::size_t BatchSize = 32;

	template <typename T>
	struct IsStdBitset : std::false_type
	{
	};
	template <std::size_t N>
	struct IsStdBitset<std::bitset<N>> : std::true_type
	{
	};

	template <typename T, std::size_t N>
	T MakeBitset()
	{
		if constexpr (std::is_same_v<T, DynamicBitset>)
		{
			return T(N);
		}
		else
		{
			return T{};
		}
	}

	template <typename T, std::size_t N>
	T RandomBitset(const std::uint64_t seed = 42)
	{
		auto bs = MakeBitset<T, N>();
		std::mt19937_64 rng(seed);
		for (std::size_t i = 0; i < N; ++i)
		{
			bs[i] = static_cast<bool>(rng() & 1u);
		}
		return bs;
	}

	template <bool Left, typename T>
	void Shift(T& bs, const std::size_t amount)
	{
		if constexpr (Left)
		{
			bs <<= amount;
		}
		else
		{
			bs >>= amount;
		}
	}

	// Reset existing allocations outside the measured interval. Batch timings
	// cover 32 operations; items_per_second reports individual bitsets per second.
	template <typename T>
	void ResetBatch(benchmark::State& state, std::vector<T>& work, T& original)
	{
		state.PauseTiming();
		benchmark::DoNotOptimize(original);
		std::fill(work.begin(), work.end(), original);
		benchmark::DoNotOptimize(work.data());
		state.ResumeTiming();
	}

	struct Assign
	{
		static constexpr const char* Name = "Assign";
		template <typename L, typename R>
		void operator()(L& lhs, const R& rhs) const
		{
			lhs = rhs;
		}
	};
	struct And
	{
		static constexpr const char* Name = "And";
		template <typename L, typename R>
		void operator()(L& lhs, const R& rhs) const
		{
			lhs &= rhs;
		}
	};
	struct Or
	{
		static constexpr const char* Name = "Or";
		template <typename L, typename R>
		void operator()(L& lhs, const R& rhs) const
		{
			lhs |= rhs;
		}
	};
	struct Xor
	{
		static constexpr const char* Name = "Xor";
		template <typename L, typename R>
		void operator()(L& lhs, const R& rhs) const
		{
			lhs ^= rhs;
		}
	};

	inline std::vector<std::size_t> ShiftAmounts(const std::size_t size)
	{
		std::vector<std::size_t> amounts{0, 1, 7, 63, 64, 65, 127, 256, size, size + 1};
		if (size != 0)
		{
			amounts.push_back(size - 1);
		}
		std::sort(amounts.begin(), amounts.end());
		amounts.erase(std::unique(amounts.begin(), amounts.end()), amounts.end());
		return amounts;
	}

	template <typename Benchmark>
	void AddShifts(Benchmark* bench, const std::size_t size)
	{
		bench->ArgName("shift");
		for (const auto shift : ShiftAmounts(size))
		{
			bench->Arg(static_cast<std::int64_t>(shift));
		}
	}

	using OwnerSizes = std::index_sequence<0, 1, 42, 63, 64, 65, 127, 128, 129, 549, 5409, 65536>;
	using OperationSizes = std::index_sequence<0, 64, 65, 129, 5409>;
	using SegmentSizes = std::index_sequence<42, 65, 129, 5409>;
}

namespace BitsetBenchmark
{
	void RegisterCopies();
	void RegisterShifts();
	void RegisterBitwise();
	void RegisterExpressions();
	void RegisterQueries();
	void RegisterEquality();
	void RegisterSegments();
}
