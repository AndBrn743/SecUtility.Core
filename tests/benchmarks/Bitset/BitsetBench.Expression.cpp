// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#include "BitsetBench.Common.hpp"

namespace
{
	enum class AllPattern
	{
		Random,
		Ones,
		LastZero
	};

	template <typename T, std::size_t N>
	void BM_ExpressionAll(benchmark::State& state, const AllPattern pattern)
	{
		auto a = RandomBitset<T, N>();
		auto b = RandomBitset<T, N>(43);
		if (pattern != AllPattern::Random)
		{
			for (std::size_t i = 0; i < N; ++i)
			{
				a[i] = true;
				b[i] = true;
			}
			if (pattern == AllPattern::LastZero)
			{
				b[N - 1] = false;
			}
		}
		for (auto _ : state)
		{
			benchmark::DoNotOptimize(a);
			benchmark::DoNotOptimize(b);
			bool result = [&]
			{
				if constexpr (IsStdBitset<T>::value)
				{
					return (a & b).all();
				}
				else
				{
					return (a & b).IsAllOnes();
				}
			}();
			benchmark::DoNotOptimize(result);
		}
	}

	template <typename T>
	auto Evaluate(const T& expression)
	{
		if constexpr (IsStdBitset<T>::value)
		{
			return expression;
		}
		else
		{
			return expression.Eval();
		}
	}

	template <typename T, std::size_t N, bool Materialize>
	void BM_Expression(benchmark::State& state)
	{
		auto a = RandomBitset<T, N>();
		auto b = RandomBitset<T, N>(69);
		auto c = RandomBitset<T, N>(123);
		auto result = MakeBitset<T, N>();
		for (auto _ : state)
		{
			benchmark::DoNotOptimize(a);
			benchmark::DoNotOptimize(b);
			benchmark::DoNotOptimize(c);
			if constexpr (Materialize)
			{
				auto intermediate = Evaluate(a & b);
				// Require an actual intermediate store even when the compiler
				// could otherwise fuse the two eager standard-library operations.
				benchmark::DoNotOptimize(intermediate);
				result = intermediate | c;
			}
			else
			{
				result = (a & b) | c;
			}
			benchmark::DoNotOptimize(result);
		}
	}

	template <typename T, std::size_t N>
	void RegisterOwner(const std::string& type)
	{
		const auto suffix = "/" + type + "/" + std::to_string(N);
		benchmark::RegisterBenchmark("Expression/Chained" + suffix, &BM_Expression<T, N, false>);
		benchmark::RegisterBenchmark("Expression/Materialized" + suffix, &BM_Expression<T, N, true>);
		// Empty all queries have different semantics in the two APIs.
		if constexpr (N != 0)
		{
			for (const auto& pattern : {std::pair{AllPattern::Random, "Random"},
			                            std::pair{AllPattern::Ones, "Ones"},
			                            std::pair{AllPattern::LastZero, "LastZero"}})
			{
				benchmark::RegisterBenchmark(
				        "Expression/AndAll" + suffix + "/" + pattern.second, &BM_ExpressionAll<T, N>, pattern.first);
			}
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

void BitsetBenchmark::RegisterExpressions()
{
	RegisterSizes(OperationSizes{});
}
