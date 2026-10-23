// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#include "BitsetBench.Read.Common.hpp"

namespace
{
	enum class ReadQuery
	{
		Count,
		All,
		Any,
		None
	};

	template <ReadQuery Query, typename Expression>
	auto QueryExpression(const Expression& expression)
	{
		if constexpr (IsStdBitset<Expression>::value)
		{
			if constexpr (Query == ReadQuery::Count)
			{
				return expression.count();
			}
			else if constexpr (Query == ReadQuery::All)
			{
				return expression.all();
			}
			else if constexpr (Query == ReadQuery::Any)
			{
				return expression.any();
			}
			else
			{
				return expression.none();
			}
		}
		else
		{
			if constexpr (Query == ReadQuery::Count)
			{
				return expression.OneCount();
			}
			else if constexpr (Query == ReadQuery::All)
			{
				return expression.IsAllOnes();
			}
			else if constexpr (Query == ReadQuery::Any)
			{
				return expression.HasOnes();
			}
			else
			{
				return expression.IsAllZeros();
			}
		}
	}

	template <typename T, std::size_t N, ReadForm Form, ReadQuery Query>
	void BM_ExpressionRead(benchmark::State& state, const ReadPattern pattern, const std::size_t head)
	{
		ReadFixture<T, N, Form> fixture(pattern, head);
		const auto query = [](const auto& expression) { return QueryExpression<Query>(expression); };
		if (!CheckReadResult(state, fixture.Apply(query), QueryExpression<Query>(fixture.expected)))
		{
			return;
		}
		for (auto _ : state)
		{
			fixture.Barrier();
			auto result = fixture.Apply(query);
			benchmark::DoNotOptimize(result);
		}
	}

	template <typename T, std::size_t N, ReadForm Form, bool Reverse>
	void BM_ExpressionEqual(benchmark::State& state, const int mismatch, const std::size_t head)
	{
		ReadFixture<T, N, Form> fixture(ReadPattern::Random, head);
		auto other = MakeBitset<T, N>();
		for (std::size_t i = 0; i < N; ++i)
		{
			other[i] = static_cast<bool>(fixture.expected[i]);
		}
		if constexpr (N != 0)
		{
			if (mismatch != 0)
			{
				const std::size_t index = mismatch == 1 ? 0 : N - 1;
				other[index] = !static_cast<bool>(other[index]);
			}
		}
		const auto query = [&other](const auto& expression)
		{
			if constexpr (Reverse)
			{
				return other == expression;
			}
			else
			{
				return expression == other;
			}
		};
		if (!CheckReadResult(state, fixture.Apply(query), mismatch == 0))
		{
			return;
		}
		for (auto _ : state)
		{
			fixture.Barrier();
			benchmark::DoNotOptimize(other);
			auto result = fixture.Apply(query);
			benchmark::DoNotOptimize(result);
		}
	}

	template <typename T, std::size_t N, ReadForm Form, ReadQuery Query>
	void RegisterReadQuery(const std::string& name, const std::string& suffix, const std::size_t head)
	{
		for (const auto& pattern : ReadPatterns)
		{
			if (N == 0 && pattern.first != ReadPattern::Random)
			{
				continue;
			}
			benchmark::RegisterBenchmark("ExpressionRead/" + name + suffix + "/" + pattern.second,
			                             &BM_ExpressionRead<T, N, Form, Query>,
			                             pattern.first,
			                             head);
		}
	}

	template <typename T, std::size_t N, ReadForm Form>
	void RegisterReadForm(const std::string& type, const std::size_t head = 0)
	{
		const auto suffix = ReadSuffix<Form>(type, N, head);
		RegisterReadQuery<T, N, Form, ReadQuery::Count>("Count", suffix, head);
		RegisterReadQuery<T, N, Form, ReadQuery::Any>("Any", suffix, head);
		if constexpr (N != 0)
		{
			// Empty all/none differ between SecUtility and std::bitset.
			RegisterReadQuery<T, N, Form, ReadQuery::All>("All", suffix, head);
			RegisterReadQuery<T, N, Form, ReadQuery::None>("None", suffix, head);
		}
		const char* mismatches[] = {"Same", "FirstMismatch", "LastMismatch"};
		for (int mismatch = 0; mismatch < (N == 0 ? 1 : 3); ++mismatch)
		{
			benchmark::RegisterBenchmark("ExpressionRead/EqualExprFirst" + suffix + "/" + mismatches[mismatch],
			                             &BM_ExpressionEqual<T, N, Form, false>,
			                             mismatch,
			                             head);
			benchmark::RegisterBenchmark("ExpressionRead/EqualOwnerFirst" + suffix + "/" + mismatches[mismatch],
			                             &BM_ExpressionEqual<T, N, Form, true>,
			                             mismatch,
			                             head);
		}
	}

	template <typename T, std::size_t N>
	void RegisterReadOwner(const std::string& type)
	{
		RegisterReadForm<T, N, ReadForm::Owner>(type);
		RegisterReadForm<T, N, ReadForm::Not>(type);
		RegisterReadForm<T, N, ReadForm::And>(type);
		RegisterReadForm<T, N, ReadForm::Nested>(type);
		if constexpr (!IsStdBitset<T>::value)
		{
			for (const std::size_t head : {0u, 11u, 63u})
			{
				RegisterReadForm<T, N, ReadForm::View>(type, head);
				RegisterReadForm<T, N, ReadForm::NotView>(type, head);
				RegisterReadForm<T, N, ReadForm::ViewAnd>(type, head);
			}
		}
	}

	template <std::size_t... Sizes>
	void RegisterReadSizes(std::index_sequence<Sizes...>)
	{
		(RegisterReadOwner<Bitset<Sizes>, Sizes>("Static"), ...);
		(RegisterReadOwner<DynamicBitset, Sizes>("Dynamic"), ...);
		(RegisterReadOwner<std::bitset<Sizes>, Sizes>("Std"), ...);
	}
}

void BitsetBenchmark::RegisterExpressionReads()
{
	RegisterReadSizes(OperationSizes{});
}
