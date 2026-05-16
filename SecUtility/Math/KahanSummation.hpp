// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <SecUtility/Math/Core.hpp>
#include <SecUtility/Meta/Identity.hpp>
#include <functional>


namespace SecUtility::Math
{
	template <typename Scalar>
	class KahanAccumulator
	{
	public:
		constexpr KahanAccumulator() noexcept = default;
		KahanAccumulator(const KahanAccumulator&) = delete;
		KahanAccumulator(KahanAccumulator&&) = delete;
		KahanAccumulator& operator=(const KahanAccumulator&) = delete;
		KahanAccumulator& operator=(KahanAccumulator&&) = delete;
		~KahanAccumulator() noexcept = default;

		constexpr KahanAccumulator& AddTerm(const Scalar value) noexcept
		{
			const Scalar corrected = value - m_Compensation;
			const Scalar next = m_Sum + corrected;
			m_Compensation = (next - m_Sum) - corrected;
			m_Sum = next;

			return *this;
		}

		constexpr Scalar Sum() const noexcept
		{
			return m_Sum;
		}

	private:
		Scalar m_Sum = {};
		Scalar m_Compensation = {};
	};

	template <typename Scalar>
	class KahanBabushkaNeumaierAccumulator
	{
	public:
		constexpr KahanBabushkaNeumaierAccumulator() noexcept = default;
		KahanBabushkaNeumaierAccumulator(const KahanBabushkaNeumaierAccumulator&) = delete;
		KahanBabushkaNeumaierAccumulator(KahanBabushkaNeumaierAccumulator&&) = delete;
		KahanBabushkaNeumaierAccumulator& operator=(const KahanBabushkaNeumaierAccumulator&) = delete;
		KahanBabushkaNeumaierAccumulator& operator=(KahanBabushkaNeumaierAccumulator&&) = delete;
		~KahanBabushkaNeumaierAccumulator() noexcept = default;

		constexpr KahanBabushkaNeumaierAccumulator& AddTerm(const Scalar value) noexcept
		{
			const auto t = m_Sum + value;
			m_Compensation += Abs(m_Sum) >= Abs(value) ? (m_Sum - t) + value : (value - t) + m_Sum;
			m_Sum = t;

			return *this;
		}

		constexpr Scalar Sum() const noexcept
		{
			return m_Sum + m_Compensation;
		}

	private:
		Scalar m_Sum = {};
		Scalar m_Compensation = {};
	};

	template <typename Scalar>
	class KahanBabushkaKleinAccumulator
	{
	public:
		constexpr KahanBabushkaKleinAccumulator() noexcept = default;
		KahanBabushkaKleinAccumulator(const KahanBabushkaKleinAccumulator&) = delete;
		KahanBabushkaKleinAccumulator(KahanBabushkaKleinAccumulator&&) = delete;
		KahanBabushkaKleinAccumulator& operator=(const KahanBabushkaKleinAccumulator&) = delete;
		KahanBabushkaKleinAccumulator& operator=(KahanBabushkaKleinAccumulator&&) = delete;
		~KahanBabushkaKleinAccumulator() noexcept = default;

		constexpr KahanBabushkaKleinAccumulator& AddTerm(const Scalar term) noexcept
		{
			const auto t1 = m_Sum + term;
			const auto c = Abs(m_Sum) >= Abs(term) ? (m_Sum - t1) + term : (term - t1) + m_Sum;
			m_Sum = t1;

			const auto t2 = m_Cs + c;
			const auto cc = Abs(m_Cs) >= Abs(c) ? (m_Cs - t2) + c : (c - t2) + m_Cs;
			m_Cs = t2;
			m_Ccs += cc;

			return *this;
		}

		constexpr Scalar Sum() const noexcept
		{
			return m_Sum + (m_Cs + m_Ccs);
		}

	private:
		Scalar m_Sum = {};
		Scalar m_Cs = {};
		Scalar m_Ccs = {};
	};


#define SEC_MATH_DEFINE_KAHAN_SUMMATION(TYPE)                                                                          \
	template <typename ForwardIterator, typename Projection = SecUtility::Identity>                                    \
	constexpr auto TYPE##Sum(ForwardIterator begin, const ForwardIterator end, Projection projection = {}) noexcept(   \
	        noexcept(++begin) && noexcept(begin != end) && noexcept(std::invoke(projection, *begin)))                  \
	{                                                                                                                  \
		TYPE##Accumulator<std::decay_t<decltype(std::invoke(projection, *begin))>> accumulator{};                      \
                                                                                                                       \
		for (/* NO CODE */; begin != end; ++begin)                                                                     \
		{                                                                                                              \
			accumulator.AddTerm(std::invoke(projection, *begin));                                                      \
		}                                                                                                              \
                                                                                                                       \
		return accumulator.Sum();                                                                                      \
	}                                                                                                                  \
                                                                                                                       \
	template <typename Range, typename Projection = SecUtility::Identity>                                              \
	constexpr auto TYPE##Sum(const Range& range, Projection projection = {}) noexcept(                                 \
	        noexcept(TYPE##Sum(std::begin(range), std::end(range)), projection))                                       \
	{                                                                                                                  \
		return TYPE##Sum(std::begin(range), std::end(range), projection);                                              \
	}

	SEC_MATH_DEFINE_KAHAN_SUMMATION(Kahan)
	SEC_MATH_DEFINE_KAHAN_SUMMATION(KahanBabushkaNeumaier)
	SEC_MATH_DEFINE_KAHAN_SUMMATION(KahanBabushkaKlein)

#undef SEC_MATH_DEFINE_KAHAN_SUMMATION
}
