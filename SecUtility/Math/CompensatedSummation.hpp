// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <SecUtility/Math/Core.hpp>
#include <SecUtility/Meta/Identity.hpp>
#include <SecUtility/Meta/TypeTrait.hpp>
#include <functional>


namespace SecUtility::Math
{
	template <typename Derived>
	class CompensatedAccumulatorBase;

	template <typename Scalar>
	class KahanAccumulator;

	template <typename Scalar>
	class KahanBabushkaNeumaierAccumulator;

	template <typename Scalar>
	class KahanBabushkaKleinAccumulator;
}

namespace SecUtility
{
	template <typename TScalar>
	struct Traits<Math::KahanAccumulator<TScalar>>
	{
		using Scalar = TScalar;
	};

	template <typename TScalar>
	struct Traits<Math::KahanBabushkaNeumaierAccumulator<TScalar>>
	{
		using Scalar = TScalar;
	};

	template <typename TScalar>
	struct Traits<Math::KahanBabushkaKleinAccumulator<TScalar>>
	{
		using Scalar = TScalar;
	};
}

namespace SecUtility::Math
{
	template <typename Derived>
	class CompensatedAccumulatorBase
	{
		constexpr const Derived& AsDerived() const noexcept
		{
			return static_cast<const Derived&>(*this);
		}

		constexpr Derived& AsDerived() noexcept
		{
			return static_cast<Derived&>(*this);
		}

		constexpr CompensatedAccumulatorBase() noexcept = default;
		constexpr CompensatedAccumulatorBase(const CompensatedAccumulatorBase&) noexcept = default;
		constexpr CompensatedAccumulatorBase(CompensatedAccumulatorBase&&) noexcept = default;
		constexpr CompensatedAccumulatorBase& operator=(const CompensatedAccumulatorBase&) noexcept = default;
		constexpr CompensatedAccumulatorBase& operator=(CompensatedAccumulatorBase&&) noexcept = default;
		~CompensatedAccumulatorBase() noexcept = default;

		friend Derived;

		using Scalar = typename Traits<Derived>::Scalar;

	public:
		constexpr Scalar Sum() const noexcept
		{
			return AsDerived().Sum_Impl();
		}

		constexpr Derived& AddTerm(Scalar term) noexcept
		{
			AsDerived().AddTerm_Impl(term);
			return AsDerived();
		}

		template <typename... Args>
		constexpr std::enable_if_t<(std::is_convertible_v<Args, Scalar> && ...), Derived&> AddTerms(Args... args)  //
		        noexcept((noexcept(static_cast<Scalar>(args)) && ...))
		{
			(AsDerived().AddTerm_Impl(static_cast<Scalar>(args)), ...);
			return AsDerived();
		}

		// clang-format off
		template <typename ForwardIterator, typename Projector = SecUtility::Identity>
		constexpr auto AddTerms(ForwardIterator begin, const ForwardIterator end, Projector projector = {})  //
		        noexcept(noexcept(static_cast<Scalar>(std::invoke(projector, *++begin))))
		                -> std::enable_if_t<std::is_same_v<std::void_t<decltype(static_cast<Scalar>(std::invoke(projector, *++begin)))>, void>, Derived&>
		// clang-format on
		{
			for (/* NO CODE */; begin != end; ++begin)
			{
				AsDerived().AddTerm_Impl(std::invoke(projector, *begin));
			}
			return AsDerived();
		}

		// clang-format off
		template <typename Range, typename Projector = SecUtility::Identity>
		constexpr auto AddTerms(const Range& range, Projector projector = {})  //
		        noexcept(noexcept(static_cast<Scalar>(std::invoke(projector, *++std::begin(range)))))
		                -> std::enable_if_t<std::is_same_v<std::void_t<decltype(static_cast<Scalar>(std::invoke(projector, *++std::begin(range))))>, void>, Derived&>
		// clang-format on
		{
			return AddTerms(std::begin(range), std::end(range), projector);
		}
	};

	template <typename Scalar>
	class KahanAccumulator : public CompensatedAccumulatorBase<KahanAccumulator<Scalar>>
	{
		using Base = CompensatedAccumulatorBase<KahanAccumulator>;
		friend Base;

		constexpr void AddTerm_Impl(const Scalar value) noexcept
		{
			const Scalar corrected = value - m_Compensation;
			const Scalar next = m_Sum + corrected;
			m_Compensation = (next - m_Sum) - corrected;
			m_Sum = next;
		}

		constexpr Scalar Sum_Impl() const noexcept
		{
			return m_Sum;
		}

		Scalar m_Sum = {};
		Scalar m_Compensation = {};
	};

	template <typename Scalar>
	class KahanBabushkaNeumaierAccumulator : public CompensatedAccumulatorBase<KahanBabushkaNeumaierAccumulator<Scalar>>
	{
		using Base = CompensatedAccumulatorBase<KahanBabushkaNeumaierAccumulator>;
		friend Base;

		constexpr void AddTerm_Impl(const Scalar value) noexcept
		{
			const auto t = m_Sum + value;
			m_Compensation += Abs(m_Sum) >= Abs(value) ? (m_Sum - t) + value : (value - t) + m_Sum;
			m_Sum = t;
		}

		constexpr Scalar Sum_Impl() const noexcept
		{
			return m_Sum + m_Compensation;
		}

		Scalar m_Sum = {};
		Scalar m_Compensation = {};
	};

	template <typename Scalar>
	class KahanBabushkaKleinAccumulator : public CompensatedAccumulatorBase<KahanBabushkaKleinAccumulator<Scalar>>
	{
		using Base = CompensatedAccumulatorBase<KahanBabushkaKleinAccumulator>;
		friend Base;

		constexpr void AddTerm_Impl(const Scalar term) noexcept
		{
			const auto t1 = m_Sum + term;
			const auto c = Abs(m_Sum) >= Abs(term) ? (m_Sum - t1) + term : (term - t1) + m_Sum;
			m_Sum = t1;

			const auto t2 = m_Cs + c;
			const auto cc = Abs(m_Cs) >= Abs(c) ? (m_Cs - t2) + c : (c - t2) + m_Cs;
			m_Cs = t2;
			m_Ccs += cc;
		}

		constexpr Scalar Sum_Impl() const noexcept
		{
			return m_Sum + (m_Cs + m_Ccs);
		}

		Scalar m_Sum = {};
		Scalar m_Cs = {};
		Scalar m_Ccs = {};
	};


#define SEC_MATH_DEFINE_KAHAN_SUMMATION(TYPE)                                                                          \
	template <typename ForwardIterator, typename Projection = SecUtility::Identity>                                    \
	constexpr auto TYPE##Sum(ForwardIterator begin, const ForwardIterator end, Projection projection = {}) noexcept(   \
	        noexcept(++begin) && noexcept(begin != end) && noexcept(std::invoke(projection, *begin)))                  \
	{                                                                                                                  \
		using Scalar = std::decay_t<decltype(std::invoke(projection, *begin))>;                                        \
		return TYPE##Accumulator<Scalar>{}.AddTerms(begin, end, projection).Sum();                                     \
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
