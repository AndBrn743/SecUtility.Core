// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2026 Andy Brown

#pragma once

#include <SecUtility/Diagnostic/Exception.hpp>
#include <SecUtility/Math/Constant.hpp>
#include <SecUtility/Math/Core.hpp>
#include <SecUtility/Meta/TypeTrait.hpp>
#include <cassert>
#include <cstddef>
#include <functional>
#include <limits>


namespace SecUtility::Math
{
	template <typename Scalar>
	struct Interval
	{
		Scalar Lower;
		Scalar Upper;
	};


	template <typename DomainScalar, typename CodomainScalar>
	struct SearchOptions
	{
		DomainScalar AbsoluteDomainTolerance = 10 * std::numeric_limits<DomainScalar>::epsilon();
		DomainScalar RelativeDomainTolerance = std::numeric_limits<DomainScalar>::epsilon();
		CodomainScalar AbsoluteCodomainTolerance = 10 * std::numeric_limits<CodomainScalar>::epsilon();
		CodomainScalar RelativeCodomainTolerance = std::numeric_limits<CodomainScalar>::epsilon();
		std::size_t MaximumIterations = 100;
	};


	template <typename DomainScalar, typename CodomainScalar>
	struct SearchResult
	{
		DomainScalar Position;
		CodomainScalar FunctionValue;
		std::size_t IterationCount;
		bool IsConverged;
	};


	template <typename Function,
	          typename...,
	          typename = std::enable_if_t<!std::is_arithmetic_v<std::decay_t<Function>>>,  // tweak me
	          typename = std::enable_if_t<FunctionTraits<Function>::Arity == 1>,
	          typename DomainScalar = std::decay_t<typename FunctionTraits<Function>::ArgTypeTuple::template TypeAt<0>>,
	          typename CodomainScalar = std::decay_t<typename FunctionTraits<Function>::ReturnType>>
	SearchResult<DomainScalar, CodomainScalar> BisectionSearch(
	        Function&& function,
	        const Interval<DomainScalar> interval,
	        const SearchOptions<DomainScalar, CodomainScalar>& options = {})
	{
		static_assert(std::is_floating_point_v<DomainScalar>);
		static_assert(std::is_floating_point_v<CodomainScalar>);
		// assert(interval.Lower < interval.Upper);

		assert(options.AbsoluteDomainTolerance >= DomainScalar{0});
		assert(options.RelativeDomainTolerance >= DomainScalar{0});

		assert(options.AbsoluteCodomainTolerance >= CodomainScalar{0});
		assert(options.RelativeCodomainTolerance >= CodomainScalar{0});

		struct Sample
		{
			std::string ToString() const
			{
				return std::string{"{"} + std::to_string(X) + ", " + std::to_string(F) + "}";
			}

			DomainScalar X;
			CodomainScalar F;
		};

		const auto TakeSampleAt = [&](const DomainScalar x) { return Sample{x, function(x)}; };

		const auto IsCodomainConverged = [&](const CodomainScalar f)
		{
			// return Abs(f) <= options.AbsoluteCodomainTolerance + options.RelativeCodomainTolerance * Abs(f);
			return Abs(f) <= options.AbsoluteCodomainTolerance;
		};

		const auto IsDomainConverged = [&](const DomainScalar a, const DomainScalar b)
		{
			const auto width = Abs(b - a);
			return width <= options.AbsoluteDomainTolerance + options.RelativeDomainTolerance * Max(Abs(a), Abs(b));
		};

		auto left = TakeSampleAt(interval.Lower);

		if (IsCodomainConverged(left.F))
		{
			return {left.X, left.F, 0, true};
		}

		auto right = TakeSampleAt(interval.Upper);

		if (IsCodomainConverged(right.F))
		{
			return {right.X, right.F, 0, true};
		}

		if (SignBit(left.F) == SignBit(right.F))
		{
			throw RuntimeException("BisectionSearch requires a bracketed root", left.ToString(), right.ToString());
		}

		for (std::size_t iteration = 0; iteration < options.MaximumIterations; ++iteration)
		{
			const auto center = TakeSampleAt(left.X + (right.X - left.X) / DomainScalar{2});

			if (IsCodomainConverged(center.F) || IsDomainConverged(left.X, right.X))
			{
				return {center.X, center.F, iteration + 1, true};
			}

			if (SignBit(left.F) != SignBit(center.F))
			{
				right = center;
			}
			else
			{
				left = center;
			}
		}

		const auto& best = Abs(left.F) < Abs(right.F) ? left : right;
		return {best.X, best.F, options.MaximumIterations, false};
	}


	template <typename DomainScalar,
	          typename CodomainScalar,
	          typename...,
	          typename DomainScalarToo = DomainScalar,
	          typename CodomainScalarToo = CodomainScalar,
	          typename Function>
	SearchResult<DomainScalar, CodomainScalar> BisectionSearch(
	        Function&& function,
	        const Interval<DomainScalarToo> interval,
	        const SearchOptions<DomainScalarToo, CodomainScalarToo>& options = {})
	{
		return BisectionSearch([&function](const DomainScalar x) { return static_cast<CodomainScalar>(function(x)); },
		                       interval,
		                       options);
	}


	template <typename Function,
	          typename Comp = std::less<>,
	          typename...,
	          typename = std::enable_if_t<!std::is_arithmetic_v<std::decay_t<Function>>>,  // tweak me
	          typename = std::enable_if_t<FunctionTraits<Function>::Arity == 1>,
	          typename DomainScalar = std::decay_t<typename FunctionTraits<Function>::ArgTypeTuple::template TypeAt<0>>,
	          typename CodomainScalar = std::decay_t<typename FunctionTraits<Function>::ReturnType>,
	          typename = std::enable_if_t<std::is_same_v<decltype(std::declval<Comp>()(std::decay_t<CodomainScalar>(),
	                                                                                   std::decay_t<CodomainScalar>())),
	                                                     bool>>>
	SearchResult<DomainScalar, CodomainScalar> GoldenSectionSearch(
	        Function&& function,
	        Interval<DomainScalar> interval,
	        const Comp comp = {},
	        const SearchOptions<DomainScalar, CodomainScalar>& options = {})
	{
		static_assert(std::is_floating_point_v<DomainScalar>);
		static_assert(std::is_floating_point_v<CodomainScalar>);
		// assert(interval.Lower < interval.Upper);

		struct Sample
		{
			DomainScalar X;
			CodomainScalar F;
		};

		const auto TakeSampleAt = [&](const DomainScalar x) { return Sample{x, function(x)}; };

		const auto IsDomainConverged = [&](const DomainScalar a, const DomainScalar b)
		{
			const auto width = Abs(b - a);
			return width <= options.AbsoluteDomainTolerance + options.RelativeDomainTolerance * Max(Abs(a), Abs(b));
		};

		const auto IsCodomainConverged = [&](const CodomainScalar f0, const CodomainScalar f1)
		{
			return Abs(f1 - f0)
			       <= options.AbsoluteCodomainTolerance + options.RelativeCodomainTolerance * Max(Abs(f0), Abs(f1));
		};

		static constexpr DomainScalar InverseGoldenRatio = DomainScalar{1} / Constant::GoldenRatio<DomainScalar>;

		auto fc = TakeSampleAt(interval.Upper - InverseGoldenRatio * (interval.Upper - interval.Lower));
		auto fd = TakeSampleAt(interval.Lower + InverseGoldenRatio * (interval.Upper - interval.Lower));

		if (IsCodomainConverged(fc.F, fd.F) || IsDomainConverged(interval.Lower, interval.Upper))
		{
			const auto& best = comp(fc.F, fd.F) ? fc : fd;
			return {best.X, best.F, 0, true};
		}

		for (std::size_t iteration = 0; iteration < options.MaximumIterations; ++iteration)
		{
			if (comp(fc.F, fd.F))
			{
				interval.Upper = fd.X;

				fd = fc;
				fc = TakeSampleAt(interval.Upper - InverseGoldenRatio * (interval.Upper - interval.Lower));
			}
			else
			{
				interval.Lower = fc.X;

				fc = fd;
				fd = TakeSampleAt(interval.Lower + InverseGoldenRatio * (interval.Upper - interval.Lower));
			}

			if (IsCodomainConverged(fc.F, fd.F) || IsDomainConverged(interval.Lower, interval.Upper))
			{
				const auto& best = comp(fc.F, fd.F) ? fc : fd;
				return {best.X, best.F, iteration + 1, true};
			}
		}

		const auto& best = comp(fc.F, fd.F) ? fc : fd;
		return {best.X, best.F, options.MaximumIterations, false};
	}

	template <typename Function,
	          typename Comp = std::less<>,
	          typename...,
	          typename = std::enable_if_t<!std::is_arithmetic_v<std::decay_t<Function>>>,  // tweak me
	          typename = std::enable_if_t<FunctionTraits<Function>::Arity == 1>,
	          typename DomainScalar = std::decay_t<typename FunctionTraits<Function>::ArgTypeTuple::template TypeAt<0>>,
	          typename CodomainScalar = std::decay_t<typename FunctionTraits<Function>::ReturnType>,
	          typename = std::enable_if_t<std::is_same_v<decltype(std::declval<Comp>()(std::decay_t<CodomainScalar>(),
	                                                                                   std::decay_t<CodomainScalar>())),
	                                                     bool>>>
	SearchResult<DomainScalar, CodomainScalar> GoldenSectionSearch(
	        Function&& function,
	        Interval<DomainScalar> interval,

	        const SearchOptions<DomainScalar, CodomainScalar>& options,
	        const Comp comp = {})
	{
		return GoldenSectionSearch(function, interval, comp, options);
	}


	template <typename DomainScalar,
	          typename CodomainScalar,
	          typename Comp = std::less<>,
	          typename...,
	          typename DomainScalarToo = DomainScalar,
	          typename CodomainScalarToo = CodomainScalar,
	          typename = std::enable_if_t<std::is_floating_point_v<DomainScalar>>,
	          typename = std::enable_if_t<std::is_floating_point_v<CodomainScalar>>,
	          typename = std::enable_if_t<std::is_same_v<decltype(std::declval<Comp>()(std::decay_t<CodomainScalar>(),
	                                                                                   std::decay_t<CodomainScalar>())),
	                                                     bool>>,
	          typename Function>
	SearchResult<DomainScalar, CodomainScalar> GoldenSectionSearch(
	        Function&& function,
	        const Interval<DomainScalarToo> interval,
	        const Comp comp = {},
	        const SearchOptions<DomainScalarToo, CodomainScalarToo>& options = {})
	{
		return GoldenSectionSearch([&function](const DomainScalar x)
		                           { return static_cast<CodomainScalar>(function(x)); },
		                           interval,
		                           comp,
		                           options);
	}


	template <typename DomainScalar,
	          typename CodomainScalar,
	          typename Comp = std::less<>,
	          typename...,
	          typename DomainScalarToo = DomainScalar,
	          typename CodomainScalarToo = CodomainScalar,
	          typename = std::enable_if_t<std::is_floating_point_v<DomainScalar>>,
	          typename = std::enable_if_t<std::is_floating_point_v<CodomainScalar>>,
	          typename = std::enable_if_t<std::is_same_v<decltype(std::declval<Comp>()(std::decay_t<CodomainScalar>(),
	                                                                                   std::decay_t<CodomainScalar>())),
	                                                     bool>>,
	          typename Function>
	SearchResult<DomainScalar, CodomainScalar> GoldenSectionSearch(
	        Function&& function,
	        const Interval<DomainScalarToo> interval,
	        const SearchOptions<DomainScalarToo, CodomainScalarToo>& options,
	        const Comp comp = {})
	{
		return GoldenSectionSearch([&function](const DomainScalar x)
		                           { return static_cast<CodomainScalar>(function(x)); },
		                           interval,
		                           comp,
		                           options);
	}
}
