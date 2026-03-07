//
// Created by Andy on 3/6/2026.
//

#pragma once

#include <SecUtility/Macro/ConstevalIf.hpp>
#include <SecUtility/Macro/ForceInline.hpp>
#include <SecUtility/Math/Constant.hpp>

#if defined(SEC_IF_CONSTEVAL) && __has_include(<gcem.hpp>)
#include <gcem.hpp>
#define SEC_MATH_CORE_CONDITIONAL_CONSTEXPR constexpr
#else
#define SEC_MATH_CORE_CONDITIONAL_CONSTEXPR
#endif

#if defined(SEC_MATH_CORE_EXTENSION)
#if __has_include(SEC_MATH_CORE_EXTENSION)
#include SEC_MATH_CORE_EXTENSION
#endif
#endif

#include <cmath>
#include <complex>
#include <utility>

namespace SecUtility::Math
{
#define SEC_EXPORT_RUNTIME_ONLY_MATH_FUNCTION(SEC_NAME, C_NAME)                                                        \
	template <typename... Args>                                                                                        \
	SEC_FORCE_INLINE auto SEC_NAME(Args&&... args) noexcept(noexcept(std::C_NAME(std::forward<Args>(args)...)))        \
	        -> decltype(std::C_NAME(std::forward<Args>(args)...))                                                      \
	{                                                                                                                  \
		return std::C_NAME(std::forward<Args>(args)...);                                                               \
	}

#define SEC_EXPORT_CONSTEXPR_MATH_FUNCTION_WITH_FALLBACK(SEC_NAME, C_NAME)                                             \
	template <typename... Args>                                                                                        \
	constexpr SEC_FORCE_INLINE auto SEC_NAME(Args&&... args) noexcept(                                                 \
	        noexcept(std::C_NAME(std::forward<Args>(args)...))) -> decltype(std::C_NAME(std::forward<Args>(args)...))  \
	{                                                                                                                  \
		SEC_IF_CONSTEVAL return gcem::C_NAME(std::forward<Args>(args)...);                                             \
		else return std::C_NAME(std::forward<Args>(args)...);                                                          \
	}

#if defined(SEC_IF_CONSTEVAL) && __has_include(<gcem.hpp>)
#define SEC_EXPORT_MATH_FUNCTION SEC_EXPORT_CONSTEXPR_MATH_FUNCTION_WITH_FALLBACK
#else
#define SEC_EXPORT_MATH_FUNCTION SEC_EXPORT_RUNTIME_ONLY_MATH_FUNCTION
#endif

	SEC_EXPORT_MATH_FUNCTION(Ceil, ceil)
	SEC_EXPORT_MATH_FUNCTION(Floor, floor)
	SEC_EXPORT_MATH_FUNCTION(Round, round)
	SEC_EXPORT_MATH_FUNCTION(Truncate, trunc)
	SEC_EXPORT_RUNTIME_ONLY_MATH_FUNCTION(NearByInt, nearbyint)

#define SEC_DEFINE_MATH_ROUNDING_FUNCTION_WITH_CAST(ROUNDING_FUNCTION)                                                 \
	template <typename Integer, typename Floating>                                                                     \
	SEC_MATH_CORE_CONDITIONAL_CONSTEXPR Integer ROUNDING_FUNCTION(const Floating& x) noexcept(                         \
	        noexcept(static_cast<Integer>(ROUNDING_FUNCTION(x))))                                                      \
	{                                                                                                                  \
		static_assert(std::is_integral_v<Integer>);                                                                    \
		return static_cast<Integer>(ROUNDING_FUNCTION(x));                                                             \
	}

	SEC_DEFINE_MATH_ROUNDING_FUNCTION_WITH_CAST(Ceil)
	SEC_DEFINE_MATH_ROUNDING_FUNCTION_WITH_CAST(Floor)
	SEC_DEFINE_MATH_ROUNDING_FUNCTION_WITH_CAST(Round)
	SEC_DEFINE_MATH_ROUNDING_FUNCTION_WITH_CAST(Truncate)

	template <typename Integer, typename Floating>
	Integer NearByInt(const Floating& x) noexcept(noexcept(static_cast<Integer>(NearByInt(x))))
	{
		static_assert(std::is_integral_v<Integer>);
		return static_cast<Integer>(NearByInt(x));
	}

#undef SEC_DEFINE_MATH_ROUNDING_FUNCTION_WITH_CAST

	SEC_EXPORT_MATH_FUNCTION(Exp, exp)
	SEC_EXPORT_MATH_FUNCTION(Log, log)
	SEC_EXPORT_MATH_FUNCTION(Log1p, log1p)
	SEC_EXPORT_MATH_FUNCTION(Log2, log2)
	SEC_EXPORT_MATH_FUNCTION(Log10, log10)

	SEC_EXPORT_MATH_FUNCTION(Pow, pow)
	SEC_EXPORT_MATH_FUNCTION(Sqrt, sqrt)

	SEC_EXPORT_MATH_FUNCTION(Sin, sin)
	SEC_EXPORT_MATH_FUNCTION(Cos, cos)
	SEC_EXPORT_MATH_FUNCTION(Tan, tan)
	SEC_EXPORT_MATH_FUNCTION(ASin, asin)
	SEC_EXPORT_MATH_FUNCTION(ACos, acos)
	SEC_EXPORT_MATH_FUNCTION(ATan, atan)
	SEC_EXPORT_MATH_FUNCTION(ATan2, atan2)

	SEC_EXPORT_MATH_FUNCTION(Sinh, sinh)
	SEC_EXPORT_MATH_FUNCTION(Cosh, cosh)
	SEC_EXPORT_MATH_FUNCTION(Tanh, tanh)
	SEC_EXPORT_MATH_FUNCTION(ASinh, asinh)
	SEC_EXPORT_MATH_FUNCTION(ACosh, acosh)
	SEC_EXPORT_MATH_FUNCTION(ATanh, atanh)

#undef SEC_EXPORT_RUNTIME_ONLY_MATH_FUNCTION
#undef SEC_EXPORT_CONSTEXPR_MATH_FUNCTION_WITH_FALLBACK
#undef SEC_EXPORT_MATH_FUNCTION

	template <typename Arg>
	constexpr SEC_FORCE_INLINE auto Abs(Arg&& arg) noexcept
	        -> std::enable_if_t<std::is_arithmetic_v<std::decay_t<Arg>>, std::decay_t<Arg>>
	{
		if constexpr (std::is_unsigned_v<std::decay_t<Arg>>)
		{
			return arg;
		}
		else
		{
			return arg < 0 ? -arg : arg;
		}
	}

	template <typename Arg>
	constexpr SEC_FORCE_INLINE auto SignBit(Arg&& arg) noexcept
	        -> std::enable_if_t<std::is_integral_v<std::decay_t<Arg>>, int>
	{
		return arg < 0 ? 1 : 0;
	}

	template <typename Arg>
	SEC_MATH_CORE_CONDITIONAL_CONSTEXPR SEC_FORCE_INLINE auto SignBit(Arg&& arg) noexcept(noexcept(
	        std::signbit(std::forward<Arg>(arg)))) -> std::enable_if_t<!std::is_integral_v<std::decay_t<Arg>>, int>
	{
#if defined(SEC_IF_CONSTEVAL) && __has_include(<gcem.hpp>)
		SEC_IF_CONSTEVAL
		{
			return gcem::signbit(std::forward<Arg>(arg));
		}
#endif

		return std::signbit(std::forward<Arg>(arg));
	}

	template <typename Scalar>
	constexpr SEC_FORCE_INLINE int Sign(const Scalar& scalar) noexcept
	{
		return scalar == 0 ? 0 : (scalar > 0 ? 1 : -1);
	}

	template <typename To, typename From>
	constexpr SEC_FORCE_INLINE auto CopySignToTheLeft(To&& to, From&& from) noexcept
	        -> std::enable_if_t<std::is_integral_v<std::decay_t<To>>, std::decay_t<To>>
	{
		return SignBit(from) == SignBit(to) ? to : -to;
	}

	template <typename To, typename From>
	SEC_MATH_CORE_CONDITIONAL_CONSTEXPR SEC_FORCE_INLINE auto CopySignToTheLeft(To&& to, From&& from) noexcept(
	        noexcept(std::copysign(to, from)))
	        -> std::enable_if_t<!std::is_integral_v<std::decay_t<To>>, decltype(std::copysign(to, from))>
	{
#if defined(SEC_IF_CONSTEVAL) && __has_include(<gcem.hpp>)
		SEC_IF_CONSTEVAL
		{
			return gcem::copysign(to, from);
		}
#endif
		return std::copysign(to, from);
	}

	template <typename Real>
	constexpr SEC_FORCE_INLINE Real Re(const std::complex<Real>& complex) noexcept
	{
		return complex.real();
	}

	template <typename Real>
	constexpr SEC_FORCE_INLINE Real Im(const std::complex<Real>& complex) noexcept
	{
		return complex.imag();
	}

	template <typename Real>
	constexpr SEC_FORCE_INLINE std::complex<Real> Conj(const std::complex<Real>& complex) noexcept
	{
		return {Re(complex), -Im(complex)};
	}

	template <typename Real>
	SEC_MATH_CORE_CONDITIONAL_CONSTEXPR SEC_FORCE_INLINE Real
	Arg(const std::complex<Real>& complex) noexcept(noexcept(ATan2(Real{}, Real{})))
	{
		if (Abs(Im(complex)) < std::numeric_limits<Real>::epsilon())
		{
			return Re(complex) < 0 ? Constant::Pi<Real> : 0;
		}

		return ATan2(Im(complex), Re(complex));
	}

	template <typename Real>
	constexpr SEC_FORCE_INLINE std::enable_if_t<std::is_arithmetic_v<std::decay_t<Real>>, Real> Re(
	        const Real& real) noexcept
	{
		return real;
	}

	template <typename Real>
	constexpr SEC_FORCE_INLINE std::enable_if_t<std::is_arithmetic_v<std::decay_t<Real>>, Real> Im(const Real&) noexcept
	{
		return 0;
	}

	template <typename Real>
	constexpr SEC_FORCE_INLINE std::enable_if_t<std::is_arithmetic_v<std::decay_t<Real>>, Real> Conj(
	        const Real& real) noexcept
	{
		return real;
	}

	template <typename Real>
	constexpr SEC_FORCE_INLINE std::enable_if_t<std::is_arithmetic_v<std::decay_t<Real>>, Real> Arg(
	        const Real& real) noexcept
	{
		return real >= 0 ? 0 : Constant::Pi<Real>;
	}

	template <typename Real>
	constexpr SEC_FORCE_INLINE std::enable_if_t<std::is_arithmetic_v<std::decay_t<Real>>, Real> SquaredNorm(
	        const Real& real) noexcept
	{
		return real * real;
	}

	template <typename Real>
	SEC_MATH_CORE_CONDITIONAL_CONSTEXPR SEC_FORCE_INLINE Real SquaredNorm(const std::complex<Real>& complex) noexcept
	{
		return SquaredNorm(Re(complex)) + SquaredNorm(Im(complex));
	}

	template <typename Scalar>
	SEC_MATH_CORE_CONDITIONAL_CONSTEXPR SEC_FORCE_INLINE auto Norm(const Scalar& scalar) noexcept(
	        noexcept(Sqrt(SquaredNorm(scalar))))
	        -> std::enable_if_t<!std::is_arithmetic_v<std::decay_t<Scalar>>, decltype(Sqrt(SquaredNorm(scalar)))>
	{
		return Sqrt(SquaredNorm(scalar));
	}

	template <typename Scalar>
	constexpr SEC_FORCE_INLINE auto Norm(const Scalar& scalar) noexcept(noexcept(Abs(scalar)))
	        -> std::enable_if_t<std::is_arithmetic_v<std::decay_t<Scalar>>, Scalar>
	{
		return Abs(scalar);
	}

	template <typename Arg>
	SEC_MATH_CORE_CONDITIONAL_CONSTEXPR SEC_FORCE_INLINE auto Abs(Arg&& arg) noexcept(
	        noexcept(std::abs(std::forward<Arg>(arg))))
	        -> std::enable_if_t<!std::is_arithmetic_v<std::decay_t<Arg>>, decltype(std::abs(std::forward<Arg>(arg)))>
	{
#if defined(SEC_IF_CONSTEVAL) && __has_include(<gcem.hpp>)
		SEC_IF_CONSTEVAL
		{
			return Norm(arg);
		}
#endif

		return std::abs(std::forward<Arg>(arg));
	}

	template <typename Scalar>
	constexpr SEC_FORCE_INLINE decltype(auto) Identity(Scalar&& scalar) noexcept
	{
		return std::forward<Scalar>(scalar);
	}

	template <typename Integer>
	constexpr SEC_FORCE_INLINE bool IsOdd(const Integer n) noexcept
	{
		static_assert(std::is_integral_v<std::decay_t<Integer>>);
		return n & 1;
	}

	template <typename Integer>
	constexpr SEC_FORCE_INLINE bool IsEven(const Integer n) noexcept
	{
		static_assert(std::is_integral_v<std::decay_t<Integer>>);
		return !IsOdd(n);
	}

	template <typename Integer>
	constexpr SEC_FORCE_INLINE bool IsPowerOfTwo(const Integer n) noexcept
	{
		static_assert(std::is_integral_v<std::decay_t<Integer>>);
		// ReSharper disable once CppRedundantParentheses
		return n > 0 && (n & (n - 1)) == 0;
	}

	template <typename Scalar, typename Exponent>
	constexpr SEC_FORCE_INLINE std::enable_if_t<std::is_integral_v<Exponent>, Scalar>  //
	MinusOneToThePowerOf(const Exponent exponent) noexcept
	{
		return exponent % 2 == 0 ? 1 : -1;
	}

	// ReSharper disable once CppTemplateParameterNeverUsed
	template <typename Comp, typename Arg>
	constexpr SEC_FORCE_INLINE decltype(auto) Extrema(Arg&& arg) noexcept
	{
		return std::forward<Arg>(arg);
	}

	template <typename Comp, typename Arg0, typename Arg1, typename... Args>
	constexpr SEC_FORCE_INLINE decltype(auto) Extrema(Arg0&& arg0, Arg1&& arg1, Args&&... args) noexcept
	{
		decltype(auto) rhs = [&]() noexcept -> decltype(auto)
		{
			if constexpr (sizeof...(Args) == 0)
			{
				return std::forward<Arg1>(arg1);
			}
			else
			{
				return Extrema<Comp>(std::forward<Arg1>(arg1), std::forward<Args>(args)...);
			}
		}();

		using Rhs = decltype(rhs);

		if constexpr (std::is_lvalue_reference_v<Arg0> && std::is_lvalue_reference_v<Rhs> && std::is_same_v<Arg0, Rhs>
		              && !std::is_const_v<Arg0>)
		{
			return Comp{}(arg0, rhs) ? std::forward<Arg0>(arg0) : std::forward<Rhs>(rhs);
		}
		else
		{
			using Result = std::common_type_t<std::decay_t<Arg0>, std::decay_t<Rhs>>;
			return Comp{}(arg0, rhs) ? static_cast<Result>(arg0) : static_cast<Result>(rhs);
		}
	}

	template <typename... Args>
	constexpr SEC_FORCE_INLINE decltype(auto) Max(Args&&... args) noexcept
	{
		return Extrema<std::greater_equal<>>(std::forward<Args>(args)...);
	}

	template <typename... Args>
	constexpr SEC_FORCE_INLINE decltype(auto) Min(Args&&... args) noexcept
	{
		return Extrema<std::less_equal<>>(std::forward<Args>(args)...);
	}

	template <typename Scalar>
	constexpr SEC_FORCE_INLINE auto Cbrt(Scalar&& arg) noexcept(noexcept(std::cbrt(std::forward<Scalar>(arg))))
	        -> decltype(std::cbrt(std::forward<Scalar>(arg)))
	{
#if defined(SEC_IF_CONSTEVAL) && __has_include(<gcem.hpp>)
		SEC_IF_CONSTEVAL
		{
			return CopySignToTheLeft(gcem::pow(Abs(arg), static_cast<Scalar>(1) / static_cast<Scalar>(3)), arg);
		}
#endif
		return std::cbrt(std::forward<Scalar>(arg));
	}

	template <typename... Scalars>
	SEC_MATH_CORE_CONDITIONAL_CONSTEXPR SEC_FORCE_INLINE auto Hypotenuse(Scalars&&... scalars)  //
	        noexcept(noexcept(Sqrt(((scalars * scalars) + ...))))
	{
		if constexpr (sizeof...(Scalars) == 2)
		{
#if defined(SEC_IF_CONSTEVAL) && __has_include(<gcem.hpp>)
			SEC_IF_CONSTEVAL
			{
				return gcem::hypot(std::forward<Scalars>(scalars)...);
			}
#endif

			return std::hypot(std::forward<Scalars>(scalars)...);
		}
		else
		{
			return Sqrt((SquaredNorm(scalars) + ...));
		}
	}

	//----------------------------------------------------------------------------------------------------------------//

	template <typename Scalar>
#if defined(SEC_IF_CONSTEVAL)
	constexpr
#endif
	        Scalar FusedMultiplyAdd(const Scalar x,
	                                const Scalar y,
	                                const Scalar z) noexcept(noexcept(std::fma(x, y, z)))
	{
#if defined(SEC_IF_CONSTEVAL)
		SEC_IF_CONSTEVAL
		{
			return x * y + z;
		}
#endif

		return std::fma(x, y, z);
	}

	template <typename T, typename Compare>
	constexpr const T& Clamp(const T& v, const T& low, const T& high, Compare comp) noexcept(noexcept(comp(low, high)))
	{
		return comp(v, low) ? low : comp(high, v) ? high : v;
	}

	template <typename T>
	constexpr const T& Clamp(const T& v, const T& low, const T& high) noexcept(noexcept(std::less<>{}(low, high)))
	{
		return Clamp(v, low, high, std::less<>{});
	}

	template <typename T>
	constexpr T MidPoint(const T a, const T b) noexcept
	{
		return a + (b - a) / 2;
	}

	template <typename T, typename FloatingPoint>
	constexpr T LinearInterpolation(const T a, const T b, const FloatingPoint t) noexcept
	{
		return Clamp(static_cast<T>(a + t * (b - a)), Min(a, b), Max(a, b));
	}

	//----------------------------------------------------------------------------------------------------------------//

	template <typename... Scalars>
	constexpr SEC_FORCE_INLINE auto Sum(Scalars&&... scalars) noexcept(noexcept((scalars + ...)))
	{
		return (scalars + ...);
	}

	template <typename... Scalars>
	constexpr SEC_FORCE_INLINE auto Prod(Scalars&&... scalars) noexcept(noexcept((scalars * ...)))
	{
		return (scalars * ...);
	}

	template <typename... Scalars>
	constexpr auto ArithmeticMean(Scalars&&... scalars) noexcept(
	        noexcept((scalars + ...) / static_cast<decltype((scalars + ...))>(sizeof...(Scalars))))
	{
		static_assert(sizeof...(Scalars) > 0);
		return (scalars + ...) / static_cast<decltype((scalars + ...))>(sizeof...(Scalars));
	}

	template <typename... Scalars>
	constexpr auto QuadraticMean(Scalars&&... scalars) noexcept(
	        noexcept(Sqrt((SquaredNorm(scalars) + ...)
	                      / static_cast<decltype(Sqrt((SquaredNorm(scalars) + ...)))>(sizeof...(Scalars)))))
	{
		static_assert(sizeof...(Scalars) > 0);
		return Sqrt((SquaredNorm(scalars) + ...)
		            / static_cast<decltype(Sqrt((SquaredNorm(scalars) + ...)))>(sizeof...(Scalars)));
	}

	template <typename... Scalars>
	constexpr auto GeometricMean(Scalars&&... scalars) noexcept(noexcept(Sqrt((scalars * ...))))
	{
		static_assert(sizeof...(Scalars) > 0);
		assert(((scalars >= 0) && ...) && "not validation for negatives");

		if constexpr (sizeof...(Scalars) == 1)
		{
			return Abs((scalars * ...));
		}
		else if constexpr (sizeof...(Scalars) == 2)
		{
			return Sqrt(Abs((scalars * ...)));
		}
		else if constexpr (sizeof...(Scalars) == 3)
		{
			return Cbrt(Abs((scalars * ...)));
		}
		else
		{
			return Pow(Abs((scalars * ...)), static_cast<decltype((scalars * ...))>(1) / sizeof...(Scalars));
		}
	}

	template <typename... Scalars>
	constexpr auto HarmonicMean(Scalars&&... scalars) noexcept(noexcept(1 / ((1 / scalars) + ...)))
	{
		static_assert(sizeof...(Scalars) > 0);
		using Scalar = decltype((scalars + ...));
		using Output = std::conditional_t<(std::is_integral_v<std::decay_t<Scalars>> && ...), double, Scalar>;
		assert(((scalars > 0) && ...) && "not validation for negatives and zeros");

		return static_cast<Output>(sizeof...(Scalars)) / ((1 / static_cast<Output>(scalars)) + ...);
	}

	//----------------------------------------------------------------------------------------------------------------//

	inline constexpr auto Deg2Rad = [](const auto& deg) constexpr noexcept
	{
		using Scalar = std::decay_t<decltype(deg)>;
		using Output = std::conditional_t<std::is_integral_v<Scalar>, double, Scalar>;
		return deg * Constant::Pi<Output> / 180;
	};

	inline constexpr auto Rad2Deg = [](const auto& deg) constexpr noexcept
	{
		using Scalar = std::decay_t<decltype(deg)>;
		using Output = std::conditional_t<std::is_integral_v<Scalar>, double, Scalar>;
		return deg * 180 / Constant::Pi<Output>;
	};

	// special functions, e.g., erf, will be exposed under Math/Special.hpp
}

#undef SEC_MATH_CORE_CONDITIONAL_CONSTEXPR
