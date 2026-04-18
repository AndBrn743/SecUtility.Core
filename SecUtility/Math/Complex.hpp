// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <complex>
#include <type_traits>

// I know non-top-level std::enable_if_t is weird, but some compiler / STL implementation of std::common_type_t
// is very eager, and cause compiler to issue error before top-level std::enable_if_t is able to disable the template

template <typename Scalar, typename Integer>
std::complex<std::common_type_t<Scalar, std::enable_if_t<std::is_integral_v<Integer>, Integer>>>  //
        constexpr operator*(const std::complex<Scalar> z, const Integer i) noexcept
{
	return {z.real() * i, z.imag() * i};
}

template <typename Integer, typename Scalar>
std::complex<std::common_type_t<Scalar, std::enable_if_t<std::is_integral_v<Integer>, Integer>>>  //
        constexpr operator*(const Integer i, const std::complex<Scalar> z) noexcept
{
	return {z.real() * i, z.imag() * i};
}

template <typename Scalar, typename Integer>
std::complex<std::common_type_t<Scalar, std::enable_if_t<std::is_integral_v<Integer>, Integer>>>  //
        constexpr operator+(const std::complex<Scalar> z, const Integer i) noexcept
{
	return {z.real() + i, z.imag()};
}

template <typename Integer, typename Scalar>
std::complex<std::common_type_t<Scalar, std::enable_if_t<std::is_integral_v<Integer>, Integer>>>  //
        constexpr operator+(const Integer i, const std::complex<Scalar> z) noexcept
{
	return {z.real() + i, z.imag()};
}

template <typename Scalar, typename Integer>
std::complex<std::common_type_t<Scalar, std::enable_if_t<std::is_integral_v<Integer>, Integer>>>  //
        constexpr operator-(const std::complex<Scalar> z, const Integer i) noexcept
{
	return {z.real() - i, z.imag()};
}

template <typename Integer, typename Scalar>
std::complex<std::common_type_t<Scalar, std::enable_if_t<std::is_integral_v<Integer>, Integer>>>  //
        constexpr operator-(const Integer i, const std::complex<Scalar> z) noexcept
{
	return {i - z.real(), -z.imag()};
}

template <typename Scalar, typename Integer>
std::complex<std::common_type_t<Scalar, std::enable_if_t<std::is_integral_v<Integer>, Integer>>>  //
        constexpr operator/(const std::complex<Scalar> z, const Integer i) noexcept
{
	return {z.real() / i, z.imag() / i};
}

template <typename Integer, typename Scalar>
std::complex<std::common_type_t<Scalar, std::enable_if_t<std::is_integral_v<Integer>, Integer>>>  //
        constexpr operator/(const Integer i, const std::complex<Scalar> z) noexcept
{
	if constexpr (std::is_same_v<std::common_type_t<Scalar, Integer>, Integer>)  // prevent infinite recursion
	{
		const auto norm = z.real() * z.real() + z.imag() * z.imag();
		return {i * z.real() / norm, -i * z.imag() / norm};
	}
	else
	{
		return static_cast<std::common_type_t<Scalar, Integer>>(i) / z;
	}
}
