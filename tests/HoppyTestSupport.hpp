// SPDX-License-Identifier: MIT

#pragma once

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <Eigen/Core>

namespace Hoppy::Test
{
	template <typename Actual, typename Expected>
	void requireApprox(const Actual& actual, const Expected& expected)
	{
		REQUIRE(actual.rows() == expected.rows());
		REQUIRE(actual.cols() == expected.cols());
		REQUIRE(actual.isApprox(expected));
	}
}  // namespace Hoppy::Test
