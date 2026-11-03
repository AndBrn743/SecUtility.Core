// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#include <SecUtility/Math/LobattoInterpolation.hpp>
#include <array>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <type_traits>

TEMPLATE_TEST_CASE_SIG("Lobatto interval interpolation returns exact sample vectors at nodes",
                       "[Interpolation][ChebyshevLobatto]",
                       ((typename Scalar, int Degree), Scalar, Degree),
                       (float, 1), (float, 3), (float, 8), (float, 12),
                       (double, 1), (double, 3), (double, 8), (double, 12))
{
	const auto& nodes = SecUtility::Math::ChebyshevLobattoNodes<Scalar, Degree>();
	Eigen::Matrix<Scalar, Degree + 1, 3, Eigen::RowMajor> samples;
	for (int i = 0; i <= Degree; ++i)
	{
		samples(i, 0) = static_cast<Scalar>(i + 2);
		samples(i, 1) = static_cast<Scalar>(-i - 3);
		samples(i, 2) = static_cast<Scalar>(i * i + 1);
	}
	CHECK(nodes[0] == Scalar{-1});
	CHECK(nodes.end()[-1] == Scalar{1});
	for (int i = 0; i <= Degree; ++i)
	{
		CAPTURE(i);
		if (i > 0)
		{
			CHECK(nodes[i] > nodes[i - 1]);
		}
		const auto actual = SecUtility::Math::InterpolateChebyshevLobatto<Degree>(nodes[i], samples);
		for (int j = 0; j < 3; ++j)
		{
			CHECK(actual[j] == samples(i, j));
		}
	}
}

TEMPLATE_TEST_CASE_SIG("Lobatto interval interpolation reproduces polynomials through its degree",
                       "[Interpolation][ChebyshevLobatto]",
                       ((typename Scalar, int Degree), Scalar, Degree),
                       (float, 1), (float, 3), (float, 5), (float, 8), (float, 12),
                       (double, 1), (double, 3), (double, 5), (double, 8), (double, 12))
{
	const auto& nodes = SecUtility::Math::ChebyshevLobattoNodes<Scalar, Degree>();
	// Distinct columns exercise vector accumulation; no quadrature rules are used.
	Eigen::Matrix<Scalar, Degree + 1, 2, Eigen::RowMajor> samples;
	const Scalar tolerance = Scalar{128} * std::numeric_limits<Scalar>::epsilon();
	for (int power = 0; power <= Degree; ++power)
	{
		for (int i = 0; i <= Degree; ++i)
		{
			const Scalar value = std::pow(nodes[i], power);
			samples(i, 0) = value;
			samples(i, 1) = Scalar{2} - value;
		}
		for (const Scalar x : {Scalar{-1}, std::nextafter(Scalar{-1}, Scalar{0}),
		                       Scalar{-0.83}, Scalar{-0.17}, Scalar{0}, Scalar{0.31}, Scalar{0.94},
		                       std::nextafter(Scalar{1}, Scalar{0}), Scalar{1}})
		{
			CAPTURE(power, x);
			const auto actual = SecUtility::Math::InterpolateChebyshevLobatto<Degree>(x, samples);
			const Scalar expected = std::pow(x, power);
			CHECK(std::abs(actual[0] - expected) <= tolerance);
			CHECK(std::abs(actual[1] - (Scalar{2} - expected)) <= tolerance);
		}
	}
}

TEMPLATE_TEST_CASE("Lobatto interpolation accepts matrix layouts, expressions, and strided views",
                   "[Interpolation][ChebyshevLobatto]", float, double)
{
	using Scalar = TestType;
	constexpr int Degree = 3;
	const auto& nodes = SecUtility::Math::ChebyshevLobattoNodes<Scalar, Degree>();
	Eigen::Matrix<Scalar, 4, 2, Eigen::RowMajor> samples;
	for (int i = 0; i < 4; ++i)
	{
		const Scalar x = nodes[i];
		samples(i, 0) = Scalar{1} + Scalar{2} * x - x * x * x;
		samples(i, 1) = Scalar{-2} + x * x;
	}
	const Scalar tolerance = Scalar{64} * std::numeric_limits<Scalar>::epsilon();
	const auto check = [&](const auto& input, const Scalar scale = Scalar{1})
	{
		for (const Scalar x : {Scalar{-1}, Scalar{-0.37}, Scalar{0.62}, Scalar{1}})
		{
			const auto actual = SecUtility::Math::InterpolateChebyshevLobatto<Degree>(x, input);
			REQUIRE(actual.size() == 2);
			CHECK(std::abs(actual[0] - scale * (Scalar{1} + Scalar{2} * x - x * x * x)) <= tolerance);
			CHECK(std::abs(actual[1] - scale * (Scalar{-2} + x * x)) <= tolerance);
		}
	};

	check(samples);
	const Eigen::Matrix<Scalar, 4, 2> columnMajor = samples;
	check(columnMajor);
	const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> dynamic = samples;
	check(dynamic);

	Eigen::Matrix<Scalar, 6, 5, Eigen::RowMajor> padded =
	        Eigen::Matrix<Scalar, 6, 5, Eigen::RowMajor>::Constant(Scalar{99});
	padded.template block<4, 2>(1, 2) = samples;
	check(padded.template block<4, 2>(1, 2));
	check(padded.block(1, 2, 4, 2));

	// Both inner and outer strides differ from those of a packed 4x2 matrix.
	std::array<Scalar, 48> storage{};
	using Stride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
	Eigen::Map<Eigen::Matrix<Scalar, 4, 2, Eigen::RowMajor>, Eigen::Unaligned, Stride>
	        strided(storage.data(), Stride{12, 2});
	strided = samples;
	check(strided);

	const Eigen::Matrix<Scalar, 2, 4> transposed = samples.transpose();
	check(transposed.transpose());
	check(samples + samples, Scalar{2});

	// One sample column produces an owning scalar-valued interpolation result.
	const auto scalarResult = SecUtility::Math::InterpolateChebyshevLobatto<Degree>(Scalar{0}, samples.col(0));
	STATIC_REQUIRE(decltype(scalarResult)::RowsAtCompileTime == 1);
	CHECK(std::abs(scalarResult[0] - Scalar{1}) <= tolerance);
}

TEMPLATE_TEST_CASE_SIG("Chebyshev-Lobatto runtime generators agree with shared fixed-degree data",
                       "[Interpolation][ChebyshevLobatto]",
                       ((typename Scalar, int Degree), Scalar, Degree),
                       (float, 1), (float, 2), (float, 5), (float, 12),
                       (double, 1), (double, 2), (double, 5), (double, 12),
                       (long double, 1), (long double, 2), (long double, 5), (long double, 12))
{
	using namespace SecUtility::Math;
	const auto nodes = ChebyshevLobattoNodes<Scalar>(Degree);
	const auto weights = ChebyshevLobattoBarycentricWeights<Scalar>(Degree);
	const auto& fixedNodes = ChebyshevLobattoNodes<Scalar, Degree>();
	const auto& fixedWeights = ChebyshevLobattoBarycentricWeights<Scalar, Degree>();
	STATIC_REQUIRE(std::is_const_v<std::remove_reference_t<decltype(fixedNodes)>>);
	STATIC_REQUIRE(std::is_const_v<std::remove_reference_t<decltype(fixedWeights)>>);
	CHECK(&fixedNodes == &ChebyshevLobattoNodes<Scalar, Degree>());
	CHECK(&fixedWeights == &ChebyshevLobattoBarycentricWeights<Scalar, Degree>());
	REQUIRE(nodes.size() == Degree + 1);
	REQUIRE(weights.size() == Degree + 1);
	const Scalar tolerance = Scalar{128} * std::numeric_limits<Scalar>::epsilon();
	for (int i = 0; i <= Degree; ++i)
	{
		CAPTURE(i);
		CHECK(nodes[i] == fixedNodes[i]);
		CHECK(weights[i] == fixedWeights[i]);
		CHECK(std::abs(nodes[i] + nodes[Degree - i]) <= tolerance);
		if (i > 0)
		{
			CHECK(nodes[i] > nodes[i - 1]);
		}
	}
	CHECK(nodes[0] == Scalar{-1});
	CHECK(nodes[Degree] == Scalar{1});
	if constexpr (Degree % 2 == 0)
	{
		CHECK(nodes[Degree / 2] == Scalar{0});
	}

	// For any interpolation nodes, w_i * product_{j!=i}(x_i-x_j)
	// must be constant. Check the closed-form weights against this independent
	// definition rather than repeating their alternating-sign formula.
	const auto weightedProduct = [&](const int i)
	{
		Scalar value = weights[i];
		for (int j = 0; j <= Degree; ++j)
		{
			if (j != i)
			{
				value *= nodes[i] - nodes[j];
			}
		}
		return value;
	};
	const Scalar reference = weightedProduct(0);
	for (int i = 1; i <= Degree; ++i)
	{
		CAPTURE(i);
		CHECK(std::abs(weightedProduct(i) / reference - Scalar{1}) <= tolerance);
	}
}

TEMPLATE_TEST_CASE("Chebyshev-Lobatto degree-one extrapolation returns an owning result",
                   "[Interpolation][ChebyshevLobatto]", float, double)
{
	using Scalar = TestType;
	Eigen::Matrix<Scalar, 2, 1> samples;
	samples << Scalar{-1}, Scalar{3};  // p(x) = 1 + 2*x
	const auto actual = SecUtility::Math::InterpolateChebyshevLobatto<1>(Scalar{2}, samples);
	samples.setZero();
	CHECK(std::abs(actual[0] - Scalar{5}) <= Scalar{16} * std::numeric_limits<Scalar>::epsilon());
}
