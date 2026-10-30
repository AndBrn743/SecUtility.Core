// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#include <SecUtility/Collection/Sequence.hpp>
#include <catch2/catch_test_macros.hpp>
#include <complex>
#include <stdexcept>
#include <type_traits>
#include <utility>

using namespace SecUtility;

namespace
{
	// Select one potentially throwing operation independently of the others.
	enum class Failure { None, Default, Copy, Move, Assignment, Addition, Multiplication };

	template <Failure F>
	struct Scalar
	{
		static inline bool Throw = false;

		template <Failure Operation>
		static void MaybeThrow() noexcept(F != Operation)
		{
			if constexpr (F == Operation)
			{
				if (Throw) throw std::runtime_error("scalar operation failed");
			}
		}

		Scalar() noexcept(F != Failure::Default) { MaybeThrow<Failure::Default>(); }
		explicit Scalar(int) noexcept {}
		Scalar(const Scalar&) noexcept(F != Failure::Copy) { MaybeThrow<Failure::Copy>(); }
		Scalar(Scalar&&) noexcept(F != Failure::Move) { MaybeThrow<Failure::Move>(); }
		Scalar& operator=(Scalar&&) noexcept(F != Failure::Assignment)
		{
			MaybeThrow<Failure::Assignment>();
			return *this;
		}
		friend Scalar operator+(const Scalar&, const Scalar&) noexcept(F != Failure::Addition && F != Failure::Default)
		{
			MaybeThrow<Failure::Addition>();
			return {};
		}
		friend Scalar operator*(const Scalar&, const Scalar&) noexcept(F != Failure::Multiplication && F != Failure::Default)
		{
			MaybeThrow<Failure::Multiplication>();
			return {};
		}
	};

	template <Failure F, std::size_t N>
	constexpr bool ArithmeticNoexcept = noexcept(MakeArithmeticSequenceArray<N>(
	        std::declval<Scalar<F>>(), std::declval<Scalar<F>>()));

	template <Failure F, std::size_t N>
	constexpr bool GeometricNoexcept = noexcept(MakeGeometricSequenceArray<N>(
	        std::declval<Scalar<F>>(), std::declval<Scalar<F>>()));

	template <Failure F>
	void CheckExceptionPropagation()
	{
		Scalar<F>::Throw = true;
		CHECK_THROWS_AS(MakeArithmeticSequenceArray<3>(Scalar<F>{0}, Scalar<F>{0}), std::runtime_error);
		CHECK_THROWS_AS(MakeGeometricSequenceArray<3>(Scalar<F>{0}, Scalar<F>{0}), std::runtime_error);
		Scalar<F>::Throw = false;
	}
}

TEST_CASE("Arithmetic sequence values and constexpr evaluation", "[Sequence]")
{
	constexpr auto ascending = MakeArithmeticSequenceArray<4>(2, 3);
	STATIC_REQUIRE(ascending[0] == 2 && ascending[1] == 5 && ascending[2] == 8 && ascending[3] == 11);
	STATIC_REQUIRE(std::is_same_v<std::remove_cv_t<decltype(ascending)>, std::array<int, 4>>);
	CHECK(MakeArithmeticSequenceArray<4>(5, -2) == std::array<int, 4>{5, 3, 1, -1});
	CHECK(MakeArithmeticSequenceArray<3>(7, 0) == std::array<int, 3>{7, 7, 7});
	CHECK(MakeArithmeticSequenceArray<3>(0.5, 0.25) == std::array<double, 3>{0.5, 0.75, 1.0});
	constexpr auto single = MakeArithmeticSequenceArray<1>(9, -3);
	STATIC_REQUIRE(single.size() == 1 && single[0] == 9);
}

TEST_CASE("Geometric sequence values and constexpr evaluation", "[Sequence]")
{
	constexpr auto positive = MakeGeometricSequenceArray<4>(2, 3);
	STATIC_REQUIRE(positive[0] == 2 && positive[1] == 6 && positive[2] == 18 && positive[3] == 54);
	STATIC_REQUIRE(std::is_same_v<std::remove_cv_t<decltype(positive)>, std::array<int, 4>>);
	CHECK(MakeGeometricSequenceArray<4>(2, -3) == std::array<int, 4>{2, -6, 18, -54});
	CHECK(MakeGeometricSequenceArray<3>(7, 0) == std::array<int, 3>{7, 0, 0});
	CHECK(MakeGeometricSequenceArray<3>(7, 1) == std::array<int, 3>{7, 7, 7});
	CHECK(MakeGeometricSequenceArray<3>(0, 2) == std::array<int, 3>{0, 0, 0});
	CHECK(MakeGeometricSequenceArray<4>(1.0, 0.5) == std::array<double, 4>{1.0, 0.5, 0.25, 0.125});
	constexpr auto single = MakeGeometricSequenceArray<1>(9, -3);
	STATIC_REQUIRE(single.size() == 1 && single[0] == 9);
}

TEST_CASE("Sequences support complex scalars", "[Sequence]")
{
	using Complex = std::complex<double>;
	CHECK(MakeArithmeticSequenceArray<3>(Complex{1, 2}, Complex{2, -1})
	      == std::array<Complex, 3>{Complex{1, 2}, Complex{3, 1}, Complex{5, 0}});
	CHECK(MakeGeometricSequenceArray<3>(Complex{1, 1}, Complex{0, 1})
	      == std::array<Complex, 3>{Complex{1, 1}, Complex{-1, 1}, Complex{-1, -1}});
}

TEST_CASE("Sequence noexcept accounts for construction and assignment", "[Sequence]")
{
	STATIC_REQUIRE(noexcept(MakeArithmeticSequenceArray<3>(1, 2)));
	STATIC_REQUIRE(noexcept(MakeGeometricSequenceArray<3>(1.0, 2.0)));
	STATIC_REQUIRE(ArithmeticNoexcept<Failure::None, 3>);
	STATIC_REQUIRE(GeometricNoexcept<Failure::None, 3>);
	STATIC_REQUIRE_FALSE(ArithmeticNoexcept<Failure::Default, 3>);
	STATIC_REQUIRE_FALSE(GeometricNoexcept<Failure::Default, 3>);
	STATIC_REQUIRE_FALSE(ArithmeticNoexcept<Failure::Copy, 3>);
	STATIC_REQUIRE_FALSE(GeometricNoexcept<Failure::Copy, 3>);
	STATIC_REQUIRE_FALSE(ArithmeticNoexcept<Failure::Move, 3>);
	STATIC_REQUIRE_FALSE(GeometricNoexcept<Failure::Move, 3>);
	STATIC_REQUIRE_FALSE(ArithmeticNoexcept<Failure::Assignment, 3>);
	STATIC_REQUIRE_FALSE(GeometricNoexcept<Failure::Assignment, 3>);
}

TEST_CASE("Sequence noexcept checks the relevant arithmetic operator", "[Sequence]")
{
	STATIC_REQUIRE_FALSE(ArithmeticNoexcept<Failure::Addition, 3>);
	STATIC_REQUIRE(GeometricNoexcept<Failure::Addition, 3>);
	STATIC_REQUIRE(ArithmeticNoexcept<Failure::Multiplication, 3>);
	STATIC_REQUIRE_FALSE(GeometricNoexcept<Failure::Multiplication, 3>);
}

TEST_CASE("Single element sequences skip default construction arithmetic and assignment", "[Sequence]")
{
	STATIC_REQUIRE(ArithmeticNoexcept<Failure::Default, 1>);
	STATIC_REQUIRE(GeometricNoexcept<Failure::Default, 1>);
	STATIC_REQUIRE(ArithmeticNoexcept<Failure::Addition, 1>);
	STATIC_REQUIRE(GeometricNoexcept<Failure::Multiplication, 1>);
	STATIC_REQUIRE(ArithmeticNoexcept<Failure::Assignment, 1>);
	STATIC_REQUIRE(GeometricNoexcept<Failure::Assignment, 1>);

	const Scalar<Failure::Default> initial;
	Scalar<Failure::Default>::Throw = true;
	CHECK_NOTHROW(MakeArithmeticSequenceArray<1>(initial, initial));
	CHECK_NOTHROW(MakeGeometricSequenceArray<1>(initial, initial));
	Scalar<Failure::Default>::Throw = false;
}

TEST_CASE("Sequence scalar exceptions reach the caller", "[Sequence]")
{
	CheckExceptionPropagation<Failure::Default>();
	CheckExceptionPropagation<Failure::Copy>();
	CheckExceptionPropagation<Failure::Assignment>();
	Scalar<Failure::Addition>::Throw = true;
	CHECK_THROWS_AS(MakeArithmeticSequenceArray<3>(Scalar<Failure::Addition>{}, Scalar<Failure::Addition>{}),
	                std::runtime_error);
	Scalar<Failure::Addition>::Throw = false;
	Scalar<Failure::Multiplication>::Throw = true;
	CHECK_THROWS_AS(MakeGeometricSequenceArray<3>(Scalar<Failure::Multiplication>{}, Scalar<Failure::Multiplication>{}),
	                std::runtime_error);
	Scalar<Failure::Multiplication>::Throw = false;
}
