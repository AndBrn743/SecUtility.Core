// SPDX-License-Identifier: MIT

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/Hoppy.hpp>

#include <Eigen/Core>

#include <cstddef>

struct CountingScalar
{
	double Value{};
	static std::size_t Additions;
	CountingScalar() = default;
	CountingScalar(double value) : Value(value) {}
	CountingScalar& operator+=(const CountingScalar& other)
	{
		++Additions;
		Value += other.Value;
		return *this;
	}
};

std::size_t CountingScalar::Additions = 0;

CountingScalar operator+(CountingScalar lhs, const CountingScalar& rhs)
{
	lhs += rhs;
	return lhs;
}

namespace Eigen
{
	template <>
	struct NumTraits<CountingScalar> : GenericNumTraits<CountingScalar>
	{
		using Real = CountingScalar;
		using NonInteger = CountingScalar;
		using Nested = CountingScalar;
		using Literal = CountingScalar;
	};
}  // namespace Eigen

namespace
{
	class EigenAllocationGuard
	{
	public:
		EigenAllocationGuard() { Eigen::internal::set_is_malloc_allowed(false); }
		~EigenAllocationGuard() { Eigen::internal::set_is_malloc_allowed(true); }
	};
}  // namespace

TEST_CASE("blockwise addition touches only stored coefficients without allocating")
{
	Hoppy::BlockDiagonalMatrix<CountingScalar> lhs{1, 1, 1, 1};
	Hoppy::BlockDiagonalMatrix<CountingScalar> rhs{1, 1, 1, 1};
	Hoppy::BlockDiagonalMatrix<CountingScalar> result{1, 1, 1, 1};
	lhs.setConstant(CountingScalar{1.0});
	rhs.setConstant(CountingScalar{2.0});
	CountingScalar::Additions = 0;
	{
		EigenAllocationGuard guard;
		result = lhs + rhs;
	}
	REQUIRE(CountingScalar::Additions == static_cast<std::size_t>(lhs.storedSize()));
	REQUIRE(CountingScalar::Additions < static_cast<std::size_t>(lhs.size()));
}

TEST_CASE("preallocated dense evaluation and dense interop create no hidden allocation")
{
	Hoppy::BlockDiagonalMatrix<double> block{1, 2, 1};
	block.setConstant(2.0);
	Eigen::MatrixXd destination(block.rows(), block.cols());
	Eigen::MatrixXd dense = Eigen::MatrixXd::Random(block.rows(), block.cols());
	{
		EigenAllocationGuard guard;
		block.evalTo(destination);
		destination = dense + block;
	}
	Hoppy::Test::requireApprox(destination, dense + block.toDense());
}
