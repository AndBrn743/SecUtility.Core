// SPDX-License-Identifier: MIT

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/Hoppy.hpp>

#include <Eigen/Core>

#include <cstddef>

struct CountingScalar
{
	double Value{};
	static std::size_t Additions;
	static std::size_t Multiplications;
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
std::size_t CountingScalar::Multiplications = 0;

CountingScalar operator+(CountingScalar lhs, const CountingScalar& rhs)
{
	lhs += rhs;
	return lhs;
}

CountingScalar operator*(const CountingScalar& lhs, const CountingScalar& rhs)
{
	++CountingScalar::Multiplications;
	return lhs.Value * rhs.Value;
}

bool operator==(const CountingScalar& lhs, const CountingScalar& rhs)
{
	return lhs.Value == rhs.Value;
}

const CountingScalar& conj(const CountingScalar& value) { return value; }

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
	constexpr Eigen::Index BlockCount = 8;

	class EigenAllocationGuard
	{
	public:
		EigenAllocationGuard() { Eigen::internal::set_is_malloc_allowed(false); }
		~EigenAllocationGuard() { Eigen::internal::set_is_malloc_allowed(true); }
	};

	template <typename T>
	void setSequential(T& value)
	{
		for (Eigen::Index index = 0; index < value.storedSize(); ++index)
			value.data()[index] = CountingScalar{static_cast<double>(index + 1)};
	}

	void resetOperationCounts()
	{
		CountingScalar::Additions = 0;
		CountingScalar::Multiplications = 0;
	}
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

TEST_CASE("reductions scale with stored coefficients rather than logical matrix size")
{
	Hoppy::BlockDiagonalMatrix<CountingScalar> matrix{1, 1, 1, 1, 1, 1, 1, 1};
	setSequential(matrix);
	resetOperationCounts();

	const auto sum = matrix.sum();
	REQUIRE(sum.Value == 36.0);
	REQUIRE(CountingScalar::Additions == static_cast<std::size_t>(matrix.storedSize()));
	REQUIRE(CountingScalar::Additions < static_cast<std::size_t>(matrix.size()));
}

TEST_CASE("block products scale with the sum of block cubes")
{
	Hoppy::BlockDiagonalMatrix<CountingScalar> lhs{1, 1, 1, 1, 1, 1, 1, 1};
	Hoppy::BlockDiagonalMatrix<CountingScalar> rhs{1, 1, 1, 1, 1, 1, 1, 1};
	Hoppy::BlockDiagonalMatrix<CountingScalar> result{1, 1, 1, 1, 1, 1, 1, 1};
	lhs.setConstant(CountingScalar{2.0});
	rhs.setConstant(CountingScalar{3.0});
	resetOperationCounts();

	result = lhs * rhs;
	REQUIRE(CountingScalar::Multiplications >= static_cast<std::size_t>(BlockCount));
	REQUIRE(CountingScalar::Multiplications <= 2 * static_cast<std::size_t>(BlockCount));
	REQUIRE(CountingScalar::Multiplications
	        < static_cast<std::size_t>(lhs.rows() * lhs.rows() * lhs.rows()));
}

TEST_CASE("block matvec scales with the sum of block squares")
{
	Hoppy::BlockDiagonalMatrix<CountingScalar> matrix{1, 1, 1, 1, 1, 1, 1, 1};
	Hoppy::BlockVector<CountingScalar> vector{1, 1, 1, 1, 1, 1, 1, 1};
	matrix.setConstant(CountingScalar{2.0});
	vector.setConstant(CountingScalar{3.0});
	resetOperationCounts();

	const auto result = matrix * vector;
	REQUIRE(result[0](0).Value == 6.0);
	REQUIRE(CountingScalar::Multiplications >= static_cast<std::size_t>(BlockCount));
	REQUIRE(CountingScalar::Multiplications <= 2 * static_cast<std::size_t>(BlockCount));
	REQUIRE(CountingScalar::Multiplications
	        < static_cast<std::size_t>(matrix.rows() * matrix.rows()));
}

TEST_CASE("congruence evaluates independently within each block")
{
	Hoppy::BlockDiagonalMatrix<CountingScalar> matrix{1, 1, 1, 1, 1, 1, 1, 1};
	Hoppy::BlockDiagonalMatrix<CountingScalar> transform{1, 1, 1, 1, 1, 1, 1, 1};
	Hoppy::BlockDiagonalMatrix<CountingScalar> result{1, 1, 1, 1, 1, 1, 1, 1};
	matrix.setConstant(CountingScalar{2.0});
	transform.setConstant(CountingScalar{3.0});
	resetOperationCounts();

	result = matrix.transformedBy(transform);
	REQUIRE(result[0](0, 0).Value == 18.0);
	REQUIRE(CountingScalar::Multiplications >= 2 * static_cast<std::size_t>(BlockCount));
	REQUIRE(CountingScalar::Multiplications <= 4 * static_cast<std::size_t>(BlockCount));
	REQUIRE(CountingScalar::Multiplications
	        < 2 * static_cast<std::size_t>(matrix.rows() * matrix.rows() * matrix.rows()));
}
