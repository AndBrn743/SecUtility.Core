// SPDX-License-Identifier: MIT

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/Hoppy.hpp>

#include <Eigen/Core>

#include <complex>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

struct ThrowingScalar
{
	double Value{};
	static int AssignmentsRemaining;
	ThrowingScalar() = default;
	explicit ThrowingScalar(double value) : Value(value) {}
	ThrowingScalar(const ThrowingScalar&) = default;
	ThrowingScalar& operator=(const ThrowingScalar& other)
	{
		if (AssignmentsRemaining == 0)
			throw std::runtime_error("injected scalar assignment failure");
		if (AssignmentsRemaining > 0)
			--AssignmentsRemaining;
		Value = other.Value;
		return *this;
	}
};

int ThrowingScalar::AssignmentsRemaining = -1;

namespace Eigen
{
	template <>
	struct NumTraits<ThrowingScalar> : GenericNumTraits<ThrowingScalar>
	{
		using Real = ThrowingScalar;
		using NonInteger = ThrowingScalar;
		using Nested = ThrowingScalar;
		using Literal = ThrowingScalar;
	};
}  // namespace Eigen

namespace
{
	template <typename, template <typename...> class, typename...>
	struct is_detected : std::false_type
	{};
	template <template <typename...> class Operation, typename... Arguments>
	struct is_detected<std::void_t<Operation<Arguments...>>, Operation, Arguments...> : std::true_type
	{};
	template <template <typename...> class Operation, typename... Arguments>
	constexpr bool IsDetected = is_detected<void, Operation, Arguments...>::value;

	template <typename T> using unary_plus = decltype(+std::declval<T>());
	template <typename T> using unary_minus = decltype(-std::declval<T>());
	template <typename T> using transpose = decltype(std::declval<T>().transpose());
	template <typename T> using adjoint = decltype(std::declval<T>().adjoint());
	template <typename T> using conjugate = decltype(std::declval<T>().conjugate());
	template <typename T> using real = decltype(std::declval<T>().real());
	template <typename T> using imag = decltype(std::declval<T>().imag());
	template <typename T> using cast_float = decltype(std::declval<T>().template cast<float>());
	template <typename L, typename R> using plus = decltype(std::declval<L>() + std::declval<R>());
	template <typename L, typename R> using minus = decltype(std::declval<L>() - std::declval<R>());
	template <typename L, typename R> using product = decltype(std::declval<L>() * std::declval<R>());
	template <typename L, typename R> using quotient = decltype(std::declval<L>() / std::declval<R>());
	template <typename T> using diagonal = decltype(std::declval<T>().diagonal());
	template <typename T> using as_diagonal = decltype(std::declval<T>().asDiagonal());
	template <typename T> using inverse = decltype(std::declval<T>().inverse());
	template <typename L, typename R> using dot = decltype(std::declval<L>().dot(std::declval<R>()));
	template <typename T> using normalized = decltype(std::declval<T>().normalized());
	template <typename T> using normalize = decltype(std::declval<T>().normalize());
	template <typename L, typename R> using solve = decltype(std::declval<L>().solve(std::declval<R>()));
	template <typename L, typename R> using transformed = decltype(std::declval<L>().transformedBy(std::declval<R>()));
	template <typename L, typename R> using transform_in_place = decltype(std::declval<L>().transformBy(std::declval<R>()));
	template <typename MatrixScalar, typename TransformScalar>
	using congruence_scalar = typename Hoppy::Detail::congruence_result_scalar<
	        MatrixScalar, TransformScalar>::type;

	using BD = Hoppy::BlockDiagonalMatrix<double>;
	using BDi = Hoppy::BlockDiagonalMatrix<int>;
	using BV = Hoppy::BlockVector<double>;
	using BVR = Hoppy::BlockVector<double, Hoppy::Row>;
	using BVi = Hoppy::BlockVector<int>;
	using Dense = Eigen::MatrixXd;
	using FixedColumn = Eigen::Vector3d;
	using FixedRow = Eigen::RowVector3d;
	using BlockingInfo = std::vector<Eigen::Index>;
	using BDExpression = decltype(std::declval<const BD&>() + std::declval<const BD&>());
	using BVExpression = decltype(std::declval<const BV&>() + std::declval<const BV&>());

	static_assert(IsDetected<unary_plus, const BD&> && IsDetected<unary_minus, const BV&>);
	static_assert(IsDetected<transpose, const BD&> && IsDetected<adjoint, const BV&>);
	static_assert(IsDetected<conjugate, const BD&> && IsDetected<real, const BV&>);
	static_assert(IsDetected<imag, const BD&> && IsDetected<cast_float, const BV&>);
	static_assert(IsDetected<plus, const BD&, const BD&> && IsDetected<minus, const BV&, const BV&>);
	static_assert(!IsDetected<plus, const BD&, const BV&> && !IsDetected<plus, const BV&, const BVR&>);
	static_assert(IsDetected<product, const BD&, double> && IsDetected<product, double, const BV&>);
	static_assert(IsDetected<quotient, const BD&, double> && IsDetected<product, const BD&, const BD&>);
	static_assert(IsDetected<product, const BD&, const BV&> && IsDetected<product, const BVR&, const BD&>);
	static_assert(!IsDetected<product, const BD&, const BVR&> && !IsDetected<product, const BV&, const BD&>);
	static_assert(IsDetected<product, const BD&, const Dense&> && IsDetected<product, const Dense&, const BD&>);
	static_assert(IsDetected<product, const BD&, const FixedColumn&>
	              && IsDetected<product, const FixedRow&, const BD&>);
	static_assert(IsDetected<plus, const Dense&, const BD&> && IsDetected<minus, const BD&, const Dense&>);
	static_assert(IsDetected<diagonal, BD&> && IsDetected<diagonal, const BD&>);
	static_assert(IsDetected<diagonal, BD&&> && IsDetected<diagonal, BDExpression&&>);
	static_assert(IsDetected<as_diagonal, const BV&> && !IsDetected<as_diagonal, const BVR&>);
	static_assert(IsDetected<as_diagonal, BV&&> && IsDetected<as_diagonal, BVExpression&&>);
	static_assert(IsDetected<dot, const BV&, const BVR&> && IsDetected<normalized, const BV&>);
	static_assert(IsDetected<normalize, BV&> && !IsDetected<normalize, BV&&>);
	static_assert(!IsDetected<normalized, const BVi&> && !IsDetected<normalize, BVi&>);
	static_assert(IsDetected<inverse, const BD&> && !IsDetected<inverse, const BDi&>);
	static_assert(IsDetected<inverse, BD&&> && IsDetected<inverse, BDExpression&&>);
	static_assert(IsDetected<solve, const BD&, const BV&> && !IsDetected<solve, const BD&, const BVR&>);
	static_assert(IsDetected<solve, const BD&, const Dense&> && !IsDetected<solve, const BD&, const FixedColumn&>);
	static_assert(IsDetected<transformed, const BD&, const BD&>);
	static_assert(IsDetected<transformed, BD&&, const BD&>
	              && IsDetected<transformed, BDExpression&&, const BD&>);
	static_assert(IsDetected<transform_in_place, BD&, const BD&>
	              && !IsDetected<transform_in_place, BD&&, const BD&>);
	static_assert(IsDetected<congruence_scalar, double, std::complex<double>>);
	static_assert(std::is_same_v<congruence_scalar<double, std::complex<double>>,
	                             std::complex<double>>);
	static_assert(!IsDetected<congruence_scalar, int, std::complex<double>>);
	static_assert(std::is_same_v<decltype(std::declval<BD&>().blockingInfo()),
	                             const BlockingInfo&>);
	static_assert(std::is_same_v<decltype(std::declval<const BD&>().blockingInfo()),
	                             const BlockingInfo&>);
	static_assert(std::is_same_v<decltype(std::declval<BD&&>().blockingInfo()), BlockingInfo>);
	static_assert(std::is_same_v<decltype(std::declval<const BD&&>().blockingInfo()), BlockingInfo>);
	static_assert(std::is_same_v<decltype(std::declval<BVExpression&>().blockingInfo()),
	                             const BlockingInfo&>);
	static_assert(std::is_same_v<decltype(std::declval<BVExpression&&>().blockingInfo()),
	                             BlockingInfo>);
}  // namespace

TEST_CASE("the Hoppy operation matrix is enforced at compile time") { SUCCEED(); }

TEST_CASE("blocking-taking fills commit only after the replacement is complete")
{
	Hoppy::BlockDiagonalMatrix<ThrowingScalar> matrix{1};
	matrix[0](0, 0).Value = 7.0;
	ThrowingScalar::AssignmentsRemaining = 1;
	REQUIRE_THROWS_AS(matrix.setConstant({1, 2}, ThrowingScalar{3.0}), std::runtime_error);
	REQUIRE(matrix.blockingInfo() == std::vector<Eigen::Index>{1});
	REQUIRE(matrix[0](0, 0).Value == 7.0);
	ThrowingScalar::AssignmentsRemaining = -1;

	Hoppy::BlockVector<ThrowingScalar> vector{1};
	vector[0](0).Value = 9.0;
	ThrowingScalar::AssignmentsRemaining = 1;
	REQUIRE_THROWS_AS(vector.setConstant({2}, ThrowingScalar{4.0}), std::runtime_error);
	REQUIRE(vector.blockingInfo() == std::vector<Eigen::Index>{1});
	REQUIRE(vector[0](0).Value == 9.0);
	ThrowingScalar::AssignmentsRemaining = -1;

	Hoppy::BlockDiagonalMatrix<ThrowingScalar> inPlace{1, 2};
	ThrowingScalar::AssignmentsRemaining = 1;
	REQUIRE_THROWS_AS(inPlace.setConstant(ThrowingScalar{5.0}), std::runtime_error);
	REQUIRE(inPlace.blockingInfo() == (std::vector<Eigen::Index>{1, 2}));
	REQUIRE(inPlace.storedSize() == 5);
	ThrowingScalar::AssignmentsRemaining = -1;
}

#ifndef EIGEN_NO_DEBUG
TEST_CASE("remaining public precondition sites use the Eigen assertion hook")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2};
	Hoppy::BlockDiagonalMatrix<double> otherMatrix{3};
	REQUIRE_THROWS_AS(matrix -= otherMatrix, Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS((void)matrix.isApprox(otherMatrix), Hoppy::Test::EigenAssertionFailure);

	Hoppy::BlockVector<double> vector{1, 2};
	Hoppy::BlockVector<double> otherVector{3};
	REQUIRE_THROWS_AS(vector += otherVector, Hoppy::Test::EigenAssertionFailure);
	Eigen::Matrix2d notVector = Eigen::Matrix2d::Zero();
	REQUIRE_THROWS_AS(vector.extractFromDense(notVector), Hoppy::Test::EigenAssertionFailure);

	Eigen::Matrix<double, 2, 3> nonsquare = Eigen::Matrix<double, 2, 3>::Zero();
	REQUIRE_THROWS_AS((Hoppy::BlockDiagonalMatrix<double>::SingleBlock(nonsquare)),
	                  Hoppy::Test::EigenAssertionFailure);
	std::vector<Eigen::MatrixXd> nonsquareBlocks{Eigen::MatrixXd::Zero(2, 3)};
	REQUIRE_THROWS_AS((Hoppy::BlockDiagonalMatrix<double>::FromBlocks(
	                          nonsquareBlocks.begin(), nonsquareBlocks.end())),
	                  Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS((Hoppy::BlockVector<double>::FromBlocks(
	                          nonsquareBlocks.begin(), nonsquareBlocks.end())),
	                  Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(Hoppy::DenseBlockPolicy::BlockElementCount(0),
	                  Hoppy::Test::EigenAssertionFailure);
}
#endif
