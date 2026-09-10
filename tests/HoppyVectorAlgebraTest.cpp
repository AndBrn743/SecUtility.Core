// SPDX-License-Identifier: MIT

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/Hoppy.hpp>

#include <Eigen/Core>

#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>
#include <utility>

namespace
{
	template <typename T, typename = void>
	struct has_normalized : std::false_type
	{};

	template <typename T>
	struct has_normalized<T, std::void_t<decltype(std::declval<const T&>().normalized())>>
	    : std::true_type
	{};

	template <typename T, typename = void>
	struct has_lvalue_normalize : std::false_type
	{};

	template <typename T>
	struct has_lvalue_normalize<T, std::void_t<decltype(std::declval<T&>().normalize())>>
	    : std::true_type
	{};

	template <typename T, typename = void>
	struct has_rvalue_normalize : std::false_type
	{};

	template <typename T>
	struct has_rvalue_normalize<T, std::void_t<decltype(std::declval<T&&>().normalize())>>
	    : std::true_type
	{};

	template <typename Lhs, typename Rhs, typename = void>
	struct has_dot : std::false_type
	{};

	template <typename Lhs, typename Rhs>
	struct has_dot<Lhs, Rhs,
	               std::void_t<decltype(std::declval<const Lhs&>().dot(
	                       std::declval<const Rhs&>()))>> : std::true_type
	{};
}  // namespace

TEST_CASE("dot supports every orientation pair and different blocking")
{
	Eigen::Vector4d lhsDense;
	lhsDense << 1.0, -2.0, 3.0, 4.0;
	Eigen::Vector4d rhsDense;
	rhsDense << 2.0, 5.0, -1.0, 3.0;
	const auto lhsColumn = Hoppy::BlockVector<double, Hoppy::BlockVectorOrientation::Column>::FromDense(lhsDense, {1, 3});
	const auto lhsRow = Hoppy::BlockVector<double, Hoppy::BlockVectorOrientation::Row>::FromDense(lhsDense, {1, 3});
	const auto rhsColumn = Hoppy::BlockVector<double, Hoppy::BlockVectorOrientation::Column>::FromDense(rhsDense, {2, 2});
	const auto rhsRow = Hoppy::BlockVector<double, Hoppy::BlockVectorOrientation::Row>::FromDense(rhsDense, {2, 2});
	const double expected = lhsDense.dot(rhsDense);

	REQUIRE(lhsColumn.dot(rhsColumn) == expected);
	REQUIRE(lhsColumn.dot(rhsRow) == expected);
	REQUIRE(lhsRow.dot(rhsColumn) == expected);
	REQUIRE(lhsRow.dot(rhsRow) == expected);

	const Hoppy::BlockVector<double> emptyColumn;
	const Hoppy::BlockVector<double, Hoppy::BlockVectorOrientation::Row> emptyRow;
	REQUIRE(emptyColumn.dot(emptyRow) == 0.0);
}

TEST_CASE("dot conjugates the lhs and follows Eigen mixed-scalar promotion")
{
	using Complex = std::complex<double>;
	Eigen::VectorXcd lhsDense(3);
	lhsDense << Complex{1.0, 2.0}, Complex{-3.0, 1.0}, Complex{2.0, -4.0};
	Eigen::Vector3d rhsDense;
	rhsDense << 2.0, -1.0, 0.5;
	const auto lhs = Hoppy::BlockVector<Complex>::FromDense(lhsDense, {2, 1});
	const auto rhs = Hoppy::BlockVector<double, Hoppy::BlockVectorOrientation::Row>::FromDense(rhsDense, {1, 2});
	const auto result = lhs.dot(rhs);
	static_assert(std::is_same_v<std::remove_cv_t<decltype(result)>, Complex>);
	REQUIRE(result == lhsDense.dot(rhsDense));

	using InvalidLhs = Hoppy::BlockVector<int>;
	using InvalidRhs = Hoppy::BlockVector<Complex>;
	static_assert(!has_dot<InvalidLhs, InvalidRhs>::value);
}

TEST_CASE("normalization uses one global norm and preserves orientation and source")
{
	Hoppy::BlockVector<double> column{1, 2};
	column.asDense() << 3.0, 0.0, 4.0;
	const Eigen::Vector3d original = column.asDense();
	const auto normalizedColumn = column.normalized();
	static_assert(std::is_same_v<std::remove_cv_t<decltype(normalizedColumn)>,
	                             Hoppy::BlockVector<double, Hoppy::BlockVectorOrientation::Column>>);
	Hoppy::Test::requireApprox(normalizedColumn.asDense(), original.normalized());
	Hoppy::Test::requireApprox(column.asDense(), original);
	REQUIRE(normalizedColumn.blockingInfo() == column.blockingInfo());

	Hoppy::BlockVector<double, Hoppy::BlockVectorOrientation::Row> row{1, 2};
	row.asDense() = original.transpose();
	const auto normalizedRow = row.normalized();
	static_assert(std::is_same_v<std::remove_cv_t<decltype(normalizedRow)>,
	                             Hoppy::BlockVector<double, Hoppy::BlockVectorOrientation::Row>>);
	Hoppy::Test::requireApprox(normalizedRow.asDense(), original.transpose().normalized());
	row.normalize();
	Hoppy::Test::requireApprox(row.asDense(), original.transpose().normalized());
}

TEST_CASE("zero nonfinite and complex normalization follow Eigen")
{
	Hoppy::BlockVector<double> zero{1, 2};
	zero.setZero();
	Hoppy::Test::requireApprox(zero.normalized().asDense(), zero.asDense());
	zero.normalize();
	Hoppy::Test::requireApprox(zero.asDense(), Eigen::Vector3d::Zero());

	Hoppy::BlockVector<double> nan{2};
	nan.asDense() << std::numeric_limits<double>::quiet_NaN(), 1.0;
	const Eigen::Vector2d expectedNan = nan.asDense().normalized();
	const auto normalizedNan = nan.normalized();
	REQUIRE(std::isnan(normalizedNan.asDense()(0)) == std::isnan(expectedNan(0)));
	REQUIRE(normalizedNan.asDense()(1) == expectedNan(1));

	Hoppy::BlockVector<double> infinity{2};
	infinity.asDense() << std::numeric_limits<double>::infinity(), 1.0;
	const Eigen::Vector2d expectedInfinity = infinity.asDense().normalized();
	const auto normalizedInfinity = infinity.normalized();
	REQUIRE(std::isnan(normalizedInfinity.asDense()(0)) == std::isnan(expectedInfinity(0)));
	REQUIRE(normalizedInfinity.asDense()(1) == expectedInfinity(1));

	using Complex = std::complex<double>;
	Hoppy::BlockVector<Complex> complex{1, 1};
	complex.asDense() << Complex{3.0, 4.0}, Complex{0.0, 12.0};
	Hoppy::Test::requireApprox(complex.normalized().asDense(), complex.asDense().normalized());
}

TEST_CASE("normalization availability is restricted to supported scalars and plain lvalues")
{
	using RealVector = Hoppy::BlockVector<double>;
	using IntegerVector = Hoppy::BlockVector<int>;
	using Expression = decltype(-std::declval<const RealVector&>());
	static_assert(has_normalized<RealVector>::value);
	static_assert(has_normalized<Expression>::value);
	static_assert(!has_normalized<IntegerVector>::value);
	static_assert(has_lvalue_normalize<RealVector>::value);
	static_assert(!has_lvalue_normalize<IntegerVector>::value);
	static_assert(!has_lvalue_normalize<Expression>::value);
	static_assert(!has_rvalue_normalize<RealVector>::value);
}

#ifdef HOPPY_TEST_EIGEN_ASSERT_THROWS
TEST_CASE("dot enforces equal logical dimensions")
{
	const Hoppy::BlockVector<double> lhs{1, 2};
	const Hoppy::BlockVector<double, Hoppy::BlockVectorOrientation::Row> rhs{2};
	REQUIRE_THROWS_AS((void)lhs.dot(rhs), Hoppy::Test::EigenAssertionFailure);
}
#endif
