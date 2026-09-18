// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Andy Brown

#include <SecUtility/Hoppy/LowRankMatrix.hpp>

#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <type_traits>
#include <utility>


namespace
{
	template <typename Left, typename Right, typename = void>
	struct CanAddAssign : std::false_type
	{
	};

	template <typename Left, typename Right>
	struct CanAddAssign<Left,
	                    Right,
	                    std::void_t<decltype(std::declval<Left&>() += std::declval<const Right&>())>> : std::true_type
	{
	};

	Hoppy::LowRankMatrixX<double> General(const double coefficient, const Eigen::Index rows = 3,
	                                     const Eigen::Index cols = 2)
	{
		Hoppy::LowRankMatrixX<double> matrix(rows, cols);
		matrix.addTerm(coefficient, Eigen::VectorXd::LinSpaced(rows, 1, static_cast<double>(rows)),
		               Eigen::VectorXd::LinSpaced(cols, -1, static_cast<double>(cols - 2)));
		return matrix;
	}
}


static_assert(!CanAddAssign<Hoppy::LowRankMatrixX<double>, Hoppy::LowRankMatrixX<float>>::value);
using FixedPlusDynamic = decltype(std::declval<const Hoppy::LowRankMatrix<double, 3, 2>&>()
                                  + std::declval<const Hoppy::LowRankMatrixX<double>&>());
static_assert(FixedPlusDynamic::RowsAtCompileTime == 3);
static_assert(FixedPlusDynamic::ColsAtCompileTime == 2);


TEST_CASE("General low-rank arithmetic preserves term order", "[Hoppy][LowRankMatrix]")
{
	const auto left = General(2);
	const auto right = General(-3);
	const Eigen::MatrixXd leftDense = left.toDense();
	const Eigen::MatrixXd rightDense = right.toDense();

	const auto sum = left + right;
	const auto difference = left - right;
	static_assert(std::is_same_v<std::decay_t<decltype(sum)>, Hoppy::LowRankMatrixX<double>>);
	REQUIRE(sum.termCount() == 2);
	REQUIRE(difference.termCount() == 2);
	CHECK(sum.coefficientOfTerm(0) == 2);
	CHECK(sum.coefficientOfTerm(1) == -3);
	CHECK(difference.coefficientOfTerm(0) == 2);
	CHECK(difference.coefficientOfTerm(1) == 3);
	CHECK(sum.toDense().isApprox(leftDense + rightDense));
	CHECK(difference.toDense().isApprox(leftDense - rightDense));

	auto inPlaceSum = left;
	inPlaceSum += right;
	CHECK(inPlaceSum.toDense().isApprox(leftDense + rightDense));
	CHECK(inPlaceSum.coefficients() == sum.coefficients());
	auto inPlaceDifference = left;
	inPlaceDifference -= right;
	CHECK(inPlaceDifference.toDense().isApprox(leftDense - rightDense));
	CHECK(inPlaceDifference.coefficients() == difference.coefficients());
}


TEST_CASE("In-place self arithmetic avoids duplicate storage and preserves dimensions", "[Hoppy][LowRankMatrix]")
{
	auto matrix = General(2, 3, 2);
	const Eigen::MatrixXd dense = matrix.toDense();
	const Eigen::Index capacity = matrix.capacity();
	matrix += matrix;
	CHECK(matrix.termCount() == 1);
	CHECK(matrix.coefficientOfTerm(0) == 4);
	CHECK(matrix.capacity() == capacity);
	CHECK(matrix.toDense().isApprox(2 * dense));

	matrix -= matrix;
	CHECK(matrix.rows() == 3);
	CHECK(matrix.cols() == 2);
	CHECK(matrix.termCount() == 0);
	CHECK(matrix.capacity() == capacity);
}


TEST_CASE("Low-rank arithmetic handles rank-zero and generated zero terms", "[Hoppy][LowRankMatrix]")
{
	const Hoppy::LowRankMatrixX<double> empty(3, 2);
	const auto matrix = General(2);
	CHECK((empty + matrix).toDense().isApprox(matrix.toDense()));
	CHECK((matrix - empty).toDense().isApprox(matrix.toDense()));
	CHECK((empty + empty).termCount() == 0);

	const auto zeroScaled = matrix * 0.0;
	const auto sum = empty + zeroScaled;
	CHECK(sum.rows() == 3);
	CHECK(sum.cols() == 2);
	CHECK(sum.termCount() == 0);
}


TEST_CASE("Structured low-rank arithmetic preserves equal policies", "[Hoppy][LowRankMatrix]")
{
	using Symmetric = Hoppy::LowRankSymmetricMatrixX<std::complex<double>>;
	using SelfAdjoint = Hoppy::LowRankSelfAdjointMatrixX<std::complex<double>>;
	const Eigen::MatrixXcd vectors1 = Eigen::MatrixXcd::Random(3, 1);
	const Eigen::MatrixXcd vectors2 = Eigen::MatrixXcd::Random(3, 1);
	const Symmetric symmetric1(std::complex<double>{2, 1}, vectors1);
	const Symmetric symmetric2(std::complex<double>{-1, 0.5}, vectors2);
	const SelfAdjoint selfAdjoint1(2.0, vectors1);
	const SelfAdjoint selfAdjoint2(-1.0, vectors2);

	const auto symmetricSum = symmetric1 + symmetric2;
	const auto selfAdjointDifference = selfAdjoint1 - selfAdjoint2;
	static_assert(std::is_same_v<typename std::decay_t<decltype(symmetricSum)>::StructurePolicy,
	                             Hoppy::Detail::SymmetricLowRankStructure>);
	static_assert(std::is_same_v<typename std::decay_t<decltype(selfAdjointDifference)>::StructurePolicy,
	                             Hoppy::Detail::SelfAdjointLowRankStructure>);
	CHECK(symmetricSum.toDense().isApprox(symmetric1.toDense() + symmetric2.toDense()));
	CHECK(selfAdjointDifference.toDense().isApprox(selfAdjoint1.toDense() - selfAdjoint2.toDense()));

	auto selfDoubled = selfAdjoint1;
	const Eigen::Index selfDoubledCapacity = selfDoubled.capacity();
	selfDoubled += selfDoubled;
	REQUIRE(selfDoubled.termCount() == 1);
	CHECK(selfDoubled.coefficientOfTerm(0) == 4);
	CHECK(selfDoubled.capacity() == selfDoubledCapacity);
	CHECK(selfDoubled.leftVectorOfTerm(0) == selfAdjoint1.leftVectorOfTerm(0));
	CHECK(selfDoubled.toDense().isApprox(2.0 * selfAdjoint1.toDense()));

	auto inPlace = selfAdjoint1;
	inPlace += selfAdjoint2;
	CHECK(inPlace.toDense().isApprox(selfAdjoint1.toDense() + selfAdjoint2.toDense()));
	inPlace -= inPlace;
	CHECK(inPlace.rows() == 3);
	CHECK(inPlace.termCount() == 0);

	auto inPlaceDifference = selfAdjoint1;
	inPlaceDifference -= selfAdjoint2;
	REQUIRE(inPlaceDifference.termCount() == 2);
	CHECK(inPlaceDifference.coefficientOfTerm(0) == 2);
	CHECK(inPlaceDifference.coefficientOfTerm(1) == 1);
	CHECK(inPlaceDifference.leftVectorOfTerm(0) == selfAdjoint1.leftVectorOfTerm(0));
	CHECK(inPlaceDifference.leftVectorOfTerm(1) == selfAdjoint2.leftVectorOfTerm(0));
	CHECK(inPlaceDifference.toDense().isApprox(selfAdjoint1.toDense() - selfAdjoint2.toDense()));
}


TEST_CASE("Mixed low-rank structures produce general matrices", "[Hoppy][LowRankMatrix]")
{
	using Symmetric = Hoppy::LowRankSymmetricMatrixX<std::complex<double>>;
	using SelfAdjoint = Hoppy::LowRankSelfAdjointMatrixX<std::complex<double>>;
	const Eigen::MatrixXcd vectors = Eigen::MatrixXcd::Random(3, 1);
	const Symmetric symmetric(std::complex<double>{2, 1}, vectors);
	const SelfAdjoint selfAdjoint(3.0, vectors);
	Hoppy::LowRankMatrixX<std::complex<double>> general(3, 3);
	general.addTerm({-1, 0.5}, vectors.col(0), vectors.col(0));

	const auto structureMix = symmetric + selfAdjoint;
	const auto generalThenStructured = general - symmetric;
	const auto structuredThenGeneral = selfAdjoint + general;
	static_assert(std::is_same_v<std::decay_t<decltype(structureMix)>,
	                             Hoppy::LowRankMatrixX<std::complex<double>>>);
	static_assert(std::is_same_v<std::decay_t<decltype(generalThenStructured)>,
	                             Hoppy::LowRankMatrixX<std::complex<double>>>);
	static_assert(std::is_same_v<std::decay_t<decltype(structuredThenGeneral)>,
	                             Hoppy::LowRankMatrixX<std::complex<double>>>);
	CHECK(structureMix.toDense().isApprox(symmetric.toDense() + selfAdjoint.toDense()));
	CHECK(generalThenStructured.toDense().isApprox(general.toDense() - symmetric.toDense()));
	CHECK(structuredThenGeneral.toDense().isApprox(selfAdjoint.toDense() + general.toDense()));
}
