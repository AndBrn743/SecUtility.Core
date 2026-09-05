// SPDX-License-Identifier: MIT

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/Hoppy.hpp>

#include <Eigen/Core>

#include <random>
#include <type_traits>
#include <vector>

namespace
{
	template <typename Object>
	void fill(Object& object, std::mt19937& generator)
	{
		std::uniform_real_distribution<double> distribution(-2.0, 2.0);
		for (Eigen::Index index = 0; index < object.storedSize(); ++index)
			object.data()[index] = distribution(generator);
	}
}  // namespace

TEST_CASE("seeded block diagonal properties agree with dense Eigen oracles")
{
	std::mt19937 generator(0x5ec07117U);
	const std::vector<std::vector<Eigen::Index>> blockings{
	        {}, {1}, {1, 1, 1}, {1, 2, 1}, {3, 1}};
	for (const auto& blocking : blockings)
	{
		CAPTURE(blocking);
		Hoppy::BlockDiagonalMatrixXd a(blocking);
		Hoppy::BlockDiagonalMatrixXd b(blocking);
		Hoppy::BlockDiagonalMatrixXd transform(blocking);
		Hoppy::BlockVectorXd column(blocking);
		Hoppy::BlockRowVectorXd row(blocking);
		fill(a, generator);
		fill(b, generator);
		fill(transform, generator);
		fill(column, generator);
		fill(row, generator);
		const Eigen::MatrixXd ad = a.toDense();
		const Eigen::MatrixXd bd = b.toDense();
		const Eigen::MatrixXd td = transform.toDense();

		Hoppy::Test::requireApprox((a + b).toDense(), ad + bd);
		Hoppy::Test::requireApprox((a - b).toDense(), ad - bd);
		Hoppy::Test::requireApprox((a * b).toDense(), ad * bd);
		Hoppy::Test::requireApprox((-a).toDense(), -ad);
		Hoppy::Test::requireApprox(a.transpose().toDense(), ad.transpose());
		Hoppy::Test::requireApprox((a * 1.5).toDense(), ad * 1.5);
		Hoppy::Test::requireApprox(a.diagonal().toDense(), ad.diagonal());
		Hoppy::Test::requireApprox(column.asDiagonal().toDense(),
		                           column.asDense().asDiagonal().toDenseMatrix());
		Hoppy::Test::requireApprox((a * column).asDense(), ad * column.asDense());
		Hoppy::Test::requireApprox((row * a).asDense(), row.asDense() * ad);
		REQUIRE(column.dot(row) == Catch::Approx(column.asDense().dot(row.asDense())));

		Eigen::MatrixXd dense(a.rows(), a.cols());
		for (Eigen::Index index = 0; index < dense.size(); ++index)
			dense(index) = std::uniform_real_distribution<double>(-2.0, 2.0)(generator);
		Hoppy::Test::requireApprox((dense + a).eval(), dense + ad);
		Hoppy::Test::requireApprox((a - dense).eval(), ad - dense);
		Eigen::MatrixXd rhs(a.rows(), 2);
		for (Eigen::Index index = 0; index < rhs.size(); ++index)
			rhs(index) = std::uniform_real_distribution<double>(-2.0, 2.0)(generator);
		Hoppy::Test::requireApprox((a * rhs).eval(), ad * rhs);
		Hoppy::Test::requireApprox((rhs.transpose() * a).eval(), rhs.transpose() * ad);
		Hoppy::Test::requireApprox(a.transformedBy(transform).toDense(), td * ad * td.adjoint());
		Hoppy::Test::requireApprox(a.backTransformedBy(transform).toDense(), td.adjoint() * ad * td);

		Hoppy::BlockDiagonalMatrixXd invertible = a;
		for (Eigen::Index index = 0; index < invertible.blockCount(); ++index)
			invertible[index].diagonal().array() += 5.0 + invertible.dimensionOfBlock(index);
		const Eigen::MatrixXd invertibleDense = invertible.toDense();
		Hoppy::Test::requireApprox(invertible.inverse().toDense(), invertibleDense.inverse());
		Hoppy::Test::requireApprox(invertible.solve(column).asDense(),
		                           invertibleDense.partialPivLu().solve(column.asDense()));
		Hoppy::Test::requireApprox(invertible.solve(rhs), invertibleDense.partialPivLu().solve(rhs));

		const auto normalized = column.normalized();
		Hoppy::Test::requireApprox(normalized.asDense(), column.asDense().normalized());
	}
}

TEST_CASE("all documented public aliases have their exact scalar and orientation")
{
	static_assert(std::is_same_v<Hoppy::BlockDiagonalMatrixXd, Hoppy::BlockDiagonalMatrix<double>>);
	static_assert(std::is_same_v<Hoppy::BlockDiagonalMatrixXf, Hoppy::BlockDiagonalMatrix<float>>);
	static_assert(std::is_same_v<Hoppy::BlockDiagonalMatrixXcd,
	                             Hoppy::BlockDiagonalMatrix<std::complex<double>>>);
	static_assert(std::is_same_v<Hoppy::BlockDiagonalMatrixXcf,
	                             Hoppy::BlockDiagonalMatrix<std::complex<float>>>);
	static_assert(std::is_same_v<Hoppy::BlockDiagonalMatrixXi, Hoppy::BlockDiagonalMatrix<int>>);
	static_assert(std::is_same_v<Hoppy::BlockDiagonalMatrixXl, Hoppy::BlockDiagonalMatrix<long>>);
	static_assert(std::is_same_v<Hoppy::BlockVectorXd, Hoppy::BlockVector<double, Hoppy::Column>>);
	static_assert(std::is_same_v<Hoppy::BlockVectorXcd,
	                             Hoppy::BlockVector<std::complex<double>, Hoppy::Column>>);
	static_assert(std::is_same_v<Hoppy::BlockRowVectorXd, Hoppy::BlockVector<double, Hoppy::Row>>);
	static_assert(std::is_same_v<Hoppy::BlockRowVectorXcd,
	                             Hoppy::BlockVector<std::complex<double>, Hoppy::Row>>);
}
