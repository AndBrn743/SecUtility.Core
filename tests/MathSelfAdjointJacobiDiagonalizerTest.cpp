//
// Created by Andy on 10/21/2025.
//

#include <SecUtility/Math/SelfAdjointJacobiEigenSolver.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <iostream>


template <typename Scalar>
Eigen::MatrixX<Scalar> HermitianMatrix(const Eigen::Index n)
{
	Eigen::MatrixX<Scalar> a = Eigen::MatrixX<Scalar>::Random(n, n);
	a += a.adjoint().eval();
	a *= 1e-1;
	a.diagonal() *= 100;
	return a;
}


TEST_CASE("SelfAdjointJacobiDiagonalizer")
{
	SECTION("The one with double should be correct")
	{
		const Eigen::MatrixXd symmetricMatrix = HermitianMatrix<double>(42);

		const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(symmetricMatrix);
		const SecUtility::Math::SelfAdjointJacobiEigenSolver jacobiSolver(symmetricMatrix);

		Eigen::VectorXd eigenvalues = jacobiSolver.UnsortedEigenvalues();
		std::sort(eigenvalues.begin(), eigenvalues.end());
		CHECK((eigenvalues - eigenSolver.eigenvalues()).norm() < 1e-9);
	}

	SECTION("The one with float should be correct")
	{
		const Eigen::MatrixXf symmetricMatrix = HermitianMatrix<float>(42);

		const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigenSolver(symmetricMatrix);
		const SecUtility::Math::SelfAdjointJacobiEigenSolver jacobiSolver(symmetricMatrix);

		Eigen::VectorXf eigenvalues = jacobiSolver.UnsortedEigenvalues();
		std::sort(eigenvalues.begin(), eigenvalues.end());
		CHECK((eigenvalues - eigenSolver.eigenvalues()).norm() < 1e-5 * eigenSolver.eigenvalues().norm());
	}

	SECTION("The one with std::complex<double> should be correct")
	{
		const Eigen::MatrixXcd symmetricMatrix = HermitianMatrix<std::complex<double>>(42);

		const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eigenSolver(symmetricMatrix);
		const SecUtility::Math::SelfAdjointJacobiEigenSolver jacobiSolver(symmetricMatrix);

		Eigen::VectorXd eigenvalues = jacobiSolver.UnsortedEigenvalues().real();
		std::sort(eigenvalues.begin(), eigenvalues.end());
		CHECK((eigenvalues - eigenSolver.eigenvalues()).norm() < 1e-9);
	}


	SECTION("Basic")
	{
		const Eigen::MatrixXd symmetricMatrix = HermitianMatrix<double>(42);

		{
			const auto t0 = std::chrono::high_resolution_clock::now();
			const SecUtility::Math::SelfAdjointJacobiEigenSolver solver(symmetricMatrix);
			std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()
			                                                                   - t0)
			                     .count()
			          << " ms" << std::endl;

			Eigen::VectorXd eigenvalues = solver.UnsortedEigenvalues();
			std::sort(eigenvalues.begin(), eigenvalues.end());
			// std::cout << solver.TransformedMatrix() << std::endl;
			std::cout << "Parallel Jacobi Eigenvalues:\n" << eigenvalues.head(10).transpose() << " . . ." << std::endl;
		}
		{
			const auto t0 = std::chrono::high_resolution_clock::now();
			const SecUtility::Math::SelfAdjointJacobiEigenSolver solver(
			        symmetricMatrix,
			        SecUtility::BiPartiteRoundRobinOrdering{ranges::views::iota(Eigen::Index{0}, Eigen::Index{5}),
			                                                ranges::views::iota(Eigen::Index{5}, Eigen::Index{42})});
			std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()
			                                                                   - t0)
			                     .count()
			          << " ms" << std::endl;

			Eigen::VectorXd eigenvalues = solver.UnsortedEigenvalues();
			std::sort(eigenvalues.begin(), eigenvalues.end());
			// std::cout << solver.TransformedMatrix() << std::endl;
			std::cout << "Parallel Jacobi Eigenvalues:\n" << eigenvalues.head(10).transpose() << " . . ." << std::endl;
		}
		{
			const auto t0 = std::chrono::high_resolution_clock::now();
			const SecUtility::Math::SelfAdjointJacobiEigenSolver solver(
			        symmetricMatrix,
			        SecUtility::RoundRobinOrdering{ranges::views::iota(Eigen::Index{0}, Eigen::Index{42}), -1},
			        1e-10,
			        32,
			        [](const Eigen::Index p, const Eigen::Index q)
			        {
				        const auto [m, n] = std::minmax(p, q);
				        return m >= 0 && n >= 0 && ((m <= 5 && n > 5) || (m <= 8 && n > 8));
			        });
			std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()
			                                                                   - t0)
			                     .count()
			          << " ms" << std::endl;

			Eigen::VectorXd eigenvalues = solver.UnsortedEigenvalues();
			std::sort(eigenvalues.begin(), eigenvalues.end());
			// std::cout << solver.TransformedMatrix() << std::endl;
			std::cout << "Parallel Jacobi Eigenvalues:\n" << eigenvalues.head(10).transpose() << " . . ." << std::endl;
		}
		{
			const auto t0 = std::chrono::high_resolution_clock::now();
			const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(symmetricMatrix);
			std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()
			                                                                   - t0)
			                     .count()
			          << " ms" << std::endl;
			std::cout << "Eigen Eigenvalues:\n" << es.eigenvalues().head(10).transpose() << " . . ." << std::endl;
		}
	}
}
