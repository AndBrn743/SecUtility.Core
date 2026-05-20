// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <Eigen/Dense>
#include <SecUtility/Math/Constant.hpp>
#include <SecUtility/Math/Core.hpp>
#include <SecUtility/Meta/TypeTrait.hpp>
#include <SecUtility/Misc/CachedFunction.hpp>
#include <utility>


namespace SecUtility::Math
{
	template <typename Scalar>
	class QuadratureGrid
	{
	public:
		QuadratureGrid() noexcept = default;

		explicit QuadratureGrid(const Eigen::Index size) : m_Data(size, 2)
		{
			/* NO CODE */
		}

		template <typename NodeVector, typename WeightVector>
		QuadratureGrid(const NodeVector& nodes, const WeightVector& weights)
		{
			static_assert(std::is_base_of_v<Eigen::EigenBase<NodeVector>, NodeVector>);
			static_assert(std::is_base_of_v<Eigen::EigenBase<WeightVector>, WeightVector>);
			eigen_assert(nodes.size() == weights.size());
			eigen_assert(nodes.rows() == 1 || nodes.cols() == 1);
			eigen_assert(weights.rows() == 1 || weights.cols() == 1);

			m_Data.resize(nodes.size(), Eigen::NoChange);
			m_Data.col(0) = nodes;
			m_Data.col(1) = weights;
		}

		constexpr Eigen::Index Size() const noexcept
		{
			return m_Data.rows();
		}

		constexpr void Resize(const Eigen::Index size)
		{
			m_Data.resize(size, Eigen::NoChange);
		}

		constexpr void ConservativeResize(const Eigen::Index size)
		{
			m_Data.conservativeResize(size, Eigen::NoChange);
		}

		decltype(auto) Nodes() const noexcept
		{
			return m_Data.col(0);
		}

		decltype(auto) Weights() const noexcept
		{
			return m_Data.col(1);
		}

		decltype(auto) Nodes() noexcept
		{
			return m_Data.col(0);
		}

		decltype(auto) Weights() noexcept
		{
			return m_Data.col(1);
		}

		const auto& Node(const Eigen::Index index) const noexcept
		{
			return m_Data(index, 0);
		}

		const auto& Weight(const Eigen::Index index) const noexcept
		{
			return m_Data(index, 1);
		}

		auto& Node(const Eigen::Index index) noexcept
		{
			return m_Data(index, 0);
		}

		auto& Weight(const Eigen::Index index) noexcept
		{
			return m_Data(index, 1);
		}

		void SetConstant(const Scalar value) noexcept
		{
			m_Data.setConstant(value);
		}

		template <typename Function>
		Scalar Integrate(Function&& function) const noexcept(noexcept(function(Scalar{})))
		{
			Scalar integral{};

			for (Eigen::Index i = 0; i < m_Data.rows(); ++i)
			{
				integral += function(Node(i)) * Weight(i);
			}

			return integral;
		}

#define SEC_MATH_QUADRATURE_GRID_DEFINE_GET(QUALIFIER)                                                                 \
	template <std::size_t I>                                                                                           \
	friend decltype(auto) get(QuadratureGrid QUALIFIER q)                                                              \
	{                                                                                                                  \
		static_assert(I == 0 || I == 1);                                                                               \
		if constexpr (I == 0)                                                                                          \
		{                                                                                                              \
			return static_cast<QuadratureGrid QUALIFIER>(q).Nodes();                                                   \
		}                                                                                                              \
		else                                                                                                           \
		{                                                                                                              \
			return static_cast<QuadratureGrid QUALIFIER>(q).Weights();                                                 \
		}                                                                                                              \
	}

		SEC_MATH_QUADRATURE_GRID_DEFINE_GET(&)
		SEC_MATH_QUADRATURE_GRID_DEFINE_GET(const&)
		SEC_MATH_QUADRATURE_GRID_DEFINE_GET(&&)
		SEC_MATH_QUADRATURE_GRID_DEFINE_GET(const&&)

#undef SEC_MATH_QUADRATURE_GRID_DEFINE_GET

	private:
		Eigen::Matrix<Scalar, Eigen::Dynamic, 2> m_Data{};
	};

	template <typename Scalar>
	struct OrthogonalPolynomialRecurrence;

	template <typename Scalar>
	QuadratureGrid<Scalar> ConstructQuadratureGrid(const OrthogonalPolynomialRecurrence<Scalar>& rule,
	                                               const Scalar zerothMoment = 1)
	{
		eigen_assert(rule.Alphas.size() == rule.Gammas.size());
		eigen_assert(rule.Alphas.size() > 1);

		const auto n = rule.Alphas.size();

		Eigen::MatrixX<Scalar> jacobian = Eigen::MatrixX<Scalar>::Zero(n, n);
		jacobian.diagonal() = rule.Alphas;
		jacobian.diagonal(1) = rule.Gammas.tail(n - 1);
		jacobian.diagonal(-1) = rule.Gammas.tail(n - 1);

		Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Scalar>> es(jacobian);

		return {std::move(es.eigenvalues()), zerothMoment * es.eigenvectors().row(0).cwiseAbs2()};
	}

	// works on [-1, 1]
	template <typename Scalar>
	const QuadratureGrid<Scalar>& FejerQuadratureGrid(const Eigen::Index size)
	{
		static CachedFunction gg{[](const Eigen::Index _size)
		                         {
			                         QuadratureGrid<Scalar> grid(_size);

			                         for (Eigen::Index i = 0; i < _size; ++i)
			                         {
				                         grid.Node(i) = Cos((2 * i + 1) * Constant::Pi<Scalar> / (2 * _size));
			                         }

			                         for (Eigen::Index i = 0; i < _size; ++i)
			                         {
				                         Scalar sum = 0;
				                         for (Eigen::Index m = 1; m <= _size / 2; ++m)
				                         {
					                         sum += Cos(2 * m * ACos(grid.Node(i))) / (4 * m * m - 1);
				                         }

				                         grid.Weight(i) = (1 - 2 * sum) * 2 / _size;
			                         }

			                         return grid;
		                         }};

		return gg(size);
	}

	// works on [0, 1]
	template <typename Scalar>
	const QuadratureGrid<Scalar>& FejerQuadratureGrid01(const Eigen::Index size)
	{
		static CachedFunction gg{[](const Eigen::Index _size)
		                         {
			                         QuadratureGrid<Scalar> grid = FejerQuadratureGrid<Scalar>(_size);
			                         grid.Nodes() =
			                                 (grid.Nodes() + Eigen::VectorX<long double>::Constant(_size, 1)) / 2;
			                         grid.Weights() *= Scalar{0.5};
			                         return grid;
		                         }};

		return gg(size);
	}

	// works on (-1, 1). notice that the first kind of Chebyshev-Gauss calculates f(x) / sqrt(1 - x^2), not f(x)
	template <typename Scalar>
	const QuadratureGrid<Scalar>& FirstKindOfChebyshevGaussQuadratureGrid(const Eigen::Index size)
	{
		static CachedFunction gg{[](const Eigen::Index _size)
		                         {
			                         QuadratureGrid<Scalar> grid(_size);

			                         for (Eigen::Index i = 0; i < _size; ++i)
			                         {
				                         grid.Node(i) = Cos((2 * i + 1) * Constant::Pi<Scalar> / (2 * _size));
			                         }

			                         grid.Weights().setConstant(Constant::Pi<Scalar> / _size);

			                         return grid;
		                         }};

		return gg(size);
	}

	// works on [-1, 1]. notice that the second kind of Chebyshev-Gauss calculates f(x) * sqrt(1 - x^2), not f(x)
	template <typename Scalar>
	const QuadratureGrid<Scalar>& SecondKindOfChebyshevGaussQuadratureGrid(const Eigen::Index size)
	{
		static CachedFunction gg{[](const Eigen::Index _size)
		                         {
			                         QuadratureGrid<Scalar> grid(_size);

			                         for (Eigen::Index i = 0; i < _size; ++i)
			                         {
				                         grid.Node(i) = Cos((i + 1) * Constant::Pi<Scalar> / (_size + 1));
				                         grid.Weight(i) = PowInt(Sin((i + 1) * Constant::Pi<Scalar> / (_size + 1)), 2)
				                                          * Constant::Pi<Scalar> / (_size + 1);
			                         }

			                         return grid;
		                         }};

		return gg(size);
	}
}


namespace std
{
	template <typename Scalar>
	struct tuple_size<SecUtility::Math::QuadratureGrid<Scalar>> : std::integral_constant<std::size_t, 2>
	{
		/* NO CODE */
	};

#define SEC_MATH_QUADRATURE_GRID_DEFINE_TUPLE_ELEMENT(QUALIFIER)                                                       \
	template <typename Scalar>                                                                                         \
	struct tuple_element<std::size_t{0}, QUALIFIER SecUtility::Math::QuadratureGrid<Scalar>>                           \
	{                                                                                                                  \
		using type = decltype(std::declval<QUALIFIER SecUtility::Math::QuadratureGrid<Scalar>&>().Nodes());            \
	};                                                                                                                 \
                                                                                                                       \
	template <typename Scalar>                                                                                         \
	struct tuple_element<std::size_t{1}, QUALIFIER SecUtility::Math::QuadratureGrid<Scalar>>                           \
	{                                                                                                                  \
		using type = decltype(std::declval<QUALIFIER SecUtility::Math::QuadratureGrid<Scalar>&>().Weights());          \
	};

	SEC_MATH_QUADRATURE_GRID_DEFINE_TUPLE_ELEMENT()
	SEC_MATH_QUADRATURE_GRID_DEFINE_TUPLE_ELEMENT(const)

#undef SEC_MATH_QUADRATURE_GRID_DEFINE_TUPLE_ELEMENT
}
