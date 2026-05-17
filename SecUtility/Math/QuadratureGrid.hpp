// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <Eigen/Dense>
#include <SecUtility/Math/Core.hpp>
#include <SecUtility/Math/Constant.hpp>
#include <SecUtility/Meta/TypeTrait.hpp>


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

	private:
		Eigen::Matrix<Scalar, Eigen::Dynamic, 2> m_Data{};
	};

	template <typename Scalar>
	QuadratureGrid<Scalar> GenerateFejerQuadratureGrid(const Eigen::Index size)
	{
		QuadratureGrid<Scalar> grid(size);

		for (Eigen::Index i = 0; i < size; ++i)
		{
			grid.Node(i) = Cos((2 * i + 1) * Constant::Pi<Scalar> / (2 * size));
		}

		for (Eigen::Index i = 0; i < size; ++i)
		{
			Scalar sum = 0;
			for (Eigen::Index m = 1; m <= size / 2; ++m)
			{
				sum += Cos(2 * m * ACos(grid.Node(i))) / (4 * m * m - 1);
			}

			grid.Weight(i) = (1 - 2 * sum) * 2 / size;
		}

		return grid;
	}
}
