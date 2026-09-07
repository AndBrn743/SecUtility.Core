// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/TrianglePacking.hpp>
#include <SecUtility/Hoppy/Detail/TriangularCompressedTags.hpp>

#include <Eigen/Core>

#include <complex>

#if EIGEN_WORLD_VERSION != 3 || EIGEN_MAJOR_VERSION != 5 || EIGEN_MINOR_VERSION != 0
#error "Hoppy requires Eigen 5.0.0; use a supported Eigen release or update Hoppy's Eigen integration."
#endif

namespace Hoppy
{
	namespace BlockVectorOrientation
	{
		struct Column
		{};
		struct Row
		{};
	}  // namespace BlockVectorOrientation

	struct DenseBlockPolicy;

	template <typename TScalar, typename TBlockPolicy = DenseBlockPolicy>
	class BlockDiagonalMatrix;

	template <typename TScalar, typename TOrientation = BlockVectorOrientation::Column>
	class BlockVector;

	template <typename Derived>
	class BlockExpressionBase;

	template <typename Derived>
	class BlockDiagonalMatrixExpr;

	template <typename Derived>
	class BlockVectorExpr;
	template <typename Derived>
	class TriangularCompressedMatrixExpr;

	using BlockDiagonalMatrixXd = BlockDiagonalMatrix<double>;
	using BlockDiagonalMatrixXf = BlockDiagonalMatrix<float>;
	using BlockDiagonalMatrixXcd = BlockDiagonalMatrix<std::complex<double>>;
	using BlockDiagonalMatrixXcf = BlockDiagonalMatrix<std::complex<float>>;
	using BlockDiagonalMatrixXi = BlockDiagonalMatrix<int>;
	using BlockDiagonalMatrixXl = BlockDiagonalMatrix<long>;

	using BlockVectorXd = BlockVector<double>;
	using BlockVectorXf = BlockVector<float>;
	using BlockVectorXcd = BlockVector<std::complex<double>>;
	using BlockVectorXcf = BlockVector<std::complex<float>>;
	using BlockVectorXi = BlockVector<int>;
	using BlockVectorXl = BlockVector<long>;

	using BlockRowVectorXd = BlockVector<double, BlockVectorOrientation::Row>;
	using BlockRowVectorXf = BlockVector<float, BlockVectorOrientation::Row>;
	using BlockRowVectorXcd = BlockVector<std::complex<double>, BlockVectorOrientation::Row>;
	using BlockRowVectorXcf = BlockVector<std::complex<float>, BlockVectorOrientation::Row>;
	using BlockRowVectorXi = BlockVector<int, BlockVectorOrientation::Row>;
	using BlockRowVectorXl = BlockVector<long, BlockVectorOrientation::Row>;

	namespace Detail
	{
		template <typename Scalar, int Dimension, TrianglePacking Packing, int Options,
		          typename NormalizedStructureTag>
		class TriangularCompressedMatrix;

		template <typename Scalar, int Dimension, TrianglePacking Packing, int Options,
		          typename StructureTag, typename = void>
		struct triangular_compressed_alias_selector
		{};

		template <typename Scalar, int Dimension, TrianglePacking Packing, int Options,
		          typename StructureTag>
		struct triangular_compressed_alias_selector<
		        Scalar, Dimension, Packing, Options, StructureTag,
		        std::enable_if_t<is_valid_triangular_dimension_v<Dimension>
		                         && has_representable_logical_size_v<Dimension>
		                         && is_valid_triangle_packing_v<Packing>
		                         && is_valid_triangular_options_v<Options>>>
		{
			using type = TriangularCompressedMatrix<
			        Scalar, Dimension, Packing, Options,
			        normalized_structure_tag_t<Scalar, StructureTag>>;
		};

		template <typename Scalar, int Dimension, TrianglePacking Packing, int Options,
		          typename StructureTag>
		using triangular_compressed_alias_t = typename triangular_compressed_alias_selector<
		        Scalar, Dimension, Packing, Options, StructureTag>::type;

		template <typename Matrix, typename Transform, bool Back>
		class CongruenceExpression;
		template <typename Source, bool Writable>
		class DiagonalReturnType;
		struct BlockDiagonalStorage;
		struct BlockVectorStorage;
		struct BlockDiagonalShape;
		struct BlockVectorShape;
	}  // namespace Detail

	template <typename Scalar, int Dimension = Eigen::Dynamic,
	          TrianglePacking Packing = TrianglePacking::Lower, int Options = 0>
	using UpperTriangularMatrix = Detail::triangular_compressed_alias_t<
	        Scalar, Dimension, Packing, Options, Detail::UpperTriangularTag>;
	template <typename Scalar, int Dimension = Eigen::Dynamic,
	          TrianglePacking Packing = TrianglePacking::Lower, int Options = 0>
	using LowerTriangularMatrix = Detail::triangular_compressed_alias_t<
	        Scalar, Dimension, Packing, Options, Detail::LowerTriangularTag>;
	template <typename Scalar, int Dimension = Eigen::Dynamic,
	          TrianglePacking Packing = TrianglePacking::Lower, int Options = 0>
	using SymmetricMatrix = Detail::triangular_compressed_alias_t<
	        Scalar, Dimension, Packing, Options, Detail::SymmetricTag>;
	template <typename Scalar, int Dimension = Eigen::Dynamic,
	          TrianglePacking Packing = TrianglePacking::Lower, int Options = 0>
	using AntiSymmetricMatrix = Detail::triangular_compressed_alias_t<
	        Scalar, Dimension, Packing, Options, Detail::AntiSymmetricTag>;
	template <typename Scalar, int Dimension = Eigen::Dynamic,
	          TrianglePacking Packing = TrianglePacking::Lower, int Options = 0>
	using HermitianMatrix = Detail::triangular_compressed_alias_t<
	        Scalar, Dimension, Packing, Options, Detail::HermitianTag>;
	template <typename Scalar, int Dimension = Eigen::Dynamic,
	          TrianglePacking Packing = TrianglePacking::Lower, int Options = 0>
	using AntiHermitianMatrix = Detail::triangular_compressed_alias_t<
	        Scalar, Dimension, Packing, Options, Detail::AntiHermitianTag>;

	using UpperTriangularMatrixXd = UpperTriangularMatrix<double>;
	using UpperTriangularMatrixXf = UpperTriangularMatrix<float>;
	using UpperTriangularMatrixXcd = UpperTriangularMatrix<std::complex<double>>;
	using UpperTriangularMatrixXcf = UpperTriangularMatrix<std::complex<float>>;
	using UpperTriangularMatrixXi = UpperTriangularMatrix<int>;
	using UpperTriangularMatrixXl = UpperTriangularMatrix<long>;
	using LowerTriangularMatrixXd = LowerTriangularMatrix<double>;
	using LowerTriangularMatrixXf = LowerTriangularMatrix<float>;
	using LowerTriangularMatrixXcd = LowerTriangularMatrix<std::complex<double>>;
	using LowerTriangularMatrixXcf = LowerTriangularMatrix<std::complex<float>>;
	using LowerTriangularMatrixXi = LowerTriangularMatrix<int>;
	using LowerTriangularMatrixXl = LowerTriangularMatrix<long>;
	using SymmetricMatrixXd = SymmetricMatrix<double>;
	using SymmetricMatrixXf = SymmetricMatrix<float>;
	using SymmetricMatrixXcd = SymmetricMatrix<std::complex<double>>;
	using SymmetricMatrixXcf = SymmetricMatrix<std::complex<float>>;
	using SymmetricMatrixXi = SymmetricMatrix<int>;
	using SymmetricMatrixXl = SymmetricMatrix<long>;
	using AntiSymmetricMatrixXd = AntiSymmetricMatrix<double>;
	using AntiSymmetricMatrixXf = AntiSymmetricMatrix<float>;
	using AntiSymmetricMatrixXcd = AntiSymmetricMatrix<std::complex<double>>;
	using AntiSymmetricMatrixXcf = AntiSymmetricMatrix<std::complex<float>>;
	using AntiSymmetricMatrixXi = AntiSymmetricMatrix<int>;
	using AntiSymmetricMatrixXl = AntiSymmetricMatrix<long>;
	using HermitianMatrixXd = HermitianMatrix<double>;
	using HermitianMatrixXf = HermitianMatrix<float>;
	using HermitianMatrixXcd = HermitianMatrix<std::complex<double>>;
	using HermitianMatrixXcf = HermitianMatrix<std::complex<float>>;
	using HermitianMatrixXi = HermitianMatrix<int>;
	using HermitianMatrixXl = HermitianMatrix<long>;
	using AntiHermitianMatrixXd = AntiHermitianMatrix<double>;
	using AntiHermitianMatrixXf = AntiHermitianMatrix<float>;
	using AntiHermitianMatrixXcd = AntiHermitianMatrix<std::complex<double>>;
	using AntiHermitianMatrixXcf = AntiHermitianMatrix<std::complex<float>>;
	using AntiHermitianMatrixXi = AntiHermitianMatrix<int>;
	using AntiHermitianMatrixXl = AntiHermitianMatrix<long>;
}  // namespace Hoppy
