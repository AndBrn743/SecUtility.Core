// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

// ReSharper disable CppUseDesignatedInitializers
#pragma once

#include <SecUtility/IO/PersistentStore.hpp>
#include <SecUtility/IO/PersistentStore/Detail/Adapter.hpp>
#include <SecUtility/IO/PersistentStore/Detail/Mapping.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>


namespace SecUtility::IO::PersistentStoreDetail
{
	inline constexpr auto EigenDenseEncodingId =
	        MakePersistentEncodingId("d48b7210-e260-5db2-983f-7d63c20f38f1");
	inline constexpr UInt64 EigenDenseFixedHeaderBytes = 80;
	inline constexpr UInt32 EigenColumnMajorCode = 0;

	template <typename T>
	struct eigen_component
	{
		using Type = T;
	};

	template <typename T>
	struct eigen_component<std::complex<T>>
	{
		using Type = T;
	};

	template <typename T>
	inline constexpr bool IsPersistentEigenRepresentation =
	        HasScalarEncoding<T>
	        && (std::is_integral_v<T>
	            || std::is_same_v<T, float> || std::is_same_v<T, double>
	            || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>)
	        && (std::is_integral_v<T>
	            || std::numeric_limits<typename eigen_component<T>::Type>::is_iec559)
	        && (!Eigen::NumTraits<T>::IsComplex
	            || sizeof(T) == 2 * sizeof(typename eigen_component<T>::Type));

	inline constexpr UInt64 EigenRequiredAlignment =
	        EIGEN_MAX_ALIGN_BYTES > 0 ? static_cast<UInt64>(EIGEN_MAX_ALIGN_BYTES) : UInt64{1};

	template <typename Scalar>
	PayloadLayout MeasureEigenDense(const Eigen::Index rows, const Eigen::Index columns)
	{
		static_assert(IsPersistentEigenRepresentation<Scalar>,
		              "PersistentStore Eigen scalar has no supported portable representation");
		if (rows < 0 || columns < 0)
		{
			throw FormatException("PersistentStore Eigen dimensions cannot be negative");
		}
		const UInt64 rowCount = static_cast<UInt64>(rows);
		const UInt64 columnCount = static_cast<UInt64>(columns);
		const UInt64 coefficientCount = CheckedMultiply(rowCount, columnCount, "Eigen coefficient count");
		const UInt64 dataBytes = CheckedMultiply(coefficientCount, sizeof(Scalar), "Eigen coefficient bytes");
		const UInt64 dataOffset = CheckedAlignUp(EigenDenseFixedHeaderBytes, EigenRequiredAlignment,
		                                               "Eigen coefficient offset");
		return {CheckedAdd(dataOffset, dataBytes, "Eigen payload bytes"), EigenRequiredAlignment};
	}

	template <typename T, typename = void>
	struct is_eigen_dense_expression : std::false_type {};

	template <typename T>
	struct is_eigen_dense_expression<
	        T, std::void_t<typename T::Scalar, decltype(std::declval<const T&>().rows()),
	                       decltype(std::declval<const T&>().cols())>>
	    : std::bool_constant<std::is_base_of_v<Eigen::DenseBase<T>, T>> {};

	template <typename T>
	inline constexpr bool IsEigenDenseExpression = is_eigen_dense_expression<T>::value;

	template <typename T>
	struct is_eigen_plain_matrix : std::false_type {};

	template <typename Scalar, int Rows, int Columns, int Options, int MaximumRows, int MaximumColumns>
	struct is_eigen_plain_matrix<Eigen::Matrix<Scalar, Rows, Columns, Options, MaximumRows, MaximumColumns>>
	    : std::true_type {};

	template <typename T>
	inline constexpr bool IsEigenPlainMatrix = is_eigen_plain_matrix<T>::value;

	template <typename Derived>
	void EncodeEigenDense(const Eigen::MatrixBase<Derived>& value, const MutableByteView bytes)
	{
		using Scalar = typename Derived::Scalar;
		const PayloadLayout layout = MeasureEigenDense<Scalar>(value.rows(), value.cols());
		if (bytes.size() != layout.Bytes)
		{
			throw FormatException("Eigen dense payload size changed after measurement");
		}
		const UInt64 dataOffset = CheckedAlignUp(EigenDenseFixedHeaderBytes, EigenRequiredAlignment,
		                                               "Eigen coefficient offset");
		const UInt64 coefficientCount = CheckedMultiply(static_cast<UInt64>(value.rows()),
		                                                static_cast<UInt64>(value.cols()),
		                                                "Eigen coefficient count");
		const UInt64 dataBytes = CheckedMultiply(coefficientCount, sizeof(Scalar), "Eigen coefficient bytes");
		WriteAdapterHeader(bytes, EigenDenseEncodingId, CheckedNarrow<UInt32>(dataOffset, "Eigen header bytes"),
		                   layout.Bytes, dataBytes);
		FormatCodecDetail::WriteLittleEndian<UInt32>(bytes, 40, scalar_encoding<Scalar>::Code,
		                                                "Eigen scalar code");
		FormatCodecDetail::WriteLittleEndian<UInt32>(bytes, 44, EigenColumnMajorCode, "Eigen storage order");
		FormatCodecDetail::WriteLittleEndian<UInt64>(bytes, 48, static_cast<UInt64>(value.rows()), "Eigen rows");
		FormatCodecDetail::WriteLittleEndian<UInt64>(bytes, 56, static_cast<UInt64>(value.cols()), "Eigen columns");
		FormatCodecDetail::WriteLittleEndian<UInt64>(bytes, 64, coefficientCount, "Eigen coefficient count");
		FormatCodecDetail::WriteLittleEndian<UInt64>(bytes, 72, dataOffset, "Eigen coefficient offset");
		std::fill(bytes.data() + EigenDenseFixedHeaderBytes, bytes.data() + dataOffset, Byte{0});
		using CanonicalMatrix = Eigen::MatrixX<Scalar>;
		Eigen::Map<CanonicalMatrix, Eigen::AlignedMax> staging(
		        reinterpret_cast<Scalar*>(bytes.data() + dataOffset), value.rows(), value.cols());
		Eigen::internal::call_assignment_no_alias(staging, value.derived());
	}

	struct EigenDenseMetadata
	{
		UInt64 Rows = 0;
		UInt64 Columns = 0;
		UInt64 CoefficientCount = 0;
		UInt64 DataOffset = 0;
		UInt32 StorageOrder = 0;
	};

	template <typename MatrixType>
	EigenDenseMetadata ValidateEigenDense(const ConstByteView bytes)
	{
		using Scalar = typename MatrixType::Scalar;
		if (bytes.size() < EigenDenseFixedHeaderBytes
		    || !std::equal(EigenDenseEncodingId.data(), EigenDenseEncodingId.data() + EigenDenseEncodingId.size(),
		                   bytes.data()))
		{
			throw FormatException("PersistentStore Eigen dense encoding type mismatch");
		}
		const UInt64 rows = FormatCodecDetail::ReadLittleEndian<UInt64>(bytes, 48, "Eigen rows");
		const UInt64 columns = FormatCodecDetail::ReadLittleEndian<UInt64>(bytes, 56, "Eigen columns");
		const UInt64 coefficientCount = FormatCodecDetail::ReadLittleEndian<UInt64>(bytes, 64,
		                                                                                 "Eigen coefficient count");
		const UInt64 dataOffset = FormatCodecDetail::ReadLittleEndian<UInt64>(bytes, 72,
		                                                                          "Eigen coefficient offset");
		const UInt64 expectedCount = CheckedMultiply(rows, columns, "Eigen coefficient count");
		const UInt64 dataBytes = CheckedMultiply(coefficientCount, sizeof(Scalar), "Eigen coefficient bytes");
		const UInt64 expectedOffset = CheckedAlignUp(EigenDenseFixedHeaderBytes, EigenRequiredAlignment,
		                                                   "Eigen coefficient offset");
		ValidateAdapterHeader(bytes, EigenDenseEncodingId, CheckedNarrow<UInt32>(dataOffset, "Eigen header bytes"),
		                      dataBytes);
		const UInt32 storageOrder = FormatCodecDetail::ReadLittleEndian<UInt32>(bytes, 44, "Eigen storage order");
		if (FormatCodecDetail::ReadLittleEndian<UInt32>(bytes, 40, "Eigen scalar code")
		            != scalar_encoding<Scalar>::Code
		    || storageOrder != EigenColumnMajorCode
		    || coefficientCount != expectedCount || dataOffset != expectedOffset
		    || dataOffset > bytes.size() || dataBytes != bytes.size() - dataOffset
		    || (MatrixType::RowsAtCompileTime != Eigen::Dynamic
		        && rows != static_cast<UInt64>(MatrixType::RowsAtCompileTime))
		    || (MatrixType::ColsAtCompileTime != Eigen::Dynamic
		        && columns != static_cast<UInt64>(MatrixType::ColsAtCompileTime)))
		{
			throw FormatException("malformed or incompatible PersistentStore Eigen dense payload");
		}
		(void)CheckedNarrow<Eigen::Index>(rows, "Eigen rows");
		(void)CheckedNarrow<Eigen::Index>(columns, "Eigen columns");
		const auto address = reinterpret_cast<std::uintptr_t>(bytes.data() + dataOffset);
		if (address % EigenRequiredAlignment != 0)
		{
			throw FormatException("misaligned PersistentStore Eigen coefficient data");
		}
		return {rows, columns, coefficientCount, dataOffset, storageOrder};
	}
}


namespace SecUtility::IO
{
	template <typename MatrixType>
	class LeasedEigenMatrix;
}


template <typename MatrixType>
struct Eigen::internal::traits<SecUtility::IO::LeasedEigenMatrix<MatrixType>>
{
	using Scalar = typename MatrixType::Scalar;
	using StorageKind = Dense;
	using StorageIndex = Eigen::Index;
	using XprKind = MatrixXpr;
	static constexpr int RowsAtCompileTime = MatrixType::RowsAtCompileTime;
	static constexpr int ColsAtCompileTime = MatrixType::ColsAtCompileTime;
	static constexpr int MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime;
	static constexpr int MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime;
	static constexpr int Flags = LinearAccessBit;
	static constexpr int CoeffReadCost = NumTraits<Scalar>::ReadCost;
};


namespace SecUtility::IO
{
	template <typename MatrixType>
	class LeasedEigenMatrix final : public Eigen::MatrixBase<LeasedEigenMatrix<MatrixType>>
	{
	public:
		using Scalar = typename MatrixType::Scalar;
		using Index = Eigen::Index;
		using Base = Eigen::MatrixBase<LeasedEigenMatrix>;

		LeasedEigenMatrix(std::shared_ptr<PersistentStoreDetail::MappingLease> leasePtr,
		                  const Scalar* dataPtr,
		                  const Index rows,
		                  const Index columns) noexcept
		    : m_LeasePtr(std::move(leasePtr)), m_DataPtr(dataPtr), m_Rows(rows), m_Columns(columns)
		{
			/* NO CODE */
		}

		LeasedEigenMatrix(const LeasedEigenMatrix&) = default;
		LeasedEigenMatrix(LeasedEigenMatrix&&) noexcept = default;
		LeasedEigenMatrix& operator=(const LeasedEigenMatrix& other)
		{
			if (this != &other)
			{
				m_LeasePtr = other.m_LeasePtr;
				m_DataPtr = other.m_DataPtr;
				m_Rows = other.m_Rows;
				m_Columns = other.m_Columns;
			}
			return *this;
		}
		LeasedEigenMatrix& operator=(LeasedEigenMatrix&& other) noexcept
		{
			if (this != &other)
			{
				m_LeasePtr = std::move(other.m_LeasePtr);
				m_DataPtr = other.m_DataPtr;
				m_Rows = other.m_Rows;
				m_Columns = other.m_Columns;
				other.m_DataPtr = nullptr;
				other.m_Rows = 0;
				other.m_Columns = 0;
			}
			return *this;
		}

		Index rows() const noexcept { return m_Rows; }
		Index cols() const noexcept { return m_Columns; }
		const Scalar& coeff(const Index row, const Index column) const noexcept
		{
			return m_DataPtr[column * m_Rows + row];
		}
		const Scalar& coeff(const Index index) const noexcept { return m_DataPtr[index]; }

		// Eigen expressions formed from this object borrow its mapped coefficients. Evaluate those
		// expressions before the leased matrix is destroyed; only direct copies retain the mapping lease.
		~LeasedEigenMatrix() = default;

	private:
		std::shared_ptr<PersistentStoreDetail::MappingLease> m_LeasePtr;
		const Scalar* m_DataPtr = nullptr;
		Index m_Rows = 0;
		Index m_Columns = 0;
	};

}


template <typename MatrixType>
struct Eigen::internal::evaluator<SecUtility::IO::LeasedEigenMatrix<MatrixType>>
    : evaluator_base<SecUtility::IO::LeasedEigenMatrix<MatrixType>>
{
	using XprType = SecUtility::IO::LeasedEigenMatrix<MatrixType>;
	using Scalar = typename XprType::Scalar;
	using CoeffReturnType = const Scalar&;

	static constexpr int CoeffReadCost = NumTraits<Scalar>::ReadCost;
	static constexpr int Flags = LinearAccessBit;
	static constexpr int Alignment = 0;

	explicit evaluator(const XprType& expression) noexcept : m_Expression(expression) { /* NO CODE */ }

	CoeffReturnType coeff(const Eigen::Index row, const Eigen::Index column) const noexcept
	{
		return m_Expression.coeff(row, column);
	}
	CoeffReturnType coeff(const Eigen::Index index) const noexcept { return m_Expression.coeff(index); }

private:
	const XprType& m_Expression;
};


namespace SecUtility::IO
{

	template <typename Scalar, int Rows, int Columns, int Options, int MaximumRows, int MaximumColumns>
	struct PersistentTraits<Eigen::Matrix<Scalar, Rows, Columns, Options, MaximumRows, MaximumColumns>,
	                        std::enable_if_t<PersistentStoreDetail::IsPersistentEigenRepresentation<Scalar>>>
	{
		using MatrixType = Eigen::Matrix<Scalar, Rows, Columns, Options, MaximumRows, MaximumColumns>;
		using LeasedType = LeasedEigenMatrix<MatrixType>;

		template <typename Derived>
		static PersistentStoreDetail::PayloadLayout Measure(const Eigen::MatrixBase<Derived>& value)
		{
			static_assert(std::is_same_v<typename Derived::Scalar, Scalar>,
			              "PersistentStore Eigen insertion cannot change scalar type");
			if ((Rows != Eigen::Dynamic && value.rows() != Rows)
			    || (Columns != Eigen::Dynamic && value.cols() != Columns)
			    || (MaximumRows != Eigen::Dynamic && value.rows() > MaximumRows)
			    || (MaximumColumns != Eigen::Dynamic && value.cols() > MaximumColumns))
			{
				throw FormatException("PersistentStore Eigen insertion dimensions do not match the stored type");
			}
			return PersistentStoreDetail::MeasureEigenDense<Scalar>(value.rows(), value.cols());
		}

		template <typename Derived>
		static void Encode(const Eigen::MatrixBase<Derived>& value,
		                   const PersistentStoreDetail::MutableByteView bytes)
		{
			(void)Measure(value);
			PersistentStoreDetail::EncodeEigenDense(value, bytes);
		}

		static MatrixType Decode(const PersistentStoreDetail::ConstByteView bytes)
		{
			using namespace PersistentStoreDetail;
			const EigenDenseMetadata metadata = ValidateEigenDense<MatrixType>(bytes);
			const auto rows = CheckedNarrow<Eigen::Index>(metadata.Rows, "Eigen rows");
			const auto columns = CheckedNarrow<Eigen::Index>(metadata.Columns, "Eigen columns");
			using ColumnMajorMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
			const Scalar* const dataPtr = reinterpret_cast<const Scalar*>(bytes.data() + metadata.DataOffset);
			return MatrixType(Eigen::Map<const ColumnMajorMatrix, Eigen::AlignedMax>(dataPtr, rows, columns));
		}

		static LeasedType DecodeLeased(std::shared_ptr<PersistentStoreDetail::MappingLease> leasePtr)
		{
			using namespace PersistentStoreDetail;
			const ConstByteView bytes(leasePtr->Data(), leasePtr->Size());
			const EigenDenseMetadata metadata = ValidateEigenDense<MatrixType>(bytes);
			return LeasedType(std::move(leasePtr),
			                  reinterpret_cast<const Scalar*>(bytes.data() + metadata.DataOffset),
			                  CheckedNarrow<Eigen::Index>(metadata.Rows, "Eigen rows"),
			                  CheckedNarrow<Eigen::Index>(metadata.Columns, "Eigen columns"));
		}
	};

	template <typename DenseExpression>
	struct PersistentTraits<
	        DenseExpression,
	        std::enable_if_t<PersistentStoreDetail::IsEigenDenseExpression<DenseExpression>
	                         && !PersistentStoreDetail::IsEigenPlainMatrix<DenseExpression>
	                         && PersistentStoreDetail::IsPersistentEigenRepresentation<
	                                 typename DenseExpression::Scalar>>>
	{
		static PersistentStoreDetail::PayloadLayout Measure(const DenseExpression& value)
		{
			return PersistentStoreDetail::MeasureEigenDense<typename DenseExpression::Scalar>(value.rows(), value.cols());
		}

		static void Encode(const DenseExpression& value, const PersistentStoreDetail::MutableByteView bytes)
		{
			PersistentStoreDetail::EncodeEigenDense(value, bytes);
		}
	};
}
