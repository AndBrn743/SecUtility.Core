#include <SecUtility/IO/PersistentStore/Adapter/EigenDense.hpp>
#include <SecUtility/IO/PersistentStore/Detail/Directory.hpp>
#include <SecUtility/IO/PersistentStore/Detail/FileBackend.hpp>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>

#include <chrono>
#include <array>
#include <algorithm>
#include <complex>
#include <filesystem>
#include <limits>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>


using namespace SecUtility;
using namespace SecUtility::IO;
using namespace SecUtility::IO::PersistentStoreDetail;


namespace
{
	class TemporaryPath final
	{
	public:
		TemporaryPath()
		{
			m_Path = std::filesystem::temp_directory_path()
			         / ("secutility-persistent-store-eigen-"
			            + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-"
			            + std::to_string(++Counter));
		}
		~TemporaryPath()
		{
			std::error_code ignoredError;
			std::filesystem::remove(m_Path, ignoredError);
		}
		const std::filesystem::path& Get() const noexcept { return m_Path; }

	private:
		inline static unsigned long long Counter = 0;
		std::filesystem::path m_Path;
	};

	template <typename Scalar>
	void CheckScalarRoundTrip()
	{
		TemporaryPath temporary;
		Eigen::MatrixX<Scalar> expected(2, 3);
		for (Eigen::Index column = 0; column < expected.cols(); ++column)
		{
			for (Eigen::Index row = 0; row < expected.rows(); ++row)
			{
				if constexpr (Eigen::NumTraits<Scalar>::IsComplex)
				{
					using Component = typename Scalar::value_type;
					expected(row, column) = Scalar(static_cast<Component>(row + column * 3 - 2),
					                               static_cast<Component>(column - row * 2 + 1));
				}
				else
				{
					expected(row, column) = static_cast<Scalar>(row + column * 3) / static_cast<Scalar>(7);
				}
			}
		}
		{
			auto store = PersistentStore::Create(temporary.Get());
			store.Insert("matrix", expected);
		}
		const auto store = PersistentStore::OpenForReadOnly(temporary.Get());
		CHECK(store.Get<Eigen::MatrixX<Scalar>>("matrix") == expected);
		CHECK(Eigen::MatrixX<Scalar>(store.GetLeased<Eigen::MatrixX<Scalar>>("matrix")) == expected);
	}

	template <typename Scalar>
	void CheckIntegralExtremes()
	{
		TemporaryPath temporary;
		Eigen::MatrixX<Scalar> expected(2, 2);
		expected << std::numeric_limits<Scalar>::lowest(), std::numeric_limits<Scalar>::max(),
		        Scalar{0}, Scalar{1};
		auto store = PersistentStore::Create(temporary.Get());
		store.Insert("matrix", expected);
		CHECK(store.Get<Eigen::MatrixX<Scalar>>("matrix") == expected);
	}

	using Lease = PersistentTraits<Eigen::MatrixXd>::LeasedType;
	using RowMajorLease = PersistentTraits<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>>::LeasedType;
	using ConstMap = Eigen::Map<const Eigen::MatrixXd, Eigen::AlignedMax>;
	static_assert(PersistentStoreDetail::HasLeasedType<Eigen::MatrixXd>);
	static_assert(PersistentStoreDetail::HasLeasedType<Eigen::MatrixXi>);
	static_assert(!std::is_convertible_v<Lease&, ConstMap&>);
	static_assert(!std::is_assignable_v<decltype(std::declval<Lease&>().coeff(0, 0)), double>);
	static_assert((Eigen::internal::traits<Lease>::Flags & Eigen::RowMajorBit) == 0);
	static_assert((Eigen::internal::traits<RowMajorLease>::Flags & Eigen::RowMajorBit) == 0);

	std::vector<Byte> ReadPayload(const std::filesystem::path& path, const std::string& key)
	{
		auto backend = FileBackend::Open(path, FileAccess::ReadOnly);
		const UInt64 physicalBytes = backend.GetPhysicalFileBytes();
		std::array<Byte, HeaderSlotBytes> slotABytes{};
		std::array<Byte, HeaderSlotBytes> slotBBytes{};
		backend.ReadExact(HeaderSlotAOffset, slotABytes.data(), slotABytes.size());
		backend.ReadExact(HeaderSlotBOffset, slotBBytes.data(), slotBBytes.size());
		const HeaderSlot slotA = ParseHeader(ConstByteView(slotABytes), physicalBytes);
		const HeaderSlot slotB = ParseHeader(ConstByteView(slotBBytes), physicalBytes);
		const HeaderSlot& selected = slotA.Generation >= slotB.Generation ? slotA : slotB;
		std::vector<Byte> directoryBytes(static_cast<std::size_t>(selected.DirectoryBytes));
		backend.ReadExact(selected.DirectoryOffset, directoryBytes.data(), directoryBytes.size());
		const Directory directory = ParseDirectory(selected, ConstByteView(directoryBytes), physicalBytes);
		const auto iterator = std::find_if(directory.Entries.begin(), directory.Entries.end(),
		                                   [&key](const DirectoryEntry& entry) { return entry.Key == key; });
		REQUIRE(iterator != directory.Entries.end());
		std::vector<Byte> payload(static_cast<std::size_t>(iterator->PayloadBytes));
		backend.ReadExact(iterator->PayloadOffset, payload.data(), payload.size());
		return payload;
	}
}


TEST_CASE("PersistentStore Eigen dense adapter preserves every supported scalar")
{
	CheckIntegralExtremes<std::int8_t>();
	CheckIntegralExtremes<std::uint8_t>();
	CheckIntegralExtremes<std::int16_t>();
	CheckIntegralExtremes<std::uint16_t>();
	CheckIntegralExtremes<std::int32_t>();
	CheckIntegralExtremes<std::uint32_t>();
	CheckIntegralExtremes<std::int64_t>();
	CheckIntegralExtremes<std::uint64_t>();
	CheckScalarRoundTrip<float>();
	CheckScalarRoundTrip<double>();
	CheckScalarRoundTrip<std::complex<float>>();
	CheckScalarRoundTrip<std::complex<double>>();
}


TEST_CASE("PersistentStore Eigen integral scalars follow their fixed-width aliases")
{
	CheckIntegralExtremes<short>();
	CheckIntegralExtremes<unsigned short>();
	CheckIntegralExtremes<int>();
	CheckIntegralExtremes<unsigned int>();
	CheckIntegralExtremes<long>();
	CheckIntegralExtremes<unsigned long>();
	CheckIntegralExtremes<long long>();
	CheckIntegralExtremes<unsigned long long>();
}


TEST_CASE("PersistentStore Eigen owning decode handles dimensions vectors storage order and empty shapes")
{
	TemporaryPath temporary;
	auto store = PersistentStore::Create(temporary.Get());
	Eigen::Matrix<double, 2, 3, Eigen::RowMajor> rowMajor;
	rowMajor << 1, 2, 3, 4, 5, 6;
	store.Insert("row-major", rowMajor);
	store.Insert("vector", Eigen::Vector3d{7, 8, 9});
	store.Insert<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>("empty", Eigen::MatrixXd(0, 3));

	CHECK((store.Get<Eigen::Matrix<double, 2, 3>>("row-major") == rowMajor));
	CHECK((Eigen::Matrix<double, 2, 3, Eigen::RowMajor>(
	               store.GetLeased<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>>("row-major"))
	       == rowMajor));
	CHECK((Eigen::Matrix<double, 2, 3>(store.GetLeased<Eigen::Matrix<double, 2, 3>>("row-major"))
	       == rowMajor));
	CHECK((store.Get<Eigen::Vector3d>("vector") == Eigen::Vector3d{7, 8, 9}));
	const Eigen::MatrixXd empty = store.Get<Eigen::MatrixXd>("empty");
	CHECK(empty.rows() == 0);
	CHECK(empty.cols() == 3);
	CHECK_THROWS_AS((store.Get<Eigen::Matrix<double, 3, 2>>("row-major")), FormatException);
}


TEST_CASE("PersistentStore Eigen insertion evaluates dense expressions once into staging")
{
	TemporaryPath temporary;
	auto store = PersistentStore::Create(temporary.Get());
	Eigen::Index evaluations = 0;
	const auto generated = Eigen::MatrixXd::NullaryExpr(4, 3, [&evaluations](const Eigen::Index row,
	                                                                       const Eigen::Index column)
	{
		++evaluations;
		return static_cast<double>(row * 10 + column);
	});
	store.Insert<Eigen::MatrixXd>("generated", generated);
	CHECK(evaluations == 12);

	const Eigen::MatrixXd left = Eigen::MatrixXd::Random(4, 3);
	const Eigen::MatrixXd right = Eigen::MatrixXd::Random(4, 3);
	store.Insert("sum", left + right);
	store.Insert("block", left.block(1, 0, 2, 2));
	store.Insert("transpose", left.transpose());
	CHECK(store.Get<Eigen::MatrixXd>("sum").isApprox(left + right));
	CHECK(store.Get<Eigen::MatrixXd>("block").isApprox(left.block(1, 0, 2, 2)));
	CHECK(store.Get<Eigen::MatrixXd>("transpose").isApprox(left.transpose()));
}


TEST_CASE("PersistentStore Eigen leases retain snapshots and support read-only expressions")
{
	TemporaryPath temporary;
	Eigen::MatrixXd original = Eigen::MatrixXd::Random(3, 2);
	Eigen::MatrixXd replacement = Eigen::MatrixXd::Random(3, 2);
	Lease lease = [&]
	{
		auto store = PersistentStore::Create(temporary.Get());
		store.Insert("matrix", original);
		store.Insert("erased", original);
		auto first = store.GetLeased<Eigen::MatrixXd>("matrix");
		auto erased = store.GetLeased<Eigen::MatrixXd>("erased");
		auto copied = first;
		auto assigned = copied;
		assigned = first;
		store.Reassign("matrix", replacement);
		CHECK(store.Erase("erased"));
		CHECK(Eigen::MatrixXd(assigned) == original);
		CHECK(Eigen::MatrixXd(erased) == original);
		CHECK((assigned.transpose() * Eigen::Vector3d::Ones()).isApprox(original.transpose()
		                                                                    * Eigen::Vector3d::Ones()));
		return copied;
	}();
	CHECK(Eigen::MatrixXd(lease) == original);
	const Eigen::MatrixXd materialized = lease;
	lease = Lease(lease);
	CHECK(materialized == original);
}


TEST_CASE("PersistentStore Eigen metadata validation rejects malformed representations")
{
	TemporaryPath temporary;
	{
		auto store = PersistentStore::Create(temporary.Get());
		store.Insert("matrix", Eigen::MatrixXd::Identity(2, 2));
	}
	const std::vector<Byte> payload = ReadPayload(temporary.Get(), "matrix");
	std::vector<Byte> alignedStorage(payload.size() + static_cast<std::size_t>(EigenRequiredAlignment));
	const auto address = reinterpret_cast<std::uintptr_t>(alignedStorage.data());
	const auto adjustment = static_cast<std::size_t>((EigenRequiredAlignment - address % EigenRequiredAlignment)
	                                                 % EigenRequiredAlignment);
	Byte* const alignedPtr = alignedStorage.data() + adjustment;

	auto CheckRejected = [&](const auto corrupt)
	{
		std::copy(payload.begin(), payload.end(), alignedPtr);
		MutableByteView bytes(alignedPtr, payload.size());
		corrupt(bytes);
		CHECK_THROWS_AS((ValidateEigenDense<Eigen::MatrixXd>(ConstByteView(bytes.data(), bytes.size()))),
		                FormatException);
	};

	CheckRejected([](const MutableByteView bytes)
	{
		FormatCodecDetail::WriteLittleEndian<UInt32>(bytes, 40, 0xFFFFU, "bad Eigen scalar code");
	});
	CheckRejected([](const MutableByteView bytes)
	{
		FormatCodecDetail::WriteLittleEndian<UInt32>(bytes, 44, 9, "bad Eigen storage order");
	});
	CheckRejected([](const MutableByteView bytes)
	{
		FormatCodecDetail::WriteLittleEndian<UInt64>(bytes, 48, std::numeric_limits<UInt64>::max(),
		                                                "bad Eigen rows");
	});
	CheckRejected([](const MutableByteView bytes)
	{
		FormatCodecDetail::WriteLittleEndian<UInt64>(bytes, 64, 3, "bad Eigen coefficient count");
	});
	CheckRejected([](const MutableByteView bytes)
	{
		FormatCodecDetail::WriteLittleEndian<UInt64>(bytes, 72, EigenDenseFixedHeaderBytes + 1,
		                                                "bad Eigen coefficient offset");
	});

	std::copy(payload.begin(), payload.end(), alignedPtr + 1);
	CHECK_THROWS_AS((ValidateEigenDense<Eigen::MatrixXd>(ConstByteView(alignedPtr + 1, payload.size()))),
	                FormatException);
}
