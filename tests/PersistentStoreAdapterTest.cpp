#include <SecUtility/IO/PersistentStore/Adapter/Array.hpp>
#include <SecUtility/IO/PersistentStore/Adapter/ByteSequence.hpp>
#include <SecUtility/IO/PersistentStore/Adapter/EigenDense.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <new>
#include <string>
#include <system_error>
#include <type_traits>
#include <vector>


#if defined(__GNUC__) && !defined(__clang__)
// This test intentionally implements the replaceable global new/delete family with malloc/free so it can
// observe allocation sizes. GCC's interprocedural -Wmismatched-new-delete analysis treats that conforming
// replacement implementation as if malloc were called directly by each optimized allocation site.
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif


using namespace SecUtility;
using namespace SecUtility::IO;
using namespace SecUtility::IO::PersistentStoreDetail;


namespace AllocationTracking
{
	inline std::atomic<bool> IsEnabled{false};
	inline std::atomic<std::size_t> MaximumBytes{0};

	void Observe(const std::size_t bytes) noexcept
	{
		if (!IsEnabled.load(std::memory_order_relaxed))
		{
			return;
		}
		std::size_t maximum = MaximumBytes.load(std::memory_order_relaxed);
		while (maximum < bytes
		       && !MaximumBytes.compare_exchange_weak(maximum, bytes, std::memory_order_relaxed))
		{
			/* NO CODE */
		}
	}
}


void* operator new(std::size_t bytes)
{
	AllocationTracking::Observe(bytes);
	if (void* const memoryPtr = std::malloc(bytes == 0 ? 1 : bytes))
	{
		return memoryPtr;
	}
	throw std::bad_alloc();
}


void operator delete(void* memoryPtr) noexcept { std::free(memoryPtr); }
void operator delete(void* memoryPtr, std::size_t) noexcept { std::free(memoryPtr); }


namespace
{
	template <std::size_t Size>
	constexpr bool AreNonzeroAndUnique(const std::array<UInt32, Size>& values) noexcept
	{
		for (std::size_t left = 0; left < values.size(); ++left)
		{
			if (values[left] == 0)
			{
				return false;
			}
			for (std::size_t right = left + 1; right < values.size(); ++right)
			{
				if (values[left] == values[right])
				{
					return false;
				}
			}
		}
		return true;
	}

	struct Configuration
	{
		std::int32_t Iterations = 0;
		double Threshold = 0;
	};

	struct ThrowingValue
	{
		inline static unsigned int MeasureCalls = 0;
		inline static unsigned int EncodeCalls = 0;
		inline static bool ShouldThrowDuringMeasure = false;
		inline static bool ShouldThrowDuringEncode = false;
	};

	struct LargeGeneratedValue
	{
		static constexpr std::size_t Bytes = 4 * 1024 * 1024;
	};

	class TemporaryPath final
	{
	public:
		TemporaryPath()
		{
			m_Path = std::filesystem::temp_directory_path()
			         / ("secutility-persistent-store-adapter-"
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

	DirectoryEntry ReadEntry(FileBackend& ref_backend, const std::string& key)
	{
		const UInt64 physicalBytes = ref_backend.GetPhysicalFileBytes();
		std::array<Byte, HeaderSlotBytes> slotABytes{};
		std::array<Byte, HeaderSlotBytes> slotBBytes{};
		ref_backend.ReadExact(HeaderSlotAOffset, slotABytes.data(), slotABytes.size());
		ref_backend.ReadExact(HeaderSlotBOffset, slotBBytes.data(), slotBBytes.size());
		const HeaderSlot slotA = ParseHeader(ConstByteView(slotABytes), physicalBytes);
		const HeaderSlot slotB = ParseHeader(ConstByteView(slotBBytes), physicalBytes);
		const HeaderSlot& selected = slotA.Generation >= slotB.Generation ? slotA : slotB;
		std::vector<Byte> directoryBytes(static_cast<std::size_t>(selected.DirectoryBytes));
		ref_backend.ReadExact(selected.DirectoryOffset, directoryBytes.data(), directoryBytes.size());
		const Directory directory = ParseDirectory(selected, ConstByteView(directoryBytes), physicalBytes);
		const auto iterator = std::find_if(directory.Entries.begin(), directory.Entries.end(),
		                                   [&key](const DirectoryEntry& entry) { return entry.Key == key; });
		REQUIRE(iterator != directory.Entries.end());
		return *iterator;
	}
}


namespace SecUtility::IO
{
	template <>
	struct NativeObjectDescriptor<Configuration>
	{
		inline static constexpr auto EncodingId =
		        MakePersistentEncodingId("a4ab9f58-6d4d-4f83-a184-c4e88f5d20bd");
		inline static constexpr UInt32 SchemaVersion = 3;
	};

	template <>
	struct PersistentTraits<ThrowingValue>
	{
		static PersistentStoreDetail::PayloadLayout Measure(const ThrowingValue&)
		{
			++ThrowingValue::MeasureCalls;
			if (ThrowingValue::ShouldThrowDuringMeasure)
			{
				throw InvalidOperationException("injected adapter measurement failure");
			}
			return {1, 1};
		}

		static void Encode(const ThrowingValue&, const PersistentStoreDetail::MutableByteView bytes)
		{
			++ThrowingValue::EncodeCalls;
			if (ThrowingValue::ShouldThrowDuringEncode)
			{
				throw InvalidOperationException("injected adapter encoding failure");
			}
			bytes.data()[0] = Byte{0x5A};
		}

		static ThrowingValue Decode(const PersistentStoreDetail::ConstByteView) { return {}; }
	};

	template <>
	struct PersistentTraits<LargeGeneratedValue>
	{
		static PersistentStoreDetail::PayloadLayout Measure(const LargeGeneratedValue&)
		{
			return {LargeGeneratedValue::Bytes, 1};
		}

		static void Encode(const LargeGeneratedValue&, const PersistentStoreDetail::MutableByteView bytes)
		{
			std::fill(bytes.data(), bytes.data() + bytes.size(), Byte{0x6D});
		}

		static LargeGeneratedValue Decode(const PersistentStoreDetail::ConstByteView) { return {}; }
	};
}


TEST_CASE("PersistentStore native scalar adapters preserve boundary representations")
{
	TemporaryPath temporary;
	auto store = PersistentStore::Create(temporary.Get());
	store.Insert("i8", std::int8_t{-128});
	store.Insert("u64", std::numeric_limits<std::uint64_t>::max());
	store.Insert("negative-zero", -0.0);
	store.Insert("infinity", std::numeric_limits<double>::infinity());
	store.Insert("nan", std::numeric_limits<double>::quiet_NaN());
	CHECK(store.Get<std::int8_t>("i8") == -128);
	CHECK(store.Get<std::uint64_t>("u64") == std::numeric_limits<std::uint64_t>::max());
	CHECK(std::signbit(store.Get<double>("negative-zero")));
	CHECK(std::isinf(store.Get<double>("infinity")));
	CHECK(std::isnan(store.Get<double>("nan")));
}


TEST_CASE("PersistentStore registered native objects round trip")
{
	static_assert(std::is_trivially_copyable_v<Configuration>);
	static_assert(!PersistentStoreDetail::HasLeasedType<Configuration>);
	static_assert(!PersistentStoreDetail::HasLeasedType<std::int32_t>);

	TemporaryPath temporary;
	{
		auto store = PersistentStore::Create(temporary.Get());
		store.Insert("configuration", Configuration{14, 1.25e-8});
	}
	const auto store = PersistentStore::OpenForReadOnly(temporary.Get());
	const Configuration result = store.Get<Configuration>("configuration");
	CHECK(result.Iterations == 14);
	CHECK(result.Threshold == 1.25e-8);
}


TEST_CASE("PersistentStore adapter encoding identifiers are stable and distinct")
{
	using namespace PersistentStoreDetail;
	constexpr auto parsed = MakePersistentEncodingId("00112233-4455-6677-8899-aabbccddeeff");
	static_assert(parsed.size() == 16);
	static_assert(parsed.data()[0] == Byte{0x00});
	static_assert(parsed.data()[7] == Byte{0x77});
	static_assert(parsed.data()[8] == Byte{0x88});
	static_assert(parsed.data()[15] == Byte{0xFF});
	CHECK(NativeScalarEncodingId != NativeObjectEncodingId);
	CHECK(NativeScalarEncodingId != ArrayEncodingId);
	CHECK(NativeScalarEncodingId != ByteSequenceEncodingId);
	CHECK(NativeObjectEncodingId != ArrayEncodingId);
	CHECK(NativeObjectEncodingId != ByteSequenceEncodingId);
	CHECK(ArrayEncodingId != ByteSequenceEncodingId);
	CHECK(EigenDenseEncodingId != NativeScalarEncodingId);
	CHECK(EigenDenseEncodingId != NativeObjectEncodingId);
	CHECK(EigenDenseEncodingId != ArrayEncodingId);
	CHECK(EigenDenseEncodingId != ByteSequenceEncodingId);

	constexpr std::array<UInt32, 12> scalarCodes{
	        scalar_encoding<std::int8_t>::Code,
	        scalar_encoding<std::uint8_t>::Code,
	        scalar_encoding<std::int16_t>::Code,
	        scalar_encoding<std::uint16_t>::Code,
	        scalar_encoding<std::int32_t>::Code,
	        scalar_encoding<std::uint32_t>::Code,
	        scalar_encoding<std::int64_t>::Code,
	        scalar_encoding<std::uint64_t>::Code,
	        scalar_encoding<float>::Code,
	        scalar_encoding<double>::Code,
	        scalar_encoding<std::complex<float>>::Code,
	        scalar_encoding<std::complex<double>>::Code};
	static_assert(AreNonzeroAndUnique(scalarCodes));
	static_assert(!IsPersistentScalar<std::complex<float>>);
	static_assert(!IsPersistentScalar<std::complex<double>>);
	CHECK(AreNonzeroAndUnique(scalarCodes));
}


TEST_CASE("PersistentStore checks key preconditions before adapters and preserves the snapshot on adapter failure")
{
	TemporaryPath temporary;
	auto store = PersistentStore::Create(temporary.Get());
	store.Insert("existing", std::int32_t{7});
	ThrowingValue::MeasureCalls = 0;
	ThrowingValue::EncodeCalls = 0;
	ThrowingValue explicitSameTypeValue;
	CHECK_THROWS_AS(store.Insert<ThrowingValue>("existing", explicitSameTypeValue), KeyAlreadyExistsException);
	CHECK(ThrowingValue::MeasureCalls == 0);
	CHECK(ThrowingValue::EncodeCalls == 0);

	ThrowingValue::ShouldThrowDuringMeasure = true;
	CHECK_THROWS_AS(store.Insert("measure-failure", ThrowingValue{}), InvalidOperationException);
	ThrowingValue::ShouldThrowDuringMeasure = false;
	CHECK_FALSE(store.Contains("measure-failure"));
	CHECK(store.Get<std::int32_t>("existing") == 7);

	ThrowingValue::ShouldThrowDuringEncode = true;
	CHECK_THROWS_AS(store.Insert("encode-failure", ThrowingValue{}), InvalidOperationException);
	ThrowingValue::ShouldThrowDuringEncode = false;
	CHECK_FALSE(store.Contains("encode-failure"));
	CHECK(store.Get<std::int32_t>("existing") == 7);
}


TEST_CASE("PersistentStore adapters reject corrupted representation metadata")
{
	TemporaryPath temporary;
	{
		auto store = PersistentStore::Create(temporary.Get());
		store.Insert("array", std::vector<std::int32_t>{1, 2, 3});
	}
	{
		auto backend = FileBackend::Open(temporary.Get(), FileAccess::ReadWrite);
		const DirectoryEntry entry = ReadEntry(backend, "array");
		std::array<Byte, 4> invalidScalarCode{};
		backend.WriteExact(entry.PayloadOffset + 40, invalidScalarCode.data(), invalidScalarCode.size());
	}
	const auto store = PersistentStore::OpenForReadOnly(temporary.Get());
	CHECK_THROWS_AS(store.Get<std::vector<std::int32_t>>("array"), FormatException);
}


TEST_CASE("PersistentStore direct staging does not allocate a payload-sized core buffer")
{
	TemporaryPath temporary;
	auto store = PersistentStore::Create(temporary.Get());
	AllocationTracking::MaximumBytes.store(0, std::memory_order_relaxed);
	AllocationTracking::IsEnabled.store(true, std::memory_order_relaxed);
	store.Insert("large", LargeGeneratedValue{});
	AllocationTracking::IsEnabled.store(false, std::memory_order_relaxed);
	CHECK(AllocationTracking::MaximumBytes.load(std::memory_order_relaxed) < LargeGeneratedValue::Bytes / 2);
}
