#include <SecUtility/IO/PersistentStore/Detail/Mapping.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <system_error>
#include <utility>


using namespace SecUtility;
using namespace SecUtility::IO::PersistentStoreDetail;


namespace
{
	class TemporaryPath final
	{
	public:
		TemporaryPath()
		{
			const auto token = std::chrono::steady_clock::now().time_since_epoch().count();
			m_Path = std::filesystem::temp_directory_path()
			         / ("secutility-persistent-store-mapping-" + std::to_string(token) + "-"
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

	FileBackend CreateBackend(const std::filesystem::path& path, const UInt64 byteCount)
	{
		auto backendOptional = FileBackend::TryCreateExclusive(path);
		REQUIRE(backendOptional.has_value());
		FileBackend backend = std::move(*backendOptional);
		backend.SetPhysicalFileBytes(byteCount);
		return backend;
	}

	void Fill(ReadWriteMappedRegion& ref_region, const Byte value)
	{
		std::fill(ref_region.Data(), ref_region.Data() + ref_region.Size(), value);
	}

	void CheckFilled(const ReadOnlyMappedRegion& region, const Byte value)
	{
		CHECK(std::all_of(region.Data(), region.Data() + region.Size(),
		                  [value](const Byte byte) { return byte == value; }));
	}
}


TEST_CASE("PersistentStore mapping exposes bounded aligned regions")
{
	TemporaryPath temporary;
	const UInt64 granularity = FileBackend::GetMappingGranularity();
	const UInt64 physicalBytes = granularity * 3 + MaximumPayloadAlignment;
	auto backend = CreateBackend(temporary.Get(), physicalBytes);

	const auto emptyAtStart = backend.MapReadOnly(0, 0, 1);
	const auto emptyAtEnd = backend.MapReadOnly(physicalBytes, 0, 1);
	CHECK(emptyAtStart.Data() == nullptr);
	CHECK(emptyAtStart.Size() == 0);
	CHECK(emptyAtEnd.Data() == nullptr);
	CHECK(emptyAtEnd.Size() == 0);

	const std::array<UInt64, 5> offsets{1, granularity - 1, granularity, granularity + 1,
	                                           granularity * 2 + 1};
	for (const UInt64 offset : offsets)
	{
		auto writable = backend.MapReadWrite(offset, 3, 1);
		CHECK(writable.Size() == 3);
		Fill(writable, Byte{0x5A});
		writable.Complete();
		CHECK(writable.Data() == nullptr);
		CHECK(writable.Size() == 0);

		const auto readable = backend.MapReadOnly(offset, 3, 1);
		CheckFilled(readable, Byte{0x5A});
	}

	for (UInt64 alignment = 1; alignment <= MaximumPayloadAlignment; alignment *= 2)
	{
		const UInt64 offset = granularity * 2;
		const auto readable = backend.MapReadOnly(offset, 1, alignment);
		CHECK(reinterpret_cast<std::uintptr_t>(readable.Data()) % alignment == 0);
	}
}


TEST_CASE("PersistentStore mapping rejects invalid ranges and access")
{
	TemporaryPath temporary;
	{
		auto backend = CreateBackend(temporary.Get(), 4096);

		CHECK_THROWS_AS(backend.MapReadOnly(4095, 2, 1), IOException);
		CHECK_THROWS_AS(backend.MapReadOnly(std::numeric_limits<UInt64>::max(), 2, 1), IOException);
		CHECK_THROWS_AS(backend.MapReadOnly(0, 1, 0), IOException);
		CHECK_THROWS_AS(backend.MapReadOnly(0, 1, 3), IOException);
		CHECK_THROWS_AS(backend.MapReadOnly(0, 1, MaximumPayloadAlignment * 2), IOException);
		CHECK_THROWS_AS(backend.MapReadOnly(1, 1, 2), IOException);
	}

	auto backend = FileBackend::Open(temporary.Get(), FileAccess::ReadOnly);
	CHECK_THROWS_AS(backend.MapReadWrite(0, 1, 1), InvalidOperationException);
}


TEST_CASE("PersistentStore retained mappings survive growth and coherent mapped reuse")
{
	TemporaryPath temporary;
	const UInt64 granularity = FileBackend::GetMappingGranularity();
	const UInt64 oldOffset = 32;
	const UInt64 oldBytes = 16;
	const UInt64 reusableOffset = granularity;
	const UInt64 reusableBytes = 16;
	const UInt64 initialBytes = granularity * 2;
	auto backend = CreateBackend(temporary.Get(), initialBytes);

	{
		auto writable = backend.MapReadWrite(oldOffset, oldBytes, 1);
		Fill(writable, Byte{0x11});
		writable.Complete();
	}
	{
		auto writable = backend.MapReadWrite(reusableOffset, reusableBytes, 1);
		Fill(writable, Byte{0x22});
		writable.Complete();
	}

	const auto oldView = backend.MapReadOnly(oldOffset, oldBytes, 1);
	const auto broadView = backend.MapReadOnly(0, initialBytes, 1);
	const Byte* const oldDataPtr = oldView.Data();
	CheckFilled(oldView, Byte{0x11});

	backend.SetPhysicalFileBytes(granularity * 3);
	const UInt64 grownOffset = granularity * 2 + 64;
	{
		auto writable = backend.MapReadWrite(grownOffset, 16, 1);
		Fill(writable, Byte{0x33});
		writable.Complete();
	}
	const auto grownView = backend.MapReadOnly(grownOffset, 16, 1);
	CheckFilled(grownView, Byte{0x33});
	CHECK(oldView.Data() == oldDataPtr);
	CheckFilled(oldView, Byte{0x11});

	const std::array<UInt64, 3> reuseOffsets{granularity - 1, granularity, granularity + 1};
	for (std::size_t index = 0; index < reuseOffsets.size(); ++index)
	{
		const UInt64 offset = reuseOffsets[index];
		const Byte replacement{static_cast<unsigned char>(0x40 + index)};
		auto replacementView = backend.MapReadWrite(offset, 1, 1);
		replacementView.Data()[0] = replacement;
		replacementView.Complete();

		const auto freshView = backend.MapReadOnly(offset, 1, 1);
		CHECK(freshView.Data()[0] == replacement);
		CHECK(broadView.Data()[offset] == replacement);
		CheckFilled(oldView, Byte{0x11});
	}

	{
		auto reusableView = backend.MapReadWrite(reusableOffset, reusableBytes, 1);
		Fill(reusableView, Byte{0x77});
		reusableView.Complete();
		const auto freshView = backend.MapReadOnly(reusableOffset, reusableBytes, 1);
		CheckFilled(freshView, Byte{0x77});
		CHECK(std::all_of(broadView.Data() + reusableOffset,
		                  broadView.Data() + reusableOffset + reusableBytes,
		                  [](const Byte byte) { return byte == Byte{0x77}; }));
	}

	CheckFilled(oldView, Byte{0x11});
}


TEST_CASE("PersistentStore mapped writes remain visible after reopen")
{
	TemporaryPath temporary;
	{
		auto backend = CreateBackend(temporary.Get(), 4096);
		auto writable = backend.MapReadWrite(127, 4, 1);
		Fill(writable, Byte{0x6C});
		writable.Complete();
	}

	auto backend = FileBackend::Open(temporary.Get(), FileAccess::ReadOnly);
	const auto readable = backend.MapReadOnly(127, 4, 1);
	CheckFilled(readable, Byte{0x6C});
}
