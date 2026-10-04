#include <SecUtility/IO/PersistentStore/Detail/ExtentAllocator.hpp>
#include <SecUtility/IO/PersistentStore/Detail/FileBackend.hpp>
#include <SecUtility/IO/PersistentStore/Detail/FormatCodec.hpp>
#include <SecUtility/IO/PersistentStore/Adapter/ByteSequence.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <random>
#include <string>
#include <system_error>
#include <vector>


using namespace SecUtility;
using namespace SecUtility::IO::PersistentStoreDetail;
using IO::PersistentStore;


namespace
{
	class TemporaryPath final
	{
	public:
		TemporaryPath()
		{
			m_Path = std::filesystem::temp_directory_path()
			         / ("secutility-persistent-store-audit-"
			            + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
		}
		~TemporaryPath()
		{
			std::error_code ignoredError;
			std::filesystem::remove(m_Path, ignoredError);
		}
		const std::filesystem::path& Get() const noexcept { return m_Path; }

	private:
		std::filesystem::path m_Path;
	};
}


TEST_CASE("PersistentStore allocator audit accounts independently for every physical byte")
{
	ExtentAllocator allocator;
	allocator.Rebuild(8192, {{5000, 400}, {6000, 300}});
	const ExtentAllocatorMetrics metrics = allocator.Audit();
	CHECK(metrics.PhysicalBytes == 8192);
	CHECK(metrics.ProtectedBytes == SuperblockBytes + 700);
	CHECK(metrics.ProtectedBytes + metrics.ReusableBytes == metrics.PhysicalBytes);
}


TEST_CASE("PersistentStore allocator rebuild exposes orphan tail space")
{
	ExtentAllocator allocator;
	allocator.Rebuild(4600, {{4096, 128}});
	const Allocation allocation = allocator.Reserve(200, 1);
	CHECK(allocation.AllocatedExtent.Offset == 4224);
	CHECK(allocator.GetPhysicalBytes() == 4600);
}


TEST_CASE("PersistentStore allocator randomized reservations preserve audit invariants")
{
	std::mt19937_64 random(0x5EC07117U);
	for (unsigned iteration = 0; iteration != 100; ++iteration)
	{
		ExtentAllocator allocator;
		allocator.Rebuild(16384, {{4096, 512}, {8192, 1024}, {14000, 400}});
		for (unsigned reservation = 0; reservation != 40; ++reservation)
		{
			const UInt64 bytes = 1 + random() % 160;
			const UInt64 alignment = UInt64{1} << (random() % 7);
			(void)allocator.Reserve(bytes, alignment);
			(void)allocator.Audit();
		}
	}
}


TEST_CASE("PersistentStore reconstructs reusable extents across root rotation and reopen")
{
	TemporaryPath temporary;
	const std::vector<Byte> first(256, Byte{0x11});
	const std::vector<Byte> second(256, Byte{0x22});
	{
		auto store = PersistentStore::Create(temporary.Get());
		store.Insert("value", first);
		store.Reassign("value", second);
		store.Reassign("value", first);
	}

	const auto beforeReuse = std::filesystem::file_size(temporary.Get());
	{
		auto store = PersistentStore::OpenForReadWrite(temporary.Get());
		CHECK(store.Get<std::vector<Byte>>("value") == first);
		store.Reassign("value", second);
		store.Reassign("value", first);
		CHECK(store.Get<std::vector<Byte>>("value") == first);
	}
	CHECK(std::filesystem::file_size(temporary.Get()) == beforeReuse);
}


TEST_CASE("PersistentStore rejects inconsistent extents shared across recoverable roots")
{
	TemporaryPath temporary;
	{
		auto store = PersistentStore::Create(temporary.Get());
		store.Insert("value", std::vector<Byte>{Byte{0x11}, Byte{0x12}});
		store.Reassign("value", std::vector<Byte>{Byte{0x21}, Byte{0x22}});
	}
	{
		auto backend = FileBackend::Open(temporary.Get(), FileAccess::ReadWrite);
		const UInt64 physicalBytes = backend.GetPhysicalFileBytes();
		std::array<Byte, HeaderSlotBytes> slotABytes{};
		std::array<Byte, HeaderSlotBytes> slotBBytes{};
		backend.ReadExact(HeaderSlotAOffset, slotABytes.data(), slotABytes.size());
		backend.ReadExact(HeaderSlotBOffset, slotBBytes.data(), slotBBytes.size());
		HeaderSlot slotA = ParseHeader(ConstByteView(slotABytes), physicalBytes);
		HeaderSlot slotB = ParseHeader(ConstByteView(slotBBytes), physicalBytes);
		const HeaderSlot& older = slotA.Generation < slotB.Generation ? slotA : slotB;
		HeaderSlot* newerPtr = slotA.Generation < slotB.Generation ? &slotB : &slotA;
		const UInt64 newerSlotOffset = newerPtr == &slotA ? HeaderSlotAOffset : HeaderSlotBOffset;
		std::vector<Byte> olderBytes(older.DirectoryBytes);
		std::vector<Byte> newerBytes(newerPtr->DirectoryBytes);
		backend.ReadExact(older.DirectoryOffset, olderBytes.data(), olderBytes.size());
		backend.ReadExact(newerPtr->DirectoryOffset, newerBytes.data(), newerBytes.size());
		const Directory olderDirectory = ParseDirectory(older, ConstByteView(olderBytes), physicalBytes);
		Directory newerDirectory = ParseDirectory(*newerPtr, ConstByteView(newerBytes), physicalBytes);
		REQUIRE(olderDirectory.Entries.size() == 1);
		REQUIRE(newerDirectory.Entries.size() == 1);
		newerDirectory.Entries[0].AllocatedExtent = olderDirectory.Entries[0].AllocatedExtent;
		newerDirectory.Entries[0].PayloadOffset = olderDirectory.Entries[0].PayloadOffset;
		newerDirectory.Entries[0].PayloadBytes = olderDirectory.Entries[0].PayloadBytes - 1;
		newerBytes = SerializeDirectory(newerDirectory);
		newerPtr->DirectoryChecksum = FormatCodecDetail::ComputeCrc32C(ConstByteView(newerBytes));
		const auto corruptedHeaderBytes = SerializeHeader(*newerPtr);
		backend.WriteExact(newerPtr->DirectoryOffset, newerBytes.data(), newerBytes.size());
		backend.WriteExact(newerSlotOffset, corruptedHeaderBytes.data(), corruptedHeaderBytes.size());
	}
	CHECK_THROWS_AS(PersistentStore::OpenForReadOnly(temporary.Get()), FormatException);
}
