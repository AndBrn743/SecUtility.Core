#include <SecUtility/IO/PersistentStore/Detail/ExtentAllocator.hpp>
#include <SecUtility/IO/PersistentStore/Adapter/ByteSequence.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
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
	allocator.Rebuild(8192, {{5000, 400}, {6000, 300}}, {{5500, 100}});
	const ExtentAllocatorMetrics metrics = allocator.Audit();
	CHECK(metrics.PhysicalBytes == 8192);
	CHECK(metrics.ProtectedBytes == SuperblockBytes + 700);
	CHECK(metrics.PendingBytes == 100);
	CHECK(metrics.ProtectedBytes + metrics.PendingBytes + metrics.ReusableBytes == metrics.PhysicalBytes);

	allocator.RetryPendingReleases();
	const ExtentAllocatorMetrics releasedMetrics = allocator.Audit();
	CHECK(releasedMetrics.PendingBytes == 0);
	CHECK(releasedMetrics.ProtectedBytes + releasedMetrics.ReusableBytes == releasedMetrics.PhysicalBytes);
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
