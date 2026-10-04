#include <SecUtility/IO/PersistentStore/Detail/ExtentAllocator.hpp>

#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <vector>


using namespace SecUtility;
using namespace SecUtility::IO::PersistentStoreDetail;


TEST_CASE("PersistentStore allocator uses exact and best-fit reusable extents")
{
	ExtentAllocator allocator;
	allocator.Rebuild(5000, {{4096, 104}, {4300, 200}, {4700, 300}});

	const Allocation exact = allocator.Reserve(100, 1);
	CHECK(exact.AllocatedExtent.Offset == 4200);
	CHECK(exact.AllocatedExtent.Capacity == 100);

	const Allocation bestFit = allocator.Reserve(120, 1);
	CHECK(bestFit.AllocatedExtent.Offset == 4500);
	CHECK(bestFit.AllocatedExtent.Capacity == 120);
	CHECK(allocator.GetFreeByOffset().at(4620) == 80);
}


TEST_CASE("PersistentStore allocator handles alignment, small fragments, zero bytes, and append")
{
	ExtentAllocator allocator;
	allocator.Rebuild(4200, {{0, 4096}});

	const Allocation aligned = allocator.Reserve(32, 64);
	CHECK(aligned.PayloadOffset % 64 == 0);
	CHECK(aligned.AllocatedExtent.Offset == 4096);
	CHECK(aligned.AllocatedExtent.Capacity == 32);

	const Allocation absorbed = allocator.Reserve(41, 1);
	CHECK(absorbed.AllocatedExtent.Capacity == 72);

	const Allocation empty = allocator.Reserve(0, 1);
	CHECK(empty.AllocatedExtent.Capacity == 1);
	CHECK(empty.PayloadOffset == empty.AllocatedExtent.Offset);

	const Allocation appended = allocator.Reserve(200, 4096);
	CHECK(appended.PayloadOffset % 4096 == 0);
	CHECK(appended.AllocatedExtent.Offset == 4201);
	CHECK(allocator.GetPhysicalBytes() == appended.AllocatedExtent.Offset + appended.AllocatedExtent.Capacity);
}


TEST_CASE("PersistentStore allocator coalesces pending releases atomically")
{
	ExtentAllocator allocator;
	allocator.Rebuild(4300, {}, {{4096, 64}, {4160, 140}});
	allocator.RetryPendingReleases();

	REQUIRE(allocator.GetFreeByOffset().size() == 1);
	CHECK(allocator.GetFreeByOffset().at(4096) == 204);
	CHECK(allocator.GetMetrics().PendingBytes == 0);
}


TEST_CASE("PersistentStore allocator rejects invalid ranges and arithmetic overflow")
{
	ExtentAllocator allocator;
	CHECK_THROWS_AS(allocator.Rebuild(4096, {{4090, 8}}), FormatException);
	allocator.Rebuild(std::numeric_limits<UInt64>::max(), {{0, std::numeric_limits<UInt64>::max()}});
	CHECK_THROWS(allocator.Reserve(1, 4096));
}
