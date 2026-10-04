#include <SecUtility/IO/PersistentStore/Adapter/Array.hpp>
#include <SecUtility/IO/PersistentStore/Adapter/ByteSequence.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>
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
			         / ("secutility-persistent-store-basic-"
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
}


TEST_CASE("PersistentStore append-only mutations persist and enforce key preconditions")
{
	TemporaryPath temporary;
	const std::string embeddedKey("a\0b", 3);
	{
		auto store = PersistentStore::Create(temporary.Get());
		store.Insert("signed", std::int64_t{-17});
		store.Insert(embeddedKey, std::vector<Byte>{Byte{1}, Byte{2}, Byte{3}});
		CHECK_THROWS_AS(store.Insert("signed", std::int64_t{4}), KeyAlreadyExistsException);
		CHECK_THROWS_AS(store.Reassign("missing", std::int64_t{4}), KeyNotFoundException);
		store.Reassign("signed", std::int64_t{29});
		store.InsertOrReassign("new", std::uint32_t{7});
		store.InsertOrReassign("new", std::uint32_t{9});
		CHECK(store.Size() == 3);
		CHECK(store.Get<std::int64_t>("signed") == 29);
		CHECK(store.Get<std::uint32_t>("new") == 9);
		CHECK(store.Get<std::vector<Byte>>(embeddedKey) == std::vector<Byte>{Byte{1}, Byte{2}, Byte{3}});
		CHECK_THROWS_AS(store.Get<double>("signed"), FormatException);
		CHECK_THROWS_AS(store.Get<std::int32_t>("missing"), KeyNotFoundException);
	}

	auto store = PersistentStore::OpenForReadOnly(temporary.Get());
	CHECK(store.Size() == 3);
	CHECK(store.Get<std::int64_t>("signed") == 29);
	CHECK_THROWS_AS(store.Reassign("signed", std::int64_t{3}), InvalidOperationException);
}


TEST_CASE("PersistentStore leased bytes retain mapped storage and the store lock")
{
	TemporaryPath temporary;
	LeasedByteView retained;
	{
		auto store = PersistentStore::Create(temporary.Get());
		store.Insert("bytes", std::vector<Byte>{Byte{0x31}, Byte{0x32}, Byte{0x33}});
		retained = store.GetLeased<std::vector<Byte>>("bytes");
		auto copied = retained;
		CHECK(copied.size() == 3);
		CHECK(copied[1] == Byte{0x32});
	}

	CHECK(retained.size() == 3);
	CHECK(retained[2] == Byte{0x33});
	CHECK_THROWS_AS(PersistentStore::OpenForReadWrite(temporary.Get()), IOException);
	retained = {};
	CHECK_NOTHROW(PersistentStore::OpenForReadWrite(temporary.Get()));
}


TEST_CASE("PersistentStore stores empty and aligned array payloads")
{
	TemporaryPath temporary;
	auto store = PersistentStore::Create(temporary.Get());
	store.Insert("empty-bytes", std::vector<Byte>{});
	store.Insert("empty-array", std::vector<double>{});
	store.Insert("fixed", std::array<std::int32_t, 4>{-1, 0, 1, 2147483647});
	CHECK(store.Get<std::vector<Byte>>("empty-bytes").empty());
	CHECK(store.Get<std::vector<double>>("empty-array").empty());
	CHECK(store.Get<std::array<std::int32_t, 4>>("fixed")
	      == std::array<std::int32_t, 4>{-1, 0, 1, 2147483647});
}


TEST_CASE("PersistentStore alternates roots and falls back from a corrupt newest publication")
{
	TemporaryPath temporary;
	{
		auto store = PersistentStore::Create(temporary.Get());
		store.Insert("value", std::int32_t{1});
		store.Reassign("value", std::int32_t{2});
	}

	{
		auto backend = FileBackend::Open(temporary.Get(), FileAccess::ReadWrite);
		std::array<Byte, HeaderSlotBytes> slotABytes{};
		std::array<Byte, HeaderSlotBytes> slotBBytes{};
		backend.ReadExact(HeaderSlotAOffset, slotABytes.data(), slotABytes.size());
		backend.ReadExact(HeaderSlotBOffset, slotBBytes.data(), slotBBytes.size());
		const UInt64 physicalBytes = backend.GetPhysicalFileBytes();
		const HeaderSlot slotA = ParseHeader(ConstByteView(slotABytes), physicalBytes);
		const HeaderSlot slotB = ParseHeader(ConstByteView(slotBBytes), physicalBytes);
		CHECK(slotA.Generation == 3);
		CHECK(slotB.Generation == 2);
		std::fill(slotABytes.begin(), slotABytes.end(), Byte{0});
		backend.WriteExact(HeaderSlotAOffset, slotABytes.data(), slotABytes.size());
	}

	const auto store = PersistentStore::OpenForReadOnly(temporary.Get());
	CHECK(store.Get<std::int32_t>("value") == 1);
}
