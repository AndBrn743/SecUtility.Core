#include <SecUtility/IO/PersistentStore/Adapter/ByteSequence.hpp>

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>


using namespace SecUtility;
using namespace SecUtility::IO;


namespace
{
	class TemporaryPath final
	{
	public:
		TemporaryPath()
		{
			m_Path = std::filesystem::temp_directory_path()
			         / ("secutility-persistent-store-lease-"
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


TEST_CASE("PersistentStore erase distinguishes absent and present keys")
{
	TemporaryPath temporary;
	{
		auto store = PersistentStore::Create(temporary.Get());
		store.Insert("value", std::vector<Byte>{Byte{1}, Byte{2}});
		CHECK_FALSE(store.Erase("absent"));
		CHECK(store.Erase("value"));
		CHECK_FALSE(store.Contains("value"));
		CHECK_FALSE(store.Erase("value"));
	}
	const auto store = PersistentStore::OpenForReadOnly(temporary.Get());
	CHECK_FALSE(store.Contains("value"));
}


TEST_CASE("PersistentStore leases retain each reassigned and erased extent")
{
	TemporaryPath temporary;
	LeasedByteView firstLease;
	LeasedByteView secondLease;
	{
		auto store = PersistentStore::Create(temporary.Get());
		store.Insert("value", std::vector<Byte>{Byte{0x11}});
		firstLease = store.GetLeased<std::vector<Byte>>("value");
		auto copiedLease = firstLease;
		store.Reassign("value", std::vector<Byte>{Byte{0x22}});
		secondLease = store.GetLeased<std::vector<Byte>>("value");
		CHECK(store.Erase("value"));
		store.Insert("other", std::vector<Byte>(1024, Byte{0x33}));
		CHECK(firstLease[0] == Byte{0x11});
		CHECK(copiedLease[0] == Byte{0x11});
		CHECK(secondLease[0] == Byte{0x22});
	}
	CHECK(firstLease[0] == Byte{0x11});
	CHECK(secondLease[0] == Byte{0x22});
}


TEST_CASE("PersistentStore retains copied and moved leases across repeated generations")
{
	TemporaryPath temporary;
	auto store = PersistentStore::Create(temporary.Get());
	std::vector<LeasedByteView> leases;
	for (unsigned int generation = 0; generation != 8; ++generation)
	{
		const std::vector<Byte> value(128, Byte{static_cast<unsigned char>(generation)});
		if (generation == 0) store.Insert("value", value);
		else store.Reassign("value", value);
		auto lease = store.GetLeased<std::vector<Byte>>("value");
		auto copy = lease;
		leases.push_back(std::move(copy));
		CHECK(lease[0] == Byte{static_cast<unsigned char>(generation)});
	}
	CHECK(store.Erase("value"));
	for (unsigned int generation = 0; generation != leases.size(); ++generation)
	{
		CHECK(leases[generation].size() == 128);
		CHECK(leases[generation][0] == Byte{static_cast<unsigned char>(generation)});
	}
}
