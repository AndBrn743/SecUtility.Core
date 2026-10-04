#include <SecUtility/IO/PersistentStoreChecker.hpp>
#include <SecUtility/IO/PersistentStore/Adapter/ByteSequence.hpp>
#include <SecUtility/IO/PersistentStore/Detail/FileBackend.hpp>

#include <catch2/catch_test_macros.hpp>

#include <chrono>
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
			         / ("secutility-persistent-store-checker-"
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

	std::vector<Byte> ReadFile(const std::filesystem::path& path)
	{
		auto backend = FileBackend::Open(path, FileAccess::ReadOnly);
		std::vector<Byte> bytes(static_cast<std::size_t>(backend.GetPhysicalFileBytes()));
		backend.ReadExact(0, bytes.data(), bytes.size());
		return bytes;
	}
}


TEST_CASE("PersistentStore checker reports both roots and accounts physical bytes without writing")
{
	TemporaryPath temporary;
	{
		auto store = PersistentStore::Create(temporary.Get());
		store.Insert("first", std::vector<Byte>(73, Byte{0x21}));
		store.Insert("second", std::vector<Byte>(91, Byte{0x42}));
		store.Reassign("first", std::vector<Byte>(73, Byte{0x63}));
	}
	{
		auto backend = FileBackend::Open(temporary.Get(), FileAccess::ReadWrite);
		backend.SetPhysicalFileBytes(backend.GetPhysicalFileBytes() + 137);
	}
	const std::vector<Byte> before = ReadFile(temporary.Get());
	const PersistentStoreCheckReport report = CheckPersistentStore(temporary.Get());
	const std::vector<Byte> after = ReadFile(temporary.Get());
	CHECK(report.SlotA.IsValid);
	CHECK(report.SlotB.IsValid);
	CHECK(report.HasSelectedRoot);
	CHECK(report.SelectedGeneration == 4);
	CHECK(report.LiveRecordCount == 2);
	CHECK(report.LivePayloadBytes > 0);
	CHECK(report.PhysicalFileBytes == before.size());
	CHECK(report.CommittedFileBytes <= report.PhysicalFileBytes);
	CHECK(report.MetadataBytes >= SuperblockBytes);
	CHECK(report.OtherRootOnlyBytes > 0);
	CHECK(report.PhysicalTailBytes == 137);
	CHECK(report.ReclaimableBytes + report.PhysicalTailBytes <= report.PhysicalFileBytes);
	CHECK(before == after);
}


TEST_CASE("PersistentStore checker reports invalid slots independently and both rejection reasons")
{
	TemporaryPath temporary;
	{
		auto store = PersistentStore::Create(temporary.Get());
		store.Insert("value", std::vector<Byte>{Byte{0x11}});
	}
	{
		auto backend = FileBackend::Open(temporary.Get(), FileAccess::ReadWrite);
		const Byte corrupt{0xFF};
		backend.WriteExact(HeaderSlotAOffset, &corrupt, 1);
	}
	const PersistentStoreCheckReport oneInvalid = CheckPersistentStore(temporary.Get());
	CHECK_FALSE(oneInvalid.SlotA.IsValid);
	CHECK_FALSE(oneInvalid.SlotA.Reason.empty());
	CHECK(oneInvalid.SlotB.IsValid);
	CHECK(oneInvalid.HasSelectedRoot);
	{
		auto backend = FileBackend::Open(temporary.Get(), FileAccess::ReadWrite);
		const Byte corrupt{0xFF};
		backend.WriteExact(HeaderSlotBOffset, &corrupt, 1);
	}
	const PersistentStoreCheckReport neitherValid = CheckPersistentStore(temporary.Get());
	CHECK_FALSE(neitherValid.SlotA.IsValid);
	CHECK_FALSE(neitherValid.SlotB.IsValid);
	CHECK_FALSE(neitherValid.SlotA.Reason.empty());
	CHECK_FALSE(neitherValid.SlotB.Reason.empty());
	CHECK_FALSE(neitherValid.HasSelectedRoot);
}


TEST_CASE("PersistentStore checker shares reader locks and fails immediately against a writer")
{
	TemporaryPath temporary;
	{
		auto store = PersistentStore::Create(temporary.Get());
		store.Insert("value", std::vector<Byte>{Byte{0x31}});
	}
	{
		const auto reader = PersistentStore::OpenForReadOnly(temporary.Get());
		CHECK(CheckPersistentStore(temporary.Get()).HasSelectedRoot);
	}
	{
		const auto writer = PersistentStore::OpenForReadWrite(temporary.Get());
		CHECK_THROWS_AS(CheckPersistentStore(temporary.Get()), IOException);
	}
}
