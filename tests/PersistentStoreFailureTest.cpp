#include <SecUtility/IO/PersistentStore/Adapter/ByteSequence.hpp>
#include <SecUtility/IO/PersistentStore/Detail/FailureInjection.hpp>

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <new>
#include <string>
#include <system_error>
#include <vector>


using namespace SecUtility;
using namespace SecUtility::IO;
using namespace SecUtility::IO::PersistentStoreDetail;


namespace
{
	struct Injection
	{
		FailurePoint Point = FailurePoint::ReservePayload;
		UInt64 Detail = 0;
		bool MatchesDetail = false;
	};

	void ThrowInjectedFailure(const FailurePoint point, const UInt64 detail, void* const contextPtr)
	{
		const Injection& injection = *static_cast<const Injection*>(contextPtr);
		if (point == injection.Point && (!injection.MatchesDetail || detail == injection.Detail))
			throw IOException("injected PersistentStore failure");
	}

	void ThrowInjectedAllocationFailure(const FailurePoint point, const UInt64, void* const contextPtr)
	{
		if (point == *static_cast<const FailurePoint*>(contextPtr)) throw std::bad_alloc();
	}

	class TemporaryPath final
	{
	public:
		TemporaryPath()
		{
			m_Path = std::filesystem::temp_directory_path()
			         / ("secutility-persistent-store-failure-"
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

	const std::vector<FailurePoint> PrePublicationPoints{
	        FailurePoint::RebuildAllocator, FailurePoint::ReservePayload, FailurePoint::GrowPayload,
	        FailurePoint::MapPayload, FailurePoint::EncodePayload, FailurePoint::CompletePayload,
	        FailurePoint::SerializeDirectory, FailurePoint::ReserveDirectory, FailurePoint::GrowDirectory,
	        FailurePoint::WriteDirectory, FailurePoint::ReadBackDirectory, FailurePoint::PrepareRuntimeState};
}


TEST_CASE("PersistentStore remains usable after every injected pre-publication failure")
{
	for (const FailurePoint point : PrePublicationPoints)
	{
		TemporaryPath temporary;
		auto store = PersistentStore::Create(temporary.Get());
		store.Insert("value", std::vector<Byte>{Byte{0x11}});
		const Injection injection{point};
		{
			ScopedFailureInjection scope(ThrowInjectedFailure, const_cast<Injection*>(&injection));
			CHECK_THROWS_AS(store.Reassign("value", std::vector<Byte>{Byte{0x22}}), IOException);
		}
		CHECK(store.Get<std::vector<Byte>>("value") == std::vector<Byte>{Byte{0x11}});
		CHECK_NOTHROW(store.Reassign("value", std::vector<Byte>{Byte{0x33}}));
		CHECK(store.Get<std::vector<Byte>>("value") == std::vector<Byte>{Byte{0x33}});
	}
}


TEST_CASE("PersistentStore poisons mutations after injected publication uncertainty")
{
	const std::vector<Injection> injections{
	        {FailurePoint::BeginPublication},
	        {FailurePoint::PublicationWriteProgress, 1, true},
	        {FailurePoint::PublicationWriteProgress, HeaderSlotBytes / 2, true},
	        {FailurePoint::PublicationWriteProgress, HeaderSlotBytes, true},
	        {FailurePoint::ReadBackPublication}};
	for (const Injection& injection : injections)
	{
		TemporaryPath temporary;
		{
			auto store = PersistentStore::Create(temporary.Get());
			store.Insert("value", std::vector<Byte>{Byte{0x11}});
			{
				ScopedFailureInjection scope(ThrowInjectedFailure, const_cast<Injection*>(&injection));
				CHECK_THROWS_AS(store.Reassign("value", std::vector<Byte>{Byte{0x22}}), IOException);
			}
			CHECK(store.Get<std::vector<Byte>>("value") == std::vector<Byte>{Byte{0x11}});
			CHECK_THROWS_AS(store.Reassign("value", std::vector<Byte>{Byte{0x33}}), InvalidOperationException);
		}
		auto reopened = PersistentStore::OpenForReadOnly(temporary.Get());
		const auto value = reopened.Get<std::vector<Byte>>("value");
		CHECK((value == std::vector<Byte>{Byte{0x11}} || value == std::vector<Byte>{Byte{0x22}}));
	}
}


TEST_CASE("PersistentStore recovers from injected preparation allocation failure")
{
	TemporaryPath temporary;
	auto store = PersistentStore::Create(temporary.Get());
	store.Insert("value", std::vector<Byte>{Byte{0x11}});
	FailurePoint point = FailurePoint::PrepareRuntimeState;
	{
		ScopedFailureInjection scope(ThrowInjectedAllocationFailure, &point);
		CHECK_THROWS_AS(store.Reassign("value", std::vector<Byte>{Byte{0x22}}), std::bad_alloc);
	}
	CHECK(store.Get<std::vector<Byte>>("value") == std::vector<Byte>{Byte{0x11}});
	CHECK_NOTHROW(store.Reassign("value", std::vector<Byte>{Byte{0x33}}));
}
