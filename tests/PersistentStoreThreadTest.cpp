#include <SecUtility/IO/PersistentStore/Adapter/NativeObject.hpp>

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>
#include <thread>
#include <vector>


using namespace SecUtility;
using namespace SecUtility::IO;


struct ReentrantValue
{
	PersistentStore* StorePtr = nullptr;
};


namespace SecUtility::IO
{
	template <>
	struct PersistentTraits<ReentrantValue>
	{
		static PersistentStoreDetail::PayloadLayout Measure(const ReentrantValue& value)
		{
			(void)value.StorePtr->Size();
			return {1, 1};
		}

		static void Encode(const ReentrantValue&, const PersistentStoreDetail::MutableByteView bytes)
		{
			bytes.data()[0] = Byte{0x5A};
		}

		static ReentrantValue Decode(const PersistentStoreDetail::ConstByteView) { return {}; }
	};
}


namespace
{
	class TemporaryPath final
	{
	public:
		TemporaryPath()
		{
			m_Path = std::filesystem::temp_directory_path()
			         / ("secutility-persistent-store-thread-"
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


TEST_CASE("PersistentStore rejects same-store adapter reentry and permits a different store")
{
	TemporaryPath firstPath;
	TemporaryPath secondPath;
	auto first = PersistentStore::Create(firstPath.Get());
	auto second = PersistentStore::Create(secondPath.Get());
	CHECK_THROWS_AS(first.Insert<ReentrantValue>("same", ReentrantValue{&first}), InvalidOperationException);
	CHECK_NOTHROW(first.Insert<ReentrantValue>("different", ReentrantValue{&second}));
	CHECK(first.Contains("different"));
}


TEST_CASE("PersistentStore serializes concurrent writers while readers observe committed values")
{
	TemporaryPath temporary;
	auto store = PersistentStore::Create(temporary.Get());
	store.Insert("value", std::int64_t{0});
	std::atomic<bool> shouldStart{false};
	std::atomic<bool> shouldStop{false};
	std::atomic<bool> hasInvalidRead{false};

	std::thread reader([&]
	{
		while (!shouldStart.load(std::memory_order_acquire)) std::this_thread::yield();
		while (!shouldStop.load(std::memory_order_acquire))
		{
			const std::int64_t value = store.Get<std::int64_t>("value");
			if (value < 0 || value > 200) hasInvalidRead.store(true, std::memory_order_release);
		}
	});
	std::vector<std::thread> writers;
	for (std::int64_t writer = 0; writer != 2; ++writer)
	{
		writers.emplace_back([&, writer]
		{
			while (!shouldStart.load(std::memory_order_acquire)) std::this_thread::yield();
			for (std::int64_t index = 1; index <= 100; ++index)
				store.Reassign("value", writer * 100 + index);
		});
	}
	shouldStart.store(true, std::memory_order_release);
	for (auto& writer : writers) writer.join();
	shouldStop.store(true, std::memory_order_release);
	reader.join();
	CHECK_FALSE(hasInvalidRead.load(std::memory_order_acquire));
	const std::int64_t finalValue = store.Get<std::int64_t>("value");
	CHECK(finalValue >= 1);
	CHECK(finalValue <= 200);
}
