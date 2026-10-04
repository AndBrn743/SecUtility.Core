#include <SecUtility/IO/PersistentStore/Adapter/NativeObject.hpp>

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <future>
#include <mutex>
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

struct PauseControl
{
	std::mutex Mutex;
	std::condition_variable Condition;
	bool HasEntered = false;
	bool CanProceed = false;
};

struct PausedDecodeValue
{
	Byte Value{};
	inline static PauseControl* ControlPtr = nullptr;
};

struct PausedEncodeValue
{
	Byte Value{};
	inline static PauseControl* ControlPtr = nullptr;
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

	template <>
	struct PersistentTraits<PausedDecodeValue>
	{
		static PersistentStoreDetail::PayloadLayout Measure(const PausedDecodeValue&) { return {1, 1}; }
		static void Encode(const PausedDecodeValue value, const PersistentStoreDetail::MutableByteView bytes)
		{
			bytes.data()[0] = value.Value;
		}
		static PausedDecodeValue Decode(const PersistentStoreDetail::ConstByteView bytes)
		{
			PauseControl& control = *PausedDecodeValue::ControlPtr;
			std::unique_lock<std::mutex> lock(control.Mutex);
			control.HasEntered = true;
			control.Condition.notify_all();
			control.Condition.wait(lock, [&control] { return control.CanProceed; });
			return {bytes.data()[0]};
		}
	};

	template <>
	struct PersistentTraits<PausedEncodeValue>
	{
		static PersistentStoreDetail::PayloadLayout Measure(const PausedEncodeValue&) { return {1, 1}; }
		static void Encode(const PausedEncodeValue value, const PersistentStoreDetail::MutableByteView bytes)
		{
			PauseControl& control = *PausedEncodeValue::ControlPtr;
			std::unique_lock<std::mutex> lock(control.Mutex);
			control.HasEntered = true;
			control.Condition.notify_all();
			control.Condition.wait(lock, [&control] { return control.CanProceed; });
			bytes.data()[0] = value.Value;
		}
		static PausedEncodeValue Decode(const PersistentStoreDetail::ConstByteView bytes)
		{
			return {bytes.data()[0]};
		}
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
	std::mutex exceptionMutex;
	std::vector<std::exception_ptr> exceptions;
	const auto recordException = [&]
	{
		std::lock_guard<std::mutex> lock(exceptionMutex);
		exceptions.push_back(std::current_exception());
		shouldStop.store(true, std::memory_order_release);
	};

	std::thread reader([&]
	{
		try
		{
			while (!shouldStart.load(std::memory_order_acquire)) std::this_thread::yield();
			while (!shouldStop.load(std::memory_order_acquire))
			{
				const std::int64_t value = store.Get<std::int64_t>("value");
					if (value < 0 || value > 200) hasInvalidRead.store(true, std::memory_order_release);
			}
		}
		catch (...) { recordException(); }
	});
	std::vector<std::thread> writers;
	for (std::int64_t writer = 0; writer != 2; ++writer)
	{
		writers.emplace_back([&, writer]
		{
			try
			{
				while (!shouldStart.load(std::memory_order_acquire)) std::this_thread::yield();
				for (std::int64_t index = 1; index <= 100 && !shouldStop.load(std::memory_order_acquire); ++index)
					store.Reassign("value", writer * 100 + index);
			}
			catch (...) { recordException(); }
		});
	}
	shouldStart.store(true, std::memory_order_release);
	for (auto& writer : writers) writer.join();
	shouldStop.store(true, std::memory_order_release);
	reader.join();
	if (!exceptions.empty()) std::rethrow_exception(exceptions.front());
	CHECK_FALSE(hasInvalidRead.load(std::memory_order_acquire));
	const std::int64_t finalValue = store.Get<std::int64_t>("value");
	CHECK(finalValue >= 1);
	CHECK(finalValue <= 200);
}


TEST_CASE("PersistentStore owning read pins its extent while decoding is paused")
{
	TemporaryPath temporary;
	auto store = PersistentStore::Create(temporary.Get());
	PauseControl control;
	PausedDecodeValue::ControlPtr = &control;
	store.Insert("value", PausedDecodeValue{Byte{0x11}});
	PausedDecodeValue decoded;
	std::exception_ptr readerException;
	std::thread reader([&]
	{
		try { decoded = store.Get<PausedDecodeValue>("value"); }
		catch (...) { readerException = std::current_exception(); }
	});
	{
		std::unique_lock<std::mutex> lock(control.Mutex);
		CHECK(control.Condition.wait_for(lock, std::chrono::seconds(5), [&control] { return control.HasEntered; }));
	}
	for (unsigned int generation = 0; generation != 4; ++generation)
		store.Reassign("value", PausedDecodeValue{Byte{static_cast<unsigned char>(0x20 + generation)}});
	{
		std::lock_guard<std::mutex> lock(control.Mutex);
		control.CanProceed = true;
	}
	control.Condition.notify_all();
	reader.join();
	PausedDecodeValue::ControlPtr = nullptr;
	CHECK(readerException == nullptr);
	CHECK(decoded.Value == Byte{0x11});
}


TEST_CASE("PersistentStore readers proceed while a writer is paused in adapter encoding")
{
	TemporaryPath temporary;
	auto store = PersistentStore::Create(temporary.Get());
	store.Insert("existing", std::int64_t{42});
	PauseControl control;
	PausedEncodeValue::ControlPtr = &control;
	std::thread writer([&] { store.Insert("paused", PausedEncodeValue{Byte{7}}); });
	{
		std::unique_lock<std::mutex> lock(control.Mutex);
		CHECK(control.Condition.wait_for(lock, std::chrono::seconds(5), [&control] { return control.HasEntered; }));
	}
	auto reader = std::async(std::launch::async, [&] { return store.Get<std::int64_t>("existing"); });
	CHECK(reader.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
	CHECK(reader.get() == 42);
	{
		std::lock_guard<std::mutex> lock(control.Mutex);
		control.CanProceed = true;
	}
	control.Condition.notify_all();
	writer.join();
	PausedEncodeValue::ControlPtr = nullptr;
}


TEST_CASE("PersistentStore readers race safely with erase and insert publication")
{
	TemporaryPath temporary;
	auto store = PersistentStore::Create(temporary.Get());
	store.Insert("value", std::int64_t{1});
	std::atomic<bool> shouldStart{false};
	std::atomic<bool> isDone{false};
	std::atomic<bool> hasInvalidObservation{false};
	std::exception_ptr readerException;
	std::thread reader([&]
	{
		while (!shouldStart.load(std::memory_order_acquire)) std::this_thread::yield();
		try
		{
			while (!isDone.load(std::memory_order_acquire))
			{
				try
				{
					const std::int64_t value = store.Get<std::int64_t>("value");
					if (value != 1 && value != 2) hasInvalidObservation.store(true, std::memory_order_release);
				}
				catch (const KeyNotFoundException&)
				{
					// Erase may linearize between any two independent reader operations.
				}
			}
		}
		catch (...) { readerException = std::current_exception(); }
	});
	shouldStart.store(true, std::memory_order_release);
	for (unsigned int iteration = 0; iteration != 100; ++iteration)
	{
		CHECK(store.Erase("value"));
		store.Insert("value", std::int64_t{iteration % 2 == 0 ? 1 : 2});
	}
	isDone.store(true, std::memory_order_release);
	reader.join();
	if (readerException != nullptr) std::rethrow_exception(readerException);
	CHECK_FALSE(hasInvalidObservation.load(std::memory_order_acquire));
}
