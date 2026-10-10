#include <SecUtility/Diagnostic/Exception.hpp>
#include <SecUtility/IO/RandomAccessFile.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <limits>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif


using namespace SecUtility;
using namespace SecUtility::IO;


namespace
{
	class TemporaryPath final
	{
	public:
		TemporaryPath()
		{
			const auto token = std::chrono::steady_clock::now().time_since_epoch().count();
			m_Path = std::filesystem::temp_directory_path()
			         / ("secutility-random-access-file-" + std::to_string(token) + "-"
			            + std::to_string(++Counter));
		}

		~TemporaryPath()
		{
			std::error_code ignoredError;
			std::filesystem::remove_all(m_Path, ignoredError);
		}

		const std::filesystem::path& Get() const noexcept { return m_Path; }

	private:
		inline static std::atomic<unsigned long long> Counter{0};
		std::filesystem::path m_Path;
	};

	class ChildLock final
	{
	public:
		ChildLock(const std::filesystem::path& path, const char* mode)
		{
#if defined(_WIN32)
			SECURITY_ATTRIBUTES attributes{};
			attributes.nLength = sizeof(attributes);
			attributes.bInheritHandle = TRUE;
			HANDLE childInput = nullptr;
			HANDLE parentInput = nullptr;
			HANDLE parentOutput = nullptr;
			HANDLE childOutput = nullptr;
			REQUIRE(::CreatePipe(&childInput, &parentInput, &attributes, 0));
			REQUIRE(::CreatePipe(&parentOutput, &childOutput, &attributes, 0));
			::SetHandleInformation(parentInput, HANDLE_FLAG_INHERIT, 0);
			::SetHandleInformation(parentOutput, HANDLE_FLAG_INHERIT, 0);
			STARTUPINFOW startup{};
			startup.cb = sizeof(startup);
			startup.dwFlags = STARTF_USESTDHANDLES;
			startup.hStdInput = childInput;
			startup.hStdOutput = childOutput;
			startup.hStdError = childOutput;
			PROCESS_INFORMATION process{};
			std::wstring command = L"\"" + std::filesystem::path(RANDOM_ACCESS_FILE_LOCK_HELPER_PATH).wstring()
			                       + L"\" " + std::filesystem::path(mode).wstring() + L" \"" + path.wstring() + L"\"";
			REQUIRE(::CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr,
			                         &startup, &process));
			::CloseHandle(childInput);
			::CloseHandle(childOutput);
			::CloseHandle(process.hThread);
			m_ProcessHandle = process.hProcess;
			m_InputHandle = parentInput;
			m_OutputHandle = parentOutput;
			WaitUntilReady();
#else
			int inputPipe[2]{};
			int outputPipe[2]{};
			REQUIRE(::pipe(inputPipe) == 0);
			REQUIRE(::pipe(outputPipe) == 0);
			m_ProcessId = ::fork();
			REQUIRE(m_ProcessId >= 0);
			if (m_ProcessId == 0)
			{
				::dup2(inputPipe[0], STDIN_FILENO);
				::dup2(outputPipe[1], STDOUT_FILENO);
				::dup2(outputPipe[1], STDERR_FILENO);
				::close(inputPipe[0]);
				::close(inputPipe[1]);
				::close(outputPipe[0]);
				::close(outputPipe[1]);
				::execl(RANDOM_ACCESS_FILE_LOCK_HELPER_PATH, RANDOM_ACCESS_FILE_LOCK_HELPER_PATH,
				        mode, path.c_str(), static_cast<char*>(nullptr));
				::_exit(127);
			}
			::close(inputPipe[0]);
			::close(outputPipe[1]);
			m_InputDescriptor = inputPipe[1];
			m_OutputDescriptor = outputPipe[0];
			WaitUntilReady();
#endif
		}

		~ChildLock()
		{
#if defined(_WIN32)
			DWORD written = 0;
			::WriteFile(m_InputHandle, "stop\n", 5, &written, nullptr);
			::CloseHandle(m_InputHandle);
			::CloseHandle(m_OutputHandle);
			::WaitForSingleObject(m_ProcessHandle, 5000);
			::CloseHandle(m_ProcessHandle);
#else
			(void)::write(m_InputDescriptor, "stop\n", 5);
			::close(m_InputDescriptor);
			::close(m_OutputDescriptor);
			int status = 0;
			(void)::waitpid(m_ProcessId, &status, 0);
#endif
		}

		ChildLock(const ChildLock&) = delete;
		ChildLock& operator=(const ChildLock&) = delete;

	private:
		void WaitUntilReady()
		{
			std::string line;
			for (;;)
			{
				char character = 0;
#if defined(_WIN32)
				DWORD readBytes = 0;
				REQUIRE(::ReadFile(m_OutputHandle, &character, 1, &readBytes, nullptr));
				REQUIRE(readBytes == 1);
#else
				REQUIRE(::read(m_OutputDescriptor, &character, 1) == 1);
#endif
				if (character == '\n')
				{
					if (!line.empty() && line.back() == '\r') line.pop_back();
					if (line == "ready") return;
					line.clear();
				}
				else
				{
					REQUIRE(line.size() < 4096);
					line.push_back(character);
				}
			}
		}

#if defined(_WIN32)
		HANDLE m_ProcessHandle = nullptr;
		HANDLE m_InputHandle = nullptr;
		HANDLE m_OutputHandle = nullptr;
#else
		pid_t m_ProcessId = -1;
		int m_InputDescriptor = -1;
		int m_OutputDescriptor = -1;
#endif
	};
}


static_assert(!std::is_copy_constructible_v<RandomAccessFile>);
static_assert(std::is_nothrow_move_constructible_v<RandomAccessFile>);
static_assert(!std::is_copy_constructible_v<RandomAccessFile::Lock>);
static_assert(std::is_nothrow_move_constructible_v<RandomAccessFile::Lock>);
static_assert(!std::is_copy_constructible_v<ReadOnlyMappedRegion>);
static_assert(!std::is_copy_constructible_v<ReadWriteMappedRegion>);


TEST_CASE("RandomAccessFile has explicit open and exclusive-create behavior")
{
	TemporaryPath temporary;
	CHECK_THROWS_AS(RandomAccessFile::OpenForReadOnly(temporary.Get()), IOException);
	CHECK_THROWS_AS(RandomAccessFile::OpenForReadWrite(temporary.Get()), IOException);

	auto created = RandomAccessFile::TryCreateExclusive(temporary.Get());
	REQUIRE(created.has_value());
	CHECK(created->Size() == 0);
	CHECK_FALSE(RandomAccessFile::TryCreateExclusive(temporary.Get()).has_value());
	CHECK_NOTHROW(RandomAccessFile::OpenForReadOnly(temporary.Get()));
	CHECK_NOTHROW(RandomAccessFile::OpenForReadWrite(temporary.Get()));
}


TEST_CASE("RandomAccessFile exclusive creation has one winner")
{
	TemporaryPath temporary;
	std::atomic<unsigned int> successCount{0};
	const auto create = [&]
	{
		if (RandomAccessFile::TryCreateExclusive(temporary.Get()).has_value()) ++successCount;
	};
	std::thread first(create);
	std::thread second(create);
	first.join();
	second.join();
	CHECK(successCount == 1);
}


TEST_CASE("RandomAccessFile performs exact positional transfers and resizing")
{
	TemporaryPath temporary;
	auto created = RandomAccessFile::TryCreateExclusive(temporary.Get());
	REQUIRE(created.has_value());
	created->Resize(131149);

	std::vector<Byte> expected(131123);
	for (std::size_t index = 0; index < expected.size(); ++index)
		expected[index] = Byte{static_cast<unsigned char>((index * 37U) & 0xFFU)};
	created->WriteExactAt(11, expected.data(), expected.size());
	std::vector<Byte> actual(expected.size());
	created->ReadExactAt(11, actual.data(), actual.size());
	CHECK(actual == expected);
	CHECK(created->Size() == 131149);

	created->Resize(17);
	CHECK(created->Size() == 17);
	std::array<Byte, 8> bytes{};
	CHECK_THROWS_AS(created->ReadExactAt(16, bytes.data(), bytes.size()), IOException);
	CHECK_THROWS_AS(created->ReadExactAt(0, nullptr, 1), InvalidArgumentException);
	CHECK_THROWS_AS(created->WriteExactAt(0, nullptr, 1), InvalidArgumentException);
	CHECK_THROWS_AS(created->ReadExactAt(std::numeric_limits<UInt64>::max(), bytes.data(), 1),
	                ArgumentOutOfRangeException);
	CHECK_THROWS_AS(created->WriteExactAt(std::numeric_limits<UInt64>::max(), bytes.data(), 1),
	                ArgumentOutOfRangeException);
	CHECK_THROWS_AS(created->Resize(std::numeric_limits<UInt64>::max()), ArgumentOutOfRangeException);
	CHECK_NOTHROW(created->ReadExactAt(0, nullptr, 0));
	CHECK_NOTHROW(created->WriteExactAt(0, nullptr, 0));
}


TEST_CASE("RandomAccessFile enforces read-only access")
{
	TemporaryPath temporary;
	REQUIRE(RandomAccessFile::TryCreateExclusive(temporary.Get()).has_value());
	auto file = RandomAccessFile::OpenForReadOnly(temporary.Get());
	std::array<Byte, 1> byte{};
	CHECK_THROWS_AS(file.Resize(1), InvalidOperationException);
	CHECK_THROWS_AS(file.WriteExactAt(0, byte.data(), byte.size()), InvalidOperationException);
	CHECK_THROWS_AS(file.MapReadWrite(0, 0), InvalidOperationException);
	CHECK_THROWS_AS(file.TryAcquireLock(RandomAccessFile::LockMode::Exclusive), InvalidOperationException);
}


TEST_CASE("RandomAccessFile exposes stable file identity")
{
	TemporaryPath firstPath;
	TemporaryPath secondPath;
	auto first = RandomAccessFile::TryCreateExclusive(firstPath.Get());
	auto second = RandomAccessFile::TryCreateExclusive(secondPath.Get());
	REQUIRE(first.has_value());
	REQUIRE(second.has_value());
	auto reopened = RandomAccessFile::OpenForReadOnly(firstPath.Get());
	CHECK(first->GetIdentity() == reopened.GetIdentity());
	CHECK(first->GetIdentity() != second->GetIdentity());
	CHECK(first->IdentifiesSameFile(firstPath.Get()));
	CHECK_FALSE(first->IdentifiesSameFile(secondPath.Get()));
	CHECK_FALSE(first->IdentifiesSameFile(firstPath.Get().string() + "-missing"));
}


TEST_CASE("RandomAccessFile maps unaligned ranges and completes mapped writes")
{
	TemporaryPath temporary;
	auto file = RandomAccessFile::TryCreateExclusive(temporary.Get());
	REQUIRE(file.has_value());
	file->Resize(8192);
	{
		auto region = file->MapReadWrite(127, 4);
		REQUIRE(region.Size() == 4);
		region.Data()[0] = Byte{0x12};
		region.Data()[1] = Byte{0x34};
		region.Data()[2] = Byte{0x56};
		region.Data()[3] = Byte{0x78};
		region.Complete();
		CHECK(region.Size() == 0);
	}
	const auto region = file->MapReadOnly(127, 4);
	CHECK(region.Data()[0] == Byte{0x12});
	CHECK(region.Data()[3] == Byte{0x78});
	CHECK(file->MapReadOnly(8192, 0).Size() == 0);
	CHECK_THROWS_AS(file->MapReadOnly(8193, 0), ArgumentOutOfRangeException);
	CHECK_THROWS_AS(file->MapReadOnly(8191, 2), ArgumentOutOfRangeException);
}


TEST_CASE("RandomAccessFile locks coordinate open-file states and retain their file")
{
	TemporaryPath temporary;
	REQUIRE(RandomAccessFile::TryCreateExclusive(temporary.Get()).has_value());
	auto firstReader = RandomAccessFile::OpenForReadOnly(temporary.Get());
	auto secondReader = RandomAccessFile::OpenForReadOnly(temporary.Get());
	auto writer = RandomAccessFile::OpenForReadWrite(temporary.Get());
	auto firstLock = firstReader.TryAcquireLock(RandomAccessFile::LockMode::Shared);
	REQUIRE(firstLock.has_value());
	CHECK_FALSE(firstReader.TryAcquireLock(RandomAccessFile::LockMode::Shared).has_value());
	auto secondLock = secondReader.TryAcquireLock(RandomAccessFile::LockMode::Shared);
	REQUIRE(secondLock.has_value());
	CHECK_FALSE(writer.TryAcquireLock(RandomAccessFile::LockMode::Exclusive).has_value());

	firstReader = RandomAccessFile::OpenForReadOnly(temporary.Get());
	firstLock.reset();
	CHECK_FALSE(writer.TryAcquireLock(RandomAccessFile::LockMode::Exclusive).has_value());
	secondReader = RandomAccessFile::OpenForReadOnly(temporary.Get());
	secondLock.reset();
	CHECK(writer.TryAcquireLock(RandomAccessFile::LockMode::Exclusive).has_value());
}


TEST_CASE("RandomAccessFile locks coordinate cooperating processes")
{
	TemporaryPath temporary;
	REQUIRE(RandomAccessFile::TryCreateExclusive(temporary.Get()).has_value());

	SECTION("shared child permits shared lock and rejects exclusive lock")
	{
		ChildLock child(temporary.Get(), "shared");
		auto reader = RandomAccessFile::OpenForReadOnly(temporary.Get());
		auto writer = RandomAccessFile::OpenForReadWrite(temporary.Get());
		CHECK(reader.TryAcquireLock(RandomAccessFile::LockMode::Shared).has_value());
		CHECK_FALSE(writer.TryAcquireLock(RandomAccessFile::LockMode::Exclusive).has_value());
	}

	SECTION("exclusive child rejects shared and exclusive locks")
	{
		ChildLock child(temporary.Get(), "exclusive");
		auto reader = RandomAccessFile::OpenForReadOnly(temporary.Get());
		auto writer = RandomAccessFile::OpenForReadWrite(temporary.Get());
		CHECK_FALSE(reader.TryAcquireLock(RandomAccessFile::LockMode::Shared).has_value());
		CHECK_FALSE(writer.TryAcquireLock(RandomAccessFile::LockMode::Exclusive).has_value());
	}
}


TEST_CASE("RandomAccessFile supports concurrent positional transfers on one open file")
{
	TemporaryPath temporary;
	auto file = RandomAccessFile::TryCreateExclusive(temporary.Get());
	REQUIRE(file.has_value());
	constexpr std::size_t BlockBytes = 4096;
	constexpr std::size_t ThreadCount = 8;
	file->Resize(BlockBytes * ThreadCount);
	std::array<std::thread, ThreadCount> threads;
	for (std::size_t threadIndex = 0; threadIndex < ThreadCount; ++threadIndex)
	{
		threads[threadIndex] = std::thread([&, threadIndex]
		{
			std::vector<Byte> bytes(BlockBytes, Byte{static_cast<unsigned char>(threadIndex + 1)});
			file->WriteExactAt(threadIndex * BlockBytes, bytes.data(), bytes.size());
		});
	}
	for (auto& thread : threads) thread.join();
	for (std::size_t threadIndex = 0; threadIndex < ThreadCount; ++threadIndex)
	{
		std::vector<Byte> bytes(BlockBytes);
		file->ReadExactAt(threadIndex * BlockBytes, bytes.data(), bytes.size());
		CHECK(bytes.front() == Byte{static_cast<unsigned char>(threadIndex + 1)});
		CHECK(bytes.back() == Byte{static_cast<unsigned char>(threadIndex + 1)});
	}
}
