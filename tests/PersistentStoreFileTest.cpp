#include <SecUtility/IO/PersistentStore.hpp>
#include <SecUtility/IO/PersistentStore/Detail/FileBackend.hpp>
#include <SecUtility/IO/PersistentStore/Detail/FormatCodec.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
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
using namespace SecUtility::IO::PersistentStoreDetail;


namespace
{
	class TemporaryPath final
	{
	public:
		TemporaryPath()
		{
			const auto token = std::chrono::steady_clock::now().time_since_epoch().count();
			m_Path = std::filesystem::temp_directory_path()
			         / ("secutility-persistent-store-" + std::to_string(token) + "-" + std::to_string(++Counter));
		}

		~TemporaryPath()
		{
			std::error_code ignoredError;
			std::filesystem::remove_all(m_Path, ignoredError);
		}

		const std::filesystem::path& Get() const noexcept { return m_Path; }

	private:
		inline static unsigned long long Counter = 0;
		std::filesystem::path m_Path;
	};

	std::vector<Byte> MakeDirectory(const UInt64 generation,
	                                const std::string& key = {},
	                                const UInt64 payloadOffset = 0)
	{
		Directory directory;
		directory.Generation = generation;
		if (!key.empty() || payloadOffset != 0)
		{
			directory.Entries.push_back({key, {payloadOffset, 8}, payloadOffset, 8});
		}
		return SerializeDirectory(directory);
	}

	HeaderSlot WriteRoot(FileBackend& ref_backend,
	                     const UInt64 slotOffset,
	                     const UInt64 generation,
	                     const UInt64 directoryOffset,
	                     const std::vector<Byte>& directoryBytes,
	                     const UInt64 committedBytes)
	{
		HeaderSlot header;
		header.Generation = generation;
		header.DirectoryOffset = directoryOffset;
		header.DirectoryBytes = directoryBytes.size();
		header.CommittedLogicalFileBytes = committedBytes;
		header.DirectoryChecksum = FormatCodecDetail::ComputeCrc32C(ConstByteView(directoryBytes));
		const auto headerBytes = SerializeHeader(header);
		ref_backend.WriteExact(directoryOffset, directoryBytes.data(), directoryBytes.size());
		ref_backend.WriteExact(slotOffset, headerBytes.data(), headerBytes.size());
		return header;
	}

	class ChildLock final
	{
	public:
		ChildLock(const std::filesystem::path& path, const char* access)
		{
#if defined(_WIN32)
			SECURITY_ATTRIBUTES attributes{};
			attributes.nLength = sizeof(attributes);
			attributes.bInheritHandle = TRUE;
			HANDLE childStdInRead = nullptr;
			HANDLE childStdInWrite = nullptr;
			HANDLE childStdOutRead = nullptr;
			HANDLE childStdOutWrite = nullptr;
			REQUIRE(::CreatePipe(&childStdInRead, &childStdInWrite, &attributes, 0));
			REQUIRE(::CreatePipe(&childStdOutRead, &childStdOutWrite, &attributes, 0));
			::SetHandleInformation(childStdInWrite, HANDLE_FLAG_INHERIT, 0);
			::SetHandleInformation(childStdOutRead, HANDLE_FLAG_INHERIT, 0);

			STARTUPINFOW startup{};
			startup.cb = sizeof(startup);
			startup.dwFlags = STARTF_USESTDHANDLES;
			startup.hStdInput = childStdInRead;
			startup.hStdOutput = childStdOutWrite;
			startup.hStdError = childStdOutWrite;
			PROCESS_INFORMATION process{};
			std::wstring command = L"\"" + std::filesystem::path(PERSISTENT_STORE_LOCK_HELPER_PATH).wstring()
			                       + L"\" " + std::filesystem::path(access).wstring() + L" \"" + path.wstring() + L"\"";
			REQUIRE(::CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr,
			                         &startup, &process));
			::CloseHandle(childStdInRead);
			::CloseHandle(childStdOutWrite);
			::CloseHandle(process.hThread);
			m_ProcessHandle = process.hProcess;
			m_InputHandle = childStdInWrite;
			m_OutputHandle = childStdOutRead;
			char ready[5]{};
			DWORD readBytes = 0;
			REQUIRE(::ReadFile(m_OutputHandle, ready, sizeof(ready), &readBytes, nullptr));
			REQUIRE(std::string(ready, readBytes) == "ready");
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
				::execl(PERSISTENT_STORE_LOCK_HELPER_PATH, PERSISTENT_STORE_LOCK_HELPER_PATH,
				        access, path.c_str(), static_cast<char*>(nullptr));
				::_exit(127);
			}
			::close(inputPipe[0]);
			::close(outputPipe[1]);
			m_InputDescriptor = inputPipe[1];
			m_OutputDescriptor = outputPipe[0];
			char ready[5]{};
			REQUIRE(::read(m_OutputDescriptor, ready, sizeof(ready)) == 5);
			REQUIRE(std::string(ready, 5) == "ready");
#endif
		}

		~ChildLock()
		{
#if defined(_WIN32)
			if (m_InputHandle != nullptr)
			{
				DWORD written = 0;
				::WriteFile(m_InputHandle, "stop\n", 5, &written, nullptr);
				::CloseHandle(m_InputHandle);
				::CloseHandle(m_OutputHandle);
				::WaitForSingleObject(m_ProcessHandle, 5000);
				::CloseHandle(m_ProcessHandle);
			}
#else
			if (m_ProcessId > 0)
			{
				(void)::write(m_InputDescriptor, "stop\n", 5);
				::close(m_InputDescriptor);
				::close(m_OutputDescriptor);
				int status = 0;
				(void)::waitpid(m_ProcessId, &status, 0);
			}
#endif
		}

		ChildLock(const ChildLock&) = delete;
		ChildLock& operator=(const ChildLock&) = delete;

	private:
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


TEST_CASE("PersistentStore factories have explicit existence behavior")
{
	TemporaryPath temporary;
	CHECK_THROWS_AS(PersistentStore::OpenForReadOnly(temporary.Get()), IOException);
	CHECK_THROWS_AS(PersistentStore::OpenForReadWrite(temporary.Get()), IOException);

	{
		auto store = PersistentStore::Create(temporary.Get());
		CHECK(store.Size() == 0);
		CHECK_FALSE(store.Contains("missing"));
	}
	CHECK_THROWS_AS(PersistentStore::Create(temporary.Get()), IOException);
	CHECK(std::filesystem::file_size(temporary.Get()) >= SuperblockBytes + DirectoryHeaderBytes);

	{
		auto store = PersistentStore::CreateIfNotExist(temporary.Get());
		CHECK(store.Size() == 0);
	}
}


TEST_CASE("PersistentStore exclusive creation has one winner")
{
	TemporaryPath temporary;
	std::atomic<unsigned int> successCount{0};
	std::atomic<unsigned int> failureCount{0};
	const auto create = [&] {
		try
		{
			auto store = PersistentStore::Create(temporary.Get());
			++successCount;
		}
		catch (const IOException&)
		{
			++failureCount;
		}
	};
	std::thread first(create);
	std::thread second(create);
	first.join();
	second.join();
	CHECK(successCount == 1);
	CHECK(failureCount == 1);
	CHECK_NOTHROW(PersistentStore::OpenForReadOnly(temporary.Get()));
}


TEST_CASE("PersistentStore supports native non-ASCII paths")
{
	TemporaryPath temporary;
	const std::filesystem::path path = temporary.Get().parent_path() / (temporary.Get().filename().native()
#if defined(_WIN32)
	                                                                    + std::wstring(L"-存储")
#else
	                                                                    + std::string("-存储")
#endif
	);
	std::error_code ignoredError;
	{
		auto store = PersistentStore::Create(path);
		CHECK(store.Size() == 0);
	}
	CHECK_NOTHROW(PersistentStore::OpenForReadOnly(path));
	std::filesystem::remove(path, ignoredError);
}


TEST_CASE("PersistentStore coordinates same-process lock owners")
{
	TemporaryPath temporary;
	{
		auto created = PersistentStore::Create(temporary.Get());
	}

	{
		auto firstReader = PersistentStore::OpenForReadOnly(temporary.Get());
		auto secondReader = PersistentStore::OpenForReadOnly(temporary.Get());
		CHECK_THROWS_AS(PersistentStore::OpenForReadWrite(temporary.Get()), IOException);
		auto movedReader = std::move(secondReader);
		CHECK(movedReader.Size() == 0);
	}
	CHECK_NOTHROW(PersistentStore::OpenForReadWrite(temporary.Get()));

	{
		auto writer = PersistentStore::OpenForReadWrite(temporary.Get());
		CHECK_THROWS_AS(PersistentStore::OpenForReadOnly(temporary.Get()), IOException);
		CHECK_THROWS_AS(PersistentStore::OpenForReadWrite(temporary.Get()), IOException);
	}
}


TEST_CASE("PersistentStore coordinates subprocess lock owners")
{
	TemporaryPath temporary;
	{
		auto created = PersistentStore::Create(temporary.Get());
	}

	SECTION("reader permits reader and rejects writer")
	{
		ChildLock child(temporary.Get(), "read");
		CHECK_NOTHROW(PersistentStore::OpenForReadOnly(temporary.Get()));
		CHECK_THROWS_AS(PersistentStore::OpenForReadWrite(temporary.Get()), IOException);
	}

	SECTION("writer rejects reader and writer")
	{
		ChildLock child(temporary.Get(), "write");
		CHECK_THROWS_AS(PersistentStore::OpenForReadOnly(temporary.Get()), IOException);
		CHECK_THROWS_AS(PersistentStore::OpenForReadWrite(temporary.Get()), IOException);
	}
}


TEST_CASE("PersistentStore backend performs exact positional I/O")
{
	TemporaryPath temporary;
	auto backendOptional = FileBackend::TryCreateExclusive(temporary.Get());
	REQUIRE(backendOptional.has_value());
	FileBackend backend = std::move(*backendOptional);
	std::vector<Byte> written(131123);
	for (std::size_t index = 0; index < written.size(); ++index)
	{
		written[index] = Byte{static_cast<unsigned char>((index * 37) & 0xFF)};
	}
	backend.SetPhysicalFileBytes(written.size() + 17);
	backend.WriteExact(11, written.data(), written.size());
	std::vector<Byte> read(written.size());
	backend.ReadExact(11, read.data(), read.size());
	CHECK(read == written);
	CHECK(backend.GetPhysicalFileBytes() == written.size() + 17);
}


TEST_CASE("PersistentStore exact-transfer loop handles interruptions and short operations")
{
	std::array<Byte, 9> bytes{};
	std::size_t callCount = 0;
	CompleteExactTransfer(100, bytes.data(), bytes.size(), true,
	                      [&](const UInt64 offset, Byte* bytesPtr, const std::size_t remainingBytes) {
		                      ++callCount;
		                      if (callCount == 1)
		                      {
			                      return ExactTransferResult{0, true};
		                      }
		                      CHECK(offset == 100 + (callCount - 2) * 2);
		                      const std::size_t transferred = std::min<std::size_t>(2, remainingBytes);
		                      std::fill(bytesPtr, bytesPtr + transferred, Byte{0x5A});
		                      return ExactTransferResult{transferred, false};
	                      });
	CHECK(callCount == 6);
	CHECK(std::all_of(bytes.begin(), bytes.end(), [](const Byte byte) { return byte == Byte{0x5A}; }));

	CHECK_THROWS_AS(
	        CompleteExactTransfer(0, bytes.data(), bytes.size(), false,
	                              [](const UInt64, Byte*, const std::size_t) { return ExactTransferResult{}; }),
	        IOException);
}


TEST_CASE("PersistentStore selects valid and newest roots")
{
	TemporaryPath temporary;
	{
		auto created = PersistentStore::Create(temporary.Get());
	}
	{
		auto backend = FileBackend::Open(temporary.Get(), FileAccess::ReadWrite);
		backend.SetPhysicalFileBytes(7000);
		const auto newerDirectory = MakeDirectory(2, "new", 4200);
		WriteRoot(backend, HeaderSlotBOffset, 2, 6000, newerDirectory, 7000);
	}
	{
		auto store = PersistentStore::OpenForReadOnly(temporary.Get());
		CHECK(store.Size() == 1);
		CHECK(store.Contains("new"));
	}

	{
		auto backend = FileBackend::Open(temporary.Get(), FileAccess::ReadWrite);
		std::array<Byte, HeaderSlotBytes> corrupt{};
		backend.WriteExact(HeaderSlotBOffset, corrupt.data(), corrupt.size());
	}
	{
		auto store = PersistentStore::OpenForReadOnly(temporary.Get());
		CHECK(store.Size() == 0);
	}
}


TEST_CASE("PersistentStore opens when only header slot B is valid")
{
	TemporaryPath temporary;
	{
		auto created = PersistentStore::Create(temporary.Get());
	}
	{
		auto backend = FileBackend::Open(temporary.Get(), FileAccess::ReadWrite);
		backend.SetPhysicalFileBytes(7000);
		const auto directory = MakeDirectory(2, "only-b", 4200);
		WriteRoot(backend, HeaderSlotBOffset, 2, 6000, directory, 7000);
		std::array<Byte, HeaderSlotBytes> corrupt{};
		backend.WriteExact(HeaderSlotAOffset, corrupt.data(), corrupt.size());
	}
	const auto store = PersistentStore::OpenForReadOnly(temporary.Get());
	CHECK(store.Size() == 1);
	CHECK(store.Contains("only-b"));
}


TEST_CASE("PersistentStore rejects conflicting equal-generation roots")
{
	TemporaryPath temporary;
	{
		auto created = PersistentStore::Create(temporary.Get());
	}
	{
		auto backend = FileBackend::Open(temporary.Get(), FileAccess::ReadWrite);
		backend.SetPhysicalFileBytes(7500);
		const auto first = MakeDirectory(2, "a", 4200);
		const auto second = MakeDirectory(2, "b", 4300);
		WriteRoot(backend, HeaderSlotAOffset, 2, 6000, first, 7500);
		WriteRoot(backend, HeaderSlotBOffset, 2, 6500, second, 7500);
	}
	CHECK_THROWS_AS(PersistentStore::OpenForReadOnly(temporary.Get()), FormatException);
}


TEST_CASE("PersistentStore rejects two invalid roots without modifying the file")
{
	TemporaryPath temporary;
	auto backendOptional = FileBackend::TryCreateExclusive(temporary.Get());
	REQUIRE(backendOptional.has_value());
	backendOptional->SetPhysicalFileBytes(SuperblockBytes);
	backendOptional.reset();
	const auto before = std::filesystem::file_size(temporary.Get());
	CHECK_THROWS_AS(PersistentStore::OpenForReadOnly(temporary.Get()), FormatException);
	CHECK(std::filesystem::file_size(temporary.Get()) == before);
}


TEST_CASE("PersistentStore rejects a committed length beyond the physical file")
{
	TemporaryPath temporary;
	{
		auto created = PersistentStore::Create(temporary.Get());
	}
	{
		auto backend = FileBackend::Open(temporary.Get(), FileAccess::ReadWrite);
		const auto directory = MakeDirectory(2);
		HeaderSlot header;
		header.Generation = 2;
		header.DirectoryOffset = SuperblockBytes;
		header.DirectoryBytes = directory.size();
		header.CommittedLogicalFileBytes = 9000;
		header.DirectoryChecksum = FormatCodecDetail::ComputeCrc32C(ConstByteView(directory));
		const auto headerBytes = SerializeHeader(header);
		std::array<Byte, HeaderSlotBytes> invalidSlot{};
		backend.WriteExact(HeaderSlotAOffset, headerBytes.data(), headerBytes.size());
		backend.WriteExact(HeaderSlotBOffset, invalidSlot.data(), invalidSlot.size());
	}
	CHECK_THROWS_AS(PersistentStore::OpenForReadOnly(temporary.Get()), FormatException);
}


TEST_CASE("PersistentStore tolerates an unreachable physical tail")
{
	TemporaryPath temporary;
	{
		auto created = PersistentStore::Create(temporary.Get());
	}
	const auto committed = std::filesystem::file_size(temporary.Get());
	{
		auto backend = FileBackend::Open(temporary.Get(), FileAccess::ReadWrite);
		backend.SetPhysicalFileBytes(committed + 4096);
	}
	CHECK_NOTHROW(PersistentStore::OpenForReadOnly(temporary.Get()));
}


TEST_CASE("PersistentStore preserves empty and embedded-NUL keys from a valid directory")
{
	TemporaryPath temporary;
	{
		auto created = PersistentStore::Create(temporary.Get());
	}
	{
		Directory directory;
		directory.Generation = 2;
		directory.Entries = {{"", {4200, 8}, 4200, 8},
		                     {std::string("a\0b", 3), {4300, 8}, 4300, 8}};
		const auto directoryBytes = SerializeDirectory(directory);
		auto backend = FileBackend::Open(temporary.Get(), FileAccess::ReadWrite);
		backend.SetPhysicalFileBytes(7000);
		WriteRoot(backend, HeaderSlotBOffset, 2, 6000, directoryBytes, 7000);
	}
	const auto store = PersistentStore::OpenForReadOnly(temporary.Get());
	CHECK(store.Size() == 2);
	CHECK(store.Contains(""));
	CHECK(store.Contains(std::string("a\0b", 3)));
}


#if !defined(_WIN32)
TEST_CASE("PersistentStore read-only factory works without write permission")
{
	if (::geteuid() == 0)
	{
		SUCCEED("permission test is not meaningful for the root user");
		return;
	}
	TemporaryPath temporary;
	{
		auto created = PersistentStore::Create(temporary.Get());
	}
	REQUIRE(::chmod(temporary.Get().c_str(), 0444) == 0);
	CHECK_NOTHROW(PersistentStore::OpenForReadOnly(temporary.Get()));
	CHECK_THROWS_AS(PersistentStore::OpenForReadWrite(temporary.Get()), IOException);
	REQUIRE(::chmod(temporary.Get().c_str(), 0644) == 0);
}
#endif


TEST_CASE("PersistentStore rejects non-regular paths")
{
	CHECK_THROWS_AS(PersistentStore::OpenForReadOnly(std::filesystem::temp_directory_path()), IOException);
}
