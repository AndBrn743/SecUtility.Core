#include <SecUtility/IO/PersistentStore/Adapter/ByteSequence.hpp>
#include <SecUtility/IO/PersistentStore/Detail/FailureInjection.hpp>

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <csignal>
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
			m_Path = std::filesystem::temp_directory_path()
			         / ("secutility-persistent-store-crash-"
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

	class CrashWorker final
	{
	public:
		CrashWorker(const std::filesystem::path& path,
		            const FailurePoint point,
		            const UInt64 detail = 0,
		            const bool matchesDetail = false)
		{
			const std::string pointText = std::to_string(static_cast<int>(point));
			const std::string detailText = std::to_string(detail);
#if defined(_WIN32)
			SECURITY_ATTRIBUTES attributes{};
			attributes.nLength = sizeof(attributes);
			attributes.bInheritHandle = TRUE;
			HANDLE inputRead = nullptr;
			HANDLE outputWrite = nullptr;
			REQUIRE(::CreatePipe(&inputRead, &m_InputWrite, &attributes, 0));
			REQUIRE(::CreatePipe(&m_OutputRead, &outputWrite, &attributes, 0));
			::SetHandleInformation(m_InputWrite, HANDLE_FLAG_INHERIT, 0);
			::SetHandleInformation(m_OutputRead, HANDLE_FLAG_INHERIT, 0);
			STARTUPINFOW startup{};
			startup.cb = sizeof(startup);
			startup.dwFlags = STARTF_USESTDHANDLES;
			startup.hStdInput = inputRead;
			startup.hStdOutput = outputWrite;
			startup.hStdError = outputWrite;
			std::wstring command = L"\"" + std::filesystem::path(PERSISTENT_STORE_CRASH_WORKER_PATH).wstring()
			        + L"\" \"" + path.wstring() + L"\" "
			        + std::filesystem::path(pointText).wstring() + L" "
			        + std::filesystem::path(detailText).wstring() + L" " + (matchesDetail ? L"1" : L"0");
			PROCESS_INFORMATION process{};
			REQUIRE(::CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr,
			                         &startup, &process));
			::CloseHandle(inputRead);
			::CloseHandle(outputWrite);
			::CloseHandle(process.hThread);
			m_Process = process.hProcess;
#else
			int inputPipe[2]{};
			int outputPipe[2]{};
			REQUIRE(::pipe(inputPipe) == 0);
			REQUIRE(::pipe(outputPipe) == 0);
			m_Process = ::fork();
			REQUIRE(m_Process >= 0);
			if (m_Process == 0)
			{
				::dup2(inputPipe[0], STDIN_FILENO);
				::dup2(outputPipe[1], STDOUT_FILENO);
				::dup2(outputPipe[1], STDERR_FILENO);
				::close(inputPipe[0]); ::close(inputPipe[1]);
				::close(outputPipe[0]); ::close(outputPipe[1]);
				::execl(PERSISTENT_STORE_CRASH_WORKER_PATH, PERSISTENT_STORE_CRASH_WORKER_PATH,
				        path.c_str(), pointText.c_str(), detailText.c_str(), matchesDetail ? "1" : "0",
				        static_cast<char*>(nullptr));
				::_exit(127);
			}
			::close(inputPipe[0]);
			::close(outputPipe[1]);
			m_InputWrite = inputPipe[1];
			m_OutputRead = outputPipe[0];
#endif
			WaitForReady();
		}

		~CrashWorker() noexcept { Kill(); }
		CrashWorker(const CrashWorker&) = delete;
		CrashWorker& operator=(const CrashWorker&) = delete;

		void Kill() noexcept
		{
#if defined(_WIN32)
			if (m_Process != nullptr)
			{
				::TerminateProcess(m_Process, 99);
				::WaitForSingleObject(m_Process, 5000);
				::CloseHandle(m_Process);
				::CloseHandle(m_InputWrite);
				::CloseHandle(m_OutputRead);
				m_Process = nullptr;
			}
#else
			if (m_Process > 0)
			{
				::kill(m_Process, SIGKILL);
				int status = 0;
				(void)::waitpid(m_Process, &status, 0);
				::close(m_InputWrite);
				::close(m_OutputRead);
				m_Process = -1;
			}
#endif
		}

	private:
		void WaitForReady()
		{
			std::string line;
			for (;;)
			{
				char character = 0;
#if defined(_WIN32)
				DWORD bytes = 0;
				REQUIRE(::ReadFile(m_OutputRead, &character, 1, &bytes, nullptr));
				REQUIRE(bytes == 1);
#else
				REQUIRE(::read(m_OutputRead, &character, 1) == 1);
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
		HANDLE m_Process = nullptr;
		HANDLE m_InputWrite = nullptr;
		HANDLE m_OutputRead = nullptr;
#else
		pid_t m_Process = -1;
		int m_InputWrite = -1;
		int m_OutputRead = -1;
#endif
	};

	void CreateOldValue(const std::filesystem::path& path)
	{
		auto store = PersistentStore::Create(path);
		store.Insert("value", std::vector<Byte>{Byte{0x11}});
	}
}


TEST_CASE("PersistentStore process crash before publication preserves the old root")
{
	const std::vector<FailurePoint> points{
	        FailurePoint::RebuildAllocator, FailurePoint::ReservePayload, FailurePoint::GrowPayload,
	        FailurePoint::MapPayload, FailurePoint::EncodePayload, FailurePoint::CompletePayload,
	        FailurePoint::SerializeDirectory, FailurePoint::ReserveDirectory, FailurePoint::GrowDirectory,
	        FailurePoint::WriteDirectory, FailurePoint::ReadBackDirectory, FailurePoint::PrepareRuntimeState};
	for (const FailurePoint point : points)
	{
		TemporaryPath temporary;
		CreateOldValue(temporary.Get());
		CrashWorker worker(temporary.Get(), point);
		worker.Kill();
		auto store = PersistentStore::OpenForReadOnly(temporary.Get());
		CHECK(store.Get<std::vector<Byte>>("value") == std::vector<Byte>{Byte{0x11}});
	}
}


TEST_CASE("PersistentStore process crash during publication recovers an old or new complete root")
{
	{
		TemporaryPath temporary;
		CreateOldValue(temporary.Get());
		CrashWorker worker(temporary.Get(), FailurePoint::BeginPublication);
		worker.Kill();
		auto store = PersistentStore::OpenForReadOnly(temporary.Get());
		CHECK(store.Get<std::vector<Byte>>("value") == std::vector<Byte>{Byte{0x11}});
	}
	for (UInt64 bytes = 1; bytes <= HeaderSlotBytes; ++bytes)
	{
		TemporaryPath temporary;
		CreateOldValue(temporary.Get());
		CrashWorker worker(temporary.Get(), FailurePoint::PublicationWriteProgress, bytes, true);
		worker.Kill();
		auto store = PersistentStore::OpenForReadOnly(temporary.Get());
		const auto value = store.Get<std::vector<Byte>>("value");
		CHECK((value == std::vector<Byte>{Byte{0x11}} || value == std::vector<Byte>{Byte{0x22}}));
	}
}


TEST_CASE("PersistentStore reclaims crash-orphaned space after root rotation")
{
	TemporaryPath temporary;
	CreateOldValue(temporary.Get());
	{
		CrashWorker worker(temporary.Get(), FailurePoint::PrepareRuntimeState);
		worker.Kill();
	}
	auto store = PersistentStore::OpenForReadWrite(temporary.Get());
	store.Reassign("value", std::vector<Byte>{Byte{0x31}});
	store.Reassign("value", std::vector<Byte>{Byte{0x32}});
	store.Reassign("value", std::vector<Byte>{Byte{0x33}});
	const auto stableBytes = std::filesystem::file_size(temporary.Get());
	store.Reassign("value", std::vector<Byte>{Byte{0x34}});
	CHECK(std::filesystem::file_size(temporary.Get()) == stableBytes);
	CHECK(store.Get<std::vector<Byte>>("value") == std::vector<Byte>{Byte{0x34}});
}
