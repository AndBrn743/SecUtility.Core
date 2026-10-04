#include <SecUtility/IO/PersistentStore/Adapter/Array.hpp>
#include <SecUtility/IO/PersistentStore/Adapter/ByteSequence.hpp>
#include <SecUtility/IO/PersistentStore/Adapter/EigenDense.hpp>
#include <SecUtility/IO/PersistentStoreChecker.hpp>

#include <catch2/catch_test_macros.hpp>

#include <Eigen/Core>

#include <array>
#include <chrono>
#include <cstdint>
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
			         / ("secutility-persistent-store-integration-"
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


TEST_CASE("PersistentStore R1 adapters compose through mutation leases checking and reopen")
{
	TemporaryPath temporary;
	const Eigen::MatrixXd firstMatrix = Eigen::MatrixXd::Random(5, 4);
	const Eigen::MatrixXd secondMatrix = Eigen::MatrixXd::Random(5, 4);
	{
		auto oldMatrixLease = [&]
		{
			auto store = PersistentStore::Create(temporary.Get());
			store.Insert("iteration", std::int32_t{7});
			store.Insert("indices", std::array<std::uint64_t, 3>{2, 3, 5});
			store.Insert("blob", std::vector<Byte>{Byte{0x10}, Byte{0x20}, Byte{0x30}});
			store.Insert("matrix", firstMatrix);
			auto lease = store.GetLeased<Eigen::MatrixXd>("matrix");
			store.Reassign("matrix", secondMatrix);
			CHECK(store.Erase("blob"));
			CHECK(store.Size() == 3);
			CHECK(Eigen::MatrixXd(lease) == firstMatrix);
			return lease;
		}();
		CHECK(Eigen::MatrixXd(oldMatrixLease) == firstMatrix);
		CHECK_THROWS_AS(CheckPersistentStore(temporary.Get()), IOException);
	}

	const PersistentStoreCheckReport report = CheckPersistentStore(temporary.Get());
	CHECK(report.HasSelectedRoot);
	CHECK(report.LiveRecordCount == 3);
	{
		const auto store = PersistentStore::OpenForReadOnly(temporary.Get());
		CHECK(store.Get<std::int32_t>("iteration") == 7);
		CHECK((store.Get<std::array<std::uint64_t, 3>>("indices")
		       == std::array<std::uint64_t, 3>{2, 3, 5}));
		CHECK_FALSE(store.Contains("blob"));
		CHECK(store.Get<Eigen::MatrixXd>("matrix").isApprox(secondMatrix));
	}
}
