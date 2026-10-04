#include <SecUtility/IO/PersistentStore/Adapter/EigenDense.hpp>
#include <SecUtility/IO/PersistentStoreChecker.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <system_error>


using namespace SecUtility::IO;


int main()
{
	const std::filesystem::path path = std::filesystem::temp_directory_path()
	                                   / "secutility-persistent-store-benchmark";
	std::error_code ignoredError;
	std::filesystem::remove(path, ignoredError);
	try
	{
		std::mt19937 random(0x5EC07117U);
		std::uniform_real_distribution<double> distribution(-1.0, 1.0);
		auto MakeMatrix = [&](const Eigen::Index rows, const Eigen::Index columns)
		{
			return Eigen::MatrixXd::NullaryExpr(rows, columns, [&] { return distribution(random); });
		};
		UInt64 highWaterBytes = 0;
		UInt64 warmupBytes = 0;
		UInt64 postWarmupRequestedBytes = 0;
		{
			auto store = PersistentStore::Create(path);
			for (int index = 0; index < 10; ++index)
			{
				store.Insert("matrix-" + std::to_string(index), MakeMatrix(64 + index * 3, 48 + index * 2));
				highWaterBytes = std::max<UInt64>(highWaterBytes, std::filesystem::file_size(path));
			}
			for (int iteration = 0; iteration < 40; ++iteration)
			{
				const int index = iteration % 10;
				store.Reassign("matrix-" + std::to_string(index), MakeMatrix(64 + index * 3, 48 + index * 2));
				highWaterBytes = std::max<UInt64>(highWaterBytes, std::filesystem::file_size(path));
				if (iteration == 19) warmupBytes = highWaterBytes;
				if (iteration >= 20)
					postWarmupRequestedBytes += static_cast<UInt64>((64 + index * 3) * (48 + index * 2) * sizeof(double));
			}
		}
		auto store = PersistentStore::OpenForReadOnly(path);
		const auto owningStart = std::chrono::steady_clock::now();
		for (int iteration = 0; iteration < 100; ++iteration)
			(void)store.Get<Eigen::MatrixXd>("matrix-" + std::to_string(iteration % 10));
		const auto owningEnd = std::chrono::steady_clock::now();
		const auto leasedStart = owningEnd;
		for (int iteration = 0; iteration < 100; ++iteration)
			(void)store.GetLeased<Eigen::MatrixXd>("matrix-" + std::to_string(iteration % 10));
		const auto leasedEnd = std::chrono::steady_clock::now();
		const auto checkerStart = std::chrono::steady_clock::now();
		const PersistentStoreCheckReport report = CheckPersistentStore(path);
		const auto checkerEnd = std::chrono::steady_clock::now();
		const UInt64 postWarmupGrowth = highWaterBytes - warmupBytes;
		const double reuseRatio = postWarmupRequestedBytes == 0 ? 0.0
		        : 1.0 - static_cast<double>(postWarmupGrowth) / static_cast<double>(postWarmupRequestedBytes);
		std::cout << "physical_bytes=" << report.PhysicalFileBytes
		          << " high_water_bytes=" << highWaterBytes
		          << " reused_ratio=" << reuseRatio
		          << " reclaimable_bytes=" << report.ReclaimableBytes
		          << " root_retained_bytes=" << report.OtherRootOnlyBytes
		          << " owning_read_us="
		          << std::chrono::duration_cast<std::chrono::microseconds>(owningEnd - owningStart).count()
		          << " leased_read_us="
		          << std::chrono::duration_cast<std::chrono::microseconds>(leasedEnd - leasedStart).count()
		          << " mapping_setup_us="
		          << std::chrono::duration_cast<std::chrono::microseconds>(leasedEnd - leasedStart).count()
		          << " checker_us="
		          << std::chrono::duration_cast<std::chrono::microseconds>(checkerEnd - checkerStart).count()
		          << '\n';
		std::filesystem::remove(path, ignoredError);
		const bool hasBoundedPostWarmupGrowth = postWarmupGrowth < 1024 * 1024;
		return report.HasSelectedRoot && report.LiveRecordCount == 10 && hasBoundedPostWarmupGrowth ? 0 : 1;
	}
	catch (...)
	{
		std::filesystem::remove(path, ignoredError);
		throw;
	}
}
