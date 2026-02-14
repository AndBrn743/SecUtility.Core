//
// Created by Claude on 2/10/2026.
//

#include "SecUtility/Text/Symbol.hpp"
#include "catch2/catch_approx.hpp"
#include <SecUtility/Diagnostic/Stopwatch.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include <chrono>
#include <cmath>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")
#endif


using namespace SecUtility;
using namespace SecUtility::Diagnostic::Stopwatch;
using Catch::Approx;
using SecUtility::TimeUnit;


static void CpuWork()
{
	volatile double x = 0.0;
	for (int i = 0; i < 5'000'000; ++i)
	{
		x += std::sin(i * 0.00001);
	}
	(void)x;
}


TEST_CASE("CpuStopwatch - Initial state")
{
	SECTION("Not running initially")
	{
		CpuStopwatch sw;
		CHECK(sw.IsRunning() == false);
	}

	SECTION("Elapsed time is zero initially")
	{
		CpuStopwatch sw;
		CHECK(sw.ElapsedTicks() == 0);
		CHECK(sw.Elapsed<TimeUnit::Milliseconds>() == Approx(0.0));
	}

	SECTION("Multiple stopwatches independent")
	{
		CpuStopwatch sw1;
		CpuStopwatch sw2;

		sw1.Start();
		sw2.Start();

		sw1.Stop();
		sw2.Stop();

		CHECK(sw1.ElapsedTicks() >= 0);
		CHECK(sw2.ElapsedTicks() >= 0);
	}
}


TEST_CASE("CpuStopwatch - Basic timing operations")
{
	SECTION("Start marks as running")
	{
		CpuStopwatch sw;
		sw.Start();

		CHECK(sw.IsRunning() == true);
	}

	SECTION("Stop marks as not running")
	{
		CpuStopwatch sw;
		sw.Start();
		sw.Stop();

		CHECK(sw.IsRunning() == false);
	}

	SECTION("Accumulates time while running")
	{
		CpuStopwatch sw;
		sw.Start();

		const auto t0 = sw.Elapsed<TimeUnit::Milliseconds>();
		CpuWork();
		const auto t1 = sw.Elapsed<TimeUnit::Seconds>();
		CpuWork();
		const auto t2 = sw.Elapsed<TimeUnit::Seconds>();

		CHECK(t0 < t1);
		CHECK(t1 < t2);
	}

	SECTION("Does not accumulate when stopped")
	{
		CpuStopwatch sw;
		sw.Start();
		sw.Stop();

		const auto before = sw.Elapsed<TimeUnit::Milliseconds>();
		CpuWork();
		const auto after = sw.Elapsed<TimeUnit::Milliseconds>();

		CHECK(after == before);
	}
}


TEST_CASE("Stopwatch - Reset behavior")
{
	SECTION("Reset clears elapsed time")
	{
		Stopwatch sw;
		sw.Start();
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		sw.Stop();

		REQUIRE(sw.Elapsed<TimeUnit::Milliseconds>() > 40.0);

		sw.Reset();

		CHECK(sw.Elapsed<TimeUnit::Milliseconds>() == 0);
		CHECK(sw.ElapsedTicks() == 0);
		CHECK(sw.IsRunning() == false);
	}

	SECTION("Reset while running stops timing")
	{
		Stopwatch sw;
		sw.Start();
		std::this_thread::sleep_for(std::chrono::milliseconds(50));

		sw.Reset();

		CHECK(sw.Elapsed<TimeUnit::Milliseconds>() == 0);
		CHECK(sw.IsRunning() == false);
	}

	SECTION("Can start again after reset")
	{
		Stopwatch sw;
		sw.Start();
		sw.Stop();
		sw.Reset();
		sw.Start();

		CHECK(sw.IsRunning() == true);
	}
}


TEST_CASE("Stopwatch - Restart behavior")
{
	SECTION("Restart clears and starts fresh")
	{
		Stopwatch sw;
		sw.Start();
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		sw.Stop();

		const auto before = sw.Elapsed<TimeUnit::Milliseconds>();

		sw.Restart();

		CHECK(sw.Elapsed<TimeUnit::Milliseconds>() < 0.05);  // approx zero
		CHECK(sw.IsRunning() == true);

		std::this_thread::sleep_for(std::chrono::milliseconds(30));

		const auto after = sw.Elapsed<TimeUnit::Milliseconds>();

		CHECK(after >= 25.0); // Allow tolerance
		CHECK(after < before); // Should be less than before reset
	}

	SECTION("Restart when already running")
	{
		Stopwatch sw;
		sw.Start();
		std::this_thread::sleep_for(std::chrono::milliseconds(50));

		sw.Restart();

		CHECK(sw.Elapsed<TimeUnit::Milliseconds>() < 0.05);  // approx zero
		CHECK(sw.IsRunning() == true);
	}
}


TEST_CASE("Stopwatch - Multiple start/stop cycles")
{
#if defined(_WIN32)
	timeBeginPeriod(1);
#endif

	SECTION("Accumulates across multiple sessions")
	{
		Stopwatch sw;
		double total = 0.0;

		for (int i = 0; i < 3; ++i)
		{
			sw.Start();
			std::this_thread::sleep_for(std::chrono::milliseconds(20));
			sw.Stop();

			const auto session = sw.Elapsed<TimeUnit::Milliseconds>() - total;
			CHECK(session >= 15.0); // Allow tolerance
			CHECK(session <= 30.0);

			total = sw.Elapsed<TimeUnit::Milliseconds>();
		}

		CHECK(total >= 55.0); // ~3 * 20ms
		CHECK(total <= 100.0);
	}

	SECTION("Second start while running does nothing")
	{
		Stopwatch sw;
		sw.Start();
		sw.Start(); // Should be ignored

		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		sw.Stop();

		// Should only count once
		const auto elapsed = sw.Elapsed<TimeUnit::Milliseconds>();
		CHECK(elapsed >= 45.0);
		CHECK(elapsed <= 65.0);
	}
}


TEST_CASE("Stopwatch - Elapsed formats")
{
#if defined(_WIN32)
	timeBeginPeriod(1);
#endif

	SECTION("Milliseconds format")
	{
		Stopwatch sw;
		sw.Start();
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		sw.Stop();

		const auto ms = sw.Elapsed<TimeUnit::Milliseconds>();
		CHECK(ms >= 95.0);
		CHECK(ms <= 120.0);
	}

	SECTION("Seconds format")
	{
		Stopwatch sw;
		sw.Start();
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		sw.Stop();

		const auto s = sw.Elapsed<TimeUnit::Seconds>();
		CHECK(s >= 0.09);
		CHECK(s <= 0.13);
	}

	SECTION("Microseconds format")
	{
		Stopwatch sw;
		sw.Start();
		std::this_thread::sleep_for(std::chrono::milliseconds(50));

		const auto us = sw.Elapsed<TimeUnit::Microseconds>();
		CHECK(us >= 45000.0);
		CHECK(us <= 60000.0);
	}

	SECTION("Ticks format returns integer")
	{
		Stopwatch sw;
		sw.Start();
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		sw.Stop();

		const Int64 ticks = sw.ElapsedTicks();
		CHECK(ticks > 0);
	}
}


TEST_CASE("Stopwatch - Formatted output")
{
#if defined(_WIN32)
	timeBeginPeriod(1);
#endif

	SECTION("ToString() returns string")
	{
		Stopwatch sw;
		sw.Start();
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		sw.Stop();

		const auto str = sw.ToString();
		CHECK(!str.empty());
		CHECK((str.find("ms") != std::string::npos || str.find("sec") != std::string::npos));
	}

	SECTION("ToString<Unit, Width, Precision>() formats correctly")
	{
		Stopwatch sw;
		sw.Start();
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		sw.Stop();

		const auto str = sw.ToString<TimeUnit::Milliseconds, 10, 2>();
		CHECK(!str.empty());
		CHECK(str.length() >= 6);
	}

	SECTION("Milliseconds format includes 'ms' suffix")
	{
		Stopwatch sw;
		sw.Start();
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		sw.Stop();

		const auto str = sw.ToString<TimeUnit::Milliseconds>();
		CHECK(str.find("ms") != std::string::npos);
	}

	SECTION("Seconds format includes 'sec' suffix")
	{
		Stopwatch sw;
		sw.Start();
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		sw.Stop();

		const auto str = sw.ToString<TimeUnit::Seconds>();
		CHECK(str.find("sec") != std::string::npos);
	}

	SECTION("Ticks format includes 'ticks' suffix")
	{
		Stopwatch sw;
		sw.Start();
		std::this_thread::sleep_for(std::chrono::milliseconds(50));

		const auto str = sw.ToString<TimeUnit::Ticks>();
		CHECK(str.find("ticks") != std::string::npos);
	}
}


TEST_CASE("Stopwatch - ToCString and ToCStringSymbol")
{
	SECTION("ToCString returns correct strings")
	{
		CHECK(std::string(ToCString(TimeUnit::Ticks)) == "Ticks");
		CHECK(std::string(ToCString(TimeUnit::Microseconds)) == "Microseconds");
		CHECK(std::string(ToCString(TimeUnit::Milliseconds)) == "Milliseconds");
		CHECK(std::string(ToCString(TimeUnit::Seconds)) == "Seconds");
	}

	SECTION("ToCStringSymbol returns correct symbols")
	{
		CHECK(std::string(ToCStringSymbol(TimeUnit::Ticks)) == "ticks");
		CHECK(std::string(ToCStringSymbol(TimeUnit::Microseconds)) == SEC_LOWER_MU "s");
		CHECK(std::string(ToCStringSymbol(TimeUnit::Milliseconds)) == "ms");
		CHECK(std::string(ToCStringSymbol(TimeUnit::Seconds)) == "sec");
	}
}


TEST_CASE("Stopwatch - Wall clock timing")
{
#if defined(_WIN32)
	timeBeginPeriod(1);
#endif

	SECTION("CpuStopwatch ignore sleeps")
	{
        
		CpuStopwatch cpuSw;
		Stopwatch sw;

		cpuSw.Start();
		sw.Start();

		std::this_thread::sleep_for(std::chrono::milliseconds(100));

		cpuSw.Stop();
		sw.Stop();
		
		const auto ms = sw.Elapsed<TimeUnit::Milliseconds>();
		CHECK(ms >= 95.0);
		CHECK(ms <= 120.0);

        CHECK(cpuSw.Elapsed<SecUtility::TimeUnit::Ticks>() < 10'000);
	}

	SECTION("CpuStopwatch should match Stopwatch with real works")
	{
		CpuStopwatch cpuSw;
		Stopwatch sw;

		cpuSw.Start();
		sw.Start();

		CpuWork();

		cpuSw.Stop();
		sw.Stop();

		CHECK(cpuSw.Elapsed<TimeUnit::Milliseconds>() == Approx(sw.Elapsed<TimeUnit::Milliseconds>()).margin(20));
	}
}


TEST_CASE("Stopwatch - Concurrent stopwatches are independent")
{
#if defined(_WIN32)
	timeBeginPeriod(1);
#endif

	SECTION("Two stopwatches can run simultaneously")
	{
		Stopwatch sw1;
		Stopwatch sw2;

		sw1.Start();
		std::this_thread::sleep_for(std::chrono::milliseconds(30));
		sw2.Start();

		std::this_thread::sleep_for(std::chrono::milliseconds(30));

		sw1.Stop();
		sw2.Stop();

		// sw1 ran for ~60ms, sw2 ran for ~30ms
		CHECK(sw1.Elapsed<TimeUnit::Milliseconds>() >= 55.0);
		CHECK(sw2.Elapsed<TimeUnit::Milliseconds>() >= 25.0);
		CHECK(sw1.Elapsed<TimeUnit::Milliseconds>() > sw2.Elapsed<TimeUnit::Milliseconds>());
	}

	SECTION("Multiple stopwatches don't interfere")
	{
		constexpr int count = 10;
		Stopwatch sw[count];

		for (int i = 0; i < count; ++i)
		{
			sw[i].Start();
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(50));

		for (int i = 0; i < count; ++i)
		{
			sw[i].Stop();
		}

		// All should measure approximately 50ms
		for (int i = 0; i < count; ++i)
		{
			const auto ms = sw[i].Elapsed<TimeUnit::Milliseconds>();
			CHECK(ms >= 45.0);
			CHECK(ms <= 65.0);
		}
	}
}


TEST_CASE("Stopwatch - Edge cases")
{
#if defined(_WIN32)
	timeBeginPeriod(1);
#endif

	SECTION("Start and stop immediately")
	{
		Stopwatch sw;
		sw.Start();
		sw.Stop();

		const auto elapsed = sw.Elapsed<TimeUnit::Microseconds>();
		CHECK(elapsed >= 0);
		CHECK(elapsed < 10000);  // Should be very small (<10ms)
	}

	SECTION("Multiple stops in a row")
	{
		Stopwatch sw;
		sw.Start();
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
		sw.Stop();
		sw.Stop();  // Should have no effect
		sw.Stop();

		const auto ms1 = sw.Elapsed<TimeUnit::Milliseconds>();
		const auto ms2 = sw.Elapsed<TimeUnit::Milliseconds>();

		CHECK(ms1 == Approx(ms2).epsilon(0.001));
	}

	SECTION("Multiple starts in a row")
	{
		Stopwatch sw;
		sw.Start();
		sw.Start();  // Should have no effect

		std::this_thread::sleep_for(std::chrono::milliseconds(50));

		const auto ms = sw.Elapsed<TimeUnit::Milliseconds>();
		CHECK(ms >= 45.0);
		CHECK(ms <= 65.0);
	}

	SECTION("Stop without start")
	{
		Stopwatch sw;
		sw.Stop();  // Should have no effect

		CHECK(sw.ElapsedTicks() == 0);
		CHECK(sw.IsRunning() == false);
	}

	SECTION("Reset when not started")
	{
		Stopwatch sw;
		sw.Reset();  // Should have no effect

		CHECK(sw.ElapsedTicks() == 0);
		CHECK(sw.IsRunning() == false);
	}
}


TEST_CASE("Stopwatch - State consistency")
{
	SECTION("Start changes IsRunning state")
	{
		Stopwatch sw;
		REQUIRE(!sw.IsRunning());

		sw.Start();
		CHECK(sw.IsRunning());

		sw.Stop();
		CHECK(!sw.IsRunning());
	}

	SECTION("Reset clears IsRunning state")
	{
		Stopwatch sw;
		sw.Start();
		REQUIRE(sw.IsRunning());

		sw.Reset();
		CHECK(!sw.IsRunning());
	}

	SECTION("Restart sets IsRunning to true")
	{
		Stopwatch sw;
		sw.Start();
		sw.Stop();
		REQUIRE(!sw.IsRunning());

		sw.Restart();
		CHECK(sw.IsRunning());
	}
}
