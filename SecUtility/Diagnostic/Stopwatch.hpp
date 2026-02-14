// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/Raw/Int.hpp>
#include <SecUtility/Text/Symbol.hpp>

#include <chrono>
#include <iomanip>
#include <limits>
#include <string>
#include <sstream>


#if defined(_WIN32)
#include <windows.h>
#include <iostream>
#endif


namespace SecUtility
{
	enum class TimeUnit
	{
		Ticks,
		Microseconds,
		Milliseconds,
		Seconds
	};

	constexpr auto Ticks = TimeUnit::Ticks;
	constexpr auto Microseconds = TimeUnit::Microseconds;
	constexpr auto Milliseconds = TimeUnit::Milliseconds;
	constexpr auto Seconds = TimeUnit::Seconds;

	constexpr const char* ToCString(const TimeUnit unit) noexcept
	{
		switch (unit)
		{
			case TimeUnit::Ticks:
				return "Ticks";
			case TimeUnit::Microseconds:
				return "Microseconds";
			case TimeUnit::Milliseconds:
				return "Milliseconds";
			case TimeUnit::Seconds:
				return "Seconds";
			default:
				return "Unknown";
		}
	}

	constexpr const char* ToCStringSymbol(const TimeUnit unit) noexcept
	{
		switch (unit)
		{
			case TimeUnit::Ticks:
				return "ticks";
			case TimeUnit::Microseconds:
				return SEC_LOWER_MU "s";
			case TimeUnit::Milliseconds:
				return "ms";
			case TimeUnit::Seconds:
				return "sec";
			default:
				return "unknown";
		}
	}

	inline std::string ToString(const TimeUnit unit)
	{
		return ToCString(unit);
	}

	inline std::string ToSymbol(const TimeUnit unit)
	{
		return ToCStringSymbol(unit);
	}


	/// <summary>
	/// Represents the number of (.NET) ticks in 1 second
	/// </summary>
	static constexpr Int64 TicksPerSecond = 1e+7;

	/// <summary>
	/// Represents the number of (.NET) ticks in 1 millisecond
	/// </summary>
	static constexpr Int64 TicksPerMillisecond = 1e+4;

	/// <summary>
	/// Represents the number of (.NET) ticks in 1 microsecond
	/// </summary>
	static constexpr Int64 TicksPerMicrosecond = 1e+1;

	/// <summary>
	/// Represents the number of nanoseconds in 1 (.NET) tick
	/// </summary>
	static constexpr Int64 NanosecondsPerTick = 1e+2;


	constexpr double Ticks2Seconds(const Int64 ticks)
	{
		return static_cast<double>(ticks) / TicksPerSecond;
	}

	constexpr double Ticks2Milliseconds(const Int64 ticks)
	{
		return static_cast<double>(ticks) / TicksPerMillisecond;
	}

	constexpr double Ticks2Microsecond(const Int64 ticks)
	{
		return static_cast<double>(ticks) / TicksPerMicrosecond;
	}
}

namespace SecUtility::Diagnostic::Stopwatch
{
	/// <summary>
	/// Provides a set of methods that you can use to accurately measure time.
	/// </summary>
	/// <remarks>
	/// The public interface is intentionally made to match <c>System.Diagnostics.Stopwatch</c> of .NET / C#.
	/// Some API docs are taken from .NET runtime source code. .NET runtime is released under MIT license, available
	/// from https://github.com/dotnet/runtime and https://source.dot.net
	/// </remarks>
	template <typename Derived>
	class StopwatchBase
	{
	public:
		/// <summary>
		/// Gets a value indicating whether the Stopwatch timer is running.
		/// </summary>
		constexpr bool IsRunning() const noexcept
		{
			return m_IsRunning;
		}


		/// <summary>
		/// Starts, or resumes, measuring elapsed time for an interval.
		/// </summary>
		void Start() noexcept
		{
			if (!m_IsRunning)
			{
				ResetStartTimestamp();
				m_IsRunning = true;
			}
		}


		/// <summary>
		/// Initializes a new Stopwatch instance, sets the elapsed time property to zero, and starts measuring elapsed
		/// time.
		/// </summary>
		Derived StartNew() const
		{
			Derived sw{};
			sw.Start();
			return sw;
		}


		/// <summary>
		/// Stops measuring elapsed time for an interval.
		/// </summary>
		void Stop() noexcept
		{
			if (m_IsRunning)
			{
				m_Elapsed += static_cast<Derived*>(this)->GetRawElapsedFromStartUntilNowTicks();
				m_IsRunning = false;

				if (m_Elapsed < 0)
				{
					m_Elapsed = 0;  // clamp to zero
				}
			}
		}


		/// <summary>
		/// Stops time interval measurement and resets the elapsed time to zero.
		/// </summary>
		void Reset() noexcept
		{
			m_Elapsed = 0;
			m_IsRunning = false;
		}


		/// <summary>
		/// Resets the elapsed time to zero, and starts measuring elapsed time.
		/// </summary>
		void Restart() noexcept
		{
			m_Elapsed = 0;
			ResetStartTimestamp();
			m_IsRunning = true;
		}


		/// <summary>
		/// Returns the elapsed time in given time unit, the default in millisecond.
		/// </summary>
		template <TimeUnit Unit = TimeUnit::Milliseconds>
		double Elapsed() const noexcept
		{
			switch (Unit)
			{
				case TimeUnit::Microseconds:
					return Ticks2Microsecond(ElapsedTicks());
				case TimeUnit::Milliseconds:
					return Ticks2Milliseconds(ElapsedTicks());
				case TimeUnit::Seconds:
					return Ticks2Seconds(ElapsedTicks());
				case TimeUnit::Ticks:
					return ElapsedTicks();
				default:
                    return std::numeric_limits<double>::signaling_NaN();
			}
		}


		/// <summary>
		/// Returns the elapsed time in given ticks.
		/// </summary>
		Int64 ElapsedTicks() const noexcept
		{
			if (m_IsRunning)
			{
				return m_Elapsed + static_cast<const Derived*>(this)->GetRawElapsedFromStartUntilNowTicks();
			}
			else
			{
				return m_Elapsed;
			}
		}


		/// <summary>
		/// Gets the total elapsed time measured by the current instance, in milliseconds.
		/// </summary>
		Int64 ElapsedMilliseconds() const noexcept
		{
			return ElapsedTicks() / TicksPerMillisecond;
		}


		/// <summary>
		/// Returns formatted the elapsed time in given time unit and precision.
		/// </summary>
		template <TimeUnit Unit = TimeUnit::Milliseconds, int Precision = 3>
		std::string ToString() const
		{
			std::ostringstream ss;
			ss << std::fixed << std::setprecision(Precision) << Elapsed<Unit>() << ' ' << ToCStringSymbol(Unit);
			return ss.str();
		}


		/// <summary>
		/// Returns formatted the elapsed time in given time unit, width, and precision.
		/// </summary>
		template <TimeUnit Unit, int Width, int Precision>
		std::string ToString() const
		{
			std::ostringstream ss;
			ss << std::fixed << std::setw(Width) << std::setprecision(Precision) << Elapsed<Unit>() << ' '
			   << ToCStringSymbol(Unit);
			return ss.str();
		}


	private:
		void ResetStartTimestamp() noexcept
		{
			static_cast<Derived*>(this)->m_StartTimestamp = static_cast<Derived*>(this)->GetTimestamp();
		}


	protected:
		constexpr StopwatchBase() noexcept = default;
		constexpr StopwatchBase(const StopwatchBase&) noexcept = default;
		constexpr StopwatchBase(StopwatchBase&&) noexcept = default;
		constexpr StopwatchBase& operator=(const StopwatchBase&) noexcept = default;
		constexpr StopwatchBase& operator=(StopwatchBase&&) noexcept = default;
		~StopwatchBase() noexcept = default;


	protected:
		Int64 m_Elapsed = 0;
		bool m_IsRunning = false;
	};


	/// <summary>
	/// Provides a set of methods that you can use to accurately measure CPU execution time.
	/// </summary>
	class CpuStopwatch : public StopwatchBase<CpuStopwatch>
	{
		friend StopwatchBase<CpuStopwatch>;

#if defined(_WIN32)
		using Timestamp = Int64;
#else
		using Timestamp = std::clock_t;
#endif

	private:
		static Timestamp GetTimestamp() noexcept
		{
#if defined(_WIN32)
			FILETIME creation{}, exit{}, kernel{}, user{};

			if (!GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user))
			{
				std::cerr << "::GetThreadTimes(...) ended with error" << std::endl;
				std::terminate();
			}

			ULARGE_INTEGER u{};
			u.LowPart  = user.dwLowDateTime;
			u.HighPart = user.dwHighDateTime;

			return static_cast<Timestamp>(u.QuadPart);
#else
			return std::clock();
#endif
		}

		Int64 GetRawElapsedFromStartUntilNowTicks() const noexcept
		{
#if defined(_WIN32)
			return GetTimestamp() - m_StartTimestamp;
#else
			return static_cast<Int64>(GetTimestamp() - m_StartTimestamp) * TicksPerSecond / CLOCKS_PER_SEC;
#endif
		}

	private:
		Timestamp m_StartTimestamp{};
	};


	/// <summary>
	/// Provides a set of methods that you can use to accurately measure wall time.
	/// </summary>
	class Stopwatch : public StopwatchBase<Stopwatch>
	{
		friend StopwatchBase<Stopwatch>;

	private:
		static std::chrono::time_point<std::chrono::high_resolution_clock> GetTimestamp() noexcept
		{
			return std::chrono::high_resolution_clock::now();
		}

		Int64 GetRawElapsedFromStartUntilNowTicks() const noexcept
		{
			const auto duration = GetTimestamp() - m_StartTimestamp;
			return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count() / NanosecondsPerTick;
		}

	private:
		std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimestamp;
	};
}  // namespace SecUtility::Diagnostic::Stopwatch
