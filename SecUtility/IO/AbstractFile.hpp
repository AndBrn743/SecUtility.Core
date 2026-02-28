// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/Diagnostic/Exception/Exception.hpp>
#include <SecUtility/Misc/Random.hpp>
#include <filesystem>
#include <ios>
#include <fstream>
#include <regex>
#include <type_traits>
#include <utility>


namespace SecUtility::IO
{
	/// <summary>
	/// CRTP base class providing common file path management and filesystem operations.
	/// Derived classes must implement:
	///   - <c>const std::string&amp; Name() const</c>       -- return the file path
	///   - <c>void SetFileNameFieldTo(std::string name)</c> -- update the stored path (called by RenameTo)
	///   - <c>void OnFileCreated() const</c>                -- hook called after TryCreateIfNotExist creates the file
	///   - <c>static Derived Void()</c>                     -- return a default-constructed, path-less instance
	///
	/// Derived classes should also declare <c>friend Base;</c> so that AbstractFile
	/// can reach the private CRTP hooks above.
	/// </summary>
	template <typename Derived>
	class AbstractFile
	{
		friend Derived;

	public:
		// -------------------------------------------------------------------------
		// Path accessors
		// -------------------------------------------------------------------------

		/// <summary>Returns the stored file path.</summary>
		/*CRTP VIRTUAL*/ const std::string& Name() const
		{
			return AsDerived().Name();
		}

		/// <summary>Returns the file path as a null-terminated C string. Lifetime tied to Name().</summary>
		const char* CName() const
		{
			return Name().c_str();
		}

		/// <summary>
		/// Explicit conversion to <c>const std::string&amp;</c>.
		/// Explicit to avoid silent ambiguity in overload resolution.
		/// </summary>
		explicit operator const std::string&() const
		{
			return Name();
		}

		/// <summary>
		/// Explicit conversion to <c>const char*</c>.
		/// Explicit to prevent accidental dangling-pointer bugs when storing the result.
		/// </summary>
		explicit operator const char*() const
		{
			return CName();
		}

		// -------------------------------------------------------------------------
		// State queries
		// -------------------------------------------------------------------------

		/// <summary>Returns true if this object holds no path (i.e. was default-constructed via Void()).</summary>
		bool IsVoid() const
		{
			return Name().empty();
		}

		/// <summary>Returns true if a filesystem entry exists at this path.</summary>
		bool Exist() const
		{
			return std::filesystem::exists(CName());
		}

		/// <summary>
		/// Attempts to open the file for reading to determine whether the current process
		/// has read access. Returns false if the file does not exist or cannot be opened.
		/// <remarks>
		/// This performs a live probe (open + close) rather than inspecting permission bits,
		/// so it correctly accounts for effective UID/GID, sudo, POSIX ACLs, and platform
		/// differences. It does NOT guarantee the file will still be readable a moment later
		/// (TOCTOU), so treat it as a best-effort diagnostic rather than a security gate.
		/// </remarks>
		/// </summary>
		bool IsReadable() const
		{
			if (IsVoid())
			{
				return false;
			}
			return std::ifstream(Name(), std::ios_base::binary).is_open();
		}

		/// <summary>
		/// Attempts to open the file for writing (append mode) to determine whether the
		/// current process has write access. Returns false if the file does not exist or
		/// cannot be opened.
		/// <remarks>
		/// Same live-probe semantics and TOCTOU caveat as IsReadable().
		/// Append mode is used so that the existing file contents are never modified
		/// by the check itself.
		/// </remarks>
		/// </summary>
		bool IsWritable() const
		{
			if (IsVoid())
			{
				return false;
			}
			return std::ofstream(Name(), std::ios::app).is_open();
		}

		/// <summary>
		/// Returns true if both IsReadable() and IsWritable() succeed.
		/// <remarks>
		/// Performs two separate probes. See IsReadable() and IsWritable() for
		/// the TOCTOU caveat that applies to each.
		/// </remarks>
		/// </summary>
		bool IsReadableAndWritable() const
		{
			return IsReadable() && IsWritable();
		}

		/// <summary>
		/// Returns the file size in bytes, or 0 if the file does not exist or cannot be read.
		/// </summary>
		std::size_t Size() const
		{
			if (IsVoid())
			{
				return 0;
			}
			std::ifstream f(Name(), std::ios_base::binary | std::ios_base::in);
			if (!f.is_open())
			{
				return 0;
			}
			f.seekg(0, std::ios_base::beg);
			const std::ifstream::pos_type begin_pos = f.tellg();
			f.seekg(0, std::ios_base::end);
			return static_cast<std::size_t>(f.tellg() - begin_pos);
		}

		// -------------------------------------------------------------------------
		// Lifecycle operations
		// -------------------------------------------------------------------------

		/// <summary>
		/// Creates the file if it does not already exist.
		/// Returns false if the path is void, or if the file could not be created.
		/// Returns true if the file already existed or was successfully created.
		/// <remarks>
		/// TODO (TOCTOU): There is a time-of-check/time-of-use race between the Exist()
		/// test and the subsequent open(). In a security-sensitive or multi-process context
		/// this could be exploited or cause spurious failures. A robust fix requires
		/// platform-specific support:
		///   - POSIX: open(O_CREAT | O_EXCL) to atomically create-or-fail
		///   - Win32: CreateFile(CREATE_NEW) for equivalent semantics
		/// std::fstream has no portable equivalent of O_EXCL today (C++17/20).
		/// </remarks>
		/// </summary>
		bool TryCreateIfNotExist() const
		{
			if (IsVoid())
			{
				return false;
			}

			if (Exist())
			{
				return true;
			}

			std::fstream file;
			file.open(Name(), std::ios::out);

			if (file.fail())
			{
				return false;
			}

			file.close();
			AsDerived().OnFileCreated();
			return true;
		}

		/// <summary>
		/// Creates the file if it does not already exist, throwing IOException on failure.
		/// </summary>
		const Derived& CreateIfNotExist() const
		{
			if (!TryCreateIfNotExist())
			{
				throw IOException("Target file cannot be created though not exist");
			}
			return AsDerived();
		}

		/// <summary>Non-const overload of CreateIfNotExist().</summary>
		Derived& CreateIfNotExist()
		{
			if (!TryCreateIfNotExist())
			{
				throw IOException("Target file cannot be created though not exist");
			}
			return AsDerived();
		}

		/// <summary>
		/// Attempts to delete the file. Returns true if the file was deleted or did not exist.
		/// Returns false on error.
		/// </summary>
		bool TryDelete() const noexcept
		{
			if (IsVoid())
			{
				return true;
			}

			return std::remove(CName()) == 0;
		}

		/// <summary>Deletes the file, throwing IOException on failure.</summary>
		void Delete() const
		{
			if (!TryDelete())
			{
				throw IOException(std::string("Failed to delete file `") + Name() + '`');
			}
		}

		/// <summary>
		/// Attempts to rename the file. On success the stored path is updated to newName.
		/// Returns false if the path is void or the rename failed.
		/// </summary>
		bool TryRenameTo(const std::string& newName)
		{
			if (IsVoid())
			{
				return false;
			}

			if (std::rename(CName(), newName.c_str()) == 0)
			{
				AsDerived().SetFileNameFieldTo(newName);
				return true;
			}

			return false;
		}

		/// <summary>Renames the file, throwing IOException on failure.</summary>
		void RenameTo(const std::string& newName)
		{
			if (!TryRenameTo(newName))
			{
				throw IOException(std::string("Failed to rename file from `") + Name() + "` to `" + newName + '`');
			}
		}

		// -------------------------------------------------------------------------
		// Factory / static utilities
		// -------------------------------------------------------------------------

		/// <summary>
		/// Return a void (path-less) instance.
		/// Useful for default member initialization and delayed-init patterns.
		/// </summary>
		/*CRTP VIRTUAL*/ static Derived Void()
		{
			return Derived::Void();
		}

		/// <summary>
		/// Searches each prefix directory in order for a file whose name matches <paramref name="regex"/>.
		/// If no match is found, calls <paramref name="action"/>.
		/// <paramref name="action"/> may either return a <c>Derived</c> instance or be [[noreturn]] (i.e. throw).
		/// </summary>
		template <typename Action>
		static Derived LocateFromOrCustomAction(const List<std::string>& prefixes,
												const std::regex& regex,
												Action&& action)
		{
			for (const auto& prefix : prefixes)
			{
				namespace fs = std::filesystem;
				try
				{
					// ReSharper disable once CppTooWideScopeInitStatement
					auto entries = fs::directory_iterator(prefix);  // scope init for loop require C++20 and up
					for (const auto& entry : entries)
					{
						if (fs::is_regular_file(entry) && std::regex_match(entry.path().filename().string(), regex))
						{
							return Derived(entry.path().string());
						}
					}
				}
				catch (...)
				{
					// This prefix is inaccessible (e.g. permission denied, not a directory).
					// Continue searching remaining prefixes rather than aborting early.
				}
			}

			if constexpr (std::is_same_v<std::decay_t<decltype(action())>, Derived>)
			{
				return action();
			}
			else
			{
				action();  // expected to throw

				// If action() returned without throwing, the caller violated the contract.
				throw UnreachableException("Within AbstractFile::LocateFromOrCustomAction(...), the given custom "
										   "action neither returned Derived nor was [[noreturn]]");
			}
		}

		/// <summary>
		/// Searches each prefix directory in order for a file named <paramref name="name"/> (exact match).
		/// If no match is found, calls <paramref name="action"/>.
		/// <paramref name="action"/> may either return a <c>Derived</c> instance or be [[noreturn]] (i.e. throw).
		/// </summary>
		template <typename Action>
		static Derived LocateFromOrCustomAction(const List<std::string>& prefixes,
												const std::string& name,
												Action&& action)
		{
			for (const auto& prefix : prefixes)
			{
				// A trailing '/' is added defensively; redundant slashes are harmless.
				// NOLINTNEXTLINE(*-inefficient-string-concatenation)
				if (Derived file(prefix + '/' + name); file.Exist())
				{
					return file;
				}
			}

			if constexpr (std::is_same_v<std::decay_t<decltype(action())>, Derived>)
			{
				return action();
			}
			else
			{
				action();  // expected to throw

				throw UnreachableException("Within AbstractFile::LocateFromOrCustomAction(...), the given custom "
										   "action neither returned Derived nor was [[noreturn]]");
			}
		}

		/// <summary>Returns the first match across prefixes, or <paramref name="defaultFile"/> if none found.</summary>
		static Derived LocateFromOrDefault(const List<std::string>& prefixes,
										   const std::regex& regex,
										   const Derived& defaultFile)
		{
			return LocateFromOrCustomAction(prefixes, regex, [&defaultFile] { return defaultFile; });
		}

		/// <summary>Returns the first exact-name match across prefixes, or <paramref name="defaultFile"/> if none
		/// found.</summary>
		static Derived LocateFromOrDefault(const List<std::string>& prefixes,
										   const std::string& name,
										   const Derived& defaultFile)
		{
			return LocateFromOrCustomAction(prefixes, name, [&defaultFile] { return defaultFile; });
		}

		/// <summary>Returns the first match across prefixes, or throws IOException if none found.</summary>
		static Derived LocateFrom(const List<std::string>& prefixes, const std::regex& regex)
		{
			return LocateFromOrCustomAction(prefixes,
											regex,
											[&prefixes]
											{
												throw IOException(
														"Cannot locate file matching given pattern under prefixes: "
														+ PathPrefixesToString(prefixes));
											});
		}

		/// <summary>Returns the first exact-name match across prefixes, or throws IOException if none found.</summary>
		static Derived LocateFrom(const List<std::string>& prefixes, const std::string& name)
		{
			return LocateFromOrCustomAction(prefixes,
											name,
											[&]
											{
												throw IOException("Cannot locate file named `" + name
																  + "` under prefixes: "
																  + PathPrefixesToString(prefixes));
											});
		}

		/// <summary>
		/// Generates a path for a file that does not currently exist, using random characters.
		/// The total filename length will be exactly <paramref name="fileNameLength"/> characters.
		/// Throws if the random portion would be empty, or if a unique name cannot be found
		/// within <paramref name="retryCount"/> attempts.
		/// </summary>
		static Derived Random(const int fileNameLength,
							  const std::string& prefix = "",
							  const std::string& suffix = "",
							  const std::string& chars = "abcdefghijklmnopqrstuvwxyz"
														 "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
														 "1234567890",
							  int retryCount = 10)
		{
			const int randomLength = fileNameLength - static_cast<int>(prefix.length() + suffix.length());
			if (randomLength < 1)
			{
				throw InvalidOperationException(
						"Cannot create random file name: prefix+suffix already meet or exceed the requested length");
			}

			for (/* NO CODE */; retryCount > 0; retryCount--)
			{
				// NOLINTNEXTLINE(*-inefficient-string-concatenation)
				if (Derived file(prefix + Random::NextString(randomLength, chars) + suffix); !file.Exist())
				{
					return file;
				}
			}

			throw IOException("Cannot generate a unique random file name within the given number of retries");
		}

		/// <summary>
		/// Overload that picks a sensible total length automatically (prefix + suffix + 16 random chars).
		/// </summary>
		static Derived Random(const std::string& prefix = "",
							  const std::string& suffix = "",
							  const std::string& chars = "abcdefghijklmnopqrstuvwxyz"
														 "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
														 "1234567890",
							  const int retryCount = 10)
		{
			return Random(static_cast<int>(prefix.length() + suffix.length() + 16), prefix, suffix, chars, retryCount);
		}


	protected:
		constexpr AbstractFile() noexcept = default;
		constexpr AbstractFile(const AbstractFile&) noexcept = default;
		constexpr AbstractFile(AbstractFile&&) noexcept = default;
		constexpr AbstractFile& operator=(const AbstractFile&) noexcept = default;
		constexpr AbstractFile& operator=(AbstractFile&&) noexcept = default;
		~AbstractFile() = default;

		/// <summary>
		/// Formats a list of path prefixes as a human-readable set literal, e.g. { "a", "b" }.
		/// Used in diagnostic exception messages.
		/// </summary>
		static std::string PathPrefixesToString(const std::vector<std::string>& prefixes)
		{
			std::ostringstream oss;
			oss << "{ ";
			for (std::size_t i = 0; i < prefixes.size(); ++i)
			{
				if (i > 0)
				{
					oss << ", ";
				}
				oss << '"' << prefixes[i] << '"';
			}
			oss << " }";
			return oss.str();
		}


	private:
		constexpr Derived& AsDerived() noexcept
		{
			return *static_cast<Derived*>(this);
		}

		constexpr const Derived& AsDerived() const noexcept
		{
			return *static_cast<const Derived*>(this);
		}
	};
}
