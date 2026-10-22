//
// Created by Andy on 6/12/2026.
//

#include <SecUtility/Misc/Bitflag.hpp>
#include <catch2/catch_test_macros.hpp>

enum class FileMode : unsigned int
{
	Read = 1 << 0,
	Write = 1 << 1,
	Exec = 1 << 2
};

template <>
struct SecUtility::is_bitmask<FileMode> : std::true_type
{
	/* NO CODE */
};

using namespace SecUtility;

// An enum that does NOT opt in — used to verify operators don't interfere
enum class NotABitmask : unsigned int
{
	A = 1,
	B = 2
};


TEST_CASE("Bitflag operator|")
{
	using U = std::underlying_type_t<FileMode>;

	SECTION("combines two flags")
	{
		const auto result = FileMode::Read | FileMode::Write;
		REQUIRE(static_cast<U>(result) == 0b011U);
	}

	SECTION("combines three flags")
	{
		const auto result = FileMode::Read | FileMode::Write | FileMode::Exec;
		REQUIRE(static_cast<U>(result) == 0b111U);
	}

	SECTION("OR with same flag is idempotent")
	{
		const auto result = FileMode::Read | FileMode::Read;
		REQUIRE(result == FileMode::Read);
	}
}


TEST_CASE("Bitflag operator&")
{
	SECTION("masks out unset bits")
	{
		const auto flags = FileMode::Read | FileMode::Write | FileMode::Exec;
		const auto result = flags & FileMode::Write;
		REQUIRE(result == FileMode::Write);
	}

	SECTION("returns zero when flag not set")
	{
		const auto flags = FileMode::Read | FileMode::Write;
		const auto result = flags & FileMode::Exec;
		REQUIRE(static_cast<std::underlying_type_t<FileMode>>(result) == 0U);
	}
}


TEST_CASE("Bitflag operator^")
{
	SECTION("toggles a flag on")
	{
		auto flags = FileMode::Read;
		const auto result = flags ^ FileMode::Write;
		REQUIRE((result & FileMode::Write) == FileMode::Write);
		REQUIRE((result & FileMode::Read) == FileMode::Read);
	}

	SECTION("toggles a flag off")
	{
		const auto flags = FileMode::Read | FileMode::Write;
		const auto result = flags ^ FileMode::Write;
		REQUIRE(result == FileMode::Read);
	}
}


TEST_CASE("Bitflag operator~")
{
	SECTION("inverts all bits")
	{
		const auto result = ~FileMode::Read;
		// Underlying is unsigned int; all bits set except bit 0
		using U = std::underlying_type_t<FileMode>;
		REQUIRE(static_cast<U>(result) == ~static_cast<U>(FileMode::Read));
	}

	SECTION("double invert is identity")
	{
		const auto flags = FileMode::Read | FileMode::Write;
		REQUIRE(~~flags == flags);
	}
}


TEST_CASE("Bitflag operator|=")
{
	SECTION("sets a flag")
	{
		auto flags = FileMode::Read;
		flags |= FileMode::Write;
		REQUIRE((flags & FileMode::Read) == FileMode::Read);
		REQUIRE((flags & FileMode::Write) == FileMode::Write);
	}

	SECTION("returns reference to lhs")
	{
		auto flags = FileMode::Read;
		auto& ref = flags |= FileMode::Write;
		REQUIRE(&ref == &flags);
	}
}


TEST_CASE("Bitflag operator&=")
{
	SECTION("clears unset bits")
	{
		auto flags = FileMode::Read | FileMode::Write | FileMode::Exec;
		flags &= FileMode::Write;
		REQUIRE(flags == FileMode::Write);
	}

	SECTION("returns reference to lhs")
	{
		auto flags = FileMode::Read;
		auto& ref = flags &= FileMode::Read;
		REQUIRE(&ref == &flags);
	}
}


TEST_CASE("Bitflag operator^=")
{
	SECTION("toggles a flag")
	{
		auto flags = FileMode::Read | FileMode::Write;
		flags ^= FileMode::Write;
		REQUIRE(flags == FileMode::Read);
	}

	SECTION("returns reference to lhs")
	{
		auto flags = FileMode::Read;
		auto& ref = flags ^= FileMode::Write;
		REQUIRE(&ref == &flags);
	}
}


TEST_CASE("Bitflag constexpr")
{
	// Verify operators work in constant expressions
	constexpr auto combined = FileMode::Read | FileMode::Write;
	static_assert(static_cast<unsigned>(combined) == 0b011U);

	constexpr auto masked = combined & FileMode::Read;
	static_assert(masked == FileMode::Read);

	constexpr auto toggled = combined ^ FileMode::Exec;
	static_assert(static_cast<unsigned>(toggled) == 0b111U);

	constexpr auto inverted = ~FileMode::Read;
	static_assert(static_cast<unsigned>(inverted) == ~1u);

	(void)combined;
	(void)masked;
	(void)toggled;
	(void)inverted;
}


TEST_CASE("Bitflag is_bitmask_v trait")
{
	STATIC_REQUIRE(SecUtility::is_bitmask_v<FileMode>);
	STATIC_REQUIRE_FALSE(SecUtility::is_bitmask_v<NotABitmask>);
}
