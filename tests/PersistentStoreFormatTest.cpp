#include <SecUtility/IO/PersistentStore.hpp>
#include <SecUtility/IO/PersistentStore/Detail/CheckedArithmetic.hpp>
#include <SecUtility/IO/PersistentStore/Detail/Format.hpp>
#include <SecUtility/IO/PersistentStore/Detail/FormatCodec.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <initializer_list>
#include <string>
#include <type_traits>
#include <vector>


using namespace SecUtility;
using namespace SecUtility::IO;
using namespace SecUtility::IO::PersistentStoreDetail;


namespace
{
	std::vector<Byte> Bytes(const std::initializer_list<unsigned int> values)
	{
		std::vector<Byte> bytes;
		bytes.reserve(values.size());
		for (const unsigned int value : values)
		{
			bytes.push_back(Byte{static_cast<unsigned char>(value)});
		}
		return bytes;
	}

	Directory EmptyDirectory()
	{
		Directory directory;
		directory.Generation = 1;
		return directory;
	}

	HeaderSlot HeaderFor(const std::vector<Byte>& directoryBytes,
	                     const std::uint64_t directoryOffset = SuperblockBytes,
	                     const std::uint64_t committedBytes = SuperblockBytes + DirectoryHeaderBytes)
	{
		HeaderSlot header;
		header.Generation = 1;
		header.DirectoryOffset = directoryOffset;
		header.DirectoryBytes = directoryBytes.size();
		header.CommittedLogicalFileBytes = committedBytes;
		header.DirectoryChecksum = FormatCodecDetail::ComputeCrc32C(ConstByteView(directoryBytes));
		return header;
	}

	void SetLittleEndian32(std::vector<Byte>& ref_bytes,
	                       const std::size_t offset,
	                       const std::uint32_t value)
	{
		FormatCodecDetail::WriteLittleEndian<std::uint32_t>(
		        MutableByteView(ref_bytes), offset, value, "test field");
	}

	void SetLittleEndian64(std::vector<Byte>& ref_bytes,
	                       const std::size_t offset,
	                       const std::uint64_t value)
	{
		FormatCodecDetail::WriteLittleEndian<std::uint64_t>(
		        MutableByteView(ref_bytes), offset, value, "test field");
	}

	HeaderSlot HeaderForMutatedDirectory(const std::vector<Byte>& directoryBytes,
	                                     const std::uint64_t committedBytes)
	{
		return HeaderFor(directoryBytes, SuperblockBytes, committedBytes);
	}
}


TEST_CASE("PersistentStore public contract is move-only and has no public access mode")
{
	STATIC_CHECK(!std::is_copy_constructible_v<PersistentStore>);
	STATIC_CHECK(!std::is_copy_assignable_v<PersistentStore>);
	STATIC_CHECK(std::is_move_constructible_v<PersistentStore>);
	STATIC_CHECK(std::is_move_assignable_v<PersistentStore>);
	STATIC_CHECK(!PersistentStoreDetail::HasLeasedType<int>);
}


TEST_CASE("Byte provides explicit C++17 byte operations")
{
	STATIC_CHECK(std::is_same_v<Byte, std::byte>);
	constexpr Byte low{0x0F};
	constexpr Byte high{0xF0};
	STATIC_CHECK(std::to_integer<unsigned int>(low | high) == 0xFF);
	STATIC_CHECK(std::to_integer<unsigned int>(low & high) == 0);
	STATIC_CHECK(std::to_integer<unsigned int>(low ^ high) == 0xFF);
	STATIC_CHECK(std::to_integer<unsigned int>(Byte{1} << 4) == 0x10);
	STATIC_CHECK(std::to_integer<unsigned int>(Byte{0x80} >> 7) == 1);
}


TEST_CASE("PersistentStore checked arithmetic rejects overflow")
{
	CHECK(CheckedAdd(10, 20, "addition") == 30);
	CHECK_THROWS_AS(CheckedAdd(std::numeric_limits<std::uint64_t>::max(), 1, "addition"), FormatException);
	CHECK(CheckedMultiply(7, 9, "multiplication") == 63);
	CHECK_THROWS_AS(CheckedMultiply(std::numeric_limits<std::uint64_t>::max(), 2, "multiplication"),
	                FormatException);
	CHECK(CheckedAlignUp(4097, 4096, "alignment") == 8192);
	CHECK_THROWS_AS(CheckedAlignUp(10, 3, "alignment"), FormatException);
	CHECK_THROWS_AS(CheckedAlignUp(std::numeric_limits<std::uint64_t>::max(), 2, "alignment"),
	                FormatException);
	CHECK(CheckedNarrow<std::uint8_t>(255, "narrow") == 255);
	CHECK_THROWS_AS(CheckedNarrow<std::uint8_t>(256, "narrow"), FormatException);
	CHECK_THROWS_AS(CheckedNarrow<std::int8_t>(128, "narrow"), FormatException);
}


TEST_CASE("PersistentStore range helpers use checked half-open ranges")
{
	CHECK(IsRangeContained(10, 20, 10, 20));
	CHECK(IsRangeContained(10, 20, 30, 0));
	CHECK_FALSE(IsRangeContained(10, 20, 30, 1));
	CHECK_FALSE(IsRangeContained(std::numeric_limits<std::uint64_t>::max(), 1, 0, 0));
	CHECK(DoRangesOverlap(10, 10, 19, 10));
	CHECK_FALSE(DoRangesOverlap(10, 10, 20, 10));
	CHECK_FALSE(DoRangesOverlap(10, 0, 10, 1));
}


TEST_CASE("PersistentStore validates UTF-8 keys")
{
	CHECK_NOTHROW(ValidateKey(""));
	CHECK_NOTHROW(ValidateKey("plain/ascii"));
	CHECK_NOTHROW(ValidateKey(std::string("a\0b", 3)));
	CHECK_NOTHROW(ValidateKey(std::string(MaximumKeyBytes, 'a')));
	CHECK_NOTHROW(ValidateKey("\xC2\xA2\xE2\x82\xAC\xF0\x90\x8D\x88"));

	CHECK_THROWS_AS(ValidateKey("\x80"), InvalidKeyException);
	CHECK_THROWS_AS(ValidateKey("\xC0\x80"), InvalidKeyException);
	CHECK_THROWS_AS(ValidateKey("\xE2\x82"), InvalidKeyException);
	CHECK_THROWS_AS(ValidateKey("\xED\xA0\x80"), InvalidKeyException);
	CHECK_THROWS_AS(ValidateKey("\xF4\x90\x80\x80"), InvalidKeyException);
	CHECK_THROWS_AS(ValidateKey(std::string(MaximumKeyBytes + 1, 'a')), InvalidKeyException);
	CHECK_THROWS_AS(ValidateStoredKey("\x80"), FormatException);
}


TEST_CASE("PersistentStore header round-trips the maximum file boundary")
{
	HeaderSlot header;
	header.Generation = std::numeric_limits<std::uint64_t>::max();
	header.DirectoryOffset = MaximumFileBytes - DirectoryHeaderBytes;
	header.DirectoryBytes = DirectoryHeaderBytes;
	header.CommittedLogicalFileBytes = MaximumFileBytes;
	header.DirectoryChecksum = std::numeric_limits<std::uint32_t>::max();

	const auto bytes = SerializeHeader(header);
	const HeaderSlot parsed = ParseHeader(ConstByteView(bytes), MaximumFileBytes);
	CHECK(parsed.Generation == std::numeric_limits<std::uint64_t>::max());
	CHECK(parsed.DirectoryOffset == MaximumFileBytes - DirectoryHeaderBytes);
	CHECK(parsed.CommittedLogicalFileBytes == MaximumFileBytes);
}


TEST_CASE("PersistentStore empty directory has a stable golden encoding")
{
	const std::vector<Byte> expected = Bytes({
	        83, 69, 67, 68, 73, 82, 49, 0, 1, 0, 0, 0, 40, 0, 0, 0, 1, 0, 0, 0,
	        0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0});
	const std::vector<Byte> bytes = SerializeDirectory(EmptyDirectory());
	CHECK(bytes == expected);
	CHECK(FormatCodecDetail::ComputeCrc32C(ConstByteView(bytes)) == 0xACE4B462U);

	const HeaderSlot header = HeaderFor(bytes);
	const Directory parsed = ParseDirectory(header, ConstByteView(bytes), header.CommittedLogicalFileBytes);
	CHECK(parsed.Generation == 1);
	CHECK(parsed.Entries.empty());
}


TEST_CASE("PersistentStore header has a stable golden encoding")
{
	const std::vector<Byte> directoryBytes = SerializeDirectory(EmptyDirectory());
	const HeaderSlot header = HeaderFor(directoryBytes);
	const auto bytes = SerializeHeader(header);

	const std::vector<Byte> expectedPrefix = Bytes({
	        83, 69, 67, 80, 83, 82, 49, 0, 1, 0, 0, 0, 4, 3, 2, 1,
	        128, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
	        0, 16, 0, 0, 0, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0,
	        40, 16, 0, 0, 0, 0, 0, 0, 98, 180, 228, 172, 61, 165, 216, 0});
	CHECK(std::equal(expectedPrefix.begin(), expectedPrefix.end(), bytes.begin()));
	CHECK(std::all_of(bytes.begin() + 64, bytes.end(), [](const Byte byte) { return byte == Byte{0}; }));

	const HeaderSlot parsed = ParseHeader(ConstByteView(bytes), header.CommittedLogicalFileBytes);
	CHECK(parsed.Generation == header.Generation);
	CHECK(parsed.DirectoryOffset == header.DirectoryOffset);
	CHECK(parsed.DirectoryBytes == header.DirectoryBytes);
	CHECK(parsed.DirectoryChecksum == header.DirectoryChecksum);
}


TEST_CASE("PersistentStore one-entry directory has a stable golden encoding")
{
	Directory directory;
	directory.Generation = 1;
	directory.Entries = {{"a", {5000, 16}, 5004, 8}};
	const std::vector<Byte> expected = Bytes({
	        83, 69, 67, 68, 73, 82, 49, 0, 1, 0, 0, 0, 40, 0, 0, 0,
	        1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
	        81, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
	        136, 19, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0,
	        140, 19, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0,
	        97});
	const auto bytes = SerializeDirectory(directory);
	CHECK(bytes == expected);
	CHECK(FormatCodecDetail::ComputeCrc32C(ConstByteView(bytes)) == 0x0A35D5C1U);
}


TEST_CASE("PersistentStore directory round-trips embedded-NUL and unsigned-byte keys")
{
	Directory directory;
	directory.Generation = 9;
	DirectoryEntry first;
	first.Key = std::string("a\0b", 3);
	first.AllocatedExtent = {5000, 64};
	first.PayloadOffset = 5008;
	first.PayloadBytes = 8;
	DirectoryEntry second;
	second.Key = "z";
	second.AllocatedExtent = {5100, 80};
	second.PayloadOffset = 5100;
	second.PayloadBytes = 80;
	directory.Entries = {first, second};

	const auto bytes = SerializeDirectory(directory);
	HeaderSlot header = HeaderFor(bytes, 6000, 7000);
	header.Generation = directory.Generation;
	header.DirectoryOffset = 6000;
	const Directory parsed = ParseDirectory(header, ConstByteView(bytes), 7000);
	REQUIRE(parsed.Entries.size() == 2);
	CHECK(parsed.Entries[0].Key == first.Key);
	CHECK(parsed.Entries[1].PayloadBytes == 80);
}


TEST_CASE("PersistentStore header parser rejects malformed candidates")
{
	const auto directoryBytes = SerializeDirectory(EmptyDirectory());
	const HeaderSlot header = HeaderFor(directoryBytes);
	const auto valid = SerializeHeader(header);

	for (std::size_t length = 0; length < valid.size(); ++length)
	{
		CHECK_THROWS_AS(ParseHeader(ConstByteView(valid.data(), length), header.CommittedLogicalFileBytes),
		                FormatException);
	}

	auto bad = valid;
	bad[0] ^= Byte{1};
	CHECK_THROWS_AS(ParseHeader(ConstByteView(bad), header.CommittedLogicalFileBytes), FormatException);
	bad = valid;
	bad[64] = Byte{1};
	CHECK_THROWS_AS(ParseHeader(ConstByteView(bad), header.CommittedLogicalFileBytes), FormatException);
	bad = valid;
	bad[24] = Byte{0};
	CHECK_THROWS_AS(ParseHeader(ConstByteView(bad), header.CommittedLogicalFileBytes), FormatException);
	bad = valid;
	bad[60] ^= Byte{1};
	CHECK_THROWS_AS(ParseHeader(ConstByteView(bad), header.CommittedLogicalFileBytes), FormatException);
}


TEST_CASE("PersistentStore directory parser rejects unchecked counts before allocation")
{
	auto bytes = SerializeDirectory(EmptyDirectory());
	SetLittleEndian64(bytes, 24, std::numeric_limits<std::uint64_t>::max());
	const HeaderSlot header = HeaderForMutatedDirectory(bytes, SuperblockBytes + bytes.size());
	CHECK_THROWS_AS(ParseDirectory(header, ConstByteView(bytes), header.CommittedLogicalFileBytes), FormatException);

	SetLittleEndian64(bytes, 24, MaximumDirectoryEntries + 1);
	const HeaderSlot excessiveHeader = HeaderForMutatedDirectory(bytes, SuperblockBytes + bytes.size());
	CHECK_THROWS_AS(ParseDirectory(excessiveHeader, ConstByteView(bytes),
	                              excessiveHeader.CommittedLogicalFileBytes),
	                FormatException);
}


TEST_CASE("PersistentStore directory parser rejects malformed records and overlaps")
{
	Directory directory;
	directory.Generation = 1;
	DirectoryEntry first{"a", {5000, 100}, 5000, 100};
	DirectoryEntry second{"b", {5200, 100}, 5200, 100};
	directory.Entries = {first, second};
	auto bytes = SerializeDirectory(directory);
	HeaderSlot header = HeaderFor(bytes, 6000, 7000);
	CHECK_NOTHROW(ParseDirectory(header, ConstByteView(bytes), 7000));

	auto malformed = bytes;
	SetLittleEndian32(malformed, DirectoryHeaderBytes + 4, 1);
	HeaderSlot malformedHeader = HeaderFor(malformed, 6000, 7000);
	CHECK_THROWS_AS(ParseDirectory(malformedHeader, ConstByteView(malformed), 7000), FormatException);

	malformed = bytes;
	SetLittleEndian64(malformed, DirectoryHeaderBytes + DirectoryEntryFixedBytes + 1 + 8, 5050);
	malformedHeader = HeaderFor(malformed, 6000, 7000);
	CHECK_THROWS_AS(ParseDirectory(malformedHeader, ConstByteView(malformed), 7000), FormatException);

	malformed = bytes;
	malformed[DirectoryHeaderBytes + DirectoryEntryFixedBytes + 1 + DirectoryEntryFixedBytes] = Byte{'a'};
	malformedHeader = HeaderFor(malformed, 6000, 7000);
	CHECK_THROWS_AS(ParseDirectory(malformedHeader, ConstByteView(malformed), 7000), FormatException);

	malformed = bytes;
	SetLittleEndian64(malformed, DirectoryHeaderBytes + 24, 5099);
	SetLittleEndian64(malformed, DirectoryHeaderBytes + 32, 2);
	malformedHeader = HeaderFor(malformed, 6000, 7000);
	CHECK_THROWS_AS(ParseDirectory(malformedHeader, ConstByteView(malformed), 7000), FormatException);

	malformed = bytes;
	SetLittleEndian64(malformed, DirectoryHeaderBytes + 8, 6000);
	malformedHeader = HeaderFor(malformed, 6000, 7000);
	CHECK_THROWS_AS(ParseDirectory(malformedHeader, ConstByteView(malformed), 7000), FormatException);

	malformed = bytes;
	SetLittleEndian64(malformed, DirectoryHeaderBytes + 8, std::numeric_limits<std::uint64_t>::max());
	SetLittleEndian64(malformed, DirectoryHeaderBytes + 16, 2);
	malformedHeader = HeaderFor(malformed, 6000, 7000);
	CHECK_THROWS_AS(ParseDirectory(malformedHeader, ConstByteView(malformed), 7000), FormatException);

	malformed = bytes;
	malformed.push_back(Byte{0});
	SetLittleEndian64(malformed, 32, malformed.size());
	malformedHeader = HeaderFor(malformed, 6000, 7000);
	CHECK_THROWS_AS(ParseDirectory(malformedHeader, ConstByteView(malformed), 7000), FormatException);
}


TEST_CASE("PersistentStore directory parser rejects every truncated record boundary")
{
	Directory directory;
	directory.Generation = 1;
	directory.Entries = {{"abc", {5000, 16}, 5004, 8}};
	const auto valid = SerializeDirectory(directory);

	for (std::size_t length = 0; length < valid.size(); ++length)
	{
		const std::vector<Byte> truncated(valid.begin(), valid.begin() + static_cast<std::ptrdiff_t>(length));
		HeaderSlot header = HeaderFor(truncated, 6000, 7000);
		CHECK_THROWS_AS(ParseDirectory(header, ConstByteView(truncated), 7000), FormatException);
	}
}


TEST_CASE("PersistentStore root candidate retains a malformed candidate reason")
{
	const auto directoryBytes = SerializeDirectory(EmptyDirectory());
	const HeaderSlot header = HeaderFor(directoryBytes);
	const auto headerBytes = SerializeHeader(header);
	const RootCandidate valid = ParseRootCandidate(
	        HeaderSlotAOffset, ConstByteView(headerBytes), ConstByteView(directoryBytes),
	        header.CommittedLogicalFileBytes);
	CHECK(valid.IsValid);
	CHECK(valid.RejectionReason.empty());

	auto malformedHeaderBytes = headerBytes;
	malformedHeaderBytes[0] ^= Byte{1};
	const RootCandidate malformed = ParseRootCandidate(
	        HeaderSlotBOffset, ConstByteView(malformedHeaderBytes), ConstByteView(directoryBytes),
	        header.CommittedLogicalFileBytes);
	CHECK_FALSE(malformed.IsValid);
	CHECK_FALSE(malformed.RejectionReason.empty());
}


TEST_CASE("PersistentStore serializers reject invalid directory state")
{
	Directory directory;
	directory.Generation = 1;
	directory.Entries = {{"b", {5000, 1}, 5000, 1}, {"a", {5001, 1}, 5001, 1}};
	CHECK_THROWS_AS(SerializeDirectory(directory), FormatException);

	directory.Entries = {{"a", {5000, 1}, 5000, 1}, {"a", {5001, 1}, 5001, 1}};
	CHECK_THROWS_AS(SerializeDirectory(directory), FormatException);

	directory.Entries = {{"a", {5000, 0}, 5000, 0}};
	CHECK_THROWS_AS(SerializeDirectory(directory), FormatException);

	HeaderSlot header;
	CHECK_THROWS_AS(SerializeHeader(header), FormatException);
}
