#include <SecUtility/IO/PersistentStore/Adapter/ByteSequence.hpp>
#include <SecUtility/IO/PersistentStore/Detail/FailureInjection.hpp>

#include <iostream>
#include <string>
#include <vector>


using namespace SecUtility;
using namespace SecUtility::IO;
using namespace SecUtility::IO::PersistentStoreDetail;


namespace
{
	struct Boundary
	{
		FailurePoint Point = FailurePoint::ReservePayload;
		UInt64 Detail = 0;
		bool MatchesDetail = false;
	};

	void PauseAtBoundary(const FailurePoint point, const UInt64 detail, void* const contextPtr)
	{
		const Boundary& boundary = *static_cast<const Boundary*>(contextPtr);
		if (point != boundary.Point || (boundary.MatchesDetail && detail != boundary.Detail)) return;
		std::cout << "ready" << std::endl;
		std::string command;
		std::getline(std::cin, command);
	}
}


int main(const int argumentCount, char** argumentValuesPtr)
{
	if (argumentCount != 5) return 2;
	try
	{
		Boundary boundary;
		boundary.Point = static_cast<FailurePoint>(std::stoi(argumentValuesPtr[2]));
		boundary.Detail = static_cast<UInt64>(std::stoull(argumentValuesPtr[3]));
		boundary.MatchesDetail = std::string(argumentValuesPtr[4]) == "1";
		auto store = PersistentStore::OpenForReadWrite(argumentValuesPtr[1]);
		ScopedFailureInjection injection(PauseAtBoundary, &boundary);
		store.Reassign("value", std::vector<Byte>{Byte{0x22}});
		return 3;
	}
	catch (const std::exception& exception)
	{
		std::cerr << exception.what() << std::endl;
		return 1;
	}
}
