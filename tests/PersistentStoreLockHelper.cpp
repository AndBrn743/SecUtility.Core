#include <SecUtility/IO/PersistentStore.hpp>

#include <iostream>
#include <string>


int main(const int argumentCount, char** argumentValuesPtr)
{
	if (argumentCount != 3)
	{
		return 2;
	}

	try
	{
		auto store = std::string(argumentValuesPtr[1]) == "read"
		                     ? SecUtility::IO::PersistentStore::OpenForReadOnly(argumentValuesPtr[2])
		                     : SecUtility::IO::PersistentStore::OpenForReadWrite(argumentValuesPtr[2]);
		std::cout << "ready" << std::endl;
		std::string command;
		std::getline(std::cin, command);
		return command == "stop" ? 0 : 3;
	}
	catch (const std::exception& exception)
	{
		std::cerr << exception.what() << std::endl;
		return 1;
	}
}
