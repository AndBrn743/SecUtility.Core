#include <SecUtility/IO/RandomAccessFile.hpp>

#include <iostream>
#include <string>


int main(const int argumentCount, char** argumentValuesPtr)
{
	if (argumentCount != 3) return 2;
	try
	{
		using SecUtility::IO::RandomAccessFile;
		const bool isShared = std::string(argumentValuesPtr[1]) == "shared";
		auto file = isShared ? RandomAccessFile::OpenForReadOnly(argumentValuesPtr[2])
		                     : RandomAccessFile::OpenForReadWrite(argumentValuesPtr[2]);
		auto lock = file.TryAcquireLock(isShared ? RandomAccessFile::LockMode::Shared
		                                         : RandomAccessFile::LockMode::Exclusive);
		if (!lock.has_value()) return 1;
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
