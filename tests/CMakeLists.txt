if (POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif ()

option(USE_SYSTEM_CATCH2 NO)

if (USE_SYSTEM_CATCH2)
    find_package(Catch2 3.7.1 CONFIG)
endif ()

if (NOT USE_SYSTEM_CATCH2 OR NOT Catch2_FOUND)
    if (NOT USE_SYSTEM_CATCH2)
        message(STATUS "Building Catch2 Locally")
    else ()
        message(STATUS "Could NOT Find Catch2 (Building Locally)")
    endif ()
    include(FetchContent)
    if (DEFINED CATCH2_SOURCE)
        message(STATUS "Building Catch2 with user specified source")
        FetchContent_Declare(
                catch2
                URL ${CATCH2_SOURCE})
    else ()
        message(STATUS "Building Catch2 with source from internet")
        FetchContent_Declare(
                catch2
                GIT_REPOSITORY https://github.com/catchorg/Catch2.git
                GIT_PROGRESS TRUE
                GIT_TAG v3.7.1)
    endif ()
    FetchContent_MakeAvailable(catch2)
else ()
    message(STATUS "Found Catch2 version ${Catch2_VERSION} from ${Catch2_DIR}")
endif ()


if (NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror -Wno-uninitialized -Wno-unknown-pragmas -Wno-unused-but-set-variable -Wno-unused-variable")

    if (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
    endif ()
endif ()

include(CTest)
include(Catch)

# Helper function to create test executable and register with CTest
function(add_secutility_test test_name source_file)
    add_executable(${test_name} ${source_file})
    target_link_libraries(${test_name} PRIVATE Catch2::Catch2WithMain SecUtility::Core)
    catch_discover_tests(${test_name})
endfunction()

# Utility tests
add_secutility_test(CachedFunctionTest CachedFunctionTest.cpp)
add_secutility_test(RandomTest RandomTest.cpp)
add_secutility_test(ChecksumTest ChecksumTest.cpp)
add_secutility_test(MacroTest MacroTest.cpp)
add_secutility_test(StopwatchTest StopwatchTest.cpp)
add_secutility_test(StaticForeachTest StaticForeachTest.cpp)
