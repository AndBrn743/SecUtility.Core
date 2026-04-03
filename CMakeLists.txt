cmake_minimum_required(VERSION 3.15)
project(SecUtility.Core LANGUAGES CXX)

if (NOT DEFINED CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 17)
    set(CMAKE_CXX_STANDARD 17)
endif ()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(SecUtility.Core INTERFACE)
add_library(SecUtility::Core ALIAS SecUtility.Core)
include(GNUInstallDirs)
target_include_directories(SecUtility.Core INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

#-----------------------------------------------------------------------------------

if (POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif ()

option(EXPORT_COMPILE_COMMANDS "Export compile_commands.json for clang-based tools" TRUE)
if (EXPORT_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif ()

if (TARGET Eigen3::Eigen)
    message(STATUS "SecUtility will use the Eigen3 target defined by ancestor project")
else ()
    if (NOT EIGEN3_SOURCE AND NOT EIGEN3_REPO AND NOT Eigen3_FOUND)
        find_package(Eigen3 5 NO_MODULE)
        if (Eigen3_FOUND)
            message(STATUS "Found Eigen3 in a default directory from `${EIGEN3_INCLUDE_DIR}`")
            list(APPEND CMAKE_INCLUDE_PATH "${EIGEN3_INCLUDE_DIR}")
        else ()
            if (NOT EIGEN3_SOURCE AND NOT EIGEN3_REPO)
                set(EIGEN3_SOURCE https://gitlab.com/libeigen/eigen/-/archive/5.0.0/eigen-5.0.0.tar.bz2)
            endif ()
        endif ()
    endif ()

    if (NOT Eigen3_FOUND)
        # Newer versions of Eigen will only generate EigenConfig.cmake when it was top level project on default,
        # see commit 9700fc84. Without EigenConfig.cmake, the following error will occur
        #     export called with target "SecUtility" which requires target "eigen" that is not in any export set.
        # The following line should fix it
        set(EIGEN_BUILD_CMAKE_PACKAGE TRUE CACHE BOOL "Enables the creation of EigenConfig.cmake and related files" FORCE)
        set(BUILD_TESTING FALSE CACHE BOOL "Enable creation of Eigen tests" FORCE)

        message(STATUS "Eigen3 (version 5.0) not found and will be fetched from `${EIGEN3_SOURCE}`")

        include(FetchContent)
        FetchContent_Declare(
                eigen
                URL ${EIGEN3_SOURCE}
                GIT_REPOSITORY ${EIGEN3_REPO}
                GIT_TAG ${EIGEN3_TAG}
                GIT_PROGRESS
                PREFIX ${CMAKE_CURRENT_BINARY_DIR}/eigen)
        FetchContent_MakeAvailable(eigen)
    endif ()
endif ()
target_link_libraries(SecUtility.Core INTERFACE Eigen3::Eigen)

if (WIN32)
    find_program(PATCH_EXECUTABLE
            NAMES patch
            PATHS
            "C:/Program Files/Git/usr/bin"
            "C:/Program Files (x86)/Git/usr/bin"
            "C:/msys64/usr/bin"
            "D:/Program Files/Git/usr/bin"
            "D:/Program Files (x86)/Git/usr/bin"
            "D:/msys64/usr/bin"
            "E:/Program Files/Git/usr/bin"
            "E:/Program Files (x86)/Git/usr/bin"
            "E:/msys64/usr/bin"
            NO_DEFAULT_PATH)
else ()
    find_program(PATCH_EXECUTABLE NAMES patch)
endif ()

if (TARGET range-v3::range-v3)
    message(WARNING "SecUtility.Core will use the range-v3 target defined by the ancestor project. "
            "For best user experience, SecUtility.Core recommends applying a patch to range-v3. "
            "SecUtility.Core will automatically apply the recommended patch for standalone builds and builds without "
            "an ancestor-defined range-v3 target, which is not the current case. SecUtility.Core recommends that you "
            "manually apply the patch at patches/range-v3/include/range/v3/view/zip_with.hpp.patch")
else ()
    # if (NOT RANGE_V3_SOURCE AND NOT RANGE_V3_REPO AND NOT RANGE_V3_TAG)
    #     find_package(range-v3)
    # endif ()

    if (NOT range-v3_FOUND)
        if (NOT RANGE_V3_SOURCE AND NOT RANGE_V3_REPO AND NOT RANGE_V3_TAG)
            set(RANGE_V3_SOURCE https://github.com/ericniebler/range-v3/archive/refs/tags/0.12.0.tar.gz)
            # set(RANGE_V3_REPO https://github.com/ericniebler/range-v3.git)
            # set(RANGE_V3_TAG 0.12.0)
            message(STATUS "Building range-v3 from ${RANGE_V3_SOURCE}")
        endif ()

        include(FetchContent)
        FetchContent_Declare(
                range-v3
                URL ${RANGE_V3_SOURCE}
                GIT_REPOSITORY ${RANGE_V3_REPO}
                GIT_TAG ${RANGE_V3_TAG}
                UPDATE_DISCONNECTED ON
                PATCH_COMMAND ${CMAKE_COMMAND} -E echo "Patching range-v3..."
                && cd ${CMAKE_BINARY_DIR}/_deps/range-v3-src/include/range/v3/view
                && ${PATCH_EXECUTABLE} --forward --batch --silent < ${CMAKE_CURRENT_SOURCE_DIR}/patches/range-v3/include/range/v3/view/zip_with.hpp.patch
                || ${CMAKE_COMMAND} -E echo "range-v3 patch skipped"
        )
        FetchContent_MakeAvailable(range-v3)
    endif ()
endif ()
target_link_libraries(SecUtility.Core INTERFACE range-v3::range-v3)

if (TARGET gcem::gcem)
    message(STATUS "SecUtility will use the gcem target defined by ancestor project")
else ()
    find_package(gcem NO_MODULE)

    if (NOT gcem_FOUND)
        if (NOT DEFINED GCEM_REPO AND NOT DEFINED GCEM_TAG AND NOT DEFINED GCEM_SOURCE)
            set(GCEM_SOURCE "https://github.com/kthohr/gcem/archive/refs/tags/v1.18.0.tar.gz")
            message(STATUS "Gcem library will be fetched from `${GCEM_SOURCE}`")
        endif ()

        set(GcemSourceDir ${CMAKE_BINARY_DIR}/external/gcem-src)

        include(FetchContent)
        FetchContent_Declare(gcem
                URL ${GCEM_SOURCE}
                GIT_REPOSITORY ${GCEM_REPO}
                GIT_TAG ${GCEM_TAG}
                SOURCE_DIR ${GcemSourceDir}
                PATCH_COMMAND ${CMAKE_COMMAND} -E echo "Patching GCEM..."
                && cd ${GcemSourceDir}
                && ${PATCH_EXECUTABLE} --forward --batch --silent  < ${CMAKE_CURRENT_SOURCE_DIR}/patches/gcem/CMakeLists.txt.patch
                || ${CMAKE_COMMAND} -E echo "GCEM patch skipped")
        FetchContent_MakeAvailable(gcem)
        add_library(gcem::gcem ALIAS gcem)
    endif ()
endif ()
target_link_libraries(SecUtility.Core INTERFACE gcem::gcem)

target_compile_definitions(SecUtility.Core INTERFACE __STDCPP_WANT_MATH_SPEC_FUNCS__)

enable_testing()
add_subdirectory(tests)

#-----------------------------------------------------------------------------------

set_target_properties(SecUtility.Core PROPERTIES EXPORT_NAME SecUtility.Core)
install(TARGETS SecUtility.Core EXPORT SecUtilityCoreTargets)
export(TARGETS SecUtility.Core NAMESPACE SecUtility:: FILE SecUtilityCoreTargets.cmake)
export(PACKAGE SecUtility.Core)
install(EXPORT SecUtilityCoreTargets NAMESPACE SecUtility:: DESTINATION "${CMAKE_INSTALL_DATADIR}/SecUtility.Core/cmake")
install(DIRECTORY "SecUtility" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
