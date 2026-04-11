cmake_minimum_required(VERSION 3.20)
project(pylibstats LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Python + nanobind
# ---------------------------------------------------------------------------
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

# ---------------------------------------------------------------------------
# libstats — prefer installed package, fall back to FetchContent
# ---------------------------------------------------------------------------
find_package(libstats QUIET)

if(NOT libstats_FOUND)
    message(STATUS "libstats not found via find_package — using FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        libstats
        GIT_REPOSITORY https://github.com/OldCrow/libstats.git
        GIT_TAG        v1.0.0
    )
    # Disable libstats tests/tools/examples when building as a subdirectory
    set(LIBSTATS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LIBSTATS_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(BUILD_DOCS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(libstats)
    set(LIBSTATS_TARGET libstats_static)
else()
    message(STATUS "Using installed libstats ${LIBSTATS_VERSION}")
    set(LIBSTATS_TARGET libstats::static)
endif()

# ---------------------------------------------------------------------------
# Extension module
# ---------------------------------------------------------------------------
nanobind_add_module(
    _core
    STABLE_ABI
    NB_STATIC
    src/pylibstats/_core.cpp
)

target_link_libraries(_core PRIVATE ${LIBSTATS_TARGET})

# Windows: prevent min/max macros from <windows.h> colliding with std::min/max
if(WIN32)
    target_compile_definitions(_core PRIVATE NOMINMAX _USE_MATH_DEFINES)
endif()

# Let nanobind / scikit-build-core handle the install destination
install(TARGETS _core LIBRARY DESTINATION pylibstats)
