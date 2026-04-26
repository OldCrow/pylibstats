cmake_minimum_required(VERSION 3.20)
project(pylibstats LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Python + nanobind
# ---------------------------------------------------------------------------
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# scikit-build-core adds site-packages to CMAKE_PREFIX_PATH automatically.
# For direct CMake invocation, discover nanobind via Python as a fallback.
if(NOT nanobind_ROOT AND NOT nanobind_DIR)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
        RESULT_VARIABLE _nanobind_rc)
    if(NOT _nanobind_rc EQUAL 0)
        set(nanobind_ROOT "")
    endif()
endif()

find_package(nanobind CONFIG REQUIRED)

# ---------------------------------------------------------------------------
# libstats — prefer installed package, fall back to FetchContent
# ---------------------------------------------------------------------------
find_package(libstats 1.1.4 QUIET)

if(NOT libstats_FOUND)
    message(STATUS "libstats not found via find_package — using FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        libstats
        GIT_REPOSITORY https://github.com/OldCrow/libstats.git
        GIT_TAG        v1.1.4
    )
    # Disable libstats tests/tools/examples when building as a subdirectory.
    set(LIBSTATS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LIBSTATS_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(LIBSTATS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_DOCS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(libstats)
    set(LIBSTATS_TARGET libstats_static)
else()
    message(STATUS "Using installed libstats ${LIBSTATS_VERSION}")
    # Installed exports use libstats::libstats_static (not libstats::static)
    if(TARGET libstats::libstats_static)
        set(LIBSTATS_TARGET libstats::libstats_static)
    elseif(TARGET libstats::static)
        set(LIBSTATS_TARGET libstats::static)
    elseif(DEFINED LIBSTATS_LIBRARIES)
        set(LIBSTATS_TARGET ${LIBSTATS_LIBRARIES})
    else()
        message(FATAL_ERROR "libstats found but no usable static target")
    endif()
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
