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
find_package(libstats QUIET)

if(NOT libstats_FOUND)
    message(STATUS "libstats not found via find_package — using FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        libstats
        GIT_REPOSITORY https://github.com/OldCrow/libstats.git
        GIT_TAG        v1.1.2
    )
    # Disable libstats tests/tools/examples when building as a subdirectory
    set(LIBSTATS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LIBSTATS_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(BUILD_DOCS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(libstats)
    # libstats currently adds examples unconditionally; exclude them from default build
    # to avoid wheel builds failing on optional example link targets.
    foreach(_libstats_example_target
        quick_start_tutorial
        basic_usage
        distribution_families_demo
        error_handling_demo
        statistical_validation_demo
        parallel_execution_demo
        performance_learning_demo
    )
        if(TARGET ${_libstats_example_target})
            set_target_properties(
                ${_libstats_example_target}
                PROPERTIES
                    EXCLUDE_FROM_ALL TRUE
                    EXCLUDE_FROM_DEFAULT_BUILD TRUE
            )
        endif()
    endforeach()

    # Some Linux builders can report AVX2 runtime support while AVX runtime checks fail.
    # In libstats, AVX2 transcendental implementations delegate to AVX functions, so this
    # combination can leave AVX symbols undefined unless simd_avx.cpp is also built.
    if(TARGET libstats_simd_obj AND LIBSTATS_HAS_AVX2 AND NOT LIBSTATS_HAS_AVX)
        message(
            STATUS
            "libstats reported AVX2 without AVX; adding simd_avx.cpp to satisfy AVX2 symbol dependencies"
        )
        target_sources(libstats_simd_obj PRIVATE ${libstats_SOURCE_DIR}/src/simd_avx.cpp)
        if(MSVC OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND WIN32))
            set_source_files_properties(
                ${libstats_SOURCE_DIR}/src/simd_avx.cpp
                PROPERTIES COMPILE_FLAGS "/arch:AVX"
            )
        else()
            set_source_files_properties(
                ${libstats_SOURCE_DIR}/src/simd_avx.cpp
                PROPERTIES COMPILE_FLAGS "-mavx"
            )
        endif()
    endif()
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
