cmake_minimum_required(VERSION 3.18)

project(rl_ale LANGUAGES C CXX)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ZLIB_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ZLIB_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ZLIB_BUILD_STATIC ON CACHE BOOL "" FORCE)
set(ZLIB_INSTALL OFF CACHE BOOL "" FORCE)
set(SKIP_INSTALL_FILES ON CACHE BOOL "" FORCE)

add_subdirectory(vendor/zlib zlib)

set_target_properties(
    zlibstatic
    PROPERTIES
        OUTPUT_NAME z
        DEBUG_POSTFIX ""
)

if(NOT TARGET ZLIB::ZLIB)
    add_library(ZLIB::ZLIB ALIAS zlibstatic)
endif()

set(
    ZLIB_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/vendor/zlib"
    CACHE PATH "" FORCE
)

set(
    ZLIB_LIBRARY
    zlibstatic
    CACHE STRING "" FORCE
)

set(BUILD_CPP_LIB OFF CACHE BOOL "" FORCE)
set(BUILD_PYTHON_LIB OFF CACHE BOOL "" FORCE)
set(BUILD_VECTOR_LIB OFF CACHE BOOL "" FORCE)
set(BUILD_WASM_LIB OFF CACHE BOOL "" FORCE)
set(SDL_SUPPORT OFF CACHE BOOL "" FORCE)

add_subdirectory(vendor/ale ale)

set_target_properties(
    ale
    PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION OFF
)

add_library(
    rl_ale
    STATIC
        bridge.cpp
        vendor/ale/src/ale/ale_interface.cpp
        $<TARGET_OBJECTS:ale>
)

target_include_directories(
    rl_ale
    PRIVATE
        vendor/ale/src
        vendor/ale/src/ale
        ${CMAKE_CURRENT_BINARY_DIR}/ale/src/ale
)

target_link_libraries(
    rl_ale
    PRIVATE
        ZLIB::ZLIB
)

install(
    TARGETS
        rl_ale
        zlibstatic
    ARCHIVE DESTINATION lib
)
