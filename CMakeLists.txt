cmake_minimum_required(VERSION 3.14)
project(Wapic LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(SRCS
    src/pattern.cc
    src/misc.cc
    src/trie.cc
    src/data.cc
    src/model.cc
    src/state.cc
    src/score.cc
    src/optimize.cc
)

add_library(wapic_lib ${SRCS})
target_include_directories(wapic_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
target_link_libraries(wapic_lib PUBLIC Threads::Threads OpenMP::OpenMP_CXX)

add_executable(wapic src/main.cc)
target_link_libraries(wapic PRIVATE wapic_lib)

# Python binding (optional; build with -DWAPIC_PYTHON=ON)
option(WAPIC_PYTHON "Build Python binding via pybind11" OFF)
if(WAPIC_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_core src/python_binding.cc)
    target_link_libraries(_core PRIVATE wapic_lib)
    set(WAPIC_PYTHON_PACKAGE_DIR "${CMAKE_BINARY_DIR}/python/wapic")
    set_target_properties(_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${WAPIC_PYTHON_PACKAGE_DIR}")
    file(MAKE_DIRECTORY "${WAPIC_PYTHON_PACKAGE_DIR}")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/python/wapic/__init__.py"
        "${WAPIC_PYTHON_PACKAGE_DIR}/__init__.py"
        COPYONLY)
endif()
