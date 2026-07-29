cmake_minimum_required(VERSION 3.14)
project(Wapic LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(SRCS
    src/pattern.cc
    src/misc.cc
    src/preprocess.cc
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
if(APPLE)
    # AppleClang does not ship an OpenMP runtime. Release wheels install
    # Homebrew libomp, which is then bundled into the wheel by delocate.
    find_library(LIBOMP_LIBRARY
        NAMES omp
        HINTS
            /opt/homebrew/opt/libomp/lib
            /usr/local/opt/libomp/lib
        REQUIRED)
    find_path(LIBOMP_INCLUDE_DIR
        NAMES omp.h
        HINTS
            /opt/homebrew/opt/libomp/include
            /usr/local/opt/libomp/include
        REQUIRED)
    target_compile_options(wapic_lib PUBLIC -Xpreprocessor -fopenmp)
    target_include_directories(wapic_lib PUBLIC ${LIBOMP_INCLUDE_DIR})
    target_link_libraries(wapic_lib PUBLIC Threads::Threads ${LIBOMP_LIBRARY})
else()
    find_package(OpenMP REQUIRED)
    target_link_libraries(wapic_lib PUBLIC Threads::Threads OpenMP::OpenMP_CXX)
endif()

add_executable(wapic src/main.cc)
target_link_libraries(wapic PRIVATE wapic_lib)

# Python binding (optional; build with -DWAPIC_PYTHON=ON)
option(WAPIC_PYTHON "Build Python binding via pybind11" OFF)
if(WAPIC_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_core src/pip.cc)
    target_link_libraries(_core PRIVATE wapic_lib)
    set(WAPIC_PYTHON_PACKAGE_DIR "${CMAKE_BINARY_DIR}/python/wapic")
    set_target_properties(_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${WAPIC_PYTHON_PACKAGE_DIR}")
    file(MAKE_DIRECTORY "${WAPIC_PYTHON_PACKAGE_DIR}")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/python/wapic/__init__.py"
        "${WAPIC_PYTHON_PACKAGE_DIR}/__init__.py"
        COPYONLY)
    # Install rules so `pip install .` (scikit-build-core) collects the package.
    install(TARGETS _core LIBRARY DESTINATION wapic)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/python/wapic/__init__.py"
            DESTINATION wapic)
endif()
