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

add_executable(wapic src/main.cc)
target_link_libraries(wapic PRIVATE wapic_lib)
