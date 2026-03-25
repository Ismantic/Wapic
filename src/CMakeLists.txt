set(SRCS
    pattern.cc
    misc.cc
    trie.cc
    data.cc
    option.cc
    model.cc
    state.cc
    score.cc
    optimize.cc
    progress.cc
)

add_library(wapic_lib ${SRCS})
target_include_directories(wapic_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(wapic main.cc)
target_link_libraries(wapic PRIVATE wapic_lib)
