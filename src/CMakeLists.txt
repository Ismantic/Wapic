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

add_library(isma_wapiti ${SRCS})
target_include_directories(isma_wapiti PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(isma_wapiti_test main.cc)
target_link_libraries(isma_wapiti_test PRIVATE isma_wapiti)
