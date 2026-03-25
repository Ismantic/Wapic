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

add_library(wapix_lib ${SRCS})
target_include_directories(wapix_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(wapix main.cc)
target_link_libraries(wapix PRIVATE wapix_lib)
