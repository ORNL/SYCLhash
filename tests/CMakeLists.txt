include(CTest)

add_executable(alloc alloc.cpp)
target_link_libraries(alloc PUBLIC syclhash)
add_test(NAME alloc COMMAND alloc)
