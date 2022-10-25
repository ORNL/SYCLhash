include(CTest)

add_executable(alloc alloc.cpp)
target_link_libraries(alloc PUBLIC syclhash)
add_test(NAME alloc COMMAND alloc)

add_executable(hash hash.cpp)
target_link_libraries(hash PUBLIC syclhash)
add_test(NAME hash COMMAND hash)

add_executable(num num.cpp)
target_link_libraries(num PUBLIC syclhash)
add_test(NAME num COMMAND hash)
