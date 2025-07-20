add_executable(helloworld helloworld.cpp)
# add_warnings_optimizations(helloworld)
target_link_libraries(helloworld PUBLIC Crow::Crow)