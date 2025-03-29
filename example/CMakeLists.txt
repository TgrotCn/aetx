# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

# yalm_add_compile_flags()

# examples
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(main)
if (YALM_BUILD_SERVER)
    add_subdirectory(server)
endif()
