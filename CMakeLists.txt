cmake_minimum_required(VERSION 3.14) # for add_link_options and implicit target directories.
project("yalm" C CXX CUDA)
include(CheckIncludeFileCXX)

#set(CMAKE_WARN_DEPRECATED YES)
# 命令行未使用的参数会报警告
set(CMAKE_WARN_UNUSED_CLI YES)

# 在CMake目录下生成一个名为 compile_commands.json 的文件。
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++标准设置
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# 添加CUDA标准设置
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Add path to modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(YALM_STANDALONE ON)
    include(git-vars)

else()
    set(YALM_STANDALONE OFF)
endif()



set(BUILD_SHARED_LIBS_DEFAULT OFF)
option(BUILD_SHARED_LIBS "build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})

# debug
option(YALM_ALL_WARNINGS           "yalm: enable all compiler warnings"                   ON)
option(YALM_ALL_WARNINGS_3RD_PARTY "yalm: enable all compiler warnings in 3rd party libs" OFF)

# build
option(YALM_FATAL_WARNINGS "yalm: enable -Werror flag" OFF)

# sanitizers  是一类用于检测程序中的潜在错误和不规范行为的工具，通常在开发和调试阶段使用。
option(YALM_SANITIZE_THREAD    "yalm: enable thread sanitizer"    OFF)
option(YALM_SANITIZE_ADDRESS   "yalm: enable address sanitizer"   OFF)
option(YALM_SANITIZE_UNDEFINED "yalm: enable undefined sanitizer" OFF)

# utils
option(YALM_BUILD_COMMON "yalm: build common utils library" ${YALM_STANDALONE})

# extra artifacts
option(YALM_BUILD_TESTS    "yalm: build tests"          ${YALM_STANDALONE})
option(YALM_BUILD_EXAMPLES "yalm: build examples"       ${YALM_STANDALONE})
option(YALM_BUILD_SERVER   "yalm: build server example" OFF)

# 3rd party libs
# option(LLAMA_CURL "llama: use libcurl to download model from an URL" OFF)
# option(LLAMA_LLGUIDANCE "llama-common: include LLGuidance library for structured output in common utils" OFF)

# Required for relocatable CMake package
# include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake)
# include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/common.cmake)

# TO DO
# override ggml options
# set(GGML_ALL_WARNINGS   ${YALM_ALL_WARNINGS})
# set(GGML_FATAL_WARNINGS ${YALM_FATAL_WARNINGS})

# change the default for these ggml options
# if (NOT DEFINED GGML_LLAMAFILE)
    # set(GGML_LLAMAFILE_DEFAULT ON)
# endif()

# if (NOT DEFINED GGML_CUDA_GRAPHS)
    # set(GGML_CUDA_GRAPHS_DEFAULT ON)
# endif()

# transition helpers
function (yalm_option_depr TYPE OLD NEW)
    if (${OLD})
        message(${TYPE} "${OLD} is deprecated and will be removed in the future.\nUse ${NEW} instead\n")
        set(${NEW} ON PARENT_SCOPE)
    endif()
endfunction()

# build the library
add_subdirectory(src)
# vendor
add_subdirectory(vendor)
    # examples
if (YALM_BUILD_EXAMPLES)
    add_subdirectory(example)
endif()




#
# install
#

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(YALM_BUILD_NUMBER        ${BUILD_NUMBER})
set(YALM_BUILD_COMMIT        ${BUILD_COMMIT})
set(YALM_INSTALL_VERSION 0.0.${BUILD_NUMBER})

set(YALM_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header  files")
set(YALM_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of library files")
set(YALM_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Location of binary  files")


# TO DO
# 公共头文件 被编译为共享库时可被外部使用
# set_target_properties(yalm
#     PROPERTIES
#         PUBLIC_HEADER "${LLAMA_PUBLIC_HEADERS}")

# install(TARGETS yalm LIBRARY PUBLIC_HEADER)

