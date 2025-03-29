# llama_add_compile_flags()

#
# libraries
find_package(CUDA REQUIRED)

# llama

add_library(aetx
            codec.cpp
            debug.cpp
            infer.cpp
            infer.cu
            model.cpp
            sampler.cpp
            test.cpp
            timeutils.cpp
            tokenizer.cpp
            codec.h
            )

target_include_directories(aetx
    PUBLIC
    ${CUDA_INCLUDE_DIRS}
    # ${CMAKE_SOURCE_DIR}/src
) 

target_link_libraries(aetx
    PUBLIC
    vendor
    ${CUDA_LIBRARIES}
)

target_compile_features   (aetx PUBLIC cxx_std_17) # don't bump

if (BUILD_SHARED_LIBS)
    set_target_properties(aetx PROPERTIES POSITION_INDEPENDENT_CODE ON)
    # 添加了预定义宏
    target_compile_definitions(aetx PRIVATE AETX_BUILD)
    target_compile_definitions(aetx PUBLIC  AETX_SHARED)
endif()
