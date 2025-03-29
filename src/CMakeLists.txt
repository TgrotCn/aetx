# llama_add_compile_flags()

#
# libraries
find_package(CUDA REQUIRED)

# llama

add_library(yalm
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

target_include_directories(yalm
    PUBLIC
    ${CUDA_INCLUDE_DIRS}
    # ${CMAKE_SOURCE_DIR}/src
) 

target_link_libraries(yalm
    PUBLIC
    vendor
    ${CUDA_LIBRARIES}
)

target_compile_features   (yalm PUBLIC cxx_std_17) # don't bump

if (BUILD_SHARED_LIBS)
    set_target_properties(yalm PROPERTIES POSITION_INDEPENDENT_CODE ON)
    # 添加了预定义宏
    target_compile_definitions(yalm PRIVATE YALM_BUILD)
    target_compile_definitions(yalm PUBLIC  YALM_SHARED)
endif()
