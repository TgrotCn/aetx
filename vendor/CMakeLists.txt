# llama_add_compile_flags()

#
# libraries
#

# llama

add_library(vendor
    format.cc
    json.hpp  # 仅为 IDE 显示，不参与编译
)
target_include_directories(vendor
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/fmt
    ${CMAKE_CURRENT_SOURCE_DIR}    # 如有需要，添加根目录
)

target_compile_features   (vendor PUBLIC cxx_std_17) # don't bump

if (BUILD_SHARED_LIBS)
    set_target_properties(vendor PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(vendor PRIVATE VENDOR_BUILD)
    target_compile_definitions(vendor PUBLIC  VENDOR_SHARED)
endif()
