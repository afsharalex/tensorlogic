cmake_minimum_required(VERSION 3.22)

project(tensorlogic)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

# Add cmake module path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Fetch dependencies
include(cmake/FetchPEGTL.cmake)
include(cmake/FetchLibTorch.cmake)
include(cmake/FetchCatch2.cmake)

# Enable testing
# enable_testing()
# include(CTest)
# include(Catch)

# Main executable
add_executable(tl
    Source/main.cpp
)

target_include_directories(tl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Include
)

target_link_libraries(tl PRIVATE
    taocpp::pegtl
    ${TORCH_LIBRARIES}
)

# Set RPATH for libtorch shared libraries
set_target_properties(tl PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Test executable
# add_executable(tl_tests
#     Tests/Unit/test_index.cpp
#     Tests/Unit/test_shape.cpp
#     Tests/Unit/test_type.cpp
# )

# target_include_directories(tl_tests PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/Include
# )

# target_link_libraries(tl_tests PRIVATE
#     Catch2::Catch2WithMain
#     taocpp::pegtl
#     ${TORCH_LIBRARIES}
# )

# Set RPATH for libtorch shared libraries in tests
# set_target_properties(tl_tests PROPERTIES
#     INSTALL_RPATH_USE_LINK_PATH TRUE
# )

# Discover tests
# catch_discover_tests(tl_tests)
