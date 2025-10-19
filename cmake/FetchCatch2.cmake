# FetchCatch2.cmake
# Fetches Catch2 (C++ testing framework) from GitHub
# Catch2 is a modern, header-only (v3 is compiled) C++ test framework

include(FetchContent)

message(STATUS "Fetching Catch2 from GitHub")

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.2  # Stable release version (v3 is the modern version)
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(Catch2)

# Add Catch2's CMake modules to the module path for test discovery
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

message(STATUS "Catch2 fetched successfully")
