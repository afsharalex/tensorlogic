# FetchLibTorch.cmake
# Downloads pre-built libtorch based on platform and architecture

include(FetchContent)

# Detect platform and architecture
if(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        set(LIBTORCH_PLATFORM "macos-arm64")
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.5.1.zip")
    else()
        set(LIBTORCH_PLATFORM "macos-x86_64")
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86-64-2.5.1.zip")
    endif()
elseif(UNIX AND NOT APPLE)
    # Linux
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
        set(LIBTORCH_PLATFORM "linux-x86_64")
        # Choose between CPU-only or CUDA version
        if(USE_CUDA)
            set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu121.zip")
        else()
            set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip")
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(LIBTORCH_PLATFORM "linux-aarch64")
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip")
    else()
        message(FATAL_ERROR "Unsupported Linux architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
elseif(WIN32)
    # Windows
    if(USE_CUDA)
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu121/libtorch-win-shared-with-deps-2.5.1%2Bcu121.zip")
    else()
        set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.5.1%2Bcpu.zip")
    endif()
    set(LIBTORCH_PLATFORM "windows")
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

message(STATUS "Fetching libtorch for ${LIBTORCH_PLATFORM}")
message(STATUS "URL: ${LIBTORCH_URL}")

FetchContent_Declare(
    libtorch
    URL ${LIBTORCH_URL}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(libtorch)

# Set up paths for find_package
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${libtorch_SOURCE_DIR})
list(APPEND CMAKE_PREFIX_PATH ${libtorch_SOURCE_DIR})

# Find the Torch package
find_package(Torch REQUIRED)

message(STATUS "libtorch fetched successfully")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
