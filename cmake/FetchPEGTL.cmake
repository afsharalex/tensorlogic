# FetchPEGTL.cmake
# Fetches PEGTL (Parsing Expression Grammar Template Library) from GitHub
# PEGTL is a header-only library

include(FetchContent)

message(STATUS "Fetching PEGTL from GitHub")

FetchContent_Declare(
    pegtl
    GIT_REPOSITORY https://github.com/taocpp/PEGTL.git
    GIT_TAG        3.2.7  # Stable release version
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(pegtl)

message(STATUS "PEGTL fetched successfully")
