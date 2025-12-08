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
enable_testing()
include(CTest)
include(Catch)

# Main executable
add_executable(tl
    Source/main.cpp
    Source/Parser.cpp
    Source/Lexer.cpp
    Source/AST.cpp
    Source/backend_libtorch.cpp
    Source/VM.cpp
    Source/Runtime/ExecutorUtils.cpp
    Source/Runtime/DatalogEngine.cpp
    Source/Runtime/LearningEngine.cpp
    Source/Runtime/Executors/ScalarAssignExecutor.cpp
    Source/Runtime/Executors/ListLiteralExecutor.cpp
    Source/Runtime/Executors/EinsumExecutor.cpp
    Source/Runtime/Executors/IndexedProductExecutor.cpp
    Source/Runtime/Executors/ReductionExecutor.cpp
    Source/Runtime/Executors/NormalizationExecutor.cpp
    Source/Runtime/Executors/IdentityExecutor.cpp
    Source/Runtime/Executors/PoolingExecutor.cpp
    Source/Runtime/Executors/GuardedClauseExecutor.cpp
    Source/Runtime/Executors/ExpressionExecutor.cpp
    Source/Runtime/Preprocessors/VirtualIndexPreprocessor.cpp
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
add_executable(tl_tests
    Tests/Integration/test_examples.cpp

    Tests/Unit/test_parse_statements.cpp
    Tests/Unit/test_tensor_operations.cpp
    Tests/Unit/test_activations.cpp
    Tests/Unit/test_pooling.cpp
    Tests/Unit/test_datalog.cpp
    Tests/Unit/test_neurosymbolic.cpp
    Tests/Unit/test_executors.cpp
    Tests/Unit/test_guarded_clauses.cpp
    Tests/Unit/test_slicing.cpp
    Tests/Unit/test_normalized_indices.cpp
    Tests/Unit/test_datalog_arithmetic.cpp
    Tests/Unit/test_tensor_comparisons.cpp
    Tests/Unit/test_learning_directives.cpp

    # Source files needed for tests
    Source/Parser.cpp
    Source/Lexer.cpp
    Source/AST.cpp
    Source/VM.cpp
    Source/backend_libtorch.cpp
    Source/Runtime/ExecutorUtils.cpp
    Source/Runtime/DatalogEngine.cpp
    Source/Runtime/LearningEngine.cpp
    Source/Runtime/Executors/ScalarAssignExecutor.cpp
    Source/Runtime/Executors/ListLiteralExecutor.cpp
    Source/Runtime/Executors/EinsumExecutor.cpp
    Source/Runtime/Executors/IndexedProductExecutor.cpp
    Source/Runtime/Executors/ReductionExecutor.cpp
    Source/Runtime/Executors/NormalizationExecutor.cpp
    Source/Runtime/Executors/IdentityExecutor.cpp
    Source/Runtime/Executors/PoolingExecutor.cpp
    Source/Runtime/Executors/GuardedClauseExecutor.cpp
    Source/Runtime/Executors/ExpressionExecutor.cpp
    Source/Runtime/Preprocessors/VirtualIndexPreprocessor.cpp
        Tests/Unit/test_virtual_indexing.cpp
)

target_include_directories(tl_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Include
)

target_link_libraries(tl_tests PRIVATE
    Catch2::Catch2WithMain
    taocpp::pegtl
    ${TORCH_LIBRARIES}
)

target_compile_definitions(tl_tests PRIVATE TL_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# Set RPATH for libtorch shared libraries in tests
set_target_properties(tl_tests PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Discover tests
catch_discover_tests(tl_tests)

# ===================================
# Installation and Packaging
# ===================================

include(GNUInstallDirs)

# Install main executable
install(
    TARGETS tl
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT Application
)

# Install libtorch dependencies
install(
    DIRECTORY ${CMAKE_BINARY_DIR}/_deps/libtorch-src/lib/
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT Runtime
    FILES_MATCHING
        PATTERN "*.so*"      # Linux
        PATTERN "*.dylib"    # macOS
        PATTERN "*.dll"      # Windows
    PATTERN "*.a" EXCLUDE    # Exclude static libraries
)

# Install example programs
install(
    DIRECTORY Examples/Programs
    DESTINATION ${CMAKE_INSTALL_DATADIR}/tensorlogic/examples
    COMPONENT Examples
    FILES_MATCHING PATTERN "*.tl"
)

# Install documentation
install(
    FILES README.md Docs/Grammar.bnf
    DESTINATION ${CMAKE_INSTALL_DOCDIR}
    COMPONENT Documentation
)

# Install installation scripts at archive root
install(
    FILES install.sh install.ps1
    DESTINATION .
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    COMPONENT Application
)

# ===================================
# CPack Configuration
# ===================================

# Basic package information
set(CPACK_PACKAGE_NAME "TensorLogic")
set(CPACK_PACKAGE_VENDOR "TensorLogic Project")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Unified programming language for AI bridging neural networks and symbolic reasoning")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "1")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")

# Package file name
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}")

# Resources
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

# Platform-specific generator selection
if(WIN32)
    set(CPACK_GENERATOR "ZIP")
elseif(APPLE)
    set(CPACK_GENERATOR "TGZ")
else()
    set(CPACK_GENERATOR "TGZ")
endif()

# Archive-specific settings
set(CPACK_ARCHIVE_COMPONENT_INSTALL OFF)  # Include all components in single archive

# Must be last
include(CPack)
