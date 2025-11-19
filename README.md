# TensorLogic

> **A unified programming language for AI that bridges neural networks and symbolic reasoning through tensor equations**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-3.22+-blue.svg)](https://cmake.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange.svg)](https://pytorch.org/)

## Overview

TensorLogic treats logical rules and Einstein summation as fundamentally equivalent operations, enabling scalable, learnable, and transparent AI systems. The language is built around a single construct: **tensor equations**.

See the Grammar in [Docs/GrammarV3.bnf](Docs/GrammarV3.bnf).

## Key Features

- **Unified Neural-Symbolic AI**: Single syntax for both neural networks and logic programs
- **Automatic Differentiation**: All tensor equations are differentiable via PyTorch
- **GPU Acceleration**: Leverages libtorch for CPU/GPU/MPS computation
- **Datalog Compatible**: Accepts standard Datalog syntax
- **Einstein Summation**: Shared indices imply automatic summation
- **Transparent Reasoning**: Forward and backward chaining inference

## Original Paper

This implementation is based on the paper **["Tensor Logic: The Language of AI"](https://arxiv.org/abs/2510.12269)** by **[Pedro Domingos](https://homes.cs.washington.edu/~pedrod/)**.
Included in Repository here: **["Tensor Logic: The Language of AI"](Docs/Tensor%20Logic%20-%20The%20Language%20of%20AI.pdf)** by **[Pedro Domingos](https://homes.cs.washington.edu/~pedrod/)**.

## Documentation

TODO: Add documentation

## Quick Start

### Prerequisites

- CMake 3.22 or higher
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2019+)
- ~2GB disk space for dependencies (downloaded automatically on first build)

### Building from Source

#### macOS (ARM64 & x86_64)

```bash
# Clone the repository
git clone https://github.com/afsharalex/tensorlogic.git
cd tensorlogic

# Configure (downloads dependencies automatically)
cmake --preset default
# OR
cmake -B build

# Build
cmake --build build

# Run
./build/tl [options] <file.tl>

# Run tests
./build/tl_tests
```

#### Linux (Ubuntu/Debian)

```bash
# Install CMake if needed
sudo apt-get update
sudo apt-get install cmake build-essential

# Clone and build
git clone https://github.com/afsharalex/tensorlogic.git
cd tensorlogic

# Configure (CPU version)
cmake --preset default

# Or configure with CUDA support
cmake --preset default -DUSE_CUDA=ON
# OR
cmake -B build

# Build
cmake --build build

# Run
./build/tl

# Run tests
./build/tl_tests
```

#### Linux (Fedora/RHEL)

```bash
# Install CMake if needed
sudo dnf install cmake gcc-c++

# Clone and build
git clone https://github.com/afsharalex/tensorlogic.git
cd tensorlogic

# Configure
cmake --preset default
# OR
cmake -B build

# Build
cmake --build build

# Run
./build/tl

# Run tests
./build/tl_tests
```

#### Windows (Visual Studio)

```powershell
# Prerequisites: Visual Studio 2019+ with C++ tools

# Clone the repository
git clone https://github.com/afsharalex/tensorlogic.git
cd tensorlogic

# Configure
cmake --preset default
# OR
cmake -B build

# Build
cmake --build build --config Release

# Run
.\build\Release\tl.exe

# Run tests
.\build\Release\tl_tests.exe
```

### Usage

#### Command Line Options

```bash
./build/tl [options] <file.tl>

Options:
  --debug, -d         Enable debug output
  --bytecode, -b      Execute using bytecode VM (experimental)
  --compile, -c       Compile to bytecode only (don't execute)
  --disassemble, -D   Show bytecode disassembly
```

#### Examples

```bash
# Run program with AST interpreter (default)
./build/tl Examples/Programs/02_matrix_multiply.tl

# Run program with bytecode VM
./build/tl --bytecode Examples/Programs/02_matrix_multiply.tl

# Compile and show disassembly without executing
./build/tl --compile --disassemble Examples/Programs/02_matrix_multiply.tl

# Enable debug output
./build/tl --debug Examples/Programs/02_matrix_multiply.tl
```

### First Build Notes

⚠️ **The first build downloads ~2GB of dependencies** (libtorch and PEGTL) which are cached in `build/_deps/`. Subsequent builds are fast.

## Platform Support

| Platform | Architecture          | Status      | Notes                  |
|----------|-----------------------|-------------|------------------------|
| macOS    | ARM64 (Apple Silicon) | ✅ Supported | CPU + MPS acceleration |
| macOS    | x86_64 (Intel)        | ✅ Supported | CPU only               |
| Linux    | x86_64                | ✅ Supported | CPU + CUDA (optional)  |
| Linux    | ARM64                 | ✅ Supported | CPU only               |
| Windows  | x86_64                | ✅ Supported | CPU + CUDA (optional)  |

## CUDA Support

To enable CUDA acceleration on Linux or Windows:

```bash
cmake --preset default -DUSE_CUDA=ON
cmake --build build
```

**Requirements**: CUDA Toolkit 11.8+ must be installed on your system.

## Project Structure

```
tensorlogic/
├── Include/          # Public header files
│   └── TensorLogic/
│       ├── Core/     # Core data structures
│       └── Parser/   # Parser grammar and AST definitions
├── Source/           # C++ implementation
│   └── main.cpp      # Entry point with CLI
├── Tests/            # Test suite
│   ├── Unit/         # Unit tests (planned)
│   ├── Integration/  # Integration tests (planned)
│   └── Benchmarks/   # Performance tests (planned)
├── Examples/         # Example .tl programs
├── cmake/            # CMake utility scripts
│   ├── FetchLibTorch.cmake  # Downloads libtorch binaries
│   ├── FetchPEGTL.cmake     # Fetches PEGTL parser library
│   └── FetchCatch2.cmake    # Fetches Catch2 testing framework
├── Docs/             # Documentation
│   ├── Tensor Logic - The Language of AI.pdf  # Original paper
│   └── GrammarV3.bnf  # Grammar definition
├── build/            # Build artifacts (gitignored)
│   ├── tl            # Main executable
│   ├── tl_tests      # Test executable
│   └── _deps/        # Downloaded dependencies (cached)
├── CMakeLists.txt    # Root build configuration
├── CMakePresets.json # CMake presets
└── README.md         # This file
```

## Dependencies

All dependencies are automatically downloaded via CMake FetchContent:

- **[PyTorch (libtorch)](https://pytorch.org/)** 2.5.1 - Tensor operations and automatic differentiation
- **[PEGTL](https://github.com/taocpp/PEGTL)** 3.2.7 - Parsing Expression Grammar Template Library
- **[Catch2](https://github.com/catchorg/Catch2)** 3.5.2 - Modern C++ testing framework

## Development

### Build Commands

```bash
# Clean build (removes cached dependencies)
rm -rf build
cmake --preset default

# OR
cmake -B build

# Build in debug mode
cmake --preset default -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Run tests
./build/tl_tests

# Run tests via CTest
cd build && ctest --output-on-failure

# Run specific test tags
./build/tl_tests "[index]"   # Run Index tests only
./build/tl_tests "[shape]"   # Run Shape tests only
./build/tl_tests "[type]"    # Run Type tests only

# Install to system
sudo cmake --install build --prefix /usr/local
```

### Architecture

TensorLogic is designed as a **thin wrapper** around PyTorch (libtorch):

- **Parser**: PEGTL-based parser for tensor equations and Datalog syntax
- **AST**: Abstract syntax tree representation
- **Compiler**: AST to bytecode compiler (optional execution path)
- **Bytecode VM**: Register-based bytecode interpreter (experimental)
- **AST Interpreter**: Direct AST execution (default)
- **Type System**: Type inference with PyTorch broadcasting rules
- **Runtime**: Forward/backward chaining inference engine
- **Autodiff**: Delegates to PyTorch's autograd
- **Backend**: PyTorch handles all tensor operations (CPU/GPU/MPS)

#### Execution Modes

TensorLogic supports two execution modes:

1. **AST Interpreter** (default): Directly interprets the abstract syntax tree
2. **Bytecode VM** (experimental): Compiles to bytecode then executes via register-based VM

The bytecode path is still experimental and may have incomplete features compared to the AST interpreter.

## Current Status

**Early Development** - This is an early-stage C++ implementation.

## Contributing

### TODO: Add contribution guide

Contributions are welcome!

- Build system details
- Code structure and conventions
- Development workflow
- Testing guidelines



## Roadmap

TODO: Add roadmap

## License

MIT - See [LICENSE](LICENSE) for details.

## Citation

If you use TensorLogic in your research, please cite:

```bibtex
@misc{domingos2025tensorlogiclanguageai,
      title={Tensor Logic: The Language of AI}, 
      author={Pedro Domingos},
      year={2025},
      eprint={2510.12269},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.12269}, 
}
```

## Contact

- **Original Paper**: Pedro Domingos - [https://arxiv.org/abs/2510.12269](https://arxiv.org/abs/2510.12269)
  - Included in Repository: [Docs/Tensor%20Logic%20-%20The%20Language%20of%20AI.pdf](Docs/Tensor%20Logic%20-%20The%20Language%20of%20AI.pdf)
- **Implementation**: Alex Afshar
- **Issues**: [GitHub Issues](https://github.com/afsharalex/tensorlogic/issues)
- **Discussions**: [GitHub Discussions](https://github.com/afsharalex/tensorlogic/discussions)

## Acknowledgments

- Built on [PyTorch (libtorch)](https://pytorch.org/) for tensor operations
- Uses [PEGTL](https://github.com/taocpp/PEGTL) for parsing
- Uses [Catch2](https://github.com/catchorg/Catch2) for testing
- Inspired by the unification of neural and symbolic AI by Pedro Domingos - [https://arxiv.org/abs/2510.12269](https://arxiv.org/abs/2510.12269)
  - Included in Repository: [Docs/Tensor%20Logic%20-%20The%20Language%20of%20AI.pdf](Docs/Tensor%20Logic%20-%20The%20Language%20of%20AI.pdf)
