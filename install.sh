#!/bin/bash
# TensorLogic Installation Script for Unix-like systems (Linux, macOS)
# Usage: ./install.sh [--prefix=PREFIX] [--uninstall]

set -e  # Exit on error

# Default installation prefix
PREFIX="/usr/local"
UNINSTALL=false

# Parse command-line arguments
for arg in "$@"; do
    case $arg in
        --prefix=*)
            PREFIX="${arg#*=}"
            shift
            ;;
        --uninstall)
            UNINSTALL=true
            shift
            ;;
        --help|-h)
            echo "TensorLogic Installation Script"
            echo ""
            echo "Usage: ./install.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --prefix=PATH    Install to PATH (default: /usr/local)"
            echo "  --uninstall      Uninstall TensorLogic from PREFIX"
            echo "  --help, -h       Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./install.sh                           # Install to /usr/local"
            echo "  ./install.sh --prefix=\$HOME/.local     # Install to ~/.local"
            echo "  ./install.sh --uninstall               # Uninstall from /usr/local"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Detect operating system
OS=$(uname -s)

# Check if we need sudo
NEED_SUDO=false
if [ "$PREFIX" = "/usr/local" ] || [ "$PREFIX" = "/usr" ] || [ "$PREFIX" = "/opt" ]; then
    if [ "$(id -u)" -ne 0 ]; then
        NEED_SUDO=true
    fi
fi

# Sudo wrapper
run_cmd() {
    if [ "$NEED_SUDO" = true ]; then
        sudo "$@"
    else
        "$@"
    fi
}

# ====================================
# Uninstallation
# ====================================
if [ "$UNINSTALL" = true ]; then
    echo "Uninstalling TensorLogic from $PREFIX..."

    # Remove binary (and wrapper if it exists)
    if [ -f "$PREFIX/bin/tl" ]; then
        echo "  Removing $PREFIX/bin/tl"
        run_cmd rm -f "$PREFIX/bin/tl"
    fi
    if [ -f "$PREFIX/bin/tl.bin" ]; then
        echo "  Removing $PREFIX/bin/tl.bin"
        run_cmd rm -f "$PREFIX/bin/tl.bin"
    fi

    # Remove libraries (be careful not to remove system libraries)
    if [ -d "$PREFIX/lib" ]; then
        echo "  Removing libtorch libraries from $PREFIX/lib"
        run_cmd find "$PREFIX/lib" -name "libtorch*" -delete 2>/dev/null || true
        run_cmd find "$PREFIX/lib" -name "libc10*" -delete 2>/dev/null || true
        run_cmd find "$PREFIX/lib" -name "libshm*" -delete 2>/dev/null || true
    fi

    # Remove examples
    if [ -d "$PREFIX/share/tensorlogic" ]; then
        echo "  Removing $PREFIX/share/tensorlogic"
        run_cmd rm -rf "$PREFIX/share/tensorlogic"
    fi

    # Remove documentation
    if [ -d "$PREFIX/share/doc/tensorlogic" ]; then
        echo "  Removing $PREFIX/share/doc/tensorlogic"
        run_cmd rm -rf "$PREFIX/share/doc/tensorlogic"
    fi

    # Update library cache (Linux only)
    if [ "$OS" = "Linux" ]; then
        echo "  Updating library cache..."
        run_cmd ldconfig 2>/dev/null || true
    fi

    echo "Uninstallation complete!"
    exit 0
fi

# ====================================
# Installation
# ====================================

echo "Installing TensorLogic to $PREFIX..."

# Get script directory (where this script is located)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Verify required directories exist in source
if [ ! -d "$SCRIPT_DIR/bin" ]; then
    echo "Error: bin/ directory not found. Are you in the extracted archive directory?"
    exit 1
fi

if [ ! -d "$SCRIPT_DIR/lib" ]; then
    echo "Error: lib/ directory not found. Are you in the extracted archive directory?"
    exit 1
fi

# Create installation directories
echo "  Creating installation directories..."
run_cmd mkdir -p "$PREFIX/bin"
run_cmd mkdir -p "$PREFIX/lib"
run_cmd mkdir -p "$PREFIX/share/tensorlogic"
run_cmd mkdir -p "$PREFIX/share/doc/tensorlogic"

# Install binary
echo "  Installing tl executable to $PREFIX/bin/tl..."
run_cmd cp "$SCRIPT_DIR/bin/tl" "$PREFIX/bin/tl"
run_cmd chmod +x "$PREFIX/bin/tl"

# Fix RPATH so binary can find libraries
echo "  Configuring library paths..."
if [ "$OS" = "Darwin" ]; then
    # macOS: Use install_name_tool to add RPATH
    if command -v install_name_tool &> /dev/null; then
        # Remove existing RPATHs that won't work
        run_cmd install_name_tool -add_rpath "$PREFIX/lib" "$PREFIX/bin/tl" 2>/dev/null || true
    fi
elif [ "$OS" = "Linux" ]; then
    # Linux: Use patchelf if available to set RPATH
    if command -v patchelf &> /dev/null; then
        run_cmd patchelf --set-rpath "$PREFIX/lib" "$PREFIX/bin/tl" 2>/dev/null || true
    else
        # If patchelf not available, create wrapper script
        echo "  Warning: patchelf not found. Creating wrapper script..."
        run_cmd mv "$PREFIX/bin/tl" "$PREFIX/bin/tl.bin"
        cat > /tmp/tl_wrapper <<'EOF'
#!/bin/bash
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LD_LIBRARY_PATH="$INSTALL_DIR/lib:$LD_LIBRARY_PATH"
exec "$INSTALL_DIR/bin/tl.bin" "$@"
EOF
        run_cmd mv /tmp/tl_wrapper "$PREFIX/bin/tl"
        run_cmd chmod +x "$PREFIX/bin/tl"
    fi
fi

# Install libraries
echo "  Installing libtorch libraries to $PREFIX/lib/..."
run_cmd cp -r "$SCRIPT_DIR/lib"/* "$PREFIX/lib/"

# Install examples (if they exist)
if [ -d "$SCRIPT_DIR/examples" ]; then
    echo "  Installing example programs to $PREFIX/share/tensorlogic/examples/..."
    run_cmd cp -r "$SCRIPT_DIR/examples" "$PREFIX/share/tensorlogic/"
elif [ -d "$SCRIPT_DIR/Programs" ]; then
    # Alternative location from CPack
    echo "  Installing example programs to $PREFIX/share/tensorlogic/examples/..."
    run_cmd mkdir -p "$PREFIX/share/tensorlogic/examples"
    run_cmd cp -r "$SCRIPT_DIR/Programs"/* "$PREFIX/share/tensorlogic/examples/"
fi

# Install documentation (if it exists)
if [ -d "$SCRIPT_DIR/doc" ]; then
    echo "  Installing documentation to $PREFIX/share/doc/tensorlogic/..."
    run_cmd cp -r "$SCRIPT_DIR/doc"/* "$PREFIX/share/doc/tensorlogic/"
fi

# Update library cache (Linux only)
if [ "$OS" = "Linux" ]; then
    echo "  Updating library cache..."
    run_cmd ldconfig 2>/dev/null || true
fi

# ====================================
# Post-installation instructions
# ====================================

echo ""
echo "Installation complete!"
echo ""
echo "TensorLogic has been installed to: $PREFIX"
echo ""

# Check if PREFIX/bin is in PATH
if echo "$PATH" | grep -q "$PREFIX/bin"; then
    echo "You can now run TensorLogic with: tl"
else
    echo "IMPORTANT: Add $PREFIX/bin to your PATH to use TensorLogic:"
    echo ""
    if [ "$OS" = "Darwin" ]; then
        # macOS
        echo "  echo 'export PATH=\"$PREFIX/bin:\$PATH\"' >> ~/.zshrc"
        echo "  source ~/.zshrc"
    else
        # Linux
        echo "  echo 'export PATH=\"$PREFIX/bin:\$PATH\"' >> ~/.bashrc"
        echo "  source ~/.bashrc"
    fi
    echo ""
fi

echo "Test the installation with:"
echo "  $PREFIX/bin/tl --version"
echo ""
echo "To uninstall, run:"
echo "  ./install.sh --uninstall --prefix=$PREFIX"
echo ""
