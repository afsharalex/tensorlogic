# TensorLogic Docker Image
# Multi-stage build for smaller final image

# ==================================
# Stage 1: Builder
# ==================================
# Force x86_64 architecture (libtorch doesn't have official ARM64 Linux builds)
FROM --platform=linux/amd64 ubuntu:22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy source code
COPY . /build/

# Configure and build
# Note: Limited parallelism to avoid OOM on systems with <8GB RAM
# Use -j1 for very limited memory, -j2 for 4-8GB, -j4 for 8GB+
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -j2

# Run tests to verify build
RUN cd build && ./tl_tests

# ==================================
# Stage 2: Runtime
# ==================================
# Must match builder platform for binary compatibility
FROM --platform=linux/amd64 ubuntu:22.04 AS runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create directories
RUN mkdir -p /usr/local/bin \
    /usr/local/lib \
    /examples \
    /workspace

# Copy binary from builder
COPY --from=builder /build/build/tl /usr/local/bin/tl

# Copy libtorch libraries from builder
COPY --from=builder /build/build/_deps/libtorch-src/lib/*.so* /usr/local/lib/

# Copy example programs
COPY --from=builder /build/Examples/Programs/*.tl /examples/

# Update library cache
RUN ldconfig

# Set working directory
WORKDIR /workspace

# Verify installation
RUN tl --version || echo "Note: --version flag may not be implemented yet"

# Default command
ENTRYPOINT ["tl"]
CMD ["--help"]

# ==================================
# Stage 3: Development (optional)
# ==================================
FROM builder AS development

# Install additional development tools
RUN apt-get update && apt-get install -y \
    gdb \
    valgrind \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set working directory to source
WORKDIR /build

# Default to bash for interactive development
ENTRYPOINT ["/bin/bash"]

# ==================================
# Metadata
# ==================================
LABEL maintainer="TensorLogic Project"
LABEL description="TensorLogic - Unified programming language for AI"
LABEL version="0.1.0"

# Usage:
# Build runtime image: docker build --target runtime -t tensorlogic:latest .
# Build dev image:     docker build --target development -t tensorlogic:dev .
# Run a program:       docker run --rm -v $(pwd):/workspace tensorlogic:latest program.tl
# Interactive shell:   docker run --rm -it tensorlogic:latest bash
