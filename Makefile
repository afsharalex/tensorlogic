.PHONY: build run_ctest test run run_examples clean

# Collect all source and header files
SOURCES := $(wildcard Source/*.cpp) $(wildcard Include/**/*.h)

build: $(SOURCES)
	cmake --build build

run_ctest: build
	ctest --test-dir build -C Debug

test: build
	./build/tl_tests

run: build
	./build/tl

run_examples: build
	find Examples -name "*.tl" -exec ./build/tl {} \;

clean:
	rm -rf build
