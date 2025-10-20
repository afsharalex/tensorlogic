
build:
	cmake --build build

run_ctest:
	ctest --test-dir build -C Debug

test:
	./build/tl_tests

run:
	./build/tl
	
run_examples:
	find Examples -name "*.tl" -exec ./build/tl {} \;

clean:
	rm -rf build
