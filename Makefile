example: main.cpp Timer.cpp
	g++ main.cpp Timer.cpp -o example -O2 -larmadillo -std=c++11 -fopenmp

debug: main.cpp
	g++ -g main.cpp Timer.cpp -o example -O3 -larmadillo
	gdb example

run:
	./example

clean:
	rm example
