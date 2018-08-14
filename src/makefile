COMPILE_ARG = -pthread -O3 -march=native -std=c++11

all: train test_link test_classification_normal test_classification_isA 

train: transC.cpp
	g++ transC.cpp -o transC $(COMPILE_ARG)

test_link: test_link_prediction.cpp
	g++ test_link_prediction.cpp -o test_link_prediction $(COMPILE_ARG)

test_classification_normal: test_classification_normal.cpp
	g++ test_classification_normal.cpp -o test_classification_normal $(COMPILE_ARG)

test_classification_isA: test_classification_isA.cpp
	g++ test_classification_isA.cpp -o test_classification_isA $(COMPILE_ARG)

