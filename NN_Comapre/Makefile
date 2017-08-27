CUDA_KDtree: KDtree.o CUDA_KDtree.o main.o
	nvcc -O2 --library-path=<./ANN> --library=ANN -I./ANN/include -L./ANN/lib -lANN -o CUDA_KDtree main.o KDtree.o CUDA_KDtree.o 

main.o: main.cu
	nvcc -O2 -c main.cu

KDtree.o: KDtree.cu
	nvcc -O2 -c KDtree.cu

CUDA_KDtree.o: CUDA_KDtree.cu
	nvcc -O2 -c CUDA_KDtree.cu

clean:
	rm -f *.o CUDA_KDtree
