# KdTree_Traversal

GPU Traversal of KdTree for N-body problems

1.Reference

The kdtree construction method and CPU traversal of kdTree is downloaded from http://nghiaho.com/?p=437 for comparation.

The correctness of the original code is tested by the ANN library. So I assume the result is correct. Then I removed the ANN library and add my autorope traversal method and compare the result, and the result is correct. Since it uses same dataset every time, the result of my implementation is also correct. Then I delete the original traversal and computation function and added my lockstep traversal method, and compare it with the result from autorope. The results are also same.

2.Files and Functions

Main.cu: handle input files and points sizes, call kdtree build function, traversal functions, calculate time for each method(lockstep and autorope).                

CUDA_KDtree.cu: Lockstep and autorope traversal for KNN. Kernal launch functions, memory copy function, kdtree transfer to gpu function, distance computation function. Local stack, can only support for up to 50000 data points, but no limit for data points as long as the GPU global memory is enough. 

SearchAtNode function: the main traversal function
Search function: extra function, will be deleted
Search Batch: Kernal function of traversal
CUDA_KDTree::CreateKDTree: copy the content of the KDtree created by main function, and then memcpy that to GPU pointer
CUDA_KDTree::Search:  handle memory allocate, kernal launch, memory copy and transfer back, and free.

CUDA_KDtree1.cu: Lockstep and autorope traversal for KNN. Stack with global memory, significantly slow, but can support large number of data points. Functions similar with CUDA_KDtree.cu

KDtree.cu: the kdtree layout and build functions on CPU. 

Timers.cpp: the timer function used

KdTree traversal implementation based on Michael Goldfarb's paper: General Transformations for GPU Execution of Tree Traversals


The result is from 50k points, 100k query points for 5 nearest neighbor.
 
Result of local stack

![capture](https://user-images.githubusercontent.com/18172104/30877497-c527e516-a2ae-11e7-9041-8e020d1ed820.JPG)
 
Result of Global memory stack

![capture1](https://user-images.githubusercontent.com/18172104/30877503-ca4d8adc-a2ae-11e7-9431-dfd36480d2e8.JPG)





Tested Result for Autorope and Lockstep Traversal(KNN)

The input data is from dataset directory

![knn_10000](https://user-images.githubusercontent.com/18172104/30877510-ce45a2dc-a2ae-11e7-9b5b-787e904653c1.JPG)
![knn_10000](https://user-images.githubusercontent.com/18172104/30877543-dcfcd3ae-a2ae-11e7-885f-4b3325d30f08.JPG)
![knn_10000](https://user-images.githubusercontent.com/18172104/30877549-dff803f8-a2ae-11e7-8681-70f264c47699.JPG)

Tested code is in GPU_tests folder(compile with CUDA_KDTree_KNN)


Result Compare with CPU(NN using ANN library)

![capture](https://user-images.githubusercontent.com/18172104/30877576-eff452b6-a2ae-11e7-9a67-75201e145289.PNG)

Tested code is in NN_Compare folder




Graph of compute time(Autorope, Lockstep, ANN)

Tested on G2-D3-1000000.csv



![capture-2](https://user-images.githubusercontent.com/18172104/30877580-f3427286-a2ae-11e7-8880-b17e850fd5ad.JPG)



Compare results for real_dataset:
5 nearest neighbor

file:IHEPC.csv

N: 2075259

autorope: 12.7538 s

lockstep: 13.4201 s

kNN_kdtree: 114.39 s


