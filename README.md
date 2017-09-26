# KdTree_Traversal

GPU Traversal of KdTree for N-body problems


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

![knn_10000](https://user-images.githubusercontent.com/18172104/30877549-dff803f8-a2ae-11e7-8681-70f264c47699.JPG)

Tested code is in NN_Compare folder




Graph of compute time(Autorope, Lockstep, ANN)

Tested on G2-D3-1000000.csv

![knn_10000](https://user-images.githubusercontent.com/18172104/30877580-f3427286-a2ae-11e7-8880-b17e850fd5ad.JPG)


![capture-2](https://user-images.githubusercontent.com/18172104/30877580-f3427286-a2ae-11e7-8880-b17e850fd5ad.JPG)



Compare results for real_dataset:
5 nearest neighbor

file:IHEPC.csv

N: 2075259

autorope: 12.7538 s

lockstep: 13.4201 s

kNN_kdtree: 114.39 s


