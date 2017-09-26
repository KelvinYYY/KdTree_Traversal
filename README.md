# KdTree_Traversal

GPU Traversal of KdTree for N-body problems


KdTree traversal implementation based on Michael Goldfarb's paper: General Transformations for GPU Execution of Tree Traversals


The result is from 50k points, 100k query points for 5 nearest neighbor.
 
Result of local stack

https://gitlab.com/Nbody-Portal/Nbody-ML/uploads/2c2663237f75505a75c0693e1e901f0b/Capture.JPG
 
Result of Global memory stack

https://gitlab.com/Nbody-Portal/Nbody-ML/uploads/ebc3e3bf4b5fa1cfbf4241538cbac485/Capture1.JPG





Tested Result for Autorope and Lockstep Traversal(KNN)

The input data is from dataset directory

![knn_10000](/uploads/e66cb29db4cfd567e5f4077de0fa3e4b/knn_10000.JPG)

![knn_100000](/uploads/8806c4528543cbce94cfc99b977b2512/knn_100000.JPG)

![knn_1000000](/uploads/b28f6fcf3aa815f3fa7f0c9001f6755b/knn_1000000.JPG)

Tested code is in GPU_tests folder(compile with CUDA_KDTree_KNN)


Result Compare with CPU(NN using ANN library)

![Capture](/uploads/037b4022dd3073c02e4ea2efca15bea1/Capture.PNG)

Tested code is in NN_Compare folder




Graph of compute time(Autorope, Lockstep, ANN)

Tested on G2-D3-1000000.csv

![Capture](/uploads/f7ab7c5a0d9dc8f834fa76035eb19a60/Capture.JPG)


https://stackoverflow.com/questions/13480213/how-to-dynamically-allocate-arrays-inside-a-kernel
[IHEPC.csv_HIGGS](/uploads/f1bf2a9b761b73ee58841eb9ff271c16/IHEPC.csv_HIGGS)



Compare results for real_dataset:
5 nearest neighbor

file:IHEPC.csv

N: 2075259

autorope: 12.7538 s

lockstep: 13.4201 s

kNN_kdtree: 114.39 s


