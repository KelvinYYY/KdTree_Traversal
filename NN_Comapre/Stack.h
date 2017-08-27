#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif
#include "device_launch_parameters.h"
#include <iostream>
#include "CUDA_KDtree.h"
#include <cuda.h>
#include <cuda_runtime.h>
struct node
{
	float dsq;
	struct CUDA_KDNode kdnode;
	struct node *next;
};
#pragma once
class Stack
{
	
	struct node *top;
public:
	int size;
	CUDA_HOSTDEV Stack();
	CUDA_HOSTDEV ~Stack();
	CUDA_HOSTDEV void push(const CUDA_KDNode a, float d); // to insert an element
	CUDA_HOSTDEV CUDA_KDNode pop();  // to delete an element
	CUDA_HOSTDEV void show(); // to show the stack
};

