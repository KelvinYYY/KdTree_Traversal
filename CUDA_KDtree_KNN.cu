


#include "CUDA_KDtree_KNN.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <cstdio>
#include "device_launch_parameters.h"
#include <iostream>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "math.h"
#define CUDA_STACK 50000 // fixed size stack elements for each thread, increase as required. Used in SearchAtNodeRange.
#define STACK_SIZE 50000
void CheckCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}


__device__ float Distance(const Point &a, const Point &b)
{
	//function to compute distance between two points
    float dist = 0;

    for(int i=0; i < KDTREE_DIM; i++) {
	
        float d = a.coords[i] - b.coords[i];
        dist += d*d;
    }

    return dist;
}



__device__ float Array_Check(int *a, float *b, int idx, float dist, int k){
	//Take a calculated distance and find its position in the array, distance array is kept in ascending order
	//Then it will return the current largest value
	//b is the array storing all the k closest distances, a is the matching index array
	if(k==1){
		*a = idx;
		*b = dist;
		return dist;
	}
	int i = 0;
						
	for (i = 0; i< k;i++){
		if(dist<*(b+i)) break;
	}
											
	float temdis;
	int temid;
	//if(i==K_SIZE){return *(b+k-1);}
	for (int j = k-1; j>=i+1;j--){
	temdis = *(b+j);
	*(b+j) = *(b+j-1);
	*(b+j-1) = temdis;
	temid = *(a+j);
	*(a+j) = *(a+j-1);
	*(a+j-1) = temid;
	}
	*(a+i) = idx;
	*(b+i) = dist;
	
	return *(b+k-1);
	
	
	
}


__device__ void Search(const CUDA_KDNode *nodes, const int *indexes, const Point *pts, const Point &query, int *ret_index, float *ret_dist, int mode, int k)
{
    //Mode 0 autorope, mode 1 lockstep
	
	//arrays to keep k nearest point index and the distances
    int idx[K_SIZE];
    float dist[K_SIZE];
    
	//initiate the array
    for (int i = 0 ; i< K_SIZE ;i++){
		dist[i] =  FLT_MAX/3;
    }
	//local stack for the kdtree points
	unsigned int s1[CUDA_STACK];

	//query is the current point
	//nodes are the treenodes
	
	if (mode == 0){
		float largest_dist = FLT_MAX/3;			
		int top1 = 0;
		int cur;
		int idxx;
		float distd;
		s1[0] = 0;
		while (top1 != -1){
			cur = s1[top1];
			top1--;
			int split_axis = nodes[cur].level % KDTREE_DIM;
			
			if (nodes[cur].left == -1){
				//if it is leaf, update largest distance and distance array 
			
				for (int i = 0; i < nodes[cur].num_indexes; i++) {
					idxx = indexes[nodes[cur].indexes + i];
					distd = Distance(query, pts[idxx]);
					if (distd < largest_dist) {
						largest_dist = Array_Check(&idx[0], &dist[0], idxx, distd, K_SIZE);
					}
				}
			        if(largest_dist ==0){
						//if largest_dist is already 0, stop searching
						break;
				}	
				continue;
			}
			else if (query.coords[split_axis] < nodes[cur].split_value){  
				//closer to left, push right node before left node

				if ((query.coords[split_axis] - nodes[cur].split_value)*(query.coords[split_axis] - nodes[cur].split_value) <= largest_dist){
					if (nodes[cur].right != -1){
						s1[++top1] = nodes[cur].right;
					}
				}
				s1[++top1] = nodes[cur].left;

			}
			else{
				//closer to right, push left node before right node
				if ((query.coords[split_axis] - nodes[cur].split_value)*(query.coords[split_axis] - nodes[cur].split_value) <= largest_dist){
					s1[++top1] = nodes[cur].left;
				}
				if (nodes[cur].right != -1){
					s1[++top1] = nodes[cur].right;
				}

			}
		}
	}
	if (mode != 0){	
		unsigned int cur = 0;
		float largest_dist = FLT_MAX/10;	
		int top1 = 0;
		int warp_mask;
		int split_axis,i;
		int idxx;
		float distd;
		s1[0] = 0;
		s1[0] |= 1<< 30;
		while (top1 > -1) {
			cur = s1[top1];
			top1--;
			warp_mask = (cur>>30)&1;
			cur &= ~(1<<30);
			split_axis = nodes[cur].level % KDTREE_DIM;
			
			if (warp_mask== 1) {
				//if thread in warp is active
				if (nodes[cur].left == -1){
				//if it is leaf, update correlation
					for (i = 0; i < nodes[cur].num_indexes; i++) {
						idxx = indexes[nodes[cur].indexes + i];
						distd = Distance(query, pts[idxx]);		
						if (distd < largest_dist) {				
							largest_dist = Array_Check(&idx[0], &dist[0], idxx, distd, K_SIZE);
						}
					}
					warp_mask = 0;
					continue;
				}
				if (largest_dist ==0) {
					break;
				}

			}
			//combine mask from all threads in warp
			
			if (__any(warp_mask != 0)) {
				if (nodes[cur].right != -1) {
						s1[++top1] = nodes[cur].right;
						if (query.coords[split_axis] < nodes[cur].split_value && (query.coords[split_axis] - nodes[cur].split_value)*(query.coords[split_axis] - nodes[cur].split_value) > largest_dist){
							//closer to left and query point distance to split axis already big enough, trancate right side.
							s1[top1] &= ~(1<<30);
						}
						else{
							s1[top1] ^= (-warp_mask ^ s1[top1] )& (1<<30);
						}
				}
				
				s1[++top1] = nodes[cur].left;
				if (query.coords[split_axis] > nodes[cur].split_value && (query.coords[split_axis] - nodes[cur].split_value)*(query.coords[split_axis] - nodes[cur].split_value) > largest_dist) {
					//if closer to right
					s1[top1] &= ~(1<<30);
				}
				else{
					s1[top1] ^= (-warp_mask^ s1[top1])&(1<<30);
				}
			}
		}
	}//mode!=0 end bracket
	
	for (int i = 0 ; i< K_SIZE;i++) {
    		*(ret_index+i) = idx[i];
    		*(ret_dist+i) = dist[i];
	}
    return;


}

__global__ void SearchBatch(const CUDA_KDNode *nodes, const int *indexes, const Point *pts, int num_pts, Point *queries, int num_queries, int *ret_index, float *ret_dist, int mode, int k)
{
    //assign job based on thread id, each thread one query point
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	

    if(idx >= num_queries)
        return;

    Search(nodes, indexes, pts, queries[idx], &ret_index[idx*K_SIZE], &ret_dist[idx*K_SIZE], mode, k);
}
	


__device__ CUDA_KDTree::~CUDA_KDTree()
{
    cudaFree(m_gpu_nodes);
    cudaFree(m_gpu_indexes);
    cudaFree(m_gpu_points);
}

void CUDA_KDTree::CreateKDTree(KDNode *root, int num_nodes, const vector <Point> &data)
{
    // Copy the entire tree from CPU to GPU 
	// Create the nodes again on the CPU, laid out nicely for the GPU transfer
    
    m_num_points = data.size();

    cudaMalloc((void**)&m_gpu_nodes, sizeof(CUDA_KDNode)*num_nodes);
    cudaMalloc((void**)&m_gpu_indexes, sizeof(int)*m_num_points);
    cudaMalloc((void**)&m_gpu_points, sizeof(Point)*m_num_points);
    CheckCUDAError("CreateKDTree");

    vector <CUDA_KDNode> cpu_nodes(num_nodes);
    vector <int> indexes(m_num_points);
    vector <KDNode*> to_visit;

    int cur_pos = 0;

    to_visit.push_back(root);

    while (to_visit.size()) {
        vector <KDNode*> next_search;

        while (to_visit.size()) {
            KDNode *cur = to_visit.back();
            to_visit.pop_back();

            int id = cur->id;

            cpu_nodes[id].level = cur->level;
            cpu_nodes[id].parent = cur->_parent;
            cpu_nodes[id].left = cur->_left;
            cpu_nodes[id].right = cur->_right;
            cpu_nodes[id].split_value = cur->split_value;
            cpu_nodes[id].num_indexes = cur->indexes.size();

            if (cur->indexes.size()) {
                for(unsigned int i=0; i < cur->indexes.size(); i++)
                    indexes[cur_pos+i] = cur->indexes[i];

                cpu_nodes[id].indexes = cur_pos;
                cur_pos += cur->indexes.size();
            }


            else {
                cpu_nodes[id].indexes = -1;
            }

            if (cur->left)
                next_search.push_back(cur->left);

            if (cur->right)
                next_search.push_back(cur->right);
        }

        to_visit = next_search;
    }

    cudaMemcpy(m_gpu_nodes, &cpu_nodes[0], sizeof(CUDA_KDNode)*cpu_nodes.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(m_gpu_indexes, &indexes[0], sizeof(int)*indexes.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(m_gpu_points, &data[0], sizeof(Point)*data.size(), cudaMemcpyHostToDevice);
 
    CheckCUDAError("CreateKDTree");
}

void CUDA_KDTree::Search(const vector <Point> &queries, vector <int> &indexes, vector <float> &dists, int mode, int k)
{	
	int threads;
    threads = 512;
    int blocks = queries.size()/threads + ((queries.size() % threads)?1:0);
    Point *gpu_queries;
    int *gpu_ret_indexes;
    float *gpu_ret_dist;
	
    indexes.resize(queries.size()*K_SIZE);
    dists.resize(queries.size()*K_SIZE);
	
	//allocate memory on GPU
    cudaMalloc((void**)&gpu_queries, sizeof(Point)*queries.size()*KDTREE_DIM);
    cudaMalloc((void**)&gpu_ret_indexes, sizeof(int)*queries.size()*KDTREE_DIM*K_SIZE);
    cudaMalloc((void**)&gpu_ret_dist, sizeof(float)*queries.size()*KDTREE_DIM*K_SIZE);
    CheckCUDAError("Search");

	//Copy query points to GPU, no need to copy data since we already copied the kdtree
    cudaMemcpy(gpu_queries, &queries[0], sizeof(float)*queries.size()*KDTREE_DIM, cudaMemcpyHostToDevice);
    CheckCUDAError("Search");
    printf("CUDA blocks/threads: %d %d\n", blocks, threads);
	
	//launch kernal 
    SearchBatch<<<blocks, threads>>>(m_gpu_nodes, m_gpu_indexes, m_gpu_points, m_num_points, gpu_queries, queries.size(), gpu_ret_indexes, gpu_ret_dist, mode,k);
	
    cudaThreadSynchronize();
    CheckCUDAError("Search");
	
	//copy result to CPU
    cudaMemcpy(&indexes[0], gpu_ret_indexes, sizeof(int)*queries.size()*K_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(&dists[0], gpu_ret_dist, sizeof(float)*queries.size()*K_SIZE, cudaMemcpyDeviceToHost);

    cudaFree(gpu_queries);
    cudaFree(gpu_ret_indexes);
    cudaFree(gpu_ret_dist);
   

}
