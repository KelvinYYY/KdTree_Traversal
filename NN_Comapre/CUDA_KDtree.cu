


#include "CUDA_KDtree.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <cstdio>
#include "device_launch_parameters.h"
#include <iostream>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "math.h"
#define STACK 40000
#define CUDA_STACK 10000 // fixed size stack elements for each thread, increase as required. Used in SearchAtNodeRange.
__device__ void push(int a, float d);
__device__ int pop();
__shared__ unsigned int mask[512 / 32][32];
__device__ unsigned int warp_and(unsigned int warp_mask, int warpid, int maskid);
struct Stack
{
	struct node
	{
		float dsq;
		int kdnode;
		struct node *next;
	};
	struct node *top;
	public:
	int size;
	__device__ Stack(){
		size = 0;
	};
	__device__ ~Stack();
	__device__ void push(int a, float d){
		struct node *ptr;
		ptr = new node;
		ptr->dsq = d;
		ptr->kdnode = a;
		if (top != NULL)
			ptr->next = top;
		top = ptr;
		size++;
	} // to insert an element
	__device__ int pop(){
		struct node *temp;
		if (top == NULL)
		{
			return;
		}
		temp = top;
		top = top->next;
		size--;
		return temp->kdnode;
	}  // to delete an element
};
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
    float dist = 0;

    for(int i=0; i < KDTREE_DIM; i++) {
        float d = a.coords[i] - b.coords[i];
        dist += d*d;
    }

    return dist;
}
__device__ unsigned int power(int a){
	unsigned int result=1;
	for (int i = 0; i < a; i++){
		result = result * 2;
	}
	return result;
}
__device__ void SearchAtNode(const CUDA_KDNode *nodes, const int *indexes, const Point *pts, int cur, const Point &query, int *ret_index, float *ret_dist, int *ret_node, int mode)
{
    // Finds the first potential candidate
	//query is the current point
	//nodes are the treenodes
	//query is the point
	
    int best_idx = 0;
    float best_dist = FLT_MAX;
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int s1[STACK];
	if (mode == 0){
		while (true) {
			int split_axis = nodes[cur].level % KDTREE_DIM;

			if (nodes[cur].left == -1) {
				*ret_node = cur;

				for (int i = 0; i < nodes[cur].num_indexes; i++) {
					int idx = indexes[nodes[cur].indexes + i];
					float dist = Distance(query, pts[idx]);
					if (dist < best_dist) {
						best_dist = dist;
						best_idx = idx;
					}
				}

				break;
			}
			else if (query.coords[split_axis] < nodes[cur].split_value) {
				cur = nodes[cur].left;
			}
			else {
				cur = nodes[cur].right;
			}
		}
	}
	if( mode==2){
		
		s1[0] = 0;
		int top1 = 0;
		int cur = 0;
		int split_axis,a,b,i;
		while (top1 > -1){
			cur = s1[top1];
			top1--;

			int split_axis = nodes[cur].level % KDTREE_DIM;
			float d = nodes[cur].split_value - query.coords[nodes[cur].level % KDTREE_DIM];

			if (nodes[cur].left == -1){//if it is leaf, update best distance
				*ret_node = cur;

				for (int i = 0; i < nodes[cur].num_indexes; i++) {
					int idx = indexes[nodes[cur].indexes + i];
					float dist = Distance(query, pts[idx]);
					if (dist < best_dist) {
						best_dist = dist;
						best_idx = idx;
					}
				}

				continue;
			}
			else if (query.coords[split_axis] < nodes[cur].split_value){  //closer to left
				
				if ((query.coords[split_axis] - nodes[cur].split_value)*(query.coords[split_axis] - nodes[cur].split_value) <= best_dist){
					if (nodes[cur].right != -1){
						s1[++top1] = nodes[cur].right;

					}
				}
				s1[++top1] = nodes[cur].left;
				
			}
			else{

				if ((query.coords[split_axis] - nodes[cur].split_value)*(query.coords[split_axis] - nodes[cur].split_value) <= best_dist){
					s1[++top1] = nodes[cur].left;
				}
				if (nodes[cur].right != -1){
					s1[++top1] = nodes[cur].right;
				}
				
			}
		}

	}
	if (mode == 1){
		/*
		s1[0] = 0;
		s2[0] = ~0;
		while (top1!=-1){
			cur = s1[top1];
			top1--;
			int mask = s2[top2];
			top2--;
			int split_axis = nodes[cur].level % KDTREE_DIM;
			float d = nodes[cur].split_value - query.coords[nodes[cur].level % KDTREE_DIM];

			i
			if (nodes[cur].left == -1){//if it is leaf, update best distance
				*ret_node = cur;

				for (int i = 0; i < nodes[cur].num_indexes; i++) {
					int idx = indexes[nodes[cur].indexes + i];
					float dist = Distance(query, pts[idx]);
					if (dist < best_dist) {
						best_dist = dist;
						best_idx = idx;
					}
				}

				continue;
			}
			else if (query.coords[split_axis] < nodes[cur].split_value){  //closer to left
				
				if ((query.coords[split_axis] - nodes[cur].split_value)*(query.coords[split_axis] - nodes[cur].split_value) <= best_dist){
					if (nodes[cur].right != -1){
						s1[++top1] = nodes[cur].right;
						s2[++top2] = 1; //closer to right
					}
				}
				else{
					//trancated
				}
				s1[++top1] = nodes[cur].left;
				s2[++top2] = 1;
				
			}
			else{

				if ((query.coords[split_axis] - nodes[cur].split_value)*(query.coords[split_axis] - nodes[cur].split_value) <= best_dist){
					s1[++top1] = nodes[cur].left;
					s2[++top2] = 1;
				}
				if (nodes[cur].right != -1){
					s1[++top1] = nodes[cur].right;
					s2[++top2] = 1; //closer to right
				}
				
			}
		}*/
		/*
		unsigned int s1[25000];
		unsigned int s2[25000];
		int pid;
		int top1 = 0;
		int top2 = 0;
		int idx = blockIdx.x*blockDim.x + threadIdx.x;
		int warpid = threadIdx.x % 32;
		int cur = 0;
		int maskid = threadIdx.x / 32;
		unsigned int warp_mask;	
//		mask[maskid][warpid] = 1;
		s1[0] = 0;
		s2[0] = 1;
		if (idx == 1){
			printf("%ud\n", s2[0]);
		}
		while (top1 > -1 && top2>-1){
			cur = s1[top1];
			top1--;
			warp_mask = s2[top2];
			top2--;
			//mask[maskid][warpid] = (warp_mask >> warpid & 1);
			int split_axis = nodes[cur].level % KDTREE_DIM;
			//float d = nodes[cur].split_value - queries[idx].coords[nodes[cur].level % KDTREE_DIM];

			if (warp_mask== 1){
				//if thread in warp is active
							if (nodes[cur].left == -1){
					//if it is leaf, update correlation
					for (int i = 0; i < nodes[cur].num_indexes; i++) {
						int id = indexes[nodes[cur].indexes + i];
						float dist = Distance(query, pts[id]);
						if (dist < best_dist) {
							
							best_dist = dist;
							best_idx = id;
						}
					}
					warp_mask = 0;
					
				 }

			}
			//combine mask from all threads in warp

			if (__any(warp_mask != 0)){
				if (nodes[cur].right != -1){
						s1[++top1] = nodes[cur].right;
						s2[++top2] = warp_mask;
				}
				s1[++top1] = nodes[cur].left;
				s2[++top2] = warp_mask;
			}
		}*/
		int top1 = 0;
		unsigned int cur = 0;
		s1[0] = 0;
		s1[0] |= 1<< 30;
		int flag = 1;
		int warp_mask;
		int split_axis,a,b,i;
		while (top1 > -1){
			cur = s1[top1];
			top1--;
			warp_mask = (cur>>30)&1;
			cur &= ~(1<<30);
			split_axis = nodes[cur].level % KDTREE_DIM;
			

			if (warp_mask== 1){
				//if thread in warp is active
				if (nodes[cur].left == -1){
					//if it is leaf, update correlation
					for (i = 0; i < nodes[cur].num_indexes; i++) {
						int id = indexes[nodes[cur].indexes + i];
						float dist = Distance(query, pts[id]);
						if (dist < best_dist) {
							
							best_dist = dist;
							best_idx = id;
						}
					}
					warp_mask = 0;
					
				 }

			}
			//combine mask from all threads in warp

			if (__any(warp_mask != 0)){
				if (nodes[cur].right != -1){
						s1[++top1] = nodes[cur].right;
						if (query.coords[split_axis] < nodes[cur].split_value && (query.coords[split_axis] - nodes[cur].split_value)*(query.coords[split_axis] - nodes[cur].split_value) > best_dist){
							//closer to left and query point distance to split axis already big enough, trancate right side.
							s1[top1] &= ~(1<<30);
						}
						else{
							s1[top1] ^= (-warp_mask ^ s1[top1] )& (1<<30);
						}
				}
				s1[++top1] = nodes[cur].left;
				if (query.coords[split_axis] > nodes[cur].split_value && (query.coords[split_axis] - nodes[cur].split_value)*(query.coords[split_axis] - nodes[cur].split_value) > best_dist){
					//if closer to right
					s1[top1] &= ~(1<<30);
				}
				else{
					s1[top1] ^= (-warp_mask^ s1[top1])&(1<<30);
				}
			}
		}
	}
	*ret_index = best_idx;
	*ret_dist = best_dist;
}
__device__ unsigned int warp_and(unsigned int warp_mask, int warpid,int maskid){
	mask[maskid][warpid] = (warp_mask >> warpid) & 1;
	int tem = 0;
	for (int i = 0; i < 32; i++){
		tem += mask[maskid][i] * power(i);
	}
	return tem;

}
__device__ void SearchAtNodeRange(const CUDA_KDNode *nodes, const int *indexes, const Point *pts, const Point &query,int cur, float range, int *ret_index, float *ret_dist)
{
    // Goes through all nodes in "range"

    int best_idx = 0;
    float best_dist = FLT_MAX;

    int to_visit[CUDA_STACK];
    int to_visit_pos = 0;

    to_visit[to_visit_pos++] = cur;

    while(to_visit_pos) {
        int next_search[CUDA_STACK];
        int next_search_pos = 0;

        while(to_visit_pos) {
            cur = to_visit[to_visit_pos-1];
            to_visit_pos--;

            int split_axis = nodes[cur].level % KDTREE_DIM;

            if(nodes[cur].left == -1) {
                for(int i=0; i < nodes[cur].num_indexes; i++) {
                    int idx = indexes[nodes[cur].indexes + i];
                    float d = Distance(query, pts[idx]);

                    if(d < best_dist) {
                        best_dist = d;
                        best_idx = idx;
                    }
                }
            }
            else {
                float d = query.coords[split_axis] - nodes[cur].split_value;

                // There are 3 possible scenarios
                // The hypercircle only intersects the left region
                // The hypercircle only intersects the right region
                // The hypercricle intersects both

                if(fabs(d) > range) {
                    if(d < 0)
                        next_search[next_search_pos++] = nodes[cur].left;
                    else
                        next_search[next_search_pos++] = nodes[cur].right;
                }
                else {
                    next_search[next_search_pos++] = nodes[cur].left;
                    next_search[next_search_pos++] = nodes[cur].right;
                }
            }
        }

        // No memcpy available??
        for(int i=0; i  < next_search_pos; i++)
            to_visit[i] = next_search[i];

        to_visit_pos = next_search_pos;
    }

    *ret_index = best_idx;
    *ret_dist = best_dist;
}


__device__ void Search(const CUDA_KDNode *nodes, const int *indexes, const Point *pts, const Point &query, int *ret_index, float *ret_dist, int mode)
{
    // Find the first closest node, this will be the upper bound for the next searches
    int best_node = 0;
    int best_idx = 0;
    float best_dist = FLT_MAX;
    float radius = 0;
    SearchAtNode(nodes, indexes, pts, 0 /* root */, query, &best_idx, &best_dist, &best_node, mode);
    if (mode !=0){
		*ret_index = best_idx;
		*ret_dist = best_dist;
		//printf("Hello from block, thread\n" );
		return;
	}

    radius = sqrt(best_dist);

    // Now find other possible candidates
    int cur = best_node;

    while(nodes[cur].parent != -1) {
        // Go up
        int parent = nodes[cur].parent;
        int split_axis = nodes[parent].level % KDTREE_DIM;

        // Search the other node
        float tmp_dist = FLT_MAX;
        int tmp_idx;

        if(fabs(nodes[parent].split_value - query.coords[split_axis]) <= radius) {
            // Search opposite node
            if(nodes[parent].left != cur)
                SearchAtNodeRange(nodes, indexes, pts, query, nodes[parent].left, radius, &tmp_idx, &tmp_dist);
            else
                SearchAtNodeRange(nodes, indexes, pts, query, nodes[parent].right, radius, &tmp_idx, &tmp_dist);
        }

        if(tmp_dist < best_dist) {
            best_dist = tmp_dist;
            best_idx = tmp_idx;
        }

        cur = parent;
    }

    *ret_index = best_idx;
    *ret_dist = best_dist;
}

__global__ void SearchBatch(const CUDA_KDNode *nodes, const int *indexes, const Point *pts, int num_pts, Point *queries, int num_queries, int *ret_index, float *ret_dist, int mode)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	

    if(idx >= num_queries)
        return;

    Search(nodes, indexes, pts, queries[idx], &ret_index[idx], &ret_dist[idx], mode);
}



	


__device__ CUDA_KDTree::~CUDA_KDTree()
{
    cudaFree(m_gpu_nodes);
    cudaFree(m_gpu_indexes);
    cudaFree(m_gpu_points);
}

void CUDA_KDTree::CreateKDTree(KDNode *root, int num_nodes, const vector <Point> &data)
{
    // Create the nodes again on the CPU, laid out nicely for the GPU transfer
    // Not exactly memory efficient, since we're creating the entire tree again
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

    while(to_visit.size()) {
        vector <KDNode*> next_search;

        while(to_visit.size()) {
            KDNode *cur = to_visit.back();
            to_visit.pop_back();

            int id = cur->id;

            cpu_nodes[id].level = cur->level;
            cpu_nodes[id].parent = cur->_parent;
            cpu_nodes[id].left = cur->_left;
            cpu_nodes[id].right = cur->_right;
            cpu_nodes[id].split_value = cur->split_value;
            cpu_nodes[id].num_indexes = cur->indexes.size();

            if(cur->indexes.size()) {
                for(unsigned int i=0; i < cur->indexes.size(); i++)
                    indexes[cur_pos+i] = cur->indexes[i];

                cpu_nodes[id].indexes = cur_pos;
                cur_pos += cur->indexes.size();
            }


            else {
                cpu_nodes[id].indexes = -1;
            }

            if(cur->left)
                next_search.push_back(cur->left);

            if(cur->right)
                next_search.push_back(cur->right);
        }

        to_visit = next_search;
    }

    cudaMemcpy(m_gpu_nodes, &cpu_nodes[0], sizeof(CUDA_KDNode)*cpu_nodes.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(m_gpu_indexes, &indexes[0], sizeof(int)*indexes.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(m_gpu_points, &data[0], sizeof(Point)*data.size(), cudaMemcpyHostToDevice);

    CheckCUDAError("CreateKDTree");
}

void CUDA_KDTree::Search(const vector <Point> &queries, vector <int> &indexes, vector <float> &dists, int mode)
{
	int threads;

	threads = 512;

    int blocks = queries.size()/threads + ((queries.size() % threads)?1:0);

    Point *gpu_queries;
    int *gpu_ret_indexes;
    float *gpu_ret_dist;

    indexes.resize(queries.size());
    dists.resize(queries.size());

    cudaMalloc((void**)&gpu_queries, sizeof(Point)*queries.size()*KDTREE_DIM);
    cudaMalloc((void**)&gpu_ret_indexes, sizeof(int)*queries.size()*KDTREE_DIM);
    cudaMalloc((void**)&gpu_ret_dist, sizeof(float)*queries.size()*KDTREE_DIM);

    CheckCUDAError("Search");

    cudaMemcpy(gpu_queries, &queries[0], sizeof(float)*queries.size()*KDTREE_DIM, cudaMemcpyHostToDevice);

    CheckCUDAError("Search");

    printf("CUDA blocks/threads: %d %d\n", blocks, threads);

		SearchBatch<<<blocks, threads >>>(m_gpu_nodes, m_gpu_indexes, m_gpu_points, m_num_points, gpu_queries, queries.size(), gpu_ret_indexes, gpu_ret_dist,mode);


    cudaThreadSynchronize();
    CheckCUDAError("Search");

    cudaMemcpy(&indexes[0], gpu_ret_indexes, sizeof(int)*queries.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(&dists[0], gpu_ret_dist, sizeof(float)*queries.size(), cudaMemcpyDeviceToHost);

    cudaFree(gpu_queries);
    cudaFree(gpu_ret_indexes);
    cudaFree(gpu_ret_dist);
}
