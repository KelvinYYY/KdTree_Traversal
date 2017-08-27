#include <cstdio>
#include <vector>
#include <cstdlib>
#include <float.h>
#include <stdint.h> // portable: uint64_t   MSVC: __int64 
#include <sys/time.h>
#include <iostream>
#include <string>
#include <sstream>
#include "KDtree.h"
#include "CUDA_KDtree_KNN.h"
#include <fstream>
#define DATA_SIZE 50000

double TimeDiff(timeval t1, timeval t2);
double SearchCPU(const vector <Point> &query, const vector <Point> &data, vector <int> &idxs, vector <float> dist_sq);

int main(int argc, char *argv[])
{
	//KDTREE_Dim of the program is a predefined variable in KDtree.h, change it according to the dimension of the data
	//K_SIZE is also in KDtree.h, change it according to how many nearest neighboor is needed, it is kept as fixed size since dynamic array will decrease the performance a lot
        
	KDtree tree;
	CUDA_KDTree GPU_tree;
	timeval t1, t2;
	int max_tree_levels = 13; 
	int size = atoi(argv[1]);
        int k = atoi(argv[2]);
	//DATA_SIZE is the size of points in the tree, maximum is 50000 since it is related to the local stack;
	vector <Point> data(DATA_SIZE);
	vector <Point> queries(size);
	vector <int> gpu_indexes, cpu_indexes;
	vector <float> gpu_dists, cpu_dists;
	std::ifstream file("datasets/HIGGS.csv");
	std::string str;
	int count = 0;
	int j = 0 ;
	while(std::getline(file, str)&& count<DATA_SIZE){
		std::istringstream ss(str);
		std::string token;
		while(std::getline(ss, token, ',')){
			data[count].coords[j++] = atof(token.c_str());
			j%=KDTREE_DIM;
		}
		count++;
		
	}
	count = 0;
	j = 0;
	while(std::getline(file, str)&& count<size){
		std::istringstream ss(str);
		std::string token;
		while(std::getline(ss, token, ',')){
			queries[count].coords[j++] = atoi(token.c_str());
			j%=KDTREE_DIM; 
		}
		count++;
	}
	/*
	for (unsigned int i = 0; i < data.size(); i++) {
		data[i].coords[0] = 0 + 10000.0*(rand() / (1.0 + RAND_MAX));
		data[i].coords[1] = 0 + 10000.0*(rand() / (1.0 + RAND_MAX));
		data[i].coords[2] = 0 + 10000.0*(rand() / (1.0 + RAND_MAX));
	}

	for (unsigned int i = 0; i < queries.size(); i++) {
		queries[i].coords[0] = 0 + 10000.0*(rand() / (1.0 + RAND_MAX));
		queries[i].coords[1] = 0 + 10000.0*(rand() / (1.0 + RAND_MAX));
		queries[i].coords[2] = 0 + 10000.0*(rand() / (1.0 + RAND_MAX));
	}*/
	
	// Time to create the tree 
	gettimeofday(&t1, NULL);
	tree.Create(data, max_tree_levels);
	GPU_tree.CreateKDTree(tree.GetRoot(), tree.GetNumNodes(), data);
	gettimeofday(&t2, NULL);
    	double gpu_create_time = TimeDiff(t1,t2);

    // Time to search the tree
    	gettimeofday(&t1, NULL);
    	GPU_tree.Search(queries, gpu_indexes, gpu_dists,0,k); //change to mode 0 to use autorope
    	gettimeofday(&t2, NULL);
    	double gpu_search_time = TimeDiff(t1,t2);

	gettimeofday(&t1, NULL);
	GPU_tree.Search(queries, cpu_indexes, cpu_dists, 1,k); //change to mode 1 to use lockstep
	gettimeofday(&t2, NULL);
    	double	Lock_Time = TimeDiff(t1, t2);


    // Verify results
	if(gpu_indexes.size()!= cpu_indexes.size()){
		printf("error");
		return 1;
}
	int i;	
    for(i=0; i< gpu_indexes.size(); i++) {
        if(gpu_dists[i] != cpu_dists[i]) {
            printf("Resuts not the same :(\n");
            printf("%d   %d != %d\n", i,  gpu_indexes[i], cpu_indexes[i]);
			printf("querydata = %f data1 = %f data2 = %f\n", queries[i/k].coords[0], queries[i/k].coords[1], queries[i/k].coords[2], data[gpu_indexes[i]].coords[0],
				data[gpu_indexes[i]].coords[1], data[gpu_indexes[i]].coords[2], data[cpu_indexes[i]].coords[0], data[cpu_indexes[i]].coords[1], data[cpu_indexes[i]].coords[2]);
            printf("%f %f\n", gpu_dists[i], cpu_dists[i]);
            return 1;
	}
    }
	
    printf("Points in the tree: %ld\n", data.size());
    printf("Query points: %ld\n", queries.size());
    printf("\n");

    printf("Results are the same!\n");

    printf("\n");

    printf("GPU max tree depth: %d\n", max_tree_levels);
    printf("AutoRopeTraversal For %dNN: %g ms\n", k,gpu_search_time);
    printf("LockstepTraversal For %dNN: %g ms\n",k, Lock_Time);
   

    return 0;
}

double TimeDiff(timeval t1, timeval t2)
{
    double t;
    t = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    t += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms

    return t;
}

double SearchCPU(const vector <Point> &queries, const vector <Point> &data, vector <int> &idxs, vector <float> dist_sq)
{
    timeval t1, t2;

    int query_pts = queries.size();
    int data_pts = data.size();

    idxs.resize(query_pts);
    dist_sq.resize(query_pts);

    gettimeofday(&t1, NULL);
    for(unsigned int i=0; i < query_pts; i++) {
        float best_dist = FLT_MAX;
        int best_idx = 0;

        for(unsigned int j=0; j < data_pts; j++) {
            float dist_sq = 0;

            for(int k=0; k < KDTREE_DIM; k++) {
                float d = queries[i].coords[k] - data[j].coords[k];
                dist_sq += d*d;
            }

            if(dist_sq < best_dist) {
                best_dist = dist_sq;
                best_idx = j;
            }
        }

        idxs[i] = best_idx;
        dist_sq[i] = best_dist;
    }

    gettimeofday(&t2, NULL);

    return TimeDiff(t1,t2);
}

/*
void SearchANN(const vector <Point> &queries, const vector <Point> &data, vector <int> &idxs, vector <float> dist_sq, double &create_time, double &search_time)
{
    int k = 1;
    timeval t1, t2;

    idxs.resize(queries.size());
    dist_sq.resize(queries.size());

    ANNidxArray nnIdx = new ANNidx[k];
    ANNdistArray dists = new ANNdist[k];
    ANNpoint queryPt = annAllocPt(KDTREE_DIM);

    ANNpointArray dataPts = annAllocPts(data.size(), KDTREE_DIM);

    for(unsigned int i=0; i < data.size(); i++) {
        for(int j=0; j < KDTREE_DIM; j++ ) {
            dataPts[i][j] = data[i].coords[j];
        }
    }

    gettimeofday(&t1, NULL);
    ANNkd_tree* kdTree = new ANNkd_tree(dataPts, data.size(), KDTREE_DIM);
    gettimeofday(&t2, NULL);
    create_time = TimeDiff(t1,t2);

    gettimeofday(&t1, NULL);
    for(int i=0; i < queries.size(); i++) {
        for(int j=0; j < KDTREE_DIM; j++) {
            queryPt[j] = queries[i].coords[j];
        }

        kdTree->annkSearch(queryPt, 1, nnIdx, dists);

        idxs[i] = nnIdx[0];
        dist_sq[i] = dists[0];
    }
    gettimeofday(&t2, NULL);
    search_time = TimeDiff(t1,t2);

	delete [] nnIdx;
	delete [] dists;
	delete kdTree;
	annDeallocPts(dataPts);
	annClose();
}
*/
