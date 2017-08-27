#include <cstdio>
#include <vector>
#include <cstdlib>
#include <float.h>
#include <stdint.h> // portable: uint64_t   MSVC: __int64 
#include <sys/time.h>
#include "KDtree.h"
#include "CUDA_KDtree.h"
#include "ANN/include/ANN/ANN.h"
#include <algorithm>
void SearchANN(const vector <Point> &queries, const vector <Point> &data, vector <int> &idxs, vector <float> dist_sq, double &create_time, double &search_time);
double SearchCPU(const vector <Point> &query, const vector <Point> &data, vector <int> &idxs, vector <float> dist_sq);
double TimeDiff(timeval t1, timeval t2);


int main(int argc, char *argv[])
{
	KDtree tree;
	CUDA_KDTree GPU_tree;
	timeval t1, t2;
	int max_tree_levels = 13; // play around with this value to get the best result
	int size = atoi(argv[1]);
	vector <Point> data(50000);
	vector <Point> queries(size);

	vector <int>   auto_indexes, gpu_indexes, cpu_indexes, ori_indexes;
	vector <float> auto_dists  , gpu_dists  , cpu_dists  , ori_dists  ;

	for (unsigned int i = 0; i < data.size(); i++) {
		data[i].coords[0] = 0 + 10000.0*(rand() / (1.0 + RAND_MAX));
		data[i].coords[1] = 0 + 10000.0*(rand() / (1.0 + RAND_MAX));
		data[i].coords[2] = 0 + 10000.0*(rand() / (1.0 + RAND_MAX));
	}

	for (unsigned int i = 0; i < queries.size(); i++) {
		queries[i].coords[0] = 0 + 10000.0*(rand() / (1.0 + RAND_MAX));
		queries[i].coords[1] = 0 + 10000.0*(rand() / (1.0 + RAND_MAX));
		queries[i].coords[2] = 0 + 10000.0*(rand() / (1.0 + RAND_MAX));
	}
	// Time to create the tree 
	gettimeofday(&t1, NULL);
	tree.Create(data, max_tree_levels);
	GPU_tree.CreateKDTree(tree.GetRoot(), tree.GetNumNodes(), data);
	gettimeofday(&t2, NULL);
    double gpu_create_time = TimeDiff(t1,t2);


    gettimeofday(&t1, NULL);
    GPU_tree.Search(queries, auto_indexes, auto_dists,2);
    gettimeofday(&t2, NULL);
    double  auto_search_time = TimeDiff(t1,t2);
    // Time to search the tree
    gettimeofday(&t1, NULL);
    GPU_tree.Search(queries, gpu_indexes, gpu_dists,1);
    gettimeofday(&t2, NULL);
    double gpu_search_time = TimeDiff(t1,t2);

    double ANN_create_time;
    double ANN_search_time;
	gettimeofday(&t1, NULL);
   	SearchANN(queries, data, cpu_indexes, cpu_dists, ANN_create_time, ANN_search_time);
	gettimeofday(&t2, NULL);
	ANN_search_time = TimeDiff(t1, t2);
	
	
    //double Orig_create_time;
    double Orig_search_time;
	gettimeofday(&t1, NULL);
	GPU_tree.Search(queries, ori_indexes, ori_dists, 0); //change to mode 1 to run new search
	gettimeofday(&t2, NULL);
	Orig_search_time = TimeDiff(t1, t2);
	
    // Verify results
	
    for(unsigned int i=0; i< gpu_indexes.size(); i++) {
		/*if (i == 1){
			printf("%d != %d\n", gpu_indexes[i], cpu_indexes[i]);
			printf("data = %f,  %f,  %f data1 = %f, %f %f data2 = %f  %f  %f\n", data[i].coords[0], data[i].coords[1], data[i].coords[2], data[gpu_indexes[i]].coords[0],
				data[gpu_indexes[i]].coords[1], data[gpu_indexes[i]].coords[2], data[cpu_indexes[i]].coords[0], data[cpu_indexes[i]].coords[1], data[cpu_indexes[i]].coords[2]);
			printf("%f %f\n", gpu_dists[i], cpu_dists[i]);
		}*/
		
        if(gpu_dists[i] != ori_dists[i]) {
            printf("Resuts not the same :(\n");
          /*  printf("%d   %d != %d\n", i,  gpu_indexes[i], cpu_indexes[i]);
			printf("querydata = %f,  %f,  %f data1 = %f, %f %f data2 = %f  %f  %f\n", queries[i].coords[0], queries[i].coords[1], queries[i].coords[2], data[gpu_indexes[i]].coords[0],
				data[gpu_indexes[i]].coords[1], data[gpu_indexes[i]].coords[2], data[cpu_indexes[i]].coords[0], data[cpu_indexes[i]].coords[1], data[cpu_indexes[i]].coords[2]);
            printf("%f %f\n", gpu_dists[i], cpu_dists[i]);*/
            return 1;
        }
    }
	
    printf("Points in the tree: %ld\n", data.size());
    printf("Query points: %ld\n", queries.size());
    printf("\n");

    printf("Results are the same!\n");

    printf("\n");

    printf("GPU max tree depth: %d\n", max_tree_levels);
	printf("autorope create + search: %g + %g = %g ms\n", gpu_create_time, auto_search_time, gpu_create_time + auto_search_time);
	printf("lockstep create + search: %g + %g = %g ms\n", gpu_create_time, gpu_search_time, gpu_create_time + gpu_search_time);
	printf("ANN create      + search: %g + %g = %g ms\n", gpu_create_time, ANN_search_time, gpu_create_time + ANN_search_time);
    printf(    "Autorope speed up for NN:    %f\n", ANN_search_time/auto_search_time);
	
	

    //printf("Speed up of GPU over CPU for searches: %.2fx\n", ANN_search_time / gpu_search_time);

/*
    Point query;
    int ret_index;
    float ret_dist;

    for(int k=0; k < 100; k++) {
        query.coords[0] = 100.0*(rand() / (1.0 + RAND_MAX));
        query.coords[1] = 100.0*(rand() / (1.0 + RAND_MAX));
        query.coords[2] = 100.0*(rand() / (1.0 + RAND_MAX));

        tree.Search(query, &ret_index, &ret_dist);

        // Brute force
        float best_dist = FLT_MAX;
        int best_idx = 0;

        for(unsigned int i=0; i < data.size(); i++) {
            float dist = 0;

            for(int j=0; j < KDTREE_DIM; j++) {
                float d = data[i].coords[j] - query.coords[j];
                dist += d*d;
            }

            if(dist < best_dist) {
                best_dist = dist;
                best_idx = i;
            }
        }

        if(ret_index != best_idx) {
            printf("RESULTS NOT THE SAME :(\n");
            printf("\n");
            printf("Query: %f %f %f\n", query.coords[0], query.coords[1], query.coords[2]);
            printf("\n");
            printf("Search result %f %f %f\n", data[ret_index].coords[0], data[ret_index].coords[1],  data[ret_index].coords[2]);
            printf("Dist: %f\n", ret_dist);
            printf("IDX: %d\n", ret_index);

            printf("\n");

            printf("Ground truth: %f %f %f\n", data[best_idx].coords[0], data[best_idx].coords[1],  data[best_idx].coords[2]);
            printf("Dist: %f\n", best_dist);
            printf("IDX: %d\n", best_idx);
            exit(1);
        }
    }

*/

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
