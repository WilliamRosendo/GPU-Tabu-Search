//============================================================================
// Name        : GPU-TS.cu
// Authors     : Bruno and William
// Version     : edit 23 September 2023
// Copyright   : bruno@ic.ufal.br (Bruno Nogueira), wgpr@ic.ufal.br (William Rosendo) 
// Description : Bruno Nogueira, William Rosendo, Eduardo Tavares, Ermeson Andrade,
//               GPU tabu search: A study on using GPU to solve massive instances of the maximum diversity problem,
//               Journal of Parallel and Distributed Computing, Volume 197, 2025, 105012, ISSN 0743-7315,
//               https://doi.org/10.1016/j.jpdc.2024.105012
//               https://www.sciencedirect.com/science/article/pii/S074373152400176X
//============================================================================


// This code incorporates functions from:
// Author      : Yangming Zhou
// Version     : edit 02 December 2016
// Copyright   : zhou.yangming@yahoo.com
// Description : Yangming Zhou et al., Opposition-based memetic search for maximum diversity problem,
//               IEEE Transaction on Evolutionary Computation, 21(5):731-745, 2017.
//               https://ieeexplore.ieee.org/abstract/document/7864317


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cub/cub.cuh"
#include <assert.h>
#include <cuda.h>

#include <utility>
#include <algorithm>
#include <iomanip>
#include <random>
#include <chrono>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <vector>
#include <math.h>
#include <unistd.h>
#include <time.h>

#define abs(x)(((x) < 0) ? -(x):(x))    // calculate the absolute value of x
#define max_integer 2147483647
#define min_integer -2147483648
#define epsilon 0.000001
#define NP 2                // number of parents
#define PS 10               // population size
#define alpha 15            // tabu tenure factor
#define T 1500              // period for determining the tabu tenure
#define max_iter 50000     // number of iterations in TS
#define scale_factor 2.0    // neighborhood size should not less than 0.5
#define is_obl true
#define MAX 100
#define INF 99999

using namespace std;


// Check for errors in the use of CUDA API functions in C/C++
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Performs an atomic addition of a double value at a memory address on the GPU
static inline __device__ double atomicAdd2(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}


// running time
double start_time,limit_time;

char *dataset;
char *instance_name;        // instance file name
char instance_path[100];     // instance file path
char result_path[100];       // result file path
char statistic_path[100];    // statistical file path
char final_result_file[100];

int allocate_space;         // allocate space for initializing
int total_nbr_v;            // total number of elements
int nbr_v;                  // number of selected elements
int delta_nbr_v;            // number of remaining elements

int max_deg;
int nbr_movs;



//-------------------- DATA STRUCTURES ON CPU --------------------

double **diversity;         // diversity matrix
double max_diversity;       // maximum distance between any two items

int *improved_sol,*best_sol;
double improved_cost,best_cost;         // global best cost
double improved_time,best_time;         // time to find the global best solution
unsigned int total_iterations = 0;
unsigned int total_evaluations = 0;

int **pop;                  // population with PS individuals
double **sol_distance;          // distance between any two solutions in the population
double *pop_distance;           // the distance from the individual to population
double *pop_cost;           // f value of each individual in the population
double *pop_score;          // score of each solution in the population

int *offspring;             // offspring by crossover operator
int *opposite_sol;          // opposite offspring
int *vertex;                // indicate each item
int nbr_gen;

int nbr_edges;

int* CSR_ROWPTR;          // The rowptr vector of the CSR matrix indicates the number of neighbors for each vertex
int* CSR_COLIND;          // The colind vector of the CSR matrix stores the neighbors of each vertex.
double* CSR_VAL;          // The val vector of the CSR matrix stores the edge values between the vertex and its neighbors.



//-------------------- DATA STRUCTURES ON GPU --------------------

int *d_crrntSolution;       // current solution
int *d_bestSolution;        // best solution found
int *d_tab;             // tabu list
int *d_pos;             // position of the vertices in the current solution
double *d_edge_gain;        // wswap vector
double *d_gain;         // swap move gains
double *d_vertex_gain;      // gains of each vertex
unsigned int *d_updated;
unsigned int *updating;

int *d_CSR_R;       // GPU CSR_ROWPTR vector
int *d_CSR_COL;     // GPU CSR_COLIND vector
double *d_CSR_VAL;  // GPU CSR_VAL vector




void *d_temp_storage = NULL;
size_t temp_storage_bytes = 0;

cub::KeyValuePair <int, double> h_out;
cub::KeyValuePair <int, double> *d_out;



/*********************basic functions******************/
void calculate_rank(int index_type,int nbr,double *a,int *b)
{
    double *c;
    c = new double[nbr];
    for(int i = 0; i < nbr; i++)
        c[i] = a[i];

    int *flag;
    flag = new int[nbr];
    for(int i = 0; i < nbr; i++)
        flag[i] = 0;

    double temp;
    for(int i = 1; i < nbr; i++)
        for(int j = nbr-1; j >= i; j--)
        {
            if(index_type == 0)
            {
                // lower score, lower rank
                if(c[j-1] > c[j])
                {
                    temp = c[j-1];
                    c[j-1] = c[j];
                    c[j] = temp;
                }
            }
            else if(index_type == 1)
            {
                // higher score, lower rank
                if(c[j-1] < c[j])
                {
                    temp = c[j-1];
                    c[j-1] = c[j];
                    c[j] = temp;
                }
            }
        }

    for(int i = 0; i < nbr; i++)
        for(int j = 0; j < nbr; j++)
        {
            if(flag[j] == 0 && a[i] == c[j])
            {
                b[i] = j + 1;
                flag[j] = 1;
                break;
            }
        }
    delete [] c;
    delete [] flag;
}

/******************Create CSR******************/
void createCSR(){
    CSR_ROWPTR = (int*)malloc((total_nbr_v + 1) * sizeof(int));
    CSR_COLIND = (int*)malloc(nbr_edges * 2 * sizeof(int));
    CSR_VAL = (double*)malloc(nbr_edges * 2 * sizeof(double));
    int i, j, k = 0;
    max_deg = 0;

    CSR_ROWPTR[0] = 0;

    for(i = 0; i < total_nbr_v; i++){
        int deg = 0;
        for(j = 0; j < total_nbr_v; j++){
            if(diversity[i][j] != 0){
                CSR_COLIND[k] = j;
                CSR_VAL[k] = diversity[i][j];
                k++;
                deg++;
            }
        }
        if (deg > max_deg)
            max_deg = deg;
        CSR_ROWPTR[i+1] = k;
    }

    // allocate GPU space
    cudaMalloc((void**)&d_CSR_R, (total_nbr_v + 1) * sizeof(int));
    cudaMalloc((void**)&d_CSR_COL, nbr_edges * 2 * sizeof(int));
    cudaMalloc((void**)&d_CSR_VAL, nbr_edges * 2 * sizeof(double));

    // copy from CPU to GPU
    cudaMemcpy(d_CSR_R, CSR_ROWPTR, (total_nbr_v + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_CSR_COL, CSR_COLIND, nbr_edges * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_CSR_VAL, CSR_VAL, nbr_edges * 2 * sizeof(double), cudaMemcpyHostToDevice);
}

/******************read instances******************/
void read_instance() {
    ifstream FIC;
    string line;
    FIC.open(instance_path);
    if (FIC.fail()) {
        cout << "### Fail to open file: " << instance_name << endl;
        getchar();
        exit(0);
    }
    if (FIC.eof()) {
        cout << "### Fail to open file: " << instance_name << endl;
        exit(0);
    }
    int nbr_pairs = 0;
    int x1, x2;
    double x3;

    cout << "Dataset: "<< dataset << endl;
    if (strcmp(dataset, "MDG-a") == 0 || strcmp(dataset, "MDG-b") == 0 ||
        strcmp(dataset, "MDG-c") == 0) // for SetIMDG-a,MDG-b and MDG-c
    {
        getline(FIC, line);
        istringstream iss(line);
        iss >> total_nbr_v >> nbr_v;
        cout << total_nbr_v << "-" << nbr_v << endl;
        nbr_pairs = (total_nbr_v * (total_nbr_v - 1)) / 2;
    } else if (strcmp(dataset, "b2500") == 0) // for SetIIb2500
    {
        getline(FIC, line);
        istringstream iss(line);
        iss >> total_nbr_v >> nbr_pairs;
        nbr_v = 1000;
    } else if (strcmp(dataset, "p30005000") == 0)// for SetIIIP3000&P5000
    {
        getline(FIC, line);
        istringstream iss(line);
        iss >> total_nbr_v >> nbr_pairs;
        nbr_v = total_nbr_v / 2;
    } else if (strcmp(dataset, "nenad") == 0) // for SetIVg_100, h_100 and h_300
    {
        getline(FIC, line);
        istringstream iss(line);
        iss >> total_nbr_v >> nbr_pairs;
        nbr_v = 100;
    } else if (strcmp(dataset, "sparse") == 0) // for Set V SuiteSparse Matrix Collection
    {
        do {
            getline(FIC, line);
        } while (line.empty() || line[0] == '%');
        istringstream iss(line);
        iss >> total_nbr_v;
        nbr_v = 100;
    } else {
        cout << "data set is wrong!" << endl;
        exit(0);
    }

    delta_nbr_v = total_nbr_v - nbr_v;
    nbr_movs = nbr_v * delta_nbr_v;

    diversity = new double*[total_nbr_v];
    for(int x = 0; x < total_nbr_v; x++)
        diversity[x] = new double[total_nbr_v];
    for(int x = 0; x < total_nbr_v; x++)
        for(int y = 0; y < total_nbr_v; y++)
            diversity[x][y] = 0;


    max_diversity = 0.0;
    while(getline(FIC, line))
    {
        istringstream iss(line);
        iss >> x1 >> x2 >> x3;

        if(strcmp(dataset,"b2500") == 0 || strcmp(dataset,"p30005000") == 0 || strcmp(dataset, "sparse") == 0)// code for SetIIb2500 and SetIIIP3000&P5000
        {
            x1--;
            x2--;
        }
        if ( x1 < 0 || x2 < 0 || x1 > total_nbr_v || x2 > total_nbr_v)
        {
            cout << "### Read Date Error : line = "<< x1 << ", column = " << x2 << endl;
            exit(0);
        }

        if(x1 != x2)
        {
            if(strcmp(dataset,"b2500") == 0) // code for SetIIb2500
                x3 = x3*2;

            diversity[x1][x2] = diversity[x2][x1] = x3;
            if(diversity[x1][x2] > max_diversity)
                max_diversity = diversity[x1][x2];

            if(strcmp(dataset, "sparse") == 0)
                nbr_pairs++;
        }
    }

    nbr_edges = nbr_pairs;
    double density = (double)nbr_pairs/(double)(total_nbr_v*(total_nbr_v-1)/2);
    double percent = (double)nbr_v/(double)(total_nbr_v);
    cout << "The statistics of the instance " << instance_name << endl;
    cout << "n = " << total_nbr_v << ", m = " << nbr_v << ", m/n = " << percent << ", and density = " << density << endl;

    cout << "Finish loading data!" << endl;

    createCSR();
}

// allocates memory space
void setup_data()
{
    // allocate CPU space
    allocate_space = total_nbr_v*sizeof(int);
    improved_sol = new int[nbr_v];
    best_sol = new int[nbr_v];
    offspring = new int[nbr_v];
    opposite_sol = new int[nbr_v];
    pop_distance = new double[PS+1];
    pop_cost = new double[PS+1];
    pop_score = new double[PS+1];
    vertex = new int[total_nbr_v];

    pop = new int*[PS+1];
    for(int x = 0; x < PS+1; x++)
        pop[x] = new int[nbr_v];

    sol_distance = new double*[PS+1];
    for(int x = 0; x < PS+1; x++)
        sol_distance[x] = new double[PS+1];


    // allocate GPU space
    cudaMalloc((void**)&d_crrntSolution, total_nbr_v * sizeof(int));
    cudaMalloc((void**)&d_bestSolution, nbr_v * sizeof(int));
    cudaMalloc((void**)&d_vertex_gain, total_nbr_v * sizeof(double));
    cudaMalloc((void**)&d_tab, total_nbr_v * sizeof(int));
    cudaMalloc((void**)&d_updated, total_nbr_v * sizeof(unsigned int));
    cudaMalloc((void**)&updating, total_nbr_v * sizeof(unsigned int));
    cudaMalloc((void**)&d_pos, total_nbr_v * sizeof(int));
    cudaMalloc((void**)&d_gain, nbr_movs * sizeof(double));
    cudaMalloc((void**)&d_edge_gain, nbr_movs * sizeof(double));
    cudaMalloc((void**)&d_out, sizeof(cub::KeyValuePair <int, double>));
}

// deallocates memory space
void clear_data()
{
    delete[] improved_sol;
    delete[] best_sol;
    delete[] pop_cost;
    delete[] pop_distance;
    delete[] pop_score;
    delete[] offspring;
    delete[] opposite_sol;
    delete[] vertex;
    for(int i = 0; i < PS+1; i++)
    {
        delete pop[i];
        delete sol_distance[i];
    }
}

// check the solution is correct or not
void check_and_store_result(int *sol,double sol_cost)
{
    FILE *out;
    // check the range of chosen elements in the solution
    for(int i = 0; i < nbr_v; i++)
        if( sol[i] < 0 || sol[i] >= total_nbr_v)
        {
            printf("### element:%d is out of range: %d-%d",sol[i],0,total_nbr_v-1);
            exit(0);
        }

    // check the cost;
    double true_sol_cost = 0.0;
    for(int i = 0; i < nbr_v; i++)
        for(int j = i + 1; j < nbr_v; j++)
            true_sol_cost += diversity[sol[i]][sol[j]];

    if(abs(true_sol_cost - sol_cost) > epsilon)
    {
        printf("Find a error solution: its sol_cost = %f, while its true_sol_ost =%f\n",sol_cost,true_sol_cost);
        exit(0);
    }
}

void update_improved_sol(int *sol,double sol_cost)
{
    improved_time = (clock() - start_time)/CLOCKS_PER_SEC;
    improved_cost = sol_cost;
    for(int i = 0; i < nbr_v; i++)
        improved_sol[i] = sol[i];
}

void update_best_sol()
{
    best_time = improved_time;
    best_cost = improved_cost;
    for(int i = 0; i < nbr_v; i++)
        best_sol[i] = improved_sol[i];
}

// determines tabu value for vertex
int determine_tabu_tenure(int iter)
{
    int delta_tenure;
    int temp = iter%T;

    if(temp > 700 && temp <= 800)
        delta_tenure = 8*alpha;
    else if((temp > 300 && temp <= 400)||(temp > 1100 && temp <= 1200))
        delta_tenure = 4*alpha;
    else if((temp > 100 && temp <= 200)||(temp > 500 && temp <= 600)||(temp > 900 && temp <= 1000)||(temp > 1300 && temp <= 1400))
        delta_tenure = 2*alpha;
    else
        delta_tenure = alpha;

    return delta_tenure;
}

// starts data structures using the sol vector
void initialSolution(int *sol, int* currentSolution, double *vertexGain, bool* isSolution, int* pos){
    int i, x = 0, y = nbr_v;

    for(i = 0; i < nbr_v; i++){
        isSolution[sol[i]] = true;
    }

    for(i = 0; i < total_nbr_v; i++){
        if(isSolution[i]){
            currentSolution[x] = i;
            vertexGain[x] = 0;
            pos[i] = x;
            x++;
        }else{
            currentSolution[y] = i;
            vertexGain[y] = 0;
            pos[i] = y;
            y++;
        }
    }
}

// calculates the gain of each vertex and the sum solution
double calculateVertexGain(int* currentSolution, double *vertexGain, bool* isSolution, int* pos){
    double best_sum = 0;
    bool visited[nbr_v] = {false};

    for(int i = 0; i < nbr_v; i++){
        int vertex = currentSolution[i];

        for(int j = CSR_ROWPTR[vertex]; j < CSR_ROWPTR[vertex+1]; j++){
            vertexGain[CSR_COLIND[j]] += CSR_VAL[j];

            if(isSolution[CSR_COLIND[j]]){
                if(!visited[pos[CSR_COLIND[j]]])best_sum += CSR_VAL[j];
            }
        }
        visited[i] = true;
    }
    return best_sum;
}

// initialize wswap
__global__ void calculateEdgeGain(int* d_crrntSolution, double* d_edge_gain, int* d_pos, int* d_CSR_R, int* d_CSR_COL, double* d_CSR_VAL, int nbr_v, int total_nbr_v, int delta_nbr_v, int max_idx){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ps, v;

    if(idx < max_idx){
        int index_i = idx/total_nbr_v;
        int index_j = idx%total_nbr_v;

        int u = d_crrntSolution[index_i];
        int qnt = d_CSR_R[u+1] - d_CSR_R[u];

        if(index_j < qnt){
            v = d_CSR_COL[d_CSR_R[u] + index_j];
            if(d_pos[v] >= nbr_v){
                ps = index_i * delta_nbr_v + d_pos[v] - nbr_v;
                d_edge_gain[ps] = d_CSR_VAL[d_CSR_R[u] + index_j];
            }
        }
    }
}

// updates the gain of each vertex and the wswap
__global__ void updateVertexEdgeGain(int* d_solution, unsigned int *d_updated, unsigned int *updating, double* d_vertex_gain, double* d_edge_gain, int *d_pos, int* d_CSR_R, int* d_CSR_COL, double* d_CSR_VAL, int index_u, int index_v, int nbr_v, int delta_nbr_v, int *d_tab, int iter, int C1, int C2){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int u;
    int v;


    while(*updating==1);
    if (*d_updated == 0) {
        u = d_solution[index_u];
        v = d_solution[index_v];
    } else {
        u = d_solution[index_v];
        v = d_solution[index_u];
    }

    int qnt_u = d_CSR_R[u+1] - d_CSR_R[u];
    if(idx < qnt_u){
        double edge_weight = d_CSR_VAL[d_CSR_R[u] + idx];
        int z = d_CSR_COL[d_CSR_R[u] + idx];
        atomicAdd2(&d_vertex_gain[z], -edge_weight);

        if (z != v) {
            if (d_pos[z] >= nbr_v) {
                int gain_idx = index_u * delta_nbr_v + d_pos[z] - nbr_v;
                atomicAdd2(&d_edge_gain[gain_idx], -edge_weight);
            } else {
                int gain_idx = d_pos[z] * delta_nbr_v + index_v - nbr_v;
                atomicAdd2(&d_edge_gain[gain_idx], edge_weight);
            }
        }
    }

    int qnt_v = d_CSR_R[v+1] - d_CSR_R[v];
    if(idx < qnt_v){
        double edge_weight = d_CSR_VAL[d_CSR_R[v] + idx];
        int z = d_CSR_COL[d_CSR_R[v] + idx];
        atomicAdd2(&d_vertex_gain[z], edge_weight);

        if (z != u) {
            if (d_pos[z] >= nbr_v) {
                int gain_idx = index_u * delta_nbr_v + d_pos[z] - nbr_v;
                atomicAdd2(&d_edge_gain[gain_idx], edge_weight);
            } else {
                int gain_idx = d_pos[z] * delta_nbr_v + index_v - nbr_v;
                atomicAdd2(&d_edge_gain[gain_idx], -edge_weight);
            }
        }
    }

    if (idx == 0) {
        atomicExch(updating, 1);

        d_solution[index_u] = v;
        d_solution[index_v] = u;

        atomicExch(d_updated, 1);
        atomicExch(updating, 0);

        d_tab[u] = iter + C1;
        d_tab[v] = iter + C2;

        d_pos[u] = index_v;
        d_pos[v] = index_u;
    }
}

// update the current solution
__global__ void updateSolution(int* d_crrntSolution, int* d_sorted_solution, int* d_pos, int* d_tab, int index_u, int index_v, int C1, int C2, int iter){
    int u = d_sorted_solution[index_u];
    int v = d_sorted_solution[index_v];
    d_tab[u] = iter + C1;
    d_tab[v] = iter + C2;

    int pos_u = d_pos[u];
    int pos_v = d_pos[v];
    d_crrntSolution[pos_u] = v;
    d_crrntSolution[pos_v] = u;

    int tmp_pos = d_pos[u];
    d_pos[u] = d_pos[v];
    d_pos[v] = tmp_pos;
}

// calculate swap gains
__global__ void calculateSwapGain(int* d_solution, unsigned int *d_updated, unsigned int *updating, double* d_vertex_gain, double* d_gain, double* d_edge_gain, int *d_pos, int *d_tab, int iter, double currentWeight, double bestWeight, int* d_CSR_R, int* d_CSR_COL, double* d_CSR_VAL, int nbr_v, int total_nbr_v, int delta_nbr_v, int nbr_movs){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < nbr_movs){
        int idx_u = idx/delta_nbr_v;
        int idx_v = (idx%delta_nbr_v) + nbr_v;

        int u = d_solution[idx_u];
        int v = d_solution[idx_v];

        int pos_uv = idx_u * delta_nbr_v + idx_v - nbr_v;
        d_gain[idx] = d_vertex_gain[v] - d_vertex_gain[u] - d_edge_gain[pos_uv];

        if( (d_tab[u] > iter || d_tab[v] > iter) && (currentWeight + d_gain[idx] <= bestWeight) ){
            d_gain[idx] = -INF;
        }

        if (idx == 0) {
            *d_updated = 0;
            *updating = 0;
        }
    }
}

// checks that the gain of the current solution has been calculated correctly
void checkSolution(int *current_solution, double *current_vertex_gain, int *h_pos, double current_cost, int nbr_v, int total_nbr_v) {
    double cost = 0;
    bool visited[total_nbr_v] = {false};
    double vertexGain[total_nbr_v] = {0};
    for(int i = 0; i < nbr_v; i++){
        int vertex = current_solution[i];

        for(int j = CSR_ROWPTR[vertex]; j < CSR_ROWPTR[vertex+1]; j++){
            int position = h_pos[CSR_COLIND[j]];
            if(position < nbr_v){
                if(!visited[CSR_COLIND[j]]) cost += CSR_VAL[j];
            }
            vertexGain[CSR_COLIND[j]] += CSR_VAL[j];
        }
        visited[vertex] = true;
    }

    for (int i = 0; i < total_nbr_v; i++) {
        assert(vertexGain[i] == current_vertex_gain[i]);
    }
    assert((current_cost == cost));
}

// tabu search initialization
void tabu_search(int *sol){
    double c_s, c_e;
    c_s = clock();

    double used_time;
    bool is_solution[total_nbr_v] = {false};
    int current_solution[total_nbr_v];
    double vertex_gain[total_nbr_v];
    int tabu[total_nbr_v] = {0};
    int pos[total_nbr_v];

    // initialization of data structures
    initialSolution(sol, current_solution, vertex_gain, is_solution, pos);
    double current_weight = calculateVertexGain(current_solution, vertex_gain, is_solution, pos);
    update_improved_sol(current_solution, current_weight);
    double bestWeight = improved_cost;

    // data transfer from CPU to GPU
    gpuErrchk( cudaMemcpy(d_crrntSolution, current_solution, total_nbr_v * sizeof(int), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_vertex_gain, vertex_gain, total_nbr_v * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_tab, tabu, total_nbr_v * sizeof(int), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_pos, pos, total_nbr_v * sizeof(int), cudaMemcpyHostToDevice) );

    // memory allocation
    if(temp_storage_bytes == 0){
        gpuErrchk( cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_gain, d_out, nbr_movs) );
        gpuErrchk( cudaMalloc(&d_temp_storage, temp_storage_bytes) );
    }

    int deg = nbr_v * total_nbr_v;
    int blcosksEdgeGain = deg / 256 + 1;
    int blocksSwapGain = nbr_movs / 256 + 1;
    int blocksVertexEdgeGain = max_deg / 32 + 1;

    // wswap initialization
    gpuErrchk( cudaMemset(d_edge_gain, 0, nbr_movs * sizeof(double)) );
    calculateEdgeGain<<<blcosksEdgeGain, 256>>>(d_crrntSolution, d_edge_gain, d_pos, d_CSR_R, d_CSR_COL,
                                                d_CSR_VAL, nbr_v, total_nbr_v, delta_nbr_v, deg);
    gpuErrchk( cudaPeekAtLastError() );

    int iter = 0;
    while(iter < max_iter){
        calculateSwapGain<<<blocksSwapGain, 256>>>(d_crrntSolution, d_updated, updating, d_vertex_gain, d_gain, d_edge_gain, d_pos, d_tab, iter,
                                                          current_weight, bestWeight, d_CSR_R, d_CSR_COL, d_CSR_VAL, nbr_v, total_nbr_v, delta_nbr_v, nbr_movs);
        gpuErrchk( cudaPeekAtLastError() );

        total_evaluations += nbr_movs;

        // choose the swap with the highest gain
        gpuErrchk( cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_gain, d_out, nbr_movs) );
        gpuErrchk( cudaMemcpy(&h_out, d_out, sizeof(cub::KeyValuePair <int, double>), cudaMemcpyDeviceToHost) );

        int idx = h_out.key;
        int index_u = idx / delta_nbr_v;
        int index_v = (idx % delta_nbr_v) + nbr_v;
        if (h_out.value == -INF ) {
            break;
        } else {
            current_weight += h_out.value;
        }

        int C1 = determine_tabu_tenure(iter);
        int C2 = round(0.7 * C1);

        // updates the vertex edge gain and the current solution by swapping
        updateVertexEdgeGain<<<blocksVertexEdgeGain, 32>>>(d_crrntSolution, d_updated, updating, d_vertex_gain, d_edge_gain, d_pos, d_CSR_R, d_CSR_COL, d_CSR_VAL, index_u, index_v, nbr_v,
                                                            delta_nbr_v, d_tab, iter, C1, C2);
        gpuErrchk( cudaPeekAtLastError() );

        if(current_weight > bestWeight){
            gpuErrchk( cudaMemcpy(d_bestSolution, d_crrntSolution, nbr_v * sizeof(int), cudaMemcpyDeviceToDevice) );
            bestWeight = current_weight;
        }

        iter++;
        total_iterations++;
        used_time = (clock() - start_time)/CLOCKS_PER_SEC;

        if(used_time > limit_time)break;

    }

    c_e = clock();

    if(bestWeight > improved_cost){
        cudaMemcpy(current_solution, d_bestSolution, nbr_v * sizeof(int), cudaMemcpyDeviceToHost);
        update_improved_sol(current_solution, bestWeight);
    }

    if(iter > 1){
        printf("Iteracoes: %d\tTime Tabu: %.10lf\tSol: %.10lf\n", iter, (c_e-c_s)/CLOCKS_PER_SEC, improved_cost);
    }
}

// Check the solution is duplicate or not in the population
int is_duplicate_sol(int **pop1,int index)
{
    int duplicate = 0;
    for(int i = 0; i < index; i++)
    {
        duplicate = 1;
        for(int j = 0; j < nbr_v; j++)
            if(improved_sol[j] != pop1[i][j])
            {
                duplicate = 0;
                break;
            }

        if(duplicate == 1)
            break;
    }
    return duplicate;
}

// Compute the distance between any two solutions in the population
double calculate_sol_distance(int x1,int x2)
{
    double distance;
    int u = 0;
    int v = 0;
    int sharing = 0;
    while((u < nbr_v) && (v < nbr_v))
    {
        if(pop[x1][u] == pop[x2][v])
        {
            sharing ++;
            u++;
            v++;
        }
        else if(pop[x1][u] < pop[x2][v])
            u++;
        else if(pop[x1][u] > pop[x2][v])
            v++;
    }
    distance = 1-(double)sharing/(double)nbr_v;
    return distance;
}

// Sort the elements in the solution in an ascend order
void ascend_sort(int *sol)
{
    int count = 0;
    memset(vertex,0,allocate_space);

    for(int i = 0; i < nbr_v; i++)
        vertex[sol[i]] = 1;

    for(int i = 0; i < total_nbr_v; i++)
        if(vertex[i] == 1)
            sol[count++] = i;
}

// Initialize the population with opposition-based learning
void build_pool_with_OBL()
{

    int nbr_item;
    int nbr_sol;
    int index;
    int *sol;
    int *opp_sol;
    int *elite_sol;
    double elite_sol_cost;

    sol = new int[nbr_v];
    opp_sol = new int[nbr_v];
    elite_sol = new int[nbr_v];
    best_cost = -1;
    nbr_sol = 0;
    while(nbr_sol < PS)
    {
        // generate a solution
        memset(vertex,0,allocate_space);
        nbr_item = 0;
        while(nbr_item < nbr_v)
        {
            index = rand()%total_nbr_v;
            if(vertex[index] == 0)
            {
                sol[nbr_item] = index;
                nbr_item++;
                vertex[index] = 1;
            }
        }

        tabu_search(sol);
        ascend_sort(improved_sol);

        elite_sol_cost = improved_cost;
        for(int i = 0; i < nbr_v; i++)
            elite_sol[i] = improved_sol[i];

        // generate its opposite solution
        if(delta_nbr_v != nbr_v)
        {
            int *available_items;
            int nbr_available_item = 0;
            available_items = new int[delta_nbr_v];

            for(int i = 0; i < total_nbr_v; i++)
                if(vertex[i] == 0)
                {
                    available_items[nbr_available_item] = i;
                    nbr_available_item++;
                }
            nbr_item = 0;
            while(nbr_item < nbr_v)
            {
                index = rand()%nbr_available_item;
                opp_sol[nbr_item] = available_items[index];
                nbr_item++;

                // delete this vertex from the array
                nbr_available_item--;
                available_items[index] = available_items[nbr_available_item];
            }
            delete [] available_items;
        }
        else
        {
            nbr_item = 0;
            for(int i = 0; i < total_nbr_v; i++)
                if(vertex[i] == 0)
                {
                    opp_sol[nbr_item] = i;
                    nbr_item++;
                }
        }

        tabu_search(opp_sol);
        ascend_sort(improved_sol);

        if(improved_cost < elite_sol_cost || (abs(improved_cost-elite_sol_cost) < epsilon && rand()%2 == 0))
        {
            improved_cost = elite_sol_cost;
            for(int i = 0; i < nbr_v; i++)
                improved_sol[i] = elite_sol[i];
        }

        memset(vertex,0,allocate_space);
        for(int i = 0; i < nbr_v; i++)
            vertex[improved_sol[i]] = 1;

        // modify it if it is same to an existing solution
        int swapin_v,swapout_v;
        int flag;
        double swapout_gain,swapin_gain;
        while(is_duplicate_sol(pop,nbr_sol) == 1)
        {
            index = rand()%nbr_v;
            swapout_v = improved_sol[index];

            swapout_gain = 0.0;
            for(int i = 0; i < nbr_v; i++)
                if(improved_sol[i] != swapout_v)
                    swapout_gain += diversity[improved_sol[i]][swapout_v];

            flag = 0;
            while(flag == 0)
            {
                swapin_v = rand()%total_nbr_v;
                if(vertex[swapin_v] == 0)
                    flag = 1;
            }

            swapin_gain = 0.0;
            for(int i = 0; i < nbr_v; i++)
                if(improved_sol[i] != swapout_v)
                    swapin_gain += diversity[improved_sol[i]][swapin_v];

            // swap
            vertex[swapin_v] = 1;
            vertex[swapout_v] = 0;
            improved_sol[index] = swapin_v;
            improved_cost += swapin_gain - swapout_gain;
            ascend_sort(improved_sol);
        }

        pop_cost[nbr_sol] = improved_cost;
        for(int i = 0; i < nbr_v; i++)
            pop[nbr_sol][i] = improved_sol[i];

        nbr_sol++;

        if(improved_cost > best_cost)
            update_best_sol();
    }

    // Calculate the distance between any two solutions in the population
    for(int i = 0; i < PS; i++)
    {
        for(int j = i + 1; j < PS; j++)
        {
            sol_distance[i][j] = calculate_sol_distance(i,j);
            sol_distance[j][i] = sol_distance[i][j];
        }
        sol_distance[i][i] = 0.0;
    }

    delete [] sol;
    delete [] opp_sol;
    delete [] elite_sol;
}

// Create an offspring and its opposition by crossover and assign the remaining items greedily
void crossover_with_greedy()
{
    int choose_p;
    int index_p;
    int nbr_p;
    int *is_choose_p;
    int *p;

    int choose_v;
    int index_remaining_v;
    int nbr_added_v;
    int *index_best_v;
    int nbr_best_v;
    int *nbr_remaining_v;
    int **remaining_v;
    double max_v_profit;
    double v_profit;

    is_choose_p = new int[PS];
    p = new int[NP];

    index_best_v = new int[delta_nbr_v];

    nbr_remaining_v = new int[NP];
    remaining_v = new int*[NP];
    for(int i = 0; i < NP; i++)
        remaining_v[i] = new int[nbr_v];

    // choose two parents
    for(int i = 0; i < PS; i++)
        is_choose_p[i] = 0;
    nbr_p = 0;
    while(nbr_p < NP)
    {
        choose_p = rand()%PS;
        if(is_choose_p[choose_p] == 0)
        {
            p[nbr_p] = choose_p;
            nbr_p++;
            is_choose_p[choose_p] = 1;
        }
    }

    // Build a partial solution S0 by preserving the common elements
    memset(vertex,0,allocate_space);
    for(int i = 0; i < NP; i++)
        for(int j = 0; j < nbr_v; j++)
            vertex[pop[p[i]][j]]++;

    // S1/S0 and S2/S0
    int v;
    for(int i = 0; i < NP; i++)
    {
        nbr_remaining_v[i] = 0;
        for(int j = 0; j < nbr_v; j++)
        {
            v = pop[p[i]][j];
            if(vertex[v] == 1)
            {
                remaining_v[i][nbr_remaining_v[i]] = v;
                nbr_remaining_v[i]++;
                vertex[v] = 0;
            }
        }
    }

    // S0
    nbr_added_v = 0;
    for(int i = 0; i < total_nbr_v; i++)
        if(vertex[i] == NP)
        {
            offspring[nbr_added_v] = i;
            vertex[i] = 1;
            nbr_added_v++;
        }

    // generate an offspring by completing the partial solution in a greedy way
    while(nbr_added_v < nbr_v)
    {
        index_p = nbr_added_v%NP;
        max_v_profit = min_integer;
        for(int i = 0; i < nbr_remaining_v[index_p]; i++)
        {
            v_profit = 0.0;
            for(int j = 0; j < nbr_added_v; j++)
                v_profit += diversity[remaining_v[index_p][i]][offspring[j]];

            if(v_profit > max_v_profit)
            {
                max_v_profit = v_profit;
                index_best_v[0] = i;
                nbr_best_v = 1;
            }
            else if(abs(v_profit-max_v_profit) < epsilon)
            {
                index_best_v[nbr_best_v] = i;
                nbr_best_v++;
            }
        }
        index_remaining_v = index_best_v[rand()%nbr_best_v];
        choose_v = remaining_v[index_p][index_remaining_v];

        offspring[nbr_added_v] = choose_v;
        nbr_added_v++;
        vertex[choose_v] = 1;
        nbr_remaining_v[index_p]--;
        remaining_v[index_p][index_remaining_v] = remaining_v[index_p][nbr_remaining_v[index_p]];
    }

    // generate an opposite solution
    if(is_obl == true)
    {
        if(delta_nbr_v > nbr_v)
        {
            int index_avaiable_v;
            int nbr_available_v;
            int *available_v;
            available_v = new int[delta_nbr_v];

            nbr_available_v = 0;
            for(int i = 0; i < total_nbr_v; i++)
                if(vertex[i] == 0)
                {
                    available_v[nbr_available_v] = i;
                    nbr_available_v++;
                }

            nbr_added_v = 0;
            while(nbr_added_v < nbr_v)
            {
                index_avaiable_v = rand()%nbr_available_v;
                opposite_sol[nbr_added_v] = available_v[index_avaiable_v];
                nbr_added_v++;

                // delete this vertex from the array
                nbr_available_v--;
                available_v[index_avaiable_v] = available_v[nbr_available_v];
            }
            delete [] available_v;
        }
        else if(delta_nbr_v == nbr_v)
        {
            nbr_added_v = 0;
            for(int i = 0; i < total_nbr_v; i++)
                if(vertex[i] == 0)
                {
                    opposite_sol[nbr_added_v] = i;
                    nbr_added_v++;
                }
        }
        else
        {
            printf("error occurs in crossover: delta_nbr_v < nbr_v\n");
            exit(-1);
        }
    }

    delete []is_choose_p;
    delete []p;
    delete []index_best_v;
    delete []nbr_remaining_v;
    for(int i = 0; i < NP; i++)
        delete remaining_v[i];
}

void rank_based_pool_updating()
{
    double avg_sol_distance;
    double min_score;
    int index_worst;

    // Insert the offspring into the population
    pop_cost[PS] = improved_cost;
    for(int i = 0; i < nbr_v; i++)
        pop[PS][i] = improved_sol[i];

    for(int i = 0; i < PS; i++)
    {
        sol_distance[i][PS] = calculate_sol_distance(i,PS);
        sol_distance[PS][i] = sol_distance[i][PS];
    }
    sol_distance[PS][PS] = 0.0;

    // Calculate the average distance of each individual with the whole population
    for(int i = 0; i < PS+1; i++)
    {
        avg_sol_distance = 0.0;
        for(int j = 0; j < PS+1; j++)
        {
            if(j != i)
                avg_sol_distance += sol_distance[i][j];
        }
        pop_distance[i] = avg_sol_distance/PS;
    }


    // Compute the score of each individual in the population
    // Calculate the rank of cost and distance respectively
    int *cost_rank,*distance_rank;
    cost_rank = new int[PS+1];
    distance_rank = new int[PS+1];

    for(int i = 0; i < PS+1; i++)
    {
        cost_rank[i] = i+1;
        distance_rank[i] = i+1;
    }

    calculate_rank(0,PS+1,pop_cost,cost_rank);
    calculate_rank(0,PS+1,pop_distance,distance_rank);

    // Compute the score of each individual in the population
    for(int i = 0; i < PS+1; i++)
        pop_score[i] = alpha*cost_rank[i] + (1.0-alpha)*distance_rank[i];

    min_score = double(max_integer);
    for(int i = 0; i < PS+1; i++)
        if(pop_score[i] < min_score)
        {
            min_score = pop_score[i];
            index_worst = i;
        }

    // Insert the offspring
    if(index_worst != PS && is_duplicate_sol(pop,PS) == 0)
    {
        pop_cost[index_worst] = improved_cost;
        for(int i = 0; i < nbr_v; i++)
            pop[index_worst][i] = improved_sol[i];

        for(int i = 0; i < PS; i++)
        {
            sol_distance[i][index_worst] = sol_distance[PS][i];
            sol_distance[index_worst][i] = sol_distance[i][index_worst];
        }
        sol_distance[index_worst][index_worst] = 0.0;
    }
}

// Oppostion-based memetic algorithm (OBMA)
void OBMA(char *result_file)
{
    int no_improve_gen = 0;
    nbr_gen = 0;
    best_cost = -1.0;
    best_time = 0.0;

    // Population Initialization
    printf("using OBL!!!\n");

    build_pool_with_OBL();

    // Population Evolution
    while(1)
    {
        //**** Create an offspring and its opposite solution by crossover operator ****
        crossover_with_greedy();


        //********************* improve offspring by tabu search **********************
        tabu_search(offspring);

        ascend_sort(improved_sol);

        // Record the best solution
        if(improved_cost > best_cost)
        {
            update_best_sol();
            no_improve_gen = 0;
        }
        else
            no_improve_gen++;

        // Update the population
        rank_based_pool_updating();


        if(is_obl == true)
        {
            //************ improve the opposite solution by tabu search *****************
            tabu_search(opposite_sol);
            ascend_sort(improved_sol);

            // Record the best solution
            if(improved_cost > best_cost)
            {
                update_best_sol();
                no_improve_gen = 0;
            }
            else
                no_improve_gen++;

            // Update the population
            rank_based_pool_updating();
        }

        // Check the cut off time
        if((clock() - start_time)/CLOCKS_PER_SEC >= limit_time)
            break;
        nbr_gen++;
        // Display the intermediate results
    }
}

int main(int argc,char **argv)
{
   FILE *sum;
    int nbr_repeat;
    int nbr_success = 0;
    double avg_best_cost = 0.0;
    double avg_best_time = 0.0;
    double max_best_cost = -1.0;
    double avg_iterations = 0.0;
    double avg_evaluations = 0.0;
    double avg_min_best_time;
    double gap_best_cost;
    double sd;

    if(argc == 5)
    {
        instance_name = argv[1];
        dataset = argv[2];
        limit_time = atof(argv[3]);
        nbr_repeat = atoi(argv[4]);
    }
    else
    {
        cout << endl << "### Input the following parameters ###" << endl;
        cout << "Instance,data set,limit time,number of repeats" << endl;
        exit(0);
    }

    strcat(instance_path,instance_name);

    //Read the instance
    read_instance();
    setup_data();
    double best_sol_cost[nbr_repeat];

    // Repeat multiple run
    for(int i = 1; i <= nbr_repeat; i++)
    {
        // Run the algorithm
        srand((unsigned)time(NULL));
        start_time = clock();
        OBMA(result_path);
        printf("finish %d-th trial,nbr_gen = %d,best_cost = %f,best_time = %f\n",i,nbr_gen,best_cost,best_time);

        // Check and store the results
        check_and_store_result(best_sol,best_cost);
 
        // Statistical results
        best_sol_cost[i-1] = best_cost;
        avg_best_cost += best_cost;
        avg_best_time += best_time;
        if(best_cost > max_best_cost)
        {
            max_best_cost = best_cost;
            avg_min_best_time = best_time;
            nbr_success = 1;
        }
        else if(abs(best_cost - max_best_cost) < 0.1)
        {
            nbr_success++;
            avg_min_best_time += best_time;
        }
    }

    // Compute the statistical results
    avg_iterations = (double) total_iterations/nbr_repeat;
    avg_evaluations = (double) total_evaluations/nbr_repeat;
    avg_best_cost = avg_best_cost/(double)nbr_repeat;
    avg_best_time = avg_best_time/(double)nbr_repeat;
    avg_min_best_time = avg_min_best_time/(double)nbr_success;
    sd = 0.0;
    for(int i = 0; i < nbr_repeat; i++)
        sd += (best_sol_cost[i] - avg_best_cost)*(best_sol_cost[i] - avg_best_cost);
    sd = sd/(double)nbr_repeat;
    sd = sqrt(sd);
    gap_best_cost = max_best_cost - avg_best_cost;

    fprintf(stdout,"---------------------------------------------------------\n");
    fprintf(stdout,"%.3lf, %.3lf\n",avg_best_cost,avg_best_time);
    fprintf(stdout,"---------------------------------------------------------\n");
    fprintf(stdout,"%.3lf, %.3lf, %.3lf, %.3lf, %d, %.3lf, %3lf\n",max_best_cost,avg_min_best_time,gap_best_cost,sd,nbr_success, avg_evaluations, avg_evaluations);

    cout << "finishing...!" << endl;
    cout << "found best objective = " << std::setprecision (10) << max_best_cost << endl;
    cout << "number of times to find the best solution = " << nbr_success << endl;
    cout << "average value of the best objective value = " << std::setprecision (10) << avg_best_cost << endl;
    cout << "average value of the best time (second) = " << std::setprecision (10) << avg_best_time << endl;
    cout << "average number of iterations per run = " << std::setprecision (10) << avg_iterations << endl;
    cout << "average number of evaluations per run = " << std::setprecision (10) << avg_evaluations << endl;

    clear_data();
    return 0;
}

