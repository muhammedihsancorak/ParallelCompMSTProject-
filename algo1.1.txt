// command to compile: mpicc -o algo1 algo1.c
// command to run: mpirun -np <numberProcs> ./algo1
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>

// size of matrix or number of vertices of the graph
#define mSize 128

int *MatrixChunk; // chunk of matrix belong to each processor
typedef struct
{
    int v1;
    int v2;
} Edge;

void primAlgorithm(int*matrix, int rank, int sendcounts, int size){
    // start to calculate running time
    double start= MPI_Wtime();
    // minimum spanning tree including the selected edges
    int* MST = (int *)malloc(sizeof(int) * mSize); 

    // initialize the MST array
    for (int i = 0; i < mSize; ++i)
    {
        MST[i] = -1;
    }

    // the first vertex is always the root of the MST
    MST[0] = 0;
    int minWeightOfGraph = 0;

    // the minimum edge of the graph
    int min, v1 = 0, v2 = 0;
    struct
    {
        int min;
        int rank;
    } minRow, row;
    Edge edge;

    // the main loop of the algorithm
    for (int k = 0; k < mSize - 1; ++k)
    {
        min = INT_MAX;

        // find the minimum edge of the graph, and the vertices of that edge
        for (int i = 0; i < sendcounts; ++i)
        {
            // if the vertex is already in the MST(already not visited)
            if (MST[i + rank*sendcounts] != -1)
            {
                // find the minimum edge of the vertex
                for (int j = 0; j < mSize; ++j)
                {
                    // if the vertex is not in the MST
                    if (MST[j] == -1)
                    {
                        // if the MatrixChunk[mSize*i+j] is less than min value
                        if (MatrixChunk[mSize * i + j] < min && MatrixChunk[mSize * i + j] != 0)
                        {
                            min = MatrixChunk[mSize * i + j];
                            // change the current edge
                            v2 = j; 
                            v1 = i;
                        }
                    }
                }
            }
        }
        // find the minimum edge of the graph
        row.min = min;
        row.rank = rank; 
        // find the minimum edge of the graph
        MPI_Allreduce(&row, &minRow, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
        // if the current processor has the minimum edge
        edge.v1 = v1 + rank*sendcounts;
        edge.v2 = v2;
        // broadcast the minimum edge to all processors
        MPI_Bcast(&edge, 1, MPI_2INT, minRow.rank, MPI_COMM_WORLD);
        // if the edge is not in the MST
        MST[edge.v2] = edge.v1;
        // add the weight of the edge to the minWeightOfGraph
        minWeightOfGraph += minRow.min;
    }
    
    double finish, calc_time;
    finish = MPI_Wtime();
    calc_time = finish - start;

    // rank 0 will writte its own process time, and values
    if (rank == 0)
    {
        FILE* f_result = fopen("output.txt", "w");
        fprintf(f_result, "\nNumber of processors: %d\nNumber of vertices: %d\nTime of execution: %f\n", size, mSize, calc_time);
        fprintf(f_result, "The minimun Weight is %d\n", minWeightOfGraph);
        for (int i = 1; i < mSize; ++i)
        {
            fprintf(f_result, "Edge %d %d. Weight: %d\n", i, MST[i], matrix[mSize * MST[i] + i]);
        }
        fclose(f_result);
    }
    free(MST);

}

int main(int argc, char *argv[])
{
    // rank of current processor
    int size,rank;      
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // to use same random value each time we are not seading the random value with time
    // srand(time(NULL) + rank);
    int sendcounts = mSize / size;
    // if number of vertices is not a multiply of number of processors then we need to handle the remainder
    // which is left as for future implementation to reduce the complexity of the problem
    int *matrix;
    if (rank == 0) 
    {
        matrix = (int *)malloc(mSize * mSize * sizeof(int));
        for (int i = 0; i < mSize * mSize; i++)
        {
            matrix[i] = 0;
        }

        // we are generating the value randomly here
        for (int i = 0; i < mSize; ++i)
        {
            matrix[mSize * i ] = 0;
            for (int j = i + 1; j < mSize; ++j)
            {
                matrix[mSize * i + j] = rand() % 10;
            }
        }
    }

    // after this each processor needs its own chunk of data
    MatrixChunk = (int *)malloc(sendcounts * mSize * sizeof(int));
    // here the chunk each processor needs will be scatter to it
    MPI_Scatter(matrix, sendcounts*mSize,MPI_INT, MatrixChunk, sendcounts*mSize, MPI_INT, 0, MPI_COMM_WORLD);
    primAlgorithm(matrix, rank, sendcounts, size);
    free(MatrixChunk);
    MPI_Finalize();
    return 0;
}