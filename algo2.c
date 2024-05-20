// command to compile: mpicc -o algo2 algo2.c
// command to run: mpirun -np <numberProcs> algo2 <input file>
// Include of libraries
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

const int ELEMENT_UNSET = -1;

// MPI variables, rank and size
int rank, size;

// Data structures for the graph and the minimum spanning tree 
// Set to store find the Disjoint and union 
typedef struct Set
{
	int elements;
	int *canonicalElements;
	int *rank;
} Set;

typedef struct Graph
{
	int edges;
	int vertices;
	int *edgeList;
} Graph;

// allocate memory on heap for the grapg
void newGraph(Graph *graph, const int Vertices, const int Edges)
{
	graph->edges = Edges;
	graph->vertices = Vertices;
	graph->edgeList = (int *)calloc(Edges* 3, sizeof(int));
}

// read a graph from a file and create the graph
void readDataOfGraphFromFile(Graph *graph, const char inputFileName[])
{
	// open the file for reading
	FILE *inputFile = fopen(inputFileName, "r");
	if (inputFile == NULL)
	{
		fprintf(stderr, "Couldn't open input file, exiting!\n");
		exit(EXIT_FAILURE);
	}
	int fscanfResult;
	// first line contains number of vertices and edges
	int vertices = 0, edges = 0;
	fscanfResult = fscanf(inputFile, "%d %d", &vertices, &edges);
	newGraph(graph, vertices, edges);

	int from, to, weight;
	for (int i = 0; i < edges; i++)
	{
		fscanfResult = fscanf(inputFile, "%d %d %d", &from, &to, &weight);
		graph->edgeList[i * 3] = from;
		graph->edgeList[i * 3 + 1] = to;
		graph->edgeList[i * 3 + 2] = weight;
	}
	fclose(inputFile);
}

// print all edges of the graph correct fromat
void printGraph(const Graph *graph)
{
	printf("##################################################\n");
	for (int i = 0; i < graph->edges; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			printf("%d\t", graph->edgeList[i * 3 + j]);
		}
		printf("\n");
	}
	printf("##################################################\n");
}

// Create a set
void createSet(Set *set, const int elements)
{
	set->elements = elements;
	set->canonicalElements = (int *)malloc(elements * sizeof(int));
	memset(set->canonicalElements, ELEMENT_UNSET, elements * sizeof(int));
	set->rank = (int *)calloc(elements, sizeof(int));
}

// return the element of a vertex with path compression
//  what is canonical element?
//  A canonical element is the representative of a set. It is the element that is used to identify the set.
//  For example, in the set {1, 2, 3}, 1 is the canonical element.
//  Path compression is a technique used to optimize the find operation in a disjoint-set data structure.
//  It involves updating the parent of each node in the path to the root of the set during the find operation.
//  This way, subsequent find operations on the same set will be faster.
int FindSet(const Set *set, const int vertex)
{
	if (set->canonicalElements[vertex] == ELEMENT_UNSET)
	{
		return vertex;
	}
	else
	{
		set->canonicalElements[vertex] = FindSet(set, set->canonicalElements[vertex]);
		return set->canonicalElements[vertex];
	}
}

// merge the set of parent1 and parent2 with union by rank
void unionSet(Set *set, const int parent1, const int parent2)
{
	int root1 = FindSet(set, parent1);
	int root2 = FindSet(set, parent2);

	if (root1 == root2)
	{
		return;
	}
	else if (set->rank[root1] < set->rank[root2])
	{
		set->canonicalElements[root1] = root2;
	}
	else if (set->rank[root1] > set->rank[root2])
	{
		set->canonicalElements[root2] = root1;
	}
	else
	{
		set->canonicalElements[root1] = root2;
		set->rank[root2] = set->rank[root1] + 1;
	}
}

// copy the value of the Edge from to to from 
void copyEdge(int *To, int *From)
{
	memcpy(To, From, 3 * sizeof(int));
}

// merge sorted lists, start and end are inclusive
void merge(int *edgeList, const int start, const int end, const int pivot)
{
	int length = end - start + 1;
	int *working = (int *)malloc(length * 3 * sizeof(int));
	// copy first part
	memcpy(working, &edgeList[start * 3], (pivot - start + 1) * 3 * sizeof(int));
	// copy second part reverse to simpify merge
	int workingEnd = end + pivot - start + 1;
	for (int i = pivot + 1; i <= end; i++)
	{
		copyEdge(&working[(workingEnd - i) * 3], &edgeList[i * 3]);
	}
	int left = 0;
	int right = end - start;
	for (int k = start; k <= end; k++)
	{
		if (working[right * 3 + 2] < working[left * 3 + 2])
		{
			copyEdge(&edgeList[k * 3], &working[right * 3]);
			right--;
		}
		else
		{
			copyEdge(&edgeList[k * 3], &working[left * 3]);
			left++;
		}
	}
	// clean up the memory
	free(working);
}

// sort the edge list using merge sort, start and end are inclusive
void mergeSort(int *edgeList, const int start, const int end)
{
	if (start != end)
	{
		// recursively divide the list in two parts and sort them
		int pivot = (start + end) / 2;
		mergeSort(edgeList, start, pivot);
		mergeSort(edgeList, pivot + 1, end);

		merge(edgeList, start, end, pivot);
	}
}

// sort the edges of the graph in parallel with mergesort in parallel
void sort(Graph *graph)
{
	int elements;
	bool parallel = size != 1;
	// send number of elements
	if (rank == 0)
	{
		elements = graph->edges;
	}
	MPI_Bcast(&elements, 1, MPI_INT, 0, MPI_COMM_WORLD);
	// scatter the edges to sort
	int elementsPart = (elements + size - 1) / size;
	int *edgeListDivided = (int *)malloc(elementsPart * 3 * sizeof(int));
	if (parallel)
	{
		if (elements % size != 0)
		{
			if (rank == 0){
				fprintf(stderr, "Please make sure number of edges are divible by number of Processor!\n");
			}
			MPI_Finalize();
			exit(0);
		}
		MPI_Scatter(graph->edgeList, elementsPart * 3, MPI_INT, edgeListDivided, elementsPart * 3, MPI_INT, 0, MPI_COMM_WORLD);
	}
	else
	{
		edgeListDivided = graph->edgeList;
	}

	// sort the part
	mergeSort(edgeListDivided, 0, elementsPart - 1);

	if (parallel)
	{
		// merge all parts
		int from, to, elementsRecieved;
		for (int step = 1; step < size; step *= 2)
		{
			if (rank % (2 * step) == 0)
			{
				from = rank + step;
				if (from < size)
				{
					MPI_Recv(&elementsRecieved, 1, MPI_INT, from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					edgeListDivided = realloc(edgeListDivided, (elementsPart + elementsRecieved) * 3 * sizeof(int));
					MPI_Recv(&edgeListDivided[elementsPart * 3], elementsRecieved * 3, MPI_INT, from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					merge(edgeListDivided, 0, elementsPart + elementsRecieved - 1, elementsPart - 1);
					elementsPart += elementsRecieved;
				}
			}
			else if (rank % step == 0)
			{
				to = rank - step;
				MPI_Send(&elementsPart, 1, MPI_INT, to, 0, MPI_COMM_WORLD);
				MPI_Send(edgeListDivided, elementsPart * 3, MPI_INT, to, 0, MPI_COMM_WORLD);
			}
		}

		// edgeListDivided is the new edgeList of the graph, cleanup other memory
		if (rank == 0)
		{
			free(graph->edgeList);
			graph->edgeList = edgeListDivided;
		}
		else
		{
			free(edgeListDivided);
		}
	}
	else
	{
		graph->edgeList = edgeListDivided;
	}
}

// find a MST of the graph using Kruskal's algorithm
void kruskalAlgorithm(Graph *graph, Graph *mst)
{
	// create needed data structures
	Set *set = &(Set){.elements = 0, .canonicalElements = NULL, .rank = NULL};
	createSet(set, graph->vertices);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// sort the edges of the graph, we are sorting the edge of the graph in parallel
	sort(graph);

	if (rank == 0)
	{
		// add edges to the MST
		int currentEdge = 0;
		for (int edgesMST = 0; edgesMST < graph->vertices - 1 || currentEdge < graph->edges;)
		{
			// check for loops if edge would be inserted
			int ElementFrom = FindSet(set, graph->edgeList[currentEdge * 3]);
			int ElementTo = FindSet(set, graph->edgeList[currentEdge * 3 + 1]);
			if (ElementFrom != ElementTo)
			{
				// add edge to MST
				copyEdge(&mst->edgeList[edgesMST * 3], &graph->edgeList[currentEdge * 3]);
				unionSet(set, ElementFrom, ElementTo);
				edgesMST++;
			}
			currentEdge++;
		}
	}

	// clean the allocated memory
	free(set->canonicalElements);
	free(set->rank);
}

// main program
int main(int argc, char *argv[])
{
	// MPI variables and initialization
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	if (argc < 2)
	{
		if (rank == 0)
		{
			printf("Usage:  mpirun -np <numberProcs> %s <input file>\n", argv[0]);
		}
		MPI_Finalize();
		return 0;
	}

	// graph Variables
	Graph *graph = &(Graph){.edges = 0, .vertices = 0, .edgeList = NULL};
	Graph *MST = &(Graph){.edges = 0, .vertices = 0, .edgeList = NULL};
	if (rank == 0)
	{
		// read the graph from the file
		readDataOfGraphFromFile(graph, argv[1]);
		newGraph(MST, graph->vertices, graph->vertices - 1);
	}

	double start = MPI_Wtime();
	// use Kruskal's algorithm
	kruskalAlgorithm(graph, MST);
	if (rank == 0)
	{
		printf("Time elapsed: %f s\n", MPI_Wtime() - start);
		// print the edges of the MST
		printf("Minimum Spanning Tree (Kruskal):\n");
		unsigned long weightMST = 0;
		for (int i = 0; i < MST->edges; i++)
		{
			weightMST += MST->edgeList[i * 3 + 2];
		}
		printf("MST weight: %lu\n", weightMST);
		printGraph(MST);
		// cleanup memory
		free(graph->edgeList);
		free(MST->edgeList);
	}
	MPI_Finalize();
	return 0;
}