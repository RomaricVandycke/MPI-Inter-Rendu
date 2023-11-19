#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define MATRIX_SIZE 4

void printMatrix(int *matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%d ", matrix[i * size + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int gridSize = sqrt(size);
    int blockSize = MATRIX_SIZE / gridSize;

    int *A = (int *)malloc(blockSize * blockSize * sizeof(int));
    int *B = (int *)malloc(blockSize * blockSize * sizeof(int));
    int *C = (int *)calloc(blockSize * blockSize, sizeof(int));

    // Initialize matrices A and B
    for (int i = 0; i < blockSize * blockSize; i++) {
        A[i] = rank + 1;  
        B[i] = rank + 1;
    }

    // Shift matrices A and B
    MPI_Cart_shift(MPI_COMM_WORLD, 1, -rank / gridSize, &rank, NULL);
    MPI_Cart_shift(MPI_COMM_WORLD, 0, -rank % gridSize, &rank, NULL);

    // Perform local block multiplication
    for (int k = 0; k < gridSize; k++) {
        for (int i = 0; i < blockSize; i++) {
            for (int j = 0; j < blockSize; j++) {
                for (int l = 0; l < blockSize; l++) {
                    C[i * blockSize + j] += A[i * blockSize + l] * B[l * blockSize + j];
                }
            }
        }

        // Shift matrices A and B 
        MPI_Sendrecv_replace(A, blockSize * blockSize, MPI_INT, (rank + gridSize - 1) % gridSize, 0,
                             (rank + 1) % gridSize, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(B, blockSize * blockSize, MPI_INT, (rank + gridSize - 1) % gridSize, 0,
                             (rank + 1) % gridSize, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Gather results to the root process
    MPI_Gather(C, blockSize * blockSize, MPI_INT, C, blockSize * blockSize, MPI_INT, 0, MPI_COMM_WORLD);

    // Print the result in the root process
    if (rank == 0) {
        printf("Matrix A:\n");
        printMatrix(A, blockSize);
        printf("Matrix B:\n");
        printMatrix(B, blockSize);
        printf("Result Matrix C:\n");
        printMatrix(C, MATRIX_SIZE);
    }

    free(A);
    free(B);
    free(C);

    MPI_Finalize();
    return 0;
}
