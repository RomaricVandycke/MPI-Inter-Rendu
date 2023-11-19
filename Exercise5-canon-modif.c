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

    MPI_Request sendReq[2], recvReq[2];
    MPI_Status status[2];

    for (int i = 0; i < blockSize * blockSize; i++) {
        A[i] = rank + 1; 
        B[i] = rank + 1;
    }

   
    MPI_Cart_shift(MPI_COMM_WORLD, 1, -rank / gridSize, &rank, NULL);
    MPI_Cart_shift(MPI_COMM_WORLD, 0, -rank % gridSize, &rank, NULL);

    for (int k = 0; k < gridSize; k++) {
 
        for (int i = 0; i < blockSize; i++) {
            for (int j = 0; j < blockSize; j++) {
                for (int l = 0; l < blockSize; l++) {
                    C[i * blockSize + j] += A[i * blockSize + l] * B[l * blockSize + j];
                }
            }
        }

    
        MPI_Isend(A, blockSize * blockSize, MPI_INT, (rank + gridSize - 1) % gridSize, 0, MPI_COMM_WORLD, &sendReq[0]);
        MPI_Irecv(A, blockSize * blockSize, MPI_INT, (rank + 1) % gridSize, 0, MPI_COMM_WORLD, &recvReq[0]);
        MPI_Isend(B, blockSize * blockSize, MPI_INT, (rank + gridSize - 1) % gridSize, 1, MPI_COMM_WORLD, &sendReq[1]);
        MPI_Irecv(B, blockSize * blockSize, MPI_INT, (rank + 1) % gridSize, 1, MPI_COMM_WORLD, &recvReq[1]);

        
        MPI_Waitall(2, sendReq, status);
        MPI_Waitall(2, recvReq, status);
    }

    MPI_Gather(C, blockSize * blockSize, MPI_INT, C, blockSize * blockSize, MPI_INT, 0, MPI_COMM_WORLD);

   
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
