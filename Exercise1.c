#include <stdio.h>
#include <mpi.h>

#define SIZE 4

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    int matrix[SIZE][SIZE] = {{0, 1, 2, 3},
                              {4, 5, 6, 7},
                              {8, 9, 10, 11},
                              {12, 13, 14, 15}};

    if (rank == 0) {
        // Process 0 creates the matrix as a one-dimensional array
        int lower_triangular[SIZE * (SIZE + 1) / 2];
        int count = 0;
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j <= i; ++j) {
                lower_triangular[count++] = matrix[i][j];
            }
        }

        // Create MPI datatype for the lower part
        MPI_Datatype lower_triangular_type;
        int block_lengths[SIZE];
        int displacements[SIZE];
        for (int i = 0; i < SIZE; ++i) {
            block_lengths[i] = i + 1;
            displacements[i] = i * SIZE + i;
        }
        MPI_Type_indexed(SIZE, block_lengths, displacements, MPI_INT, &lower_triangular_type);
        MPI_Type_commit(&lower_triangular_type);

        // Send the lower part with MPI_Send
        MPI_Send(lower_triangular, 1, lower_triangular_type, 1, 0, MPI_COMM_WORLD);

        MPI_Type_free(&lower_triangular_type);
    } else if (rank == 1) {
        // Process 1 receives the lower triangular part with a single call to MPI_Recv
        int received_lower_triangular[SIZE * (SIZE + 1) / 2];

        MPI_Recv(received_lower_triangular, SIZE * (SIZE + 1) / 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Print the received data
        printf("Received lower triangular part:\n");
        int count = 0;
        for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j <= i; ++j) {
                printf("%d ", received_lower_triangular[count++]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}
