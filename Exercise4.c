#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define IMAGE_SIZE 12

void create_image(int size, unsigned char *image) {
    for (int i = 0; i < size * size; i++) {
        image[i] = rand() % 256;
    }
}

void smooth_image(int size, unsigned char *image, unsigned char *smoothed_image) {
    for (int i = 1; i < size - 1; i++) {
        for (int j = 1; j < size - 1; j++) {
            int index = i * size + j;
            smoothed_image[index] = (image[index] +
                                     image[(i - 1) * size + j] +
                                     image[(i + 1) * size + j] +
                                     image[i * size + (j - 1)] +
                                     image[i * size + (j + 1)]) / 5;
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);

    int coords[2];
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, (int[]){1, 1}, 0, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    // Image parameters
    int local_image_size = IMAGE_SIZE / dims[0];
    int image_size = IMAGE_SIZE;

    unsigned char *image = NULL;
    unsigned char *local_image = (unsigned char *)malloc(local_image_size * local_image_size * sizeof(unsigned char));
    unsigned char *local_smoothed_image = (unsigned char *)malloc(local_image_size * local_image_size * sizeof(unsigned char));

    if (rank == 0) {
        
        image = (unsigned char *)malloc(image_size * image_size * sizeof(unsigned char));
        create_image(image_size, image);
    }

    
    MPI_Scatter(image, local_image_size * local_image_size, MPI_UNSIGNED_CHAR,
                local_image, local_image_size * local_image_size, MPI_UNSIGNED_CHAR,
                0, cart_comm);

    
    smooth_image(local_image_size, local_image, local_smoothed_image);

    
    MPI_Gather(local_smoothed_image, local_image_size * local_image_size, MPI_UNSIGNED_CHAR,
               image, local_image_size * local_image_size, MPI_UNSIGNED_CHAR,
               0, cart_comm);

    if (rank == 0) {
        printf("Original Image:\n");
        for (int i = 0; i < image_size; i++) {
            for (int j = 0; j < image_size; j++) {
                printf("%d ", image[i * image_size + j]);
            }
            printf("\n");
        }

        printf("\nSmoothed Image:\n");
        for (int i = 0; i < image_size; i++) {
            for (int j = 0; j < image_size; j++) {
                printf("%d ", image[i * image_size + j]);
            }
            printf("\n");
        }

        free(image);
    }

    free(local_image);
    free(local_smoothed_image);

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();

    return 0;
}
