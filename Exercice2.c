#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#define CHUNKSIZE1 1000
#define CHUNKSIZE2 2000
/* message tags */
#define REQUEST  1
#define REPLY    2

int main(int argc, char *argv[]) {
    int iter;
    int in, out, i, iters, max, ix, iy, ranks[1], done, temp;
    double x, y, Pi, error, epsilon;
    int numprocs, myid, server, totalin, totalout, workerid;
    int rands[CHUNKSIZE1 > CHUNKSIZE2 ? CHUNKSIZE1 : CHUNKSIZE2], request;
    MPI_Comm world, workers, workers1, workers2;
    MPI_Group world_group, worker_group, workers1_group, workers2_group;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    world = MPI_COMM_WORLD;
    MPI_Comm_size(world, &numprocs);
    MPI_Comm_rank(world, &myid);
    server = numprocs - 1; /* last proc is server */
    if (myid == 0) {
        if (argc < 2) {
            fprintf(stderr, "Usage: %s epsilon\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        sscanf(argv[1], "%lf", &epsilon);
    }
    MPI_Bcast(&epsilon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Comm_group(world, &world_group);
    ranks[0] = server;
    MPI_Group_excl(world_group, 1, ranks, &worker_group);
    MPI_Comm_create(world, worker_group, &workers);
    MPI_Group_free(&worker_group);

    // Separate workers into workers1 and workers2
    MPI_Comm_group(workers, &worker_group);
    MPI_Group_incl(worker_group, numprocs / 2, myid % 2 == 1 ? &myid : MPI_GROUP_EMPTY, &workers1_group);
    MPI_Group_excl(worker_group, numprocs / 2, myid % 2 == 1 ? &myid : MPI_GROUP_EMPTY, &workers2_group);

    MPI_Comm_create(world, workers1_group, &workers1);
    MPI_Comm_create(world, workers2_group, &workers2);

    MPI_Group_free(&worker_group);
    MPI_Group_free(&workers1_group);
    MPI_Group_free(&workers2_group);

    if (myid == server) { /* I am the rand server */
        do {
            MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST, world, &status);
            if (request) {
                int chunksize = myid % 2 == 1 ? CHUNKSIZE1 : CHUNKSIZE2;
                for (i = 0; i < chunksize; ) {
                    rands[i] = random();
                    if (rands[i] <= INT_MAX) i++;
                }
                MPI_Send(rands, chunksize, MPI_INT, status.MPI_SOURCE, REPLY, world);
            }
        } while (request > 0);
    } else { /* I am a worker process */
        request = 1;
        done = in = out = 0;
        max = INT_MAX; /* max int, for normalization */
        MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);
        MPI_Comm_rank(world, &workerid);
        iter = 0;
        while (!done) {
            iter++;
            request = 1;
            MPI_Recv(rands, myid % 2 == 1 ? CHUNKSIZE1 : CHUNKSIZE2, MPI_INT, server, REPLY, world, MPI_STATUS_IGNORE);
            int chunksize = myid % 2 == 1 ? CHUNKSIZE1 : CHUNKSIZE2;
            for (i = 0; i < chunksize; ) {
                x = (((double) rands[i++]) / max) * 2 - 1;
                y = (((double) rands[i++]) / max) * 2 - 1;
                if (x * x + y * y < 1.0)
                    in++;
                else
                    out++;
            }
            MPI_Allreduce(&in, &totalin, 1, MPI_INT, MPI_SUM, myid % 2 == 1 ? workers1 : workers2);
            MPI_Allreduce(&out, &totalout, 1, MPI_INT, MPI_SUM, myid % 2 == 1 ? workers1 : workers2);
            Pi = (4.0 * totalin) / (totalin + totalout);
            error = fabs(Pi - 3.141592653589793238462643);
            done = (error < epsilon || (totalin + totalout) > 100000000);
            request = (done) ? 0 : 1;
            if (myid == 0) {
                printf("\rpi = %23.20f", Pi);
                MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);
            } else {
                if (request)
                    MPI_Send(&request, 1, MPI_INT, server, REQUEST, world);
            }
        }
        MPI_Comm_free(myid % 2 == 1 ? &workers1 : &workers2);
    }

    if (myid == 0) {
        printf("\npoints: %d\nin: %d, out: %d, <ret> to exit\n", totalin + totalout, totalin, totalout);
        getchar();
    }

    MPI_Finalize();
    return 0;
}
