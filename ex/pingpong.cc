#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <malloc.h>

#include "hhrt.h"

#define BUFSIZE (256*1024*1024)

typedef struct {
  double st;
  char hostname[64];
} msg_t;

int mysend(int dst, void *sbuf, size_t size)
{
  msg_t *msg = (msg_t*)sbuf;
  msg->st = MPI_Wtime();
  //fprintf(stderr, "sending to %d\n", dst);
  MPI_Send(sbuf, size, MPI_CHAR, dst, 100, MPI_COMM_WORLD);
  return 0;
}

int main(int argc, char *argv[])
{
    int i;
    int rank;
    int size;

    void *sbuf;
    void *rbuf;
    msg_t *msg;
    char hostname[64];

    MPI_Init(&argc, &argv);

    gethostname(hostname, 63);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size <= 1) {
      fprintf(stderr, "This program requires >= 2 procs\n");
      exit(1);
    }

    sbuf = valloc(BUFSIZE);
    memset(sbuf, (unsigned char)rank, BUFSIZE);
    msg = (msg_t*)sbuf;
    strcpy(msg->hostname, hostname);

    rbuf = valloc(BUFSIZE);
    memset(rbuf, (unsigned char)rank, BUFSIZE);

    if (1 || rank < 10) {
      fprintf(stderr, "[pingpong] %s:%d rank=%d, size=%d\n", 
	      hostname, getpid(), rank, size);
    }

    size_t msgsize = 1;
    for (msgsize = 1; msgsize <= BUFSIZE; msgsize *= 2) {
      int dst;
      int i;
      for (i = 0; i < 3; i++) {
	MPI_Status stat;
	if (rank == 0) {
	  for (dst = 1; dst < size; dst++) {
	    double st = MPI_Wtime();
	    mysend(dst, sbuf, msgsize);
	    MPI_Recv(rbuf, 4, MPI_CHAR, MPI_ANY_SOURCE, 
		     MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
	    double et = MPI_Wtime();
	    long us = (long)((et-st)*1000000.0);
	    fprintf(stderr, "pingpong with rank %d, size %ld: %ld us -> %ldMB/s\n",
		    dst, msgsize, us, msgsize/us);
	  }
	}
	else {
	  fprintf(stderr, "%d starts to recv\n", rank);
	  MPI_Recv(rbuf, msgsize, MPI_CHAR, MPI_ANY_SOURCE, 
		   MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
	  /* pong */
	  mysend(0, sbuf, 4);
	}
      }
    }

    MPI_Finalize();
}
