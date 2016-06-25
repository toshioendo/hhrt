#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <mpi.h>
#include <hhrt.h>

int main(int argc, char **argv)
{
  int buflen = 4*1024*1024;
  long *buf;
  long *buf0;
  int rank;

  MPI_Init(&argc, &argv);

#if 0
  // test print
  int tsize;
  MPI_Type_size(MPI_LONG, &tsize);
  printf("sizeof MPI_LONG = %d\n", tsize);
  MPI_Type_size(MPI_PACKED, &tsize);
  printf("sizeof MPI_PACKED = %d\n", tsize);

#endif

  buf = (long*)malloc(sizeof(long)*buflen);
  int i;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (rank == 0) {
    for (i = 0; i < buflen; i++) {
      buf[i] = i%8+1;
    }
    buf0 = (long*)malloc(sizeof(long)*buflen);
  }

  for (i = 0; i < 20; i++) {
    MPI_Bcast(buf, buflen, MPI_LONG, 0, MPI_COMM_WORLD);
    MPI_Reduce(buf, buf0, buflen, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      memcpy(buf, buf0, buflen*sizeof(long));
      printf("buf[%d] = %ld\n", 0, buf[0]);
      printf("buf[%d] = %ld\n", buflen-1, buf[buflen-1]);
    }
  }

  MPI_Finalize();
}
