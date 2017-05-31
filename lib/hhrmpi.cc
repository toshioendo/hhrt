/* Under construction */
/* Implementing MPI rendezvous communication for reducing pinned memory */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>
//#include "hhrt_impl.h"

#ifndef piadd
#define piadd(p, i) (void*)((char*)(p) + (size_t)(i))
#endif
#ifndef ppsub
#define ppsub(p1, p2) (size_t)((char*)(p1) - (char*)(p2))
#endif

enum {
  HHRM_SEND = 200,
  HHRM_RECV,
};

#define DEFAULT_CHUNKSIZE (1024*1024)

typedef struct {
  int count;
  MPI_Datatype datatype;
} commhdr;

typedef struct {
  int kind;
  void *ptr;
  int count;
  MPI_Datatype datatype;
  int partner;
  int tag;
  MPI_Comm comm;

  /* inner request */
  MPI_Request ireq;
  commhdr hdr;
} commtask;

typedef struct {
  commtask *ctp;
} HHRMPI_Request;

class rmpig {
 public:
  /* list of floating communications */
  list<commtask *> commtasks;
};

/* global variables */
rmpig hhrls;
rmpig *HHRL = &hhrls;

/* Returns 1 if type is "contiguous" on memory */
/* (This may not mean a type made by MPI_Type_contiguous) */
/* Returns 0 if type has holes, or unknown */
static int isTypeContiguous(MPI_Datatype type)
{
  int nint, naddr, ntype, combiner;
  MPI_Type_get_envelope(type, &nint, &naddr, &ntype, &combiner);
  if (combiner == MPI_COMBINER_NAMED) {
    return 1;
  }
  /* should be more accurate */
  return 0;
}

static int addCommTask(commtask *ctp)
{
  HHRL->commtasks.push_front(ctp);
  return 0;
}

static int progressSend(commtask *ctp)
{
  /* divide original message and send */
  int cur;
  int chunksize = DEFAULT_CHUNKSIZE;
  int psize;

  void *commbuf = malloc(chunksize);

  MPI_Pack_size(ctp->count, ctp->datatype, ctp->comm, &psize);
  /* we still assume data are contiguous */
  for (cur = 0; cur < psize; cur += chunksize) {
    int size = chunksize;
    if (cur+size > psize) size = psize-cur;

    memcpy(commbuf, piadd(ctp->ptr, cur), size);
    /* This should be Isend, and returns for each send */
    MPI_Send(commbuf, size, MPI_BYTE, ctp->partner, ctp->tag,
	     ctp->comm);
  }

  free(commbuf);
  ctp->fin = 1;
  return 0;
}

static int progressRecv(commtask *ctp)
{
  /* divide original message and send */
  int cur;
  int chunksize = DEFAULT_CHUNKSIZE;
  int psize;

  void *commbuf = malloc(chunksize);

  MPI_Pack_size(ctp->count, ctp->datatype, ctp->comm, &psize);
  /* we still assume data are contiguous */
  for (cur = 0; cur < psize; cur += chunksize) {
    int size = chunksize;
    MPI_Status stat;
    if (cur+size > psize) size = psize-cur;

    /* This should be Irecv, and returns for each send */
    MPI_Recv(commbuf, size, MPI_BYTE, ctp->partner, ctp->tag,
	     ctp->comm, &stat);

    memcpy(piadd(ctp->ptr, cur), commbuf, size);
  }

  free(commbuf);
  ctp->fin = 1;
  return 0;
}

static int progress1(commtask *ctp)
{
  int flag;
  MPI_Status stat;

  if (ctp->fin) {
    /* already finished. do nothing */
    return 0;
  }

  MPI_Test(ctp->ireq, &flag, &stat);
  if (flag == 0) {
    /* do nothing */
  }

  /* now ctp->hdr is valid */
  if (ctp->kind == HHRM_SEND) {
    progressSend(ctp);
  }
  else if (ctp->kind == HHRM_RECV) {
    progressRecv(ctp);
  }
  else {
    fprintf(stderr, "[HHRM:progress1] Unknown kind %d\n", ctp->kind);
    exit(1);
  }

  return 0;
}

/* should be called to progress communication */
int HHRM_progress()
{
  list<commtask *>::iter it;
  for (it = HHRL->commtasks.begin(); it != HHRL->commtasks.end(); it++) {
    commtask *ctp = *it;
    progress1(ctp);
  }
}

int HHRMPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
		int tag, MPI_Comm comm, HHRMPI_Request *reqp)
{
  commtask *ctp = new commtask;
  ctp->kind = HHRM_SEND;
  ctp->ptr = buf;
  ctp->count = count;
  ctp->datatype = datatype;
  ctp->partner = dest;
  ctp->tag = tag;
  ctp->comm = comm;

  if (!isTypeContiguous(datatype)) {
    fprintf(stderr, "[HHRTMPI_Isend] non contiguous datatype is specified. not supported yet\n");
    exit(1);
  }

  /* Send the first header */
  commhdr hdr;
  hdr.count = count;
  hdr.datatype = datatype;

  /* This should be Isend */
  /* We assume that hdr is enough small to eager communication works */
  MPI_Send((void*)&hdr, sizeof(hdr), MPI_BYTE, dest, tag, comm);
  /* start to wait ack */
  MPI_Irecv((void*)&comm.hdr, sizeof(hdr), MPI_BYTE, dest, tag, comm, &ctp->ireq);
  /* finish of ctp->ireq should be checked later */

  reqp->ctp = ctp;
  addCommTask(ctp);

  return 0;
}

int HHRMPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source,
		int tag, MPI_Comm comm, HHRMPI_Request *reqp)
{
  commtask *ctp = new commtask;
  ctp->kind = HHRM_RECV;
  ctp->ptr = buf;
  ctp->count = count;
  ctp->datatype = datatype;
  ctp->partner = source;
  ctp->tag = tag;
  ctp->comm = comm;

  if (!isTypeContiguous(datatype)) {
    fprintf(stderr, "[HHRTMPI_Isend] non contiguous datatype is specified. not supported yet\n");
    exit(1);
  }

  /* Start recving the first header */
  MPI_Irecv((void*)&comm.hdr, sizeof(hdr), MPI_BYTE, source, tag, comm, &ctp->ireq);

  reqp->ctp = ctp;
  addCommTask(ctp);

  return 0;
}

int HHRMPI_Waitall(int n, HHRMPI_Request *reqs, MPI_Status *stats)
{
}
