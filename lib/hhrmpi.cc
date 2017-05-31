/* Under construction */
/* Implementing MPI rendezvous communication for reducing pinned memory */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>
//#include "hhrt_impl.h"

#include <list>
#include <map>
using namespace std;

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

struct commtask {
  commtask() {fin = 0;};
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
  int fin;
};

typedef commtask *HHRMPI_Request;

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
  int flag;
  MPI_Status stat;
  MPI_Test(&ctp->ireq, &flag, &stat);
  if (flag == 0) {
    /* do nothing */
    return 0;
  }

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
    printf("[HHR:progressSend] calling Internal MPI_Send\n");
    MPI_Send(commbuf, size, MPI_BYTE, ctp->partner, ctp->tag,
	     ctp->comm);
  }

  free(commbuf);
  ctp->fin = 1;
  return 0;
}

static int progressRecv(commtask *ctp)
{
  int flag;
  MPI_Status stat;
  MPI_Test(&ctp->ireq, &flag, &stat);
  if (flag == 0) {
    /* do nothing */
    return 0;
  }

  /* Now first header arrived. */
  ctp->partner = stat.MPI_SOURCE;
  ctp->tag = stat.MPI_TAG;
  fprintf(stderr, "[HHRM:progressRecv] found src=%d, tag=%d\n",
	  ctp->partner, ctp->tag);
  /* Send ack */
  MPI_Send((void*)&ctp->hdr, sizeof(commhdr), MPI_BYTE, 
	   ctp->partner, ctp->tag, ctp->comm);

  /* recv divided messages */
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
    printf("[HHR:progressRecv] calling Internal MPI_Recv\n");
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
  if (ctp->fin) {
    /* already finished. do nothing */
    return 0;
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
  list<commtask *>::iterator it;
  for (it = HHRL->commtasks.begin(); it != HHRL->commtasks.end(); it++) {
    commtask *ctp = *it;
    progress1(ctp);

    if (ctp->fin) {
      /* finished. remove from list */
      HHRL->commtasks.erase(it);
      break;
    }
  }
  return 0;
}

int HHRMPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
		int tag, MPI_Comm comm, HHRMPI_Request *reqp)
{
  commtask *ctp = new commtask;
  ctp->kind = HHRM_SEND;
  ctp->ptr = (void *)buf;
  ctp->count = count;
  ctp->datatype = datatype;
  ctp->partner = dest;
  ctp->tag = tag;
  ctp->comm = comm;

  fprintf(stderr, "[HHRMPI_Isend] called: dest=%d, tag=%d\n", dest, tag);

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
  MPI_Send((void*)&hdr, sizeof(commhdr), MPI_BYTE, dest, tag, comm);
  /* start to wait ack */
  MPI_Irecv((void*)&ctp->hdr, sizeof(commhdr), MPI_BYTE, dest, tag, comm, &ctp->ireq);
  /* finish of ctp->ireq should be checked later */

  addCommTask(ctp);
  *reqp = ctp;

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

  fprintf(stderr, "[HHRMPI_Irecv] called: src=%d, tag=%d\n", source, tag);

  if (!isTypeContiguous(datatype)) {
    fprintf(stderr, "[HHRMPI_Irecv] non contiguous datatype is specified. not supported yet\n");
    exit(1);
  }

  /* Start recving the first header */
  MPI_Irecv((void*)&ctp->hdr, sizeof(commhdr), MPI_BYTE, source, tag, comm, &ctp->ireq);

  addCommTask(ctp);
  *reqp = ctp;

  return 0;
}

int HHRMPI_Testall(int n, HHRMPI_Request *reqs, int *flagp, MPI_Status *stats)
{
  int i;
  *flagp = 0;
  for (i = 0; i < n; i++) {
    if (reqs[i] == NULL) continue;
    if (reqs[i]->fin == 0) break;
  }
  
  if (i < n) {
    /* some request is not finished */
    return 0;
  }
  *flagp = 1;
  return MPI_SUCCESS;
}

int HHRMPI_Waitall(int n, HHRMPI_Request *reqs, MPI_Status *stats)
{
  do {
    int flag;
    int rc;
    rc = HHRMPI_Testall(n, reqs, &flag, stats);
    if (rc != MPI_SUCCESS) {
      fprintf(stderr, "[HHRMPI_Waltall] failed! rc=%d\n", rc);
      exit(1);
    }
    if (flag != 0) {
      break;
    }

    //HH_progressSched();
    HHRM_progress();
    usleep(1);
  } while (1);
  return MPI_SUCCESS;
}

int HHRMPI_Wait(HHRMPI_Request *reqp, MPI_Status *statp)
{
  return HHRMPI_Waitall(1, reqp, statp);
}

int HHRMPI_Send( void *buf, int count, MPI_Datatype dt, int dst, 
		int tag, MPI_Comm comm )
{
  int rc;
  HHRMPI_Request mreq;
  MPI_Status stat;
  rc = HHRMPI_Isend(buf, count, dt, dst, tag, comm, &mreq);
  if (rc != MPI_SUCCESS) {
    fprintf(stderr, "[HHRMPI] ERROR: HHRMPI_Isend failed\n");
    return rc;
  }

  rc = HHRMPI_Wait(&mreq, &stat);
  if (rc != MPI_SUCCESS) {
    fprintf(stderr, "[HHRMPI] ERROR: HHRMPI_Wait failed\n");
    return rc;
  }

  return rc;
}

int HHRMPI_Recv( void *buf, int count, MPI_Datatype dt, int src, 
		 int tag, MPI_Comm comm, MPI_Status *status )
{
  int rc;
  HHRMPI_Request mreq;
  MPI_Status stat;
  rc = HHRMPI_Irecv(buf, count, dt, src, tag, comm, &mreq);
  if (rc != MPI_SUCCESS) {
    fprintf(stderr, "[HHRMPI] ERROR: HHRMPI_Irecv failed\n");
    return rc;
  }

  rc = HHRMPI_Wait(&mreq, &stat);
  if (rc != MPI_SUCCESS) {
    fprintf(stderr, "[HHRMPI] ERROR: HHRMPI_Wait failed\n");
    return rc;
  }
  *status = stat;

  return rc;
}


