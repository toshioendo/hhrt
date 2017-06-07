/* Under construction */
/* Implementing MPI rendezvous communication for reducing pinned memory */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>
#include "hhrt_impl.h"

#ifdef HHMPI_REND

#include <list>
#include <map>
using namespace std;

#define HHMR_TAG_OFFS 15000
#define HHMR_TAG_ACK(t) ((t)+HHMR_TAG_OFFS)
#define HHMR_TAG_BODY(t) ((t)+HHMR_TAG_OFFS)

enum {
  HHMR_SEND = 200,
  HHMR_RECV,
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

class rmpig {
 public:
  /* list of floating communications */
  list<commtask *> commtasks;
};

/* global variables */
rmpig hhrls;
rmpig *HHLMR = &hhrls;

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
  HHLMR->commtasks.push_front(ctp);
  return 0;
}

static int progressSend(commtask *ctp)
{
  int flag;
  MPI_Status stat;
  MPI_Test(&ctp->ireq, &flag, &stat);
  if (flag == 0) {
    /* do nothing */
    fprintf(stderr, "[HHMPI(R):progressSend@p%d] src=%d, tag=%d. not yet...\n",
	    HH_MYID, ctp->partner, ctp->tag);
    usleep(1);

    return 0;
  }

  {
    int count;
    MPI_Get_count(&stat, MPI_BYTE, &count);
    fprintf(stderr, "[HHMPI(R):progressSend@p%d] recvd ack (%dbytes,SOURCE=%d,TAG=%d) from %d\n",
	    HH_MYID, count, stat.MPI_SOURCE, stat.MPI_TAG, ctp->partner);
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

#if 1
    /* read from heap structure */
    HH_accessRec('R', piadd(ctp->ptr, cur), commbuf, HHM_HOST, size);
#else
    memcpy(commbuf, piadd(ctp->ptr, cur), size);
#endif
    /* This should be Isend, and returns for each send */
    printf("[HHMPI(R):progressSend@p%d] calling Internal MPI_Send(%ldbytes,dst=%d,tag=%d)\n",
	   HH_MYID, size, ctp->partner, HHMR_TAG_BODY(ctp->tag));
    MPI_Send(commbuf, size, MPI_BYTE, ctp->partner, HHMR_TAG_BODY(ctp->tag),
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
    fprintf(stderr, "[HHMPI(R):progressRecv@p%d] src=%d, tag=%d. not yet...\n",
	    HH_MYID, ctp->partner, ctp->tag);
    usleep(1);

    return 0;
  }

  /* Now first header arrived. */
  ctp->partner = stat.MPI_SOURCE;
  ctp->tag = stat.MPI_TAG;
  fprintf(stderr, "[HHMPI(R):progressRecv@p%d] found src=%d, tag=%d. sending ack...\n",
	  HH_MYID, ctp->partner, ctp->tag);
  fflush(0);
  /* Send ack */
  MPI_Send((void*)&ctp->hdr, sizeof(commhdr), MPI_BYTE, 
	   ctp->partner, HHMR_TAG_ACK(ctp->tag), ctp->comm);

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
    printf("[HHMPI(R):progressRecv@p%d] calling Internal MPI_Recv(%ldbytes,src=%d,tag=%d)\n",
	   HH_MYID, size, ctp->partner, HHMR_TAG_BODY(ctp->tag));
    MPI_Recv(commbuf, size, MPI_BYTE, ctp->partner, HHMR_TAG_BODY(ctp->tag),
	     ctp->comm, &stat);

#if 1
    /* write to heap structure */
    HH_accessRec('W', piadd(ctp->ptr, cur), commbuf, HHM_HOST, size);
#else
    memcpy(piadd(ctp->ptr, cur), commbuf, size);
#endif
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
  if (ctp->kind == HHMR_SEND) {
    progressSend(ctp);
  }
  else if (ctp->kind == HHMR_RECV) {
    progressRecv(ctp);
  }
  else {
    fprintf(stderr, "[HHMPI(R):progress1] Unknown kind %d\n", ctp->kind);
    exit(1);
  }

  return 0;
}

/* should be called to progress communication */
int HHMPIR_progress()
{
  list<commtask *>::iterator it;
  for (it = HHLMR->commtasks.begin(); it != HHLMR->commtasks.end(); it++) {
    commtask *ctp = *it;
    progress1(ctp);

    if (ctp->fin) {
      /* finished. remove from list */
      HHLMR->commtasks.erase(it);
      break;
    }
  }
  return 0;
}

int HHMPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
		int tag, MPI_Comm comm, HHMPI_Request *reqp)
{
  commtask *ctp = new commtask;
  ctp->kind = HHMR_SEND;
  ctp->ptr = (void *)buf;
  ctp->count = count;
  ctp->datatype = datatype;
  ctp->partner = dest;
  ctp->tag = tag;
  ctp->comm = comm;

  fprintf(stderr, "[HHMPI_Isend(R)@p%d] called: dest=%d, tag=%d\n", 
	  HH_MYID, dest, tag);

  if (!isTypeContiguous(datatype)) {
    fprintf(stderr, "[HHRTMPI_Isend(R)@p%d] non contiguous datatype is specified. not supported yet\n",
	    HH_MYID);
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
  MPI_Irecv((void*)&ctp->hdr, sizeof(commhdr), MPI_BYTE, dest, 
	    HHMR_TAG_ACK(tag), comm, &ctp->ireq);
  /* finish of ctp->ireq should be checked later */

  fprintf(stderr, "[HHMPI_Isend(R)@p%d] Irecv for ack--> request=0x%lx\n",
	  HH_MYID, ctp->ireq);

  addCommTask(ctp);
  *((commtask **)reqp) = ctp;

  return 0;
}

int HHMPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source,
		int tag, MPI_Comm comm, HHMPI_Request *reqp)
{
  commtask *ctp = new commtask;
  ctp->kind = HHMR_RECV;
  ctp->ptr = buf;
  ctp->count = count;
  ctp->datatype = datatype;
  ctp->partner = source;
  ctp->tag = tag;
  ctp->comm = comm;

  fprintf(stderr, "[HHMPI_Irecv(R)@p%d] called: src=%d, tag=%d\n", 
	  HH_MYID, source, tag);

  if (!isTypeContiguous(datatype)) {
    fprintf(stderr, "[HHMPI_Irecv(R)@p%d] non contiguous datatype is specified. not supported yet\n",
	    HH_MYID);
    exit(1);
  }

  /* Start recving the first header */
  MPI_Irecv((void*)&ctp->hdr, sizeof(commhdr), MPI_BYTE, source, tag, comm, &ctp->ireq);

  fprintf(stderr, "[HHMPI_Irecv(R)@p%d] Irecv for first hdr--> request=0x%lx\n",
	  HH_MYID, ctp->ireq);

  addCommTask(ctp);
  *((commtask **)reqp) = ctp;

  return 0;
}

int HHMPI_Testall(int n, HHMPI_Request *reqs, int *flagp, MPI_Status *stats)
{
  int i;
  *flagp = 0;
  for (i = 0; i < n; i++) {
    commtask *ctp = (commtask *)reqs[i];
    if (ctp  == NULL) continue;
    if (ctp->fin == 0) break;
  }
  
  if (i < n) {
    /* some request is not finished */
    return 0;
  }
  *flagp = 1;
  return MPI_SUCCESS;
}

int HHMPI_Waitall(int n, HHMPI_Request *reqs, MPI_Status *stats)
{
  HH_enterBlocking();

  do {
    int flag;
    int rc;
    rc = HHMPI_Testall(n, reqs, &flag, stats);
    if (rc != MPI_SUCCESS) {
      fprintf(stderr, "[HHMPI_Waltall(R)] failed! rc=%d\n", rc);
      exit(1);
    }
    if (flag != 0) {
      break;
    }

    HH_progressSched();
    HHMPIR_progress();

    usleep(1);
  } while (1);

  HH_exitBlocking();
  return MPI_SUCCESS;
}

int HHMPI_Wait(HHMPI_Request *reqp, MPI_Status *statp)
{
  return HHMPI_Waitall(1, reqp, statp);
}

int HHMPI_Send( void *buf, int count, MPI_Datatype dt, int dst, 
		int tag, MPI_Comm comm )
{
  int rc;
  HHMPI_Request mreq;
  MPI_Status stat;
  rc = HHMPI_Isend(buf, count, dt, dst, tag, comm, &mreq);
  if (rc != MPI_SUCCESS) {
    fprintf(stderr, "[HHMPI(R)] ERROR: HHMPI_Isend failed\n");
    return rc;
  }

  rc = HHMPI_Wait(&mreq, &stat);
  if (rc != MPI_SUCCESS) {
    fprintf(stderr, "[HHMPI(R)] ERROR: HHMPI_Wait failed\n");
    return rc;
  }

  return rc;
}

int HHMPI_Recv( void *buf, int count, MPI_Datatype dt, int src, 
		 int tag, MPI_Comm comm, MPI_Status *status )
{
  int rc;
  HHMPI_Request mreq;
  MPI_Status stat;
  rc = HHMPI_Irecv(buf, count, dt, src, tag, comm, &mreq);
  if (rc != MPI_SUCCESS) {
    fprintf(stderr, "[HHMPI(R)] ERROR: HHMPI_Irecv failed\n");
    return rc;
  }

  rc = HHMPI_Wait(&mreq, &stat);
  if (rc != MPI_SUCCESS) {
    fprintf(stderr, "[HHMPI(R)] ERROR: HHMPI_Wait failed\n");
    return rc;
  }
  *status = stat;

  return rc;
}


int HHMPI_Barrier(MPI_Comm comm)
{
  int rc;
  HH_enterGComm("Barrier");
  rc = MPI_Barrier(comm);
  HH_exitGComm();

  return rc;
}

// TODO: other collective communications

#endif // HHMPI_REND
