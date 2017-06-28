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

#define HHMR_CHUNKSIZE (16*1024*1024)

typedef struct {
  int count;
  MPI_Datatype datatype;
} commhdr;

struct commtask {
  commtask() {
    fin = 0; cursor = 0; curmem = 0; ireq = MPI_REQUEST_NULL; commbuf = NULL;
    tm_start = Wtime(); tm_rend = 0.0; tm_start_retry = 0.0;
  };
  int kind;
  void *ptr;
  int count;
  int psize;
  MPI_Datatype datatype;
  int partner;
  int tag;
  MPI_Comm comm;

  /* inner request */
  MPI_Request ireq;
  void *commbuf;
  int cursor;
  int curmem; // cursor for accessRec
  commhdr hdr;
  int fin;

  /* statistics */
  double tm_start; // time of creation of this structure
  double tm_rend; // time of rendezvous (first ACK is sent or recvd)
  double tm_start_retry; // time when blocking is started for busy heap
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

/* Rendezvous established in send side */
int sendRend(commtask *ctp)
{
  assert(ctp->tm_rend == 0.0);
  ctp->tm_rend = Wtime();

  HH_lockSched();
  HHL2->ncurcomms++;
  HH_unlockSched();

  return 0;
}

int sendFin(commtask *ctp)
{
  double et = Wtime();
  fprintf(stderr, "[HHMPI(R):progressSend@p%d] [%.2lf-%.2lf] sending %ld bytes finish in %.2lfsecs (dst=%d,tag=%d)\n",
	  HH_MYID, Wtime_conv_prt(ctp->tm_rend), Wtime_conv_prt(et), 
	  ctp->psize, et-ctp->tm_rend, ctp->partner, ctp->tag);
  free(ctp->commbuf);
  ctp->commbuf = NULL;
  HH_addHostMemStat(HHST_MPIBUFCOPY, -HHMR_CHUNKSIZE);
  ctp->ireq = MPI_REQUEST_NULL;
  ctp->fin = 1;

  HH_lockSched();
  HHL2->ncurcomms--;
  assert(HHL2->ncurcomms >= 0);
  HH_unlockSched();

  return 0;
}

int progressSend(commtask *ctp)
{
  int flag;
  MPI_Status stat;

  assert(ctp->fin == 0);

  if (ctp->ireq != MPI_REQUEST_NULL) {
    MPI_Test(&ctp->ireq, &flag, &stat);
    if (flag == 0) {
      /* ireq is not finished. do nothing */
      return 0;
    }
    ctp->ireq = MPI_REQUEST_NULL;

    if (ctp->cursor == 0) {
      /* first ack is recvd. */
      sendRend(ctp);
    }

  }

  assert(ctp->ireq == MPI_REQUEST_NULL);

#if 0
  if (ctp->cursor == 0 && ctp->tm_rend == 0.0) {
    /* first ack is recvd. */
    sendRend(ctp);
  }
#endif

  if (ctp->cursor >= ctp->psize) {
    sendFin(ctp);
    return 0;
  }

  /* divide original message and send */
  int size = HHMR_CHUNKSIZE;
  if (ctp->cursor+size > ctp->psize) size = ctp->psize-ctp->cursor;

  /* read from heap structure */
  int rc;
  rc = HH_accessRec('R', piadd(ctp->ptr, ctp->cursor), ctp->commbuf, HHM_HOST, size);
  if (rc != HHSS_OK) {
    /* do nothing and retry accessRec later */
    if (ctp->tm_start_retry == 0.0) {
      fprintf(stderr, "[HHMPI(R):progressSend@p%d] [%.2lf] accessRec ('R', cur=%d) failes, retry...\n",
	      HH_MYID, Wtime_prt(), ctp->cursor);
      ctp->tm_start_retry = Wtime();
    }
    usleep(10000);
    return 0;
  }

  if (ctp->tm_start_retry != 0.0) {
    double et = Wtime();
    double rt = et - ctp->tm_start_retry;
    fprintf(stderr, "[HHMPI(R):progressSend@p%d] [%.2lf-%.2lf] I have been blocked for busy heap for %.3lfsec\n",
	    HH_MYID, Wtime_conv_prt(ctp->tm_start_retry), Wtime_conv_prt(et), rt);
    ctp->tm_start_retry = 0.0;
  }

#if 0
  printf("[HHMPI(R):progressSend@p%d] calling Internal MPI_Isend(%ldbytes,dst=%d,tag=%d)\n",
	 HH_MYID, size, ctp->partner, HHMR_TAG_BODY(ctp->tag));
#endif
  MPI_Isend(ctp->commbuf, size, MPI_BYTE, ctp->partner, HHMR_TAG_BODY(ctp->tag),
	    ctp->comm, &ctp->ireq);

  ctp->cursor += size;

  return 0;
}

/* Rendezvous established in recv side */
int recvRend(commtask *ctp, MPI_Status *statp)
{
  assert(ctp->tm_rend == 0.0);
  ctp->tm_rend = Wtime();
  ctp->partner = statp->MPI_SOURCE;
  ctp->tag = statp->MPI_TAG;
  fprintf(stderr, "[HHMPI(R):progressRecv@p%d] [%.2lf] sending ACK for (src=%d, tag=%d)!\n",
	  HH_MYID, Wtime_prt(), ctp->partner, ctp->tag);

  HH_lockSched();
  HHL2->ncurcomms++;
  HH_unlockSched();
  return 0;
}

int recvFin(commtask *ctp)
{
  double et = Wtime();
  fprintf(stderr, "[HHMPI(R):progressRecv@p%d] [%.2lf-%.2lf] recving %ld bytes finish in %.2lfsecs (dst=%d,tag=%d)\n",
	  HH_MYID, Wtime_conv_prt(ctp->tm_rend), Wtime_conv_prt(et), 
	  ctp->psize, et-ctp->tm_rend, ctp->partner, ctp->tag);
  free(ctp->commbuf);
  ctp->commbuf = NULL;
  HH_addHostMemStat(HHST_MPIBUFCOPY, -HHMR_CHUNKSIZE);
  ctp->ireq = MPI_REQUEST_NULL;
  ctp->fin = 1;
  /* finished */

  HH_madvise(ctp->ptr, ctp->psize, HHMADV_NORMAL);

  HH_lockSched();
  HHL2->ncurcomms--;
  assert(HHL2->ncurcomms >= 0);
  HH_unlockSched();
  return 0;
}

int progressRecv(commtask *ctp)
{
  int flag;
  MPI_Status stat;
  assert(ctp->fin == 0);
  assert(ctp->curmem <= ctp->cursor);

  if (ctp->ireq != MPI_REQUEST_NULL) {
    MPI_Test(&ctp->ireq, &flag, &stat);
    if (flag == 0) {
      /* do nothing */
      return 0;
    }
    ctp->ireq = MPI_REQUEST_NULL;
  }

  assert(ctp->ireq == MPI_REQUEST_NULL);

  if (ctp->cursor > ctp->curmem) {
    int size = ctp->cursor - ctp->curmem;
    assert(size >= 0 && size <= HHMR_CHUNKSIZE);
    /* write to heap structure */
    int rc;
    rc = HH_accessRec('W', piadd(ctp->ptr, ctp->curmem), ctp->commbuf, HHM_HOST, size);

    if (rc != HHSS_OK) {
      /* do nothing and retry accessRec later */
      if (ctp->tm_start_retry == 0.0) {
	fprintf(stderr, "[HHMPI(R):progressSend@p%d] [%.2lf] accessRec ('W', cur=%d) failes, retry...\n",
		HH_MYID, Wtime_prt(), ctp->cursor);
	ctp->tm_start_retry = Wtime();
      }
      usleep(10000);
      return 0;
    }

    /* write to heap finished */
    ctp->curmem = ctp->cursor;

    if (ctp->tm_start_retry != 0.0) {
      double et = Wtime();
      double rt = et - ctp->tm_start_retry;
      fprintf(stderr, "[HHMPI(R):progressRecv@p%d] [%.2lf-%.2lf] I have been blocked for busy heap for %.3lfsec\n",
	      HH_MYID, Wtime_conv_prt(ctp->tm_start_retry), Wtime_conv_prt(et), rt);
      ctp->tm_start_retry = 0.0;
    }
  }

  assert(ctp->curmem == ctp->cursor);

  if (ctp->cursor == 0) {
    /* Now first header arrived. */
    recvRend(ctp, &stat);
    /* Send ack */
    MPI_Send((void*)&ctp->hdr, sizeof(commhdr), MPI_BYTE, 
	     ctp->partner, HHMR_TAG_ACK(ctp->tag), ctp->comm);
  }

  if (ctp->cursor >= ctp->psize) {
    recvFin(ctp);
    return 0;
  }

  /* start to recv divided messages */
  int size = HHMR_CHUNKSIZE;
  if (ctp->cursor+size > ctp->psize) size = ctp->psize-ctp->cursor;

#if 0
  printf("[HHMPI(R):progressRecv@p%d] calling Internal MPI_Irecv(%ldbytes,src=%d,tag=%d)\n",
	 HH_MYID, size, ctp->partner, HHMR_TAG_BODY(ctp->tag));
#endif
  MPI_Irecv(ctp->commbuf, size, MPI_BYTE, ctp->partner, HHMR_TAG_BODY(ctp->tag),
	    ctp->comm, &ctp->ireq);

  assert(ctp->ireq != MPI_REQUEST_NULL);

  ctp->cursor += size;

  return 0;
}

int progress1(commtask *ctp)
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
  MPI_Pack_size(ctp->count, ctp->datatype, ctp->comm, &ctp->psize);

  ctp->commbuf = valloc(HHMR_CHUNKSIZE);
  HH_addHostMemStat(HHST_MPIBUFCOPY, HHMR_CHUNKSIZE);

  fprintf(stderr, "[HHMPI_Isend(R)@p%d] [%.2lf] start: dest=%d, tag=%d\n", 
	  HH_MYID, Wtime_conv_prt(ctp->tm_start), dest, tag);

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

  /* change target's priority */
  HH_lockSched();
  HH_prioInc(dest, 1);
  HH_unlockSched();

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
  MPI_Pack_size(ctp->count, ctp->datatype, ctp->comm, &ctp->psize);

  ctp->commbuf = valloc(HHMR_CHUNKSIZE);
  HH_addHostMemStat(HHST_MPIBUFCOPY, HHMR_CHUNKSIZE);

  fprintf(stderr, "[HHMPI_Irecv(R)@p%d] [%.2lf] start: src=%d, tag=%d\n", 
	  HH_MYID, Wtime_conv_prt(ctp->tm_start), source, tag);

  if (!isTypeContiguous(datatype)) {
    fprintf(stderr, "[HHMPI_Irecv(R)@p%d] non contiguous datatype is specified. not supported yet\n",
	    HH_MYID);
    exit(1);
  }

  HH_madvise(ctp->ptr, ctp->psize, HHMADV_RECVONLY);

  /* Start recving the first header */
  MPI_Irecv((void*)&ctp->hdr, sizeof(commhdr), MPI_BYTE, source, tag, comm, &ctp->ireq);

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
  HH_enterBlockingForColl("Barrier");
  rc = MPI_Barrier(comm);
  HH_exitBlocking();

  return rc;
}

// TODO: other collective communications

#endif // HHMPI_REND
