#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>
#include "hhrt_impl.h"

#ifdef HHMPI_BASE

class mpig {
 public:
  /* list of floating communications */
  std::map<MPI_Request, reqfin> reqfins;
};

/* global variables */
mpig hhlms;
mpig *HHLM = &hhlms;

/***/

#if defined USE_SWAPHOST

/* Returns 0 if type is "contiguous" on memory */
/* (This may not mean a type made by MPI_Type_contiguous) */
/* Returns 1 if type has holes, or unknown */
static int doesTypeNeedPacked(MPI_Datatype type)
{
  int nint, naddr, ntype, combiner;
  MPI_Type_get_envelope(type, &nint, &naddr, &ntype, &combiner);
  if (combiner == MPI_COMBINER_NAMED) {
    return 1;
  }
  return 0;
}

int HH_reqfin_setup_send(reqfin *finp, const void *buf, int count, MPI_Datatype datatype,
			 MPI_Comm comm)
{
  finp->send.packflag = doesTypeNeedPacked(datatype);

  int tsize;
  size_t bsize;

  MPI_Type_size(datatype, &tsize);
  bsize = (size_t)tsize*count;

  if (bsize >= (size_t)1<<31) {
    fprintf(stderr, "[HH_reqfin_setup_send@p%d] MESSAGE SIZE %ld TOO BIG!\n",
	    HH_MYID, bsize);
    exit(1);
  }

  finp->mode |= HHRF_SEND;
  finp->comm = comm;
  finp->send.orgptr = NULL; /* org* are not used later */

  if (finp->send.packflag) {
    int psize;
    int pos = 0;
    MPI_Pack_size(count, datatype, comm, &psize);

#if 0
    fprintf(stderr, "[HH_reqfin_setup_send@p%d] MPI_Pack_size ->%d (%d x %d)\n",
	    HH_MYID, psize, tsize, count);
#endif
    
    /* copy(MPI_Pack) to dedicated buffer before sending */
    finp->send.cptr = valloc(psize);
    finp->send.csize = psize;
    finp->send.ctype = MPI_PACKED;
    HH_addHostMemStat(HHST_MPIBUFCOPY, psize);
    MPI_Pack(buf, count, datatype, 
	     finp->send.cptr, psize, &pos, comm);
  }
  else {
    /* copy(memcpy) to dedicated buffer before sending */
    finp->send.cptr = valloc(bsize);
    finp->send.csize = count;
    finp->send.ctype = datatype;
    HH_addHostMemStat(HHST_MPIBUFCOPY, bsize);
    memcpy(finp->send.cptr, buf, bsize);
  }
  return 0;
}


int HH_reqfin_setup_recv(reqfin *finp, void *buf, int count, MPI_Datatype datatype,
			 MPI_Comm comm)
{
  finp->recv.packflag = doesTypeNeedPacked(datatype);

  int tsize;
  size_t bsize;

  MPI_Type_size(datatype, &tsize);
  bsize = (size_t)tsize*count;

  if (bsize >= (size_t)1<<31) {
    fprintf(stderr, "[HH_reqfin_setup_recv@p%d] MESSAGE SIZE %ld TOO BIG!\n",
	    HH_MYID, bsize);
    exit(1);
  }

  finp->mode |= HHRF_RECV;
  finp->comm = comm;
  if (finp->recv.packflag) {
    int psize;
    int pos = 0;
    
    MPI_Pack_size(count, datatype, comm, &psize);

    /* we should copy(Unpack) from dedicated buffer after receiving */
    
    finp->recv.cptr = valloc(psize);
    finp->recv.csize = psize;
    finp->recv.ctype = MPI_PACKED;
    finp->recv.orgptr = buf;
    finp->recv.orgsize = count;
    finp->recv.orgtype = datatype;
    HH_addHostMemStat(HHST_MPIBUFCOPY, psize);
  }
  else {
    /* we should copy(memcpy) from dedicated buffer after receiving */
    finp->recv.cptr = valloc(bsize);
    finp->recv.csize = count;
    finp->recv.ctype = datatype;
    finp->recv.orgptr = buf;
    finp->recv.orgsize = count;
    finp->recv.orgtype = datatype;
    HH_addHostMemStat(HHST_MPIBUFCOPY, bsize);
  }
  return 0;
}

int HH_reqfin_finalize(reqfin *finp)
{
  if (finp->mode & HHRF_SEND) {
    free(finp->send.cptr);
    HH_addHostMemStat(HHST_MPIBUFCOPY, -finp->send.csize);
  }

  if (finp->mode & HHRF_RECV) {
    if (finp->recv.packflag) {
      int pos = 0;
      MPI_Unpack(finp->recv.cptr, finp->recv.csize, &pos,
		 finp->recv.orgptr, finp->recv.orgsize, finp->recv.orgtype, 
		 finp->comm);
    }
    else {
      int tsize;
      size_t bsize;
      assert(finp->recv.ctype == finp->recv.orgtype);

      MPI_Type_size(finp->recv.ctype, &tsize);
      bsize = (size_t)tsize*finp->recv.csize;
      memcpy(finp->recv.orgptr, finp->recv.cptr, bsize);
    }
    free(finp->recv.cptr);
    HH_addHostMemStat(HHST_MPIBUFCOPY, -finp->recv.csize);
  }

  return 0;
}

int HH_req_finalize(MPI_Request req)
{
  if (req == MPI_REQUEST_NULL) {
    fprintf(stderr, "[HH_req_finalize@p%d] req = ..NULL. TO BE FIXED\n",
	    HH_MYID);
  }

  if (HHLM->reqfins.find(req) != HHLM->reqfins.end()) {
    /* find */
    reqfin fin = HHLM->reqfins[req];
    HH_reqfin_finalize(&fin);
    HHLM->reqfins.erase(req);
  }
  else {
    fprintf(stderr, "[HH_req_finalize@p%d] No request structure for req=%ld finalized, is it OK?\n",
	    HH_MYID, (long)req);
    std::map<MPI_Request, reqfin>::iterator iter = HHLM->reqfins.begin();
    for (; iter != HHLM->reqfins.end(); iter++) {
      fprintf(stderr, "   req=%ld in map\n", (long)iter->first);
    }
  }
  return 0;
}

#else // !USE_SWAPHOST

int HH_reqfin_setup_send(reqfin *finp, const void *buf, int count, MPI_Datatype datatype,
			 MPI_Comm comm)
{
  finp->send.cptr = (void*)buf;
  finp->send.csize = count;
  finp->send.ctype = datatype;
  return 0;
}

int HH_reqfin_setup_recv(reqfin *finp, void *buf, int count, MPI_Datatype datatype,
			 MPI_Comm comm)
{
  finp->recv.cptr = (void*)buf;
  finp->recv.csize = count;
  finp->recv.ctype = datatype;
  return 0;
}

int HH_reqfin_finalize(reqfin* finp)
{
  return 0;
}

int HH_req_finalize(MPI_Request req)
{
  // do not nothing 
  return 0;
}

#endif // USE_SWAPHOST


int HHMPI_Wait(MPI_Request *reqp, MPI_Status *statp)
{
  return HHMPI_Waitall(1, reqp, statp);
}

// With USE_SWAPHOST, caller should gurantee that contents of reqs, stats
// are not swapped out.
int HHMPI_Waitall_i(int n, MPI_Request *reqs, MPI_Status *stats)
{
  do {
    int flag;
    int rc;
    rc = MPI_Testall(n, reqs, &flag, stats);
    if (rc != MPI_SUCCESS) {
      fprintf(stderr, "[HHMPI_Waltall] failed! rc=%d\n", rc);
      exit(1);
    }
    if (flag != 0) {
      break;
    }

    HH_progressSched();
    usleep(1);
  } while (1);

  return MPI_SUCCESS;
}

int HHMPI_Waitall(int n, MPI_Request *reqs, MPI_Status *stats)
{
  double st, et;
  MPI_Request *bakreqs;
  MPI_Status *bakstats;

  bakreqs = (MPI_Request*)malloc(sizeof(MPI_Request)*n);
  memcpy(bakreqs, reqs, sizeof(MPI_Request)*n);
  bakstats = (MPI_Status*)malloc(sizeof(MPI_Status)*n);

  HH_enterBlocking();

  st = Wtime();
  HHMPI_Waitall_i(n, bakreqs, bakstats);

#if 0
  et = Wtime();
  long ms = (long)(1000.0*(et-st));
  if (1) {
    fprintf(stderr, "[HHMPI_Waitall@p%d] [%.2lf] BLOCKED (%ldms) -> RUNNABLE\n", 
	    HH_MYID, Wtime_conv_prt(et), ms]);
  }
#endif

  HH_exitBlocking();

  {
    double t0 = Wtime(), t1;
    /* request finalizer */
#if 0
    fprintf(stderr, "[HHMPI_Waitall@p%d] calling HH_req_finalize() for %d reqs\n",
	    HH_MYID, n);
#endif
    for (int i = 0; i < n; i++) {
      HH_req_finalize(reqs[i]);
      reqs[i] = bakreqs[i];
      stats[i] = bakstats[i];
    }
    t1 = Wtime();
#if 1
    if (t1-t0 > 0.1) {
      fprintf(stderr, "[HHMPI_Waitall@p%d] req fin took %.1lf msec\n",
	      HH_MYID, (t1-t0)*1000.0);
    }
#endif
  }
  free(bakstats);
  free(bakreqs);
  return 0;
}

/****/
int HHMPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
		int tag, MPI_Comm comm, MPI_Request *reqp)
{
  reqfin fin; /* task to be done later */
  fin.mode = 0;
  HH_reqfin_setup_send(&fin, buf, count, datatype, comm);

  MPI_Isend(fin.send.cptr, fin.send.csize, fin.send.ctype, 
	    dest, tag, comm, reqp);

  HHLM->reqfins[*reqp] = fin;
#if 0
  fprintf(stderr, "[HHMPI_Isend@p%d] added req=%ld\n",
	  HH_MYID, (long)(*reqp));
#endif

  return MPI_SUCCESS;
}

int HHMPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source,
		int tag, MPI_Comm comm, MPI_Request *reqp)
{
  reqfin fin; /* task to be done later */
  fin.mode = 0;
  HH_reqfin_setup_recv(&fin, buf, count, datatype, comm);

  MPI_Irecv(fin.recv.cptr, fin.recv.csize, fin.recv.ctype, 
	    source, tag, comm, reqp);

  HHLM->reqfins[*reqp] = fin;
#if 0
  fprintf(stderr, "[HHMPI_Irecv@p%d] added req=%ld\n",
	  HH_MYID, (long)(*reqp));
#endif

  return MPI_SUCCESS;
}

int HHMPI_Send( void *buf, int count, MPI_Datatype dt, int dst, 
		int tag, MPI_Comm comm )
{
  int rc;
  MPI_Request mreq;
  MPI_Status stat;
  rc = HHMPI_Isend(buf, count, dt, dst, tag, comm, &mreq);
  if (rc != MPI_SUCCESS) {
    fprintf(stderr, "[HHMPI] ERROR: HHMPI_Isend failed\n");
    return rc;
  }

  rc = HHMPI_Wait(&mreq, &stat);
  if (rc != MPI_SUCCESS) {
    fprintf(stderr, "[HHMPI] ERROR: HHMPI_Wait failed\n");
    return rc;
  }

  return rc;
}

int HHMPI_Recv( void *buf, int count, MPI_Datatype dt, int src, 
		int tag, MPI_Comm comm, MPI_Status *status )
{
  int rc;
  MPI_Request mreq;
  MPI_Status stat;
  rc = HHMPI_Irecv(buf, count, dt, src, tag, comm, &mreq);
  if (rc != MPI_SUCCESS) {
    fprintf(stderr, "[HHMPI] ERROR: HHMPI_Irecv failed\n");
    return rc;
  }

  rc = HHMPI_Wait(&mreq, &stat);
  if (rc != MPI_SUCCESS) {
    fprintf(stderr, "[HHMPI] ERROR: HHMPI_Wait failed\n");
    return rc;
  }
  *status = stat;

  return rc;
}

/****** group comm */

static int is_me(MPI_Comm comm, int root)
{
  int myrank;
  MPI_Comm_rank(comm, &myrank);
  return (myrank == root);
}

int HHMPI_Barrier(MPI_Comm comm)
{
  int rc;
  HH_enterBlockingForColl("Barrier");
  rc = MPI_Barrier(comm);
  HH_exitBlocking();

  return rc;
}

int HHMPI_Bcast( void *buffer, int count, MPI_Datatype datatype, int root, 
		 MPI_Comm comm )
{
  int rc;
  reqfin fin; /* task to be done later */
  fin.mode = 0;
  if (is_me(comm, root)) {
    HH_reqfin_setup_send(&fin, buffer, count, datatype, comm);
  }
  else {
    HH_reqfin_setup_recv(&fin, buffer, count, datatype, comm);
  }

  HH_enterBlockingForColl("Bcast");
  if (is_me(comm, root)) {
    rc = MPI_Bcast(fin.send.cptr, fin.send.csize, fin.send.ctype, root, comm);
  }
  else {
    rc = MPI_Bcast(fin.recv.cptr, fin.recv.csize, fin.recv.ctype, root, comm);
  }
  HH_exitBlocking();

  HH_reqfin_finalize(&fin);

  return rc;
}

int HHMPI_Reduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
		 MPI_Op op, int root, MPI_Comm comm)
{
  int rc;
  reqfin fin; /* task to be done later */
  fin.mode = 0;
  HH_reqfin_setup_send /*Red*/(&fin, sendbuf, count, datatype, comm);
  if (is_me(comm, root)) {
    HH_reqfin_setup_recv /*Red*/(&fin, recvbuf, count, datatype, comm);
  }

  HH_enterBlockingForColl("Reduce");
  rc = MPI_Reduce(fin.send.cptr, fin.recv.cptr, fin.send.csize, fin.send.ctype,
		  op, root, comm);
  HH_exitBlocking();

  HH_reqfin_finalize(&fin);

  return rc;
}

int HHMPI_Allreduce(void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
  int rc;
  reqfin fin; /* task to be done later */
  fin.mode = 0;
  HH_reqfin_setup_send /*Red*/(&fin, sendbuf, count, datatype, comm);
  HH_reqfin_setup_recv /*Red*/(&fin, recvbuf, count, datatype, comm);

  HH_enterBlockingForColl("Allreduce");
  rc = MPI_Allreduce(fin.send.cptr, fin.recv.cptr, fin.send.csize, fin.send.ctype,
		     op, comm);
  HH_exitBlocking();

  HH_reqfin_finalize(&fin);

  return rc;
}

int HHMPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
		 void *recvbuf, int recvcount, MPI_Datatype recvtype, 
		 int root, MPI_Comm comm)
{
  int rc;
  reqfin fin; /* task to be done later */
  int np;
  fin.mode = 0;
  MPI_Comm_size(comm, &np);
  HH_reqfin_setup_send(&fin, sendbuf, sendcount, sendtype, comm);
  if (is_me(comm, root)) {
    HH_reqfin_setup_recv(&fin, recvbuf, recvcount*np, recvtype, comm);
  }

  HH_enterBlockingForColl("Gather");
  MPI_Gather(fin.send.cptr, fin.send.csize, fin.send.ctype,
	     fin.recv.cptr, fin.recv.csize/np, fin.recv.ctype, root, comm);
  HH_exitBlocking();

  HH_reqfin_finalize(&fin);

  return rc;
}

int HHMPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
		    void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
{
  int rc;
  reqfin fin; /* task to be done later */
  int np;
  fin.mode = 0;
  MPI_Comm_size(comm, &np);
  HH_reqfin_setup_send(&fin, sendbuf, sendcount, sendtype, comm);
  HH_reqfin_setup_recv(&fin, recvbuf, recvcount*np, recvtype, comm);

  HH_enterBlockingForColl("Allgather");
  MPI_Allgather(fin.send.cptr, fin.send.csize, fin.send.ctype,
		fin.recv.cptr, fin.recv.csize/np, fin.recv.ctype, comm);
  HH_exitBlocking();

  HH_reqfin_finalize(&fin);

  return rc;
}

int HHMPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
		  void *recvbuf, int recvcount, MPI_Datatype recvtype, 
		  int root, MPI_Comm comm)
{
  int rc;
  reqfin fin; /* task to be done later */
  int np;
  fin.mode = 0;
  MPI_Comm_size(comm, &np);
  if (is_me(comm, root)) {
    HH_reqfin_setup_send(&fin, sendbuf, sendcount*np, sendtype, comm);
  }
  HH_reqfin_setup_recv(&fin, recvbuf, recvcount, recvtype, comm);

  HH_enterBlockingForColl("Scatter");
  MPI_Scatter(fin.send.cptr, fin.send.csize/np, fin.send.ctype,
	      fin.recv.cptr, fin.recv.csize, fin.recv.ctype, root, comm);
  HH_exitBlocking();

  HH_reqfin_finalize(&fin);

  return rc;
}

int HHMPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
		   void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
{
  int rc;
  reqfin fin; /* task to be done later */
  int np;
  fin.mode = 0;
  MPI_Comm_size(comm, &np);
  HH_reqfin_setup_send(&fin, sendbuf, sendcount*np, sendtype, comm);
  HH_reqfin_setup_recv(&fin, recvbuf, recvcount*np, recvtype, comm);

  HH_enterBlockingForColl("Alltoall");
  MPI_Alltoall(fin.send.cptr, fin.send.csize/np, fin.send.ctype,
	      fin.recv.cptr, fin.recv.csize/np, fin.recv.ctype, comm);
  HH_exitBlocking();

  HH_reqfin_finalize(&fin);

  return rc;
}

int HHMPI_Comm_split(MPI_Comm comm, int color, int key,
		     MPI_Comm *newcomm)
{
  int rc;
  MPI_Comm newcomm2;
  HH_enterBlockingForColl("Comm_Split");
  rc = MPI_Comm_split(comm, color, key, &newcomm2);
  HH_exitBlocking();
  *newcomm = newcomm2;
  return rc;
}

#endif // HHMPI_BASE
