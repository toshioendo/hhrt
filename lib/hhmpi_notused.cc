/* internally used in group comm */
int HHMPI_Send_i( void *buf, int count, MPI_Datatype dt, int dst, 
		int tag, MPI_Comm comm )
{
  int rc;
  MPI_Request mreq;
  MPI_Status stat;
  rc = MPI_Isend(buf, count, dt, dst, tag, comm, &mreq);
  if (rc != 0) {
    fprintf(stderr, "[HHMPI] ERROR: MPI_Isend failed\n");
    return rc;
  }

  rc = HHMPI_Waitall_i(1, &mreq, &stat);
  if (rc != 0) {
    fprintf(stderr, "[HHMPI] ERROR: HHMPI_Waitall_i failed\n");
    return rc;
  }

  return rc;
}

int HHMPI_Recv_i( void *buf, int count, MPI_Datatype dt, int src, 
		int tag, MPI_Comm comm, MPI_Status *status )
{
  int rc;
  MPI_Request mreq;
  MPI_Status stat;
  rc = MPI_Irecv(buf, count, dt, src, tag, comm, &mreq);
  if (rc != 0) {
    fprintf(stderr, "[HHMPI] ERROR: HHMPI_Irecv failed\n");
    return rc;
  }

  rc = HHMPI_Waitall_i(1, &mreq, &stat);
  if (rc != 0) {
    fprintf(stderr, "[HHMPI] ERROR: HHMPI_Wait failed\n");
    return rc;
  }

  return rc;
}


int HHMPI_Barrier_inner(MPI_Comm comm)
{
  int rank, size;
  int pa, ch1, ch2;
  MPI_Status stat;
  long padummy = 123, ch1dummy = 234, ch2dummy = 345;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  pa = (rank-1)/2;
  ch1 = 2*rank+1;
  ch2 = 2*rank+2;

  /* gather phase */
  if (ch1 < size) {
    /* recv from ch1 */
    HHMPI_Recv_i(&ch1dummy, 1, MPI_LONG, ch1, HHTAG_BARRIER, comm, &stat);
  }
  if (ch2 < size) {
    /* recv from ch2 */
    HHMPI_Recv_i(&ch2dummy, 1, MPI_LONG, ch2, HHTAG_BARRIER, comm, &stat);
  }
  if (rank > 0) {
    /* send to pa */
    HHMPI_Send_i(&padummy, 1, MPI_LONG, pa, HHTAG_BARRIER, comm);
  }

  /* scatter phase */
  if (rank > 0) {
    /* recv from pa */
    HHMPI_Recv_i(&padummy, 1, MPI_LONG, pa, HHTAG_BARRIER, comm, &stat);
  }
  if (ch1 < size) {
    /* send to ch1 */
    HHMPI_Send_i(&ch1dummy, 1, MPI_LONG, ch1, HHTAG_BARRIER, comm);
  }
  if (ch2 < size) {
    /* send to ch1 */
    HHMPI_Send_i(&ch2dummy, 1, MPI_LONG, ch2, HHTAG_BARRIER, comm);
  }

  return 0;
}



