/* A sample code of HHRT */
/* 7-point 3D stencil with 2D decomposition */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <assert.h>
#include <mpi.h>
#include <cuda_runtime.h>

#include "hhrt.h"

/* utility functions */
/* walltime clock */
static double Wtime()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec * 0.000001;
}

#if 1
#  define REAL float
#  define REAL_MT MPI_FLOAT
#else
#  define REAL double
#  define REAL_MT MPI_DOUBLE
#endif

#define USE_MADVISE
#define INIT_ON_GPU

#define GPUS_PER_NODE 1 //-1 // -1 means all GPUs on the node

#define BSX 32
#define BSY 8

#define IDX(x0, y0, z0) ((x0)+(y0)*bufx+(z0)*bufx*bufy)

/* communication information for each direction */
struct comminfo {
  int poffy; // -1 or 0 or +1
  int poffz; // -1 or 0 or +1
  int county;
  int countz;
  int sidxy;
  int sidxz;
  int ridxy;
  int ridxz;

  size_t bufsize; // bufx * county * countz * sizeof(REAL)
  REAL *sbuf; // send buffer 
  REAL *rbuf; // recv buffer
} comminfo[8];


int init();
int mainloop();
int call_kernel(int bufid, int halox, int haloy, int haloz);
int comm_boundary(int bufid);

/* global variables */
int myid;
int nprocs;
int myy;
int myz;
int npy;
int npz;

int nx;
int ny;
int nz;
int nt;
int sy;
int ey;
int sz;
int ez;

int bt; /* temporal blocking factor */

/* buffer size */
int bufx;
int bufy;
int bufz;
size_t bufsize;

REAL *hp; /* buffer on host memory */
REAL *dps[2]; /* buffers on device memory (double buffer) */

/* for boundary comm */
size_t ybsize;
REAL *ybuf[4]; 
size_t zbsize;
REAL *zbuf[4]; 

size_t ss; /* statistics. sent amount in REAL type */
size_t rs; /* statistics. rcvd amount in REAL type */


/*********************************/
int setup_comminfo()
{
  struct comminfo *cip;
  int i;

  // [0,-1,-1]
  cip = &comminfo[0];
  cip->poffy = -1;
  cip->poffz = -1;
  cip->county = bt;
  cip->countz = bt;
  cip->sidxy = bt;
  cip->sidxz = bt;
  cip->ridxy = 0;
  cip->ridxz = 0;

  // [0,0,-1]
  cip = &comminfo[1];
  cip->poffy = 0;
  cip->poffz = -1;
  cip->county = bufy-2*bt;
  cip->countz = bt;
  cip->sidxy = bt;
  cip->sidxz = bt;
  cip->ridxy = bt;
  cip->ridxz = 0;

  // [0,+1,-1]
  cip = &comminfo[2];
  cip->poffy = +1;
  cip->poffz = -1;
  cip->county = bt;
  cip->countz = bt;
  cip->sidxy = bufy-2*bt;
  cip->sidxz = bt;
  cip->ridxy = bufy-bt;
  cip->ridxz = 0;

  // [0,-1,0]
  cip = &comminfo[3];
  cip->poffy = -1;
  cip->poffz = 0;
  cip->county = bt;
  cip->countz = bufz-2*bt;
  cip->sidxy = bt;
  cip->sidxz = bt;
  cip->ridxy = 0;
  cip->ridxz = bt;

  // [0,+1,0]
  cip = &comminfo[4];
  cip->poffy = +1;
  cip->poffz = 0;
  cip->county = bt;
  cip->countz = bufz-2*bt;
  cip->sidxy = bufy-2*bt;
  cip->sidxz = bt;
  cip->ridxy = bufy-bt;
  cip->ridxz = bt;

  // [0,-1,+1]
  cip = &comminfo[5];
  cip->poffy = -1;
  cip->poffz = +1;
  cip->county = bt;
  cip->countz = bt;
  cip->sidxy = bt;
  cip->sidxz = bufz-2*bt;
  cip->ridxy = 0;
  cip->ridxz = bufz-bt;

  // [0,0,+1]
  cip = &comminfo[6];
  cip->poffy = 0;
  cip->poffz = +1;
  cip->county = bufy-2*bt;
  cip->countz = bt;
  cip->sidxy = bt;
  cip->sidxz = bufz-2*bt;
  cip->ridxy = bt;
  cip->ridxz = bufz-bt;

  // [0,+1,+1]
  cip = &comminfo[7];
  cip->poffy = +1;
  cip->poffz = +1;
  cip->county = bt;
  cip->countz = bt;
  cip->sidxy = bufy-2*bt;
  cip->sidxz = bufz-2*bt;
  cip->ridxy = bufy-bt;
  cip->ridxz = bufz-bt;

  /* setup buffers */
  size_t sum = 0L;
  for (i = 0; i < 8; i++) {
    cudaError_t crc;
    int bpy;
    int bpz;
    cip = &comminfo[i];
    bpy = myy+cip->poffy;
    bpz = myz+cip->poffz;
    if (bpy >= 0 && bpy < npy && bpz >= 0 && bpz < npz) {
      cip->bufsize = bufx * cip->county * cip->countz * sizeof(REAL);
      crc = cudaMallocHost((void**)&cip->sbuf, cip->bufsize);
      if (crc != cudaSuccess) {perror("cudaMallocHost");exit(1);}
      sum += cip->bufsize;
      crc = cudaMallocHost((void**)&cip->rbuf, cip->bufsize);
      if (crc != cudaSuccess) {perror("cudaMallocHost");exit(1);}
      sum += cip->bufsize;
    }
    else {
      cip->sbuf = NULL;
      cip->rbuf = NULL;
    }
  }
  fprintf(stderr, "Process %d allocated %ldMiB for MPI buffer\n",
	  myid, sum >> 20L);
  return 0;
}

#ifdef INIT_ON_GPU
__global__ void init_array_gpu(REAL *buf,
			       int nx, int ny, int nz,
			       int sy, int sz, int bt,
			       int bufx, int bufy, int bufz)
{
  int jx = blockDim.x*blockIdx.x + threadIdx.x;
  int jy = blockDim.y*blockIdx.y + threadIdx.y;
  int jz;
  int idx;
  size_t offz = bufx*bufy;

  if (jx >= bufx || jy >= bufy) {
    return;
  }

  REAL ry = (REAL)(jy+sy-bt)/ny;
  if (ry < (REAL)0.0) ry = (REAL)0.0;
  if (ry > (REAL)1.0) ry = (REAL)1.0;
  REAL rx = (REAL)(jx-1)/nx;
  if (rx < (REAL)0.0) rx = (REAL)0.0;
  if (rx > (REAL)1.0) rx = (REAL)1.0;

  idx = IDX(jx, jy, 0);

#pragma unroll
  for (jz = 0; jz < bufz; jz++) { /* marching in z direction */
    REAL rz = (REAL)(jz+sz-bt)/nz;
    if (rz < (REAL)0.0) rz = (REAL)0.0;
    if (rz > (REAL)1.0) rz = (REAL)1.0;
    REAL v = rx*rx+ry*ry+rz*rz;
    buf[idx] = v;

    idx += offz;
    __syncthreads();
  }

}
#else /* !INIT_ON_GPU */

void init_array_cpu(REAL *buf,
		    int nx, int ny, int nz,
		    int sy, int sz, int bt,
		    int bufx, int bufy, int bufz)
{
  int jz;
#pragma omp parallel for
  for (jz = 0; jz < bufz; jz++) {
    int jy;
    REAL rz = (REAL)(jz+sz-bt)/nz;
    if (rz < (REAL)0.0) rz = (REAL)0.0;
    if (rz > (REAL)1.0) rz = (REAL)1.0;
    for (jy = 0; jy < bufy; jy++) {
      int jx;
      REAL ry = (REAL)(jy+sy-bt)/ny;
      if (ry < (REAL)0.0) ry = (REAL)0.0;
      if (ry > (REAL)1.0) ry = (REAL)1.0;
      for (jx = 0; jx < nx+2; jx++) {
	REAL rx = (REAL)(jx-1)/nx;
	if (rx < (REAL)0.0) rx = (REAL)0.0;
	if (rx > (REAL)1.0) rx = (REAL)1.0;
	REAL v = rx*rx+ry*ry+rz*rz;
	buf[IDX(jx, jy, jz)] = v;
      }
    }
  }
  return;
}

#endif

int init()
{
  cudaError_t crc;
  int bufid;

  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (nx <= 0 || ny <= 0 || nz <= 0 || nt <= 0) {
    fprintf(stderr, "ERROR! [nx, ny, nz, nt] = [%d, %d, %d, %d] INVALID\n",
	    nx, ny, nz, nt);
    MPI_Finalize();
    exit(1);
  }

  if (npy < 0) {
    /* default: 1d division */
    npy = 1;
    npz = nprocs;
  }

  if (npy <= 0 || npz <= 0 || npy*npz != nprocs) {
    fprintf(stderr, "ERROR! [npy, npz, nprocs] = [%d, %d, %d] INVALID\n",
	    npy, npz, nprocs);
    MPI_Finalize();
    exit(1);
  }

  myy = myid % npy;
  myz = myid / npy;

  if (myid == 0) {
    fprintf(stderr, "usercode(): n-temporal-block=%d, %dx%dx%dx%d\n",
	    bt, nx, ny, nz, nt);
  }

#ifdef GPUS_PER_NODE
  int devid;
  int ndevs = 1;
  if (GPUS_PER_NODE > 0) {
    ndevs = GPUS_PER_NODE;
  }
  else {
    cudaGetDeviceCount(&ndevs);
  }
  devid = myid % ndevs;
  fprintf(stderr, "Process %d uses gpu %d (ndevs=%d)\n", myid, devid, ndevs);
  cudaSetDevice(devid);
#endif

  /* data region to be computed by this process */
  sy = (ny*myy+npy-1)/npy;
  ey = (ny*(myy+1)+npy-1)/npy;
  if (ey > ny) ey = ny;
  sz = (nz*myz+npz-1)/npz;
  ez = (nz*(myz+1)+npz-1)/npz;
  if (ez > nz) ez = nz;

  /* buffer size including halo */
  bufx = nx+2;
  bufy = ey-sy+2*bt;
  bufz = ez-sz+2*bt;
  bufsize = sizeof(REAL)*bufx*bufy*bufz;

  setup_comminfo();

  /* allocate device memory */
  for (bufid = 0; bufid < 2; bufid++) {
    crc = cudaMalloc((void**)&(dps[bufid]), bufsize);
    if (crc != cudaSuccess) {perror("cudaMalloc");exit(1);}
  }
#ifdef USE_MADVISE
  HH_madvise(dps[1], bufsize, HHMADV_CANDISCARD); /* hint for optimization */
#endif

#ifdef INIT_ON_GPU
  hp = NULL;

  init_array_gpu<<<dim3((bufx+BSX-1)/BSX, (bufy+BSY-1)/BSY, 1),
    dim3(BSX, BSY, 1)>>>
    (dps[0], nx, ny, nz,
     sy, sz, bt, bufx, bufy, bufz);

#else /* !INIT_ON_GPU */

  hp = (REAL*)malloc(bufsize);
  if (hp == NULL) {perror("malloc");exit(1);}

  init_array_cpu(hp, nx, ny, nz,
     sy, sz, bt, bufx, bufy, bufz);

  cudaMemcpy(dps[0], hp, bufsize, cudaMemcpyHostToDevice);
#ifdef USE_MADVISE
  HH_madvise(hp, bufsize, HHMADV_CANDISCARD); /* hint for optimization */
#endif

#endif /* !INIT_ON_GPU */

  return 0;
}


/*********************************/
__global__ void gpu_kernel(REAL *frombuf, REAL *tobuf, 
			   int lsx, int lex, int lsy, int ley, int lsz, int lez,
			   int bufx, int bufy,
			   REAL ce,REAL cw,REAL cn,REAL cs,
			   REAL ct,REAL cb,REAL cc)
{
  int jx = blockDim.x*blockIdx.x + threadIdx.x;
  int jy = blockDim.y*blockIdx.y + threadIdx.y;
  int jz;
  int idx;
  size_t offy = bufx;
  size_t offz = bufx*bufy;

  if (jx < lsx || jy < lsy || jx >= lex || jy >= ley) {
    return;
  }

  idx = IDX(jx, jy, lsz);

  REAL vc = frombuf[idx-offz];
  REAL vt = frombuf[idx];
#pragma unroll
  for (jz = lsz; jz < lez; jz++) { /* marching in z direction */
    REAL ve = frombuf[idx+1];
    REAL vw = frombuf[idx-1];
    REAL vn = frombuf[idx+offy];
    REAL vs = frombuf[idx-offy];

    REAL vb = vc;
    vc = vt;
    vt = frombuf[idx+offz];

    tobuf[idx] = cc*vc + ce*ve + cw*vw + cn*vn + cs*vs + ct*vt + cb*vb;
    idx += offz;
    __syncthreads();
  }

#define FLOP_PER_POINT 13
}

int call_kernel(int bufid, int halox, int haloy, int haloz) 
{
  REAL c = 0.3/6.0;
  REAL ce = c, cw = c, cn = c, cs = c, ct = c, cb = c;
  REAL cc = 1.0-(ce+cw+cn+cs+ct+cb);
  int lsx = halox, lex = bufx-halox;
  int lsy = haloy, ley = bufy-haloy;
  int lsz = haloz, lez = bufz-haloz;

  gpu_kernel<<<dim3((bufx+BSX-1)/BSX, (bufy+BSY-1)/BSY, 1),
    dim3(BSX, BSY, 1)>>>
    (dps[bufid], dps[1-bufid], lsx, lex, lsy, ley, lsz, lez,
     bufx, bufy, 
     ce, cw, cn, cs, ct, cb, cc);

  return 0;
}

int comm_boundary(int bufid)
{
  int idx;
  MPI_Status stats[16];
  double st, et;
  long ms;
  int i;
  //int logflag = (myy < 2 && myz < 3);
  int logflag = 1;

  MPI_Request reqs[16];
  int nreqs;

  nreqs = 0;
  ss = 0;
  rs = 0;

  /* CUDA D2H */
  st = Wtime();
  for (i = 0; i < 8; i++) {
    struct comminfo *cip = &comminfo[i];
    /* communication buddy */
    int bpy = myy+cip->poffy;
    int bpz = myz+cip->poffz;
    if (bpy >= 0 && bpy < npy && bpz >= 0 && bpz < npz) {
      idx = IDX(0, cip->sidxy, cip->sidxz);
      cudaMemcpy2D(cip->sbuf, sizeof(REAL)*bufx*cip->county, dps[bufid]+idx, sizeof(REAL)*bufx*bufy,
		   sizeof(REAL)*bufx*cip->county, cip->countz, cudaMemcpyDeviceToHost);
    }
  }
  
  cudaDeviceSynchronize();
  et = Wtime();

  /* MPI Isend/Irecv */
  for (i = 0; i < 8; i++) {
    struct comminfo *cip = &comminfo[i];
    /* communication buddy */
    int bpy = myy+cip->poffy;
    int bpz = myz+cip->poffz;
    if (bpy >= 0 && bpy < npy && bpz >= 0 && bpz < npz) {
      int bp = bpy+bpz*npy;
      if(bp >= 0 && bp < npy*npz) {
      }
      else {
	fprintf(stderr, "ERROR myy=%d, myz=%d, bpy=%d, bpz=%d, bp=%d\n",
		myy, myz, bpy, bpz, bp);
      }
      assert(bp >= 0 && bp < npy*npz);

      MPI_Isend(cip->sbuf, cip->bufsize/sizeof(REAL), REAL_MT, bp, 
		0, MPI_COMM_WORLD, &reqs[nreqs]);
      nreqs++; ss += cip->bufsize/sizeof(REAL);

      MPI_Irecv(cip->rbuf, cip->bufsize/sizeof(REAL), REAL_MT, bp,
		0, MPI_COMM_WORLD, &reqs[nreqs]);
      nreqs++; rs += cip->bufsize/sizeof(REAL);
    }
  }

  ms = (long)((et-st)*1000);
  if (logflag) {
    fprintf(stderr,
	    "[comm_boundary@p%d] cudaMemcpy D2H %ldMB took %ldms. call Waitall...\n",
	    myid, (sizeof(REAL)*ss)>>20, ms);
  }
  
  /* wait all */
  st = Wtime();
  MPI_Waitall(nreqs, reqs, stats);
  et = Wtime();

  if (myy < 2 && myz < 2) {
    fprintf(stderr,
	    "[comm_boundary@p%d] MPI_Waitall took %5.2lfsec\n",
	    myid, et-st);
  }

  st = Wtime();
  /* CUDA H2D */
  for (i = 0; i < 8; i++) {
    struct comminfo *cip = &comminfo[i];
    /* communication buddy */
    int bpy = myy+cip->poffy;
    int bpz = myz+cip->poffz;
    if (bpy >= 0 && bpy < npy && bpz >= 0 && bpz < npz) {
      idx = IDX(0, cip->ridxy, cip->ridxz);
      cudaMemcpy2D(dps[bufid]+idx, sizeof(REAL)*bufx*bufy, cip->rbuf, sizeof(REAL)*bufx*cip->county, 
		   sizeof(REAL)*bufx*cip->county, cip->countz, cudaMemcpyHostToDevice);
    }
  }
  cudaDeviceSynchronize();
  et = Wtime();
  ms = (long)((et-st)*1000);
  if (logflag) {
    fprintf(stderr,
	    "[comm_boundary@p%d] cudaMemcpy H2D %ldMB took %ldms\n",
	    myid, (sizeof(REAL)*rs)>>20, ms);
  }

  return 0;
}

int mainloop()
{
  int iter;
  int bufid = 0;
  //int logflag = (myy < 2 && myz < 3);
  int logflag = 1;
  double st0 = Wtime();

  for (iter = 0; iter < nt; iter += bt) {
    int ntinner;
    int ii;
    if (nt-iter < bt) {
      ntinner = nt-iter;
    }
    else ntinner = bt;

    if (logflag) {
      fprintf(stderr, "### Rank %d: Iter [%d,%d) start\n", 
	      myid, iter, iter+ntinner);
    }

    double st = Wtime();
#ifdef USE_MADVISE
    HH_madvise(dps[1-bufid], bufsize, HHMADV_CANDISCARD); /* hint for optimization */
#endif
    comm_boundary(bufid);
#ifdef USE_MADVISE
    HH_madvise(dps[1-bufid], bufsize, HHMADV_NORMAL); /* hint for optimization */
#endif
    double et = Wtime();
    long ms = (long)((et-st)*1000);
    if (logflag) {
      fprintf(stderr,
	      "[mainloop@p%d] COMM %ld ms to send %ldMB, recv %ldMB\n",
	      myid, ms, (ss*sizeof(REAL)) >>20 , (rs*sizeof(REAL)) >>20);
    }

    /* bt-steps local computation */
    st = Wtime();
    for (ii = 0; ii < ntinner; ii++) {
      
      /* computation */
      /* domain is shrinking along ii */
      call_kernel(bufid, 1, ii+1, ii+1);

      bufid = 1-bufid;
    }
    cudaDeviceSynchronize();

    if (logflag) {
      et = Wtime();
      ms = (long)((et-st)*1000);
      long flop = (long)nx*(ey-sy)*(ez-sz)*ntinner*FLOP_PER_POINT;
      double gflops = (double)flop/(et-st)/1.0e+9;
#if 0
      fprintf(stderr, "[mainloop@p%d] COMP %ld ms for %dx%dx%d (%.3lfGFlops) t=[%d,%d)\n",
	      myid, ms, nx, ey-sy, ez-sz, gflops, iter, iter+ntinner);
#else
      fprintf(stderr, "[mainloop@p%d] [%.2lf-%.2lf]  COMP %ld ms for %dx%dx%d (%.3lfGFlops) t=[%d,%d)\n",
	      myid, HH_wtime_conv_prt(st), HH_wtime_conv_prt(et),
	      ms, nx, ey-sy, ez-sz, gflops, iter, iter+ntinner);
#endif
    }

    if (myid == 0) {
      double et0 = Wtime();
      long flop = (long)nx*ny*nz*(iter+ntinner)*FLOP_PER_POINT;
      double gflops = (double)flop/(et0-st0)/1.0e+9;
      fprintf(stderr, "[mainloop@p%d] APPROX TOTAL PERFORMANCE: %.3lfsec, %.3lfGFlops\n",
	      myid, et0-st0, gflops);
    }

    //HH_yield();
  }

  cudaFree(dps[1-bufid]);

  return bufid;
}


int main(int argc, char **argv)
{
  int bufid;

  MPI_Init(&argc, &argv);

  /* default setting option */
  bt = 1;
  nt = 256;
  nx = 64;
  ny = 64;
  nz = 64;
  npy = -1;
  npz = -1;

  while (argc >= 2 && argv[1][0] == '-') {
    if (strcmp(argv[1], "-ntb") == 0 || strcmp(argv[1], "-bt") == 0) {
      bt = atoi(argv[2]);
      argc -= 2;
      argv += 2;
    }
    else if (strcmp(argv[1], "-nt") == 0) {
      nt = atoi(argv[2]);
      argc -= 2;
      argv += 2;
    }
    else if (strcmp(argv[1], "-p") == 0) {
      npy = atoi(argv[2]);
      npz = atoi(argv[3]);
      argc -= 3;
      argv += 3;
    }
    else break;
  }

  if (argc >= 4) {
    nx = atoi(argv[1]);
    ny = atoi(argv[2]);
    nz = atoi(argv[3]);
  }

  /* initialize application */
  init();

  /* copy local stencil data to device */
  bufid = 0;

  struct timeval st, et;
  if (myid == 0) {
    fprintf(stderr, "######## %dx%dx%dx%d, Total array size %ldMiB (w/doule buf, w/o halo)\n",
	    nx, ny, nz, nt, (size_t)nx*ny*nz*2*sizeof(REAL)/(1024*1024));
      fprintf(stderr, "Rank %d starts first barrier...\n", myid);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  gettimeofday(&st, NULL);
  
  /* main loop */
  bufid = mainloop();
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  gettimeofday(&et, NULL);
  if (myid == 0) {
    long ms = (et.tv_sec-st.tv_sec)*1000+(et.tv_usec-st.tv_usec)/1000;
    long flop = (long)nx*ny*nz*nt*FLOP_PER_POINT;
    double gf = (double)flop/ms/1000000.0;
    fprintf(stderr, "%dx%dx%dx%dx%d (bt=%d)\n", nx, ny, nz, nt, FLOP_PER_POINT, bt);
    fprintf(stderr, "#########  mainloop() took %ld msec, %.3lf GFlops  ######## \n",
	    ms, gf);
  }

#ifndef INIT_ON_GPU
  /* copy back results */
#ifdef USE_MADVISE
  HH_madvise(hp, bufsize, HHMADV_NORMAL); /* hint for optimization */
#endif
  bufid = (nt%2);
  cudaMemcpy(hp, dps[bufid], bufsize, cudaMemcpyDeviceToHost);
#endif

  cudaFree((void*)dps[bufid]);

  MPI_Finalize();

  return 0;
}
