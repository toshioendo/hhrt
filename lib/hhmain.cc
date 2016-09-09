#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <signal.h>
#include <execinfo.h>
#include <assert.h>
#include "hhrt_impl.h"

/* node shared data */
shdata *HHS;

/* process data (on shared memory) */
proc *HHL;

/* 2nd process data (private) */
proc2 hhl2s;
proc2 *HHL2 = &hhl2s;

int HH_default_devid(int lrank)
{
  return (lrank/HHL2->conf.mvp)%HHS->ndevs;
}

dev *HH_curdev()
{
  if (HHL->curdevid < 0) {
    fprintf(stderr, 
	    "[HH_curdev@p%d] ERROR: curdevid is not set\n",
	    HH_MYID);
    exit(1);
  }
  return &HHS->devs[HHL->curdevid];
}

heap *HH_curdevheap()
{
  if (HHL->curdevid < 0) {
    fprintf(stderr, 
	    "[HH_curdevheap@p%d] ERROR: curdevid is not set\n",
	    HH_MYID);
    exit(1);
  }
  heap *h = HHL2->devheaps[HHL->curdevid];
  if (h == NULL) {
    fprintf(stderr, 
	    "[HH_curdevheap@p%d] ERROR: devid %d is not initialized\n",
	    HH_MYID, HHL->curdevid);
    exit(1);
  }
  return h;
}

fsdir *HH_curfsdir()
{
  if (HHL->curfsdirid < 0) {
    fprintf(stderr, 
	    "[HH_curfsdir@p%d] ERROR: curfsdirid is not set\n",
	    HH_MYID);
    exit(1);
  }
  return &HHS->fsdirs[HHL->curfsdirid];
}

int HH_mutex_init(pthread_mutex_t *ml)
{
  int rc;
  pthread_mutexattr_t mat;
  pthread_mutexattr_init(&mat);
  pthread_mutexattr_setpshared(&mat, PTHREAD_PROCESS_SHARED);
  rc = pthread_mutex_init(ml, &mat);
  pthread_mutexattr_destroy(&mat);
  return rc;
}

int get_local_rank_size(int *lrankp, int *lsizep)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    int size;
    int lrank = -1;
    int lsize = -1;
    char myhost[HOSTNAMELEN];
    char *hosts;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    memset(myhost, 0, HOSTNAMELEN);
    gethostname(myhost, HOSTNAMELEN-1);

    hosts = NULL;
    hosts = (char*)malloc(HOSTNAMELEN * size);

    MPI_Allgather(myhost, HOSTNAMELEN, MPI_BYTE,
		  hosts, HOSTNAMELEN, MPI_BYTE,
		  comm);

    {
      int j;
      char *hosti;
      hosti = &hosts[HOSTNAMELEN*rank];
      lrank = 0;
      lsize = 0;
      for (j = 0; j < size; j++) {
	char *hostj;
	hostj = &hosts[HOSTNAMELEN*j];
	if (strcmp(hosti, hostj) == 0) {
	  /* the same host is found */
	  if (j < rank) {
	    lrank++;
	  }
	  lsize++;
	}
      }
    }

    free(hosts);
#if 0
    fprintf(stderr, "[get_local_r_s] rank=%d, size=%d --> lrank=%d, lsize=%d\n",
	    rank, size, lrank, lsize);
#endif

    assert(lrank >= 0);
    assert(lsize >= 0);
    if (lrankp != NULL) *lrankp = lrank;
    if (lsizep != NULL) *lsizep = lsize;
    return 0;
}

/* similar to atol, but this allows notation like "2.5G", "500M" */
long getLP(char *p)
{
  double v;
  long lv;
  long mul = 1;

  v = strtod(p, &p);
  if (*p == 'k' || *p == 'K') {
    mul = (long)1024;
  }
  else if (*p == 'm' || *p == 'M') {
    mul = (long)1024*1024;
  }
  else if (*p == 'g' || *p == 'G') {
    mul = (long)1024*1024*1024;
  }
  else if (*p == 't' || *p == 'T') {
    mul = (long)1024*1024*1024*1024;
  }
  else {
    mul = (long)1;
  }

  lv = (long)(v*(double)mul);
  
  return lv;
}

static int initSharedDevmem(dev *d)
{
  cudaError_t crc;
  int devid = d->devid;
  crc = cudaSetDevice(devid);
  if (crc != cudaSuccess) {
    fprintf(stderr, "[HH:initSharedDevmem@%s:dev%d] cudaSetDevice for heap failed!\n",
	    HHS->hostname, devid);
    exit(1);
  }
  
  size_t heapsize = d->default_heapsize;
  crc = cudaMalloc(&d->hp_baseptr0, heapsize * HHS->ndhslots);
  if (crc != cudaSuccess) {
    fprintf(stderr, "[HH:initSharedDevmem@%s:dev%d] cudaMalloc(%ldMiB) for heap failed!\n",
	    HHS->hostname, devid, heapsize>>20);
    exit(1);
  }

  crc = cudaIpcGetMemHandle(&d->hp_handle, d->hp_baseptr0);
  if (crc != cudaSuccess) {
    fprintf(stderr, "[HH:initSharedDevmem@%s:dev%d] cudaIpcGetMemhandle for heap failed!\n",
	    HHS->hostname, devid);
    exit(1);
  }
#if 1
  fprintf(stderr, "[HH:initSharedDevmem@%s:dev%d] exporting pointer %p\n",
	  HHS->hostname, devid, d->hp_baseptr0);
#endif
  return 0;
}

int initDev(int i, int lsize, int size, hhconf *confp)
{
  cudaError_t crc;
  dev *d = &HHS->devs[i];

  d->devid = i;
  HH_mutex_init(&d->ml);
  HH_mutex_init(&d->userml);
  if (confp->devmem > 0) {
    d->memsize = confp->devmem;
  }
  else {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    d->memsize = prop.totalGlobalMem;
#if 0
    fprintf(stderr, "[HHRT] dev %d: memsize=%ld\n",
	    i, d->memsize);
#endif
  }
  
  /* determine device heap size */
  size_t avail = d->memsize - d->memsize/64L;
#ifdef USE_CUDA_MPS
  avail -= DEVMEM_USED_BY_PROC * 1;
#else
  avail -= DEVMEM_USED_BY_PROC * lsize;
#endif
  d->default_heapsize = avail / HHS->ndhslots;
  d->default_heapsize = (d->default_heapsize/HEAP_ALIGN)*HEAP_ALIGN;
  
#if 1
  fprintf(stderr, "[HH:initDev@%s:dev%d] memsize=%ld -> default_heapsize=%ld\n",
	  HHS->hostname, i, d->memsize, d->default_heapsize);
#endif

  initSharedDevmem(d);

  int ih;
  /* setup heap slots on device */
  for (ih = 0; ih < HHS->ndhslots; ih++) {
    d->dhslot_users[ih] = -1;
  }
  
  d->np_in = 0;
  d->np_out = 0;
  return 0;
}

/* Called only by leader (lrank=0) process */
static int initNode(int lsize, int size, hhconf *confp)
{
  int ndevs; // # of physical devices
  int i;
  cudaError_t crc;

  /* for debug print */
  char hostname[HOSTNAMELEN];
  memset(hostname, 0, HOSTNAMELEN);
  gethostname(hostname, HOSTNAMELEN-1);

#if 1
  /* delete garbage shm */
  char comm[256];
  key_t key = HH_IPSM_KEY;
  //sprintf(comm, "ipcrm -a > /dev/null 2>&1 ");
  sprintf(comm, "ipcrm -M 0x%x > /dev/null 2>&1 ", key);
  system(comm);
#endif
  /* initialize node level info (HHS) */
  HHS = (struct shdata*)ipsm_init(HH_IPSM_KEY, sizeof(struct shdata) );
  if (HHS == NULL) {
    int rc = ipsm_getlasterror();
    fprintf(stderr, "[HH:initNode@%s] ipsm_init failed!! lasterror=0x%x. abort..\n", 
	    hostname, rc);
    exit(1);
  }

  HHS->stime = Wtime();
  HHS->nlprocs = lsize;
  HHS->nprocs = size;
  strcpy(HHS->hostname, hostname);

  HH_hsc_init_node();

  HH_mutex_init(&HHS->sched_ml);

#ifdef USE_CUDA_MPS
  // start MPS server
  // this must be done before any CUDA API calls
  fprintf(stderr, "[HH:initNode@%s] Starting MPS...\n", HHS->hostname);
  system("nvidia-cuda-mps-control -d 2>&1 > /dev/null");
  sleep(2);
#endif

  ndevs = -1;
  crc = cudaGetDeviceCount(&ndevs);
  if (crc != cudaSuccess || ndevs < 0 || ndevs > MAX_LDEVS) {
    fprintf(stderr, "[HH:initNode@%s] cudaGetDeviceCount ERROR. rc=%d, ndevs=%d\n",
	    HHS->hostname, crc, ndevs);
    exit(1);
  }
  assert(ndevs <= MAX_LDEVS);
  HHS->ndevs = ndevs;

#if 1
  fprintf(stderr, "[HH:initNode@%s] I have %d visible devices\n",
	  HHS->hostname, ndevs);
#endif

  HHS->ndhslots = confp->maxrp;
  if (HHS->ndhslots > lsize) HHS->ndhslots = lsize;
  for (i = 0; i < HHS->ndhslots; i++) {
    HHS->nhostusers[i] = 0;
  }

  /* init device structures */
  {
    int mydevid;
    crc = cudaGetDevice(&mydevid);
    
    for (i = 0; i < ndevs; i++) {
      initDev(i, lsize, size, confp);
    }
    
    crc = cudaSetDevice(mydevid); // restore
  }

  /* init swap directories */
  {
    for (i = 0; i < confp->n_fileswap_dirs; i++) {
      fsdir *fsd = &HHS->fsdirs[i];
      strcpy(fsd->dirname, confp->fileswap_dirs[i]);
      fsd->np_filein = 0;
      fsd->np_fileout = 0;
    }
  }

  /* Now HHS is made public */
  ipsm_share();

  MPI_Barrier(MPI_COMM_WORLD);

  return 0;
}

/* called by non-leader (lrank != 0) processes */
/* after this function, the process can read/write HHS */
static int joinProc(int lrank, int rank)
{
  int nretry = 0;
  /* for debug print */
  char hostname[HOSTNAMELEN];
  memset(hostname, 0, HOSTNAMELEN);
  gethostname(hostname, HOSTNAMELEN-1);
  
  MPI_Barrier(MPI_COMM_WORLD); // see MPI_Barrier in initNode()
  
 retry:
  HHS = (struct shdata*)ipsm_join(HH_IPSM_KEY);
  //HHS = (struct shdata*)ipsm_tryjoin(HH_IPSM_KEY, 100*1000, 300);
  if (HHS == NULL) {
    int rc;
    rc = ipsm_getlasterror();
    if (rc == IPSM_ENOTREADY) {
      if (nretry > 5) {
	fprintf(stderr, "[HHRT@p%d(%s:L%d)] ipsm_join retried for %d times. I abandon...\n", 
		rank, hostname, lrank, nretry);
	exit(1);
      }
      /* It's ok, and retry */
      sleep(1);
      nretry++;
      goto retry;
    }
    fprintf(stderr, "[HHRT@p%d(%s:L%d)] ipsm_join failed!! lasterror=0x%x. abort..\n", 
	    rank, hostname, lrank, rc);
    exit(1);
  }
  return 0;
}

/* Called by every process */
/* This may cause waiting */
static int initProc(int lrank, int lsize, int rank, int size, hhconf *confp)
{
  assert(lsize <= MAX_LSIZE);

  HHL = &HHS->lprocs[lrank];
  HHL->rank = rank;
  HHL->lrank = lrank;
  HHL->pid = getpid();

  /* default mode */
  HHL->pmode = HHP_RUNNABLE;
  HHL->dmode = HHD_NONE;
  HHL->devmode = HHDEV_NORMAL;
  HHL->in_api = 0;

  /* statistics */
  int i;
  for (i = 0; i < HHST_MAX; i++) {
    HHL->hmstat.used[i] = 0;
  }

  HH_profInit();

  HHL->curdevid = -1;
  {
    // default device id
    // TODO: this should be lazy
    cudaError_t crc;
    crc = cudaGetDevice(&HHL->curdevid);
    if (crc != cudaSuccess) {
      fprintf(stderr, "[HH:inic_proc@p%d] cudaGetDevice failed. ignored\n", HH_MYID);
      HHL->curdevid = 0;
    }
  }
  // default fsdir id
  if (confp->n_fileswap_dirs == 0) {
    HHL->curfsdirid = -1;
  }
  else {
    HHL->curfsdirid = lrank % confp->n_fileswap_dirs;
  }

  HH_hsc_init_proc();

  // see also HH_canSwapIn()
  HHL->hpid = lrank % HHS->ndhslots;

  // setup heap structures
  HHL2->nheaps = 0;
  for (int id = 0; id < MAX_LDEVS; id++) {
    HHL2->heaps[id] = NULL;
  }

  heap *h;
#ifdef USE_SWAPHOST
  h = HH_hostheapCreate();
  HHL2->heaps[HHL2->nheaps++] = h;
  HHL2->hostheap = h;
#endif

#if 0
  h = HH_devheapCreate(HH_curdev());
  HHL2->heaps[HHL2->nheaps++] = h;
  HHL2->devheaps[HHL->curdevid] = h;
#else
  fprintf(stderr, "[HH:initProc@p%d] skip init devheap. it will be done later\n",
	  HH_MYID);
#endif

  // blocked until heaps are accessible
  HH_sleepForMemory();
  HHL->pmode = HHP_RUNNING;
  HH_profSetMode("RUNNING");

  return 0;
}

int HH_readConf(hhconf *confp)
{
  char *p;

  /* HH_MVP */
  confp->mvp = 99999;
  p = getenv("HH_MVP");
  if (p != NULL) {
    confp->mvp = (int)getLP(p);
  }
  
  /* HH_DEVMEM */
  confp->devmem = 0;
  p = getenv("HH_DEVMEM");
  if (p != NULL) {
    confp->devmem = getLP(p);
  }

  /* HH_MAXRP */
  confp->maxrp = 2;
  p = getenv("HH_MAXRP");
  if (p != NULL) {
    confp->maxrp = (int)getLP(p);
  }
  if (confp->maxrp > MAX_MAXRP) {
    fprintf(stderr, "[HH_readConf] HH_MAXRP(%d) cannot exceed %d\n",
	    confp->maxrp, MAX_MAXRP);
    exit(1);
  }

  /* HH_NLPHOST */
  confp->nlphost = 99999;
  p = getenv("HH_NLPHOST");
  if (p != NULL) {
    confp->nlphost = (int)getLP(p);
  }

  /* HH_SWAP_PATH */
  /* default value */
  confp->n_fileswap_dirs = 1;
  strcpy(confp->fileswap_dirs[0], "./hhswap");

  p = getenv("HH_FILESWAP_PATH");
  if (p != NULL) {
    confp->n_fileswap_dirs = 0;
    /* parse the string, separated by ':'  */
    while (*p != '\0' && confp->n_fileswap_dirs < MAX_FILESWAP_DIRS) {
      char *endp = strchr(p, ':');
      int len = (endp)? (endp-p): strlen(p);
      if (len == 0) {
	/* ignore empty string */
      }
      else if (len >= CONFSTRLEN) {
	fprintf(stderr, "[HH_readConf] ERROR: Too long string in HH_SWAP_PATH:%s\n",
		p);
	exit(1);
      }
      else {
	/* get a string */
	strncpy(confp->fileswap_dirs[confp->n_fileswap_dirs], p, len);
	confp->fileswap_dirs[confp->n_fileswap_dirs][len] = '\0';
	confp->n_fileswap_dirs++;
      }
      p += len;
      if (*p == ':') p++;
    }
  }


  return 0;
}

int HH_printConf(FILE *ofp, hhconf *confp)
{
  fprintf(ofp, "[HH_printConf] configuration:\n");
  fprintf(ofp, "  HH_MVP=%d\n", confp->mvp);
  fprintf(ofp, "  HH_DEVMEM=%ld\n", confp->devmem);
  fprintf(ofp, "  HH_MAXRP=%d\n", confp->maxrp);
  fprintf(ofp, "  HH_NLPHOST=%d\n", confp->nlphost);
  fprintf(ofp, "  HH_FILESWAP_PATH= ");
  int i;
  for (i = 0; i < confp->n_fileswap_dirs; i++) {
    fprintf(ofp, "[%s] ", confp->fileswap_dirs[i]);
  }
  fprintf(ofp, "\n");
  return 0;
}

void HHstacktrace()
{
#define NTR 128
  void *trace[NTR];
  int n = backtrace(trace, NTR);
  backtrace_symbols_fd(trace, n, 1);
}

/* for debug */
void HHsighandler(int sn, siginfo_t *si, void *sc)
{
  fprintf(stderr, "[HHsighandler@p%d] SIGNAL %d received! pid=%d\n",
	  HH_MYID, sn, getpid());
  fprintf(stderr, "[HHsighandler@p%d] si_addr=%p\n",
	  HH_MYID, si->si_addr);

  HHstacktrace();

#ifdef USE_SWAPHOST
  if (HHL2->hostheap != NULL) {
    HHL2->hostheap->dump();
  }
#endif

  exit(1);
}


int HHMPI_Init(int *argcp, char ***argvp)
{
  int rank, size;
  int lrank, lsize;

  MPI_Init(argcp, argvp);

#if 1
  {
    // debug 
    struct sigaction s;
    s.sa_flags = SA_SIGINFO;
    s.sa_sigaction = HHsighandler;
    sigaction(SIGSEGV, &s, NULL);
    sigaction(SIGFPE, &s, NULL);
    sigaction(SIGABRT, &s, NULL);
  }
#endif

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  get_local_rank_size(&lrank, &lsize);

  ipsm_setloglevel(IPSM_LOG_WARN);
  //ipsm_setloglevel(IPSM_LOG_INFO);
  putenv((char*)"IPSM_SHMSIZE=100000"); /* in hex */

  hhconf conf;
  if (rank == 0) {
    /* Called only by rank 0 */
    HH_readConf(&conf);
#if 1
    HH_printConf(stderr, &conf);
#endif
  }
  MPI_Bcast(&conf, sizeof(hhconf), MPI_BYTE, 0, MPI_COMM_WORLD);
  HHL2->conf = conf;

  if (lrank == 0) {
    /* Called only once per node */
    initNode(lsize, size, &conf);
  }
  else {
    joinProc(lrank, rank);
  }

  MPI_Barrier(MPI_COMM_WORLD);
#ifdef EAGER_ICS_DESTROY
  ipsm_destroy(); /* key is retained until processes finish */
#endif

  /* initialize process info */
  initProc(lrank, lsize, rank, size, &conf);

  return 0;
}

int HHMPI_Finalize()
{
  int rank = HHL->rank;
  int lrank = HHL->lrank;
#if 1
  if (rank == 0) {
    HH_printConf(stderr, &HHL2->conf);
  }
#endif

  HH_finalizeHeaps();

#ifdef USE_CUDA_MPS
  MPI_Barrier(MPI_COMM_WORLD);
  if (lrank == 0) {
    system("killall nvidia-cuda-mps-control");
  }
#endif

#ifndef EAGER_ICS_DESTROY
  MPI_Barrier(MPI_COMM_WORLD);
  ipsm_destroy();
#endif
  if (lrank == 0) {
    HH_hsc_fin_node();
  }


  MPI_Finalize();

  return 0;
}

