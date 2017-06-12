#ifndef HHRT_IMPL_H
#define HHRT_IMPL_H

#include "hhrt_common.h"
extern "C" {
#include "ipsm.h"
}

#include <list>
#include <map>
using namespace std;

#ifndef HHMPI_REND
#  define HHMPI_BASE
#endif

/*** definitions for HHRT implementation */

#define MAX_LDEVS 16  /* max #GPUs per node */
#define MAX_LSIZE 256 /* max #procs per node */
#define MAX_HEAPS (MAX_LDEVS+2) /* max # of heaps per proc */
#define MAX_DH_SLOTS 8 /* HH_DH_SLOTS (env var) cannot exceed this */
#define HOSTNAMELEN 64
#define CONFSTRLEN 128
#define HEAP_ALIGN (size_t)(1024*1024)
#define MAX_FILESWAP_DIRS 8
#define MAX_UPPERS 16 /* max upper memory layers per layer */

#define HH_IPSM_KEY ((key_t)0x1234)

// If you use hhview for debug, disable this
#define EAGER_IPSM_DESTROY 1

#define USE_SWAP_THREAD 1

#define HHLOG_SCHED
#define HHLOG_SWAP
#define HHLOG_API

#define HH_MYID (HHL->rank)

enum {
  HHM_HOST = 0,
  HHM_PINNED,
  HHM_DEV,
  HHM_FILE,
};


enum {
  HHP_RUNNING = 0,
  HHP_RUNNABLE,
  HHP_BLOCKED,
};

static const char *hhp_names[] = {
  "RUNNING",
  "RUNNABLE",
  "BLOCKED",
  "XXX",
  NULL,
};

enum {
  HHSW_NONE = 0,
  HHSW_OUT,
  HHSW_IN,
  HHSW_SWAPPED,
};

static const char *hhsw_names[] = {
  "NONE",
  "SWOUT",
  "SWIN",
  "SWPD",
  "XXX",
  NULL,
};

// result of checkRes
enum {
  HHSS_OK = 0, // swapping can be started
  HHSS_EBUSY, // swapping must be suspended since resource is unavailable
  HHSS_NONEED, // no need for swapping
  HHSS_ERROR, // other errors
};

static const char *hhss_names[] = {
  "OK",
  "EBUSY",
  "NONEED",
  "XXX",
  NULL,
};

/* host memory statistics */
enum {
  HHST_HOSTHEAP = 0,
  HHST_MPIBUFCOPY,
  HHST_ETC,
  //
  HHST_MAX,
};

static const char *hhst_names[] = {
  "HOSTHEAP",
  "MPIBUFCOPY",
  "ETC",
  NULL,
};

#ifdef USE_CUDA
/* (physical) GPU device */
/* Shared by multiple processes, so this should have flat and relocatable structure */
struct dev {
  int devid;
  pthread_mutex_t ml;
  pthread_mutex_t userml;
  size_t memsize;
  size_t default_heapsize;

  int np_in; /* # of procs that is being swapped in now */
  int np_out; /* # of procs that is being swapped out now */

  cudaIpcMemHandle_t hp_handle;
  void *hp_baseptr0; /* usable only for leader process. ugly */

  int dhslot_users[MAX_DH_SLOTS];
};
#endif

// info about fileswap dir */
struct fsdir {
  char dirname[CONFSTRLEN];

  int np_filein;
  int np_fileout;
};

struct membuf {
#if 1
  membuf(void *ptr0, size_t size0, size_t usersize0, int kind0) {
    ptr = ptr0; size = size0; usersize = usersize0;
    kind = kind0; sptr = NULL;
  }
#else
  membuf(ssize_t doffs0, size_t size0, size_t usersize0, int kind0) {
    doffs = doffs0; size = size0; usersize = usersize0;
    kind = kind0; soffs = (ssize_t)-1;
  }
#endif

#if 1
  void *ptr;
  void *sptr; /* pointer of swapped out buffer. if NULL, not swapped out */
#else
  ssize_t doffs; /* offset is used instead of address */
  ssize_t soffs; /* offset of swapped out buffer. if -1, not swapped out */
#endif
  size_t size;
  size_t usersize; // valid if used by user. alignup(usersize) == size
  int kind; /* HHM_* */

};


/*************/
class heap {
 public:
  heap(size_t size0);
  virtual int finalize();

  virtual int addLower(heap *h);
  virtual int addUpper(heap *h);
  virtual int delLower(heap *h);
  virtual int delUpper(heap *h);

  virtual void* alloc(size_t size);
  virtual int free(void *p);
  virtual size_t getobjsize(void *p);

  virtual list<membuf *>::iterator findMembufIter(void *ptr /*ssize_t doffs*/);
  virtual membuf *findMembuf(void *p);

  virtual void* offs2ptr(ssize_t offs);
  virtual ssize_t ptr2offs(void* p);
  virtual int doesInclude(void* p);

  virtual int expandHeap(size_t reqsize);
  virtual int releaseHeap();
  virtual int allocHeap();
  virtual int restoreHeap();

  // scheduling swap
  virtual int checkSwapResSelf(int kind, int *pline) {};
  virtual int checkSwapResAsLower(int kind, int *pline) {};
  virtual int checkSwapRes(int kind);

  virtual int reserveSwapResSelf(int kind) {};
  virtual int reserveSwapResAsLower(int kind) {};
  virtual int reserveSwapRes(int kind);

  virtual int swapOut();
  virtual int swapIn();
  virtual int doSwap();

  virtual int releaseSwapResSelf(int kind) {};
  virtual int releaseSwapResAsLower(int kind) {};
  virtual int releaseSwapRes();

  virtual int writeSeq(ssize_t offs, void *buf, int bufkind, size_t size) {};
  virtual int readSeq(ssize_t offs, void *buf, int bufkind, size_t size) {};
  virtual int accessRec(char rwtype, void *tgt, void *buf, int bufkind, size_t size);

  virtual int madvise(void *p, size_t size, int kind);

  virtual int dump();

  // description of memory layer tree
  heap *lower;
  heap *uppers[MAX_UPPERS];

  char name[16]; /* for debug */

  list <membuf *> membufs;
  void *heapptr;
  size_t heapsize;
  size_t align;
  int expandable;
  int memkind; // HHM_*
  int swap_stat; // HHSW_*
};

#ifdef USE_CUDA
class devheap: public heap {
 public:
  devheap(size_t size0, dev *device0);
  virtual int finalize();

  virtual int releaseHeap();
  virtual int allocHeap();
  virtual int restoreHeap();

  virtual int checkSwapResSelf(int kind, int *pline);
  virtual int checkSwapResAsLower(int kind, int *pline) {return HHSS_NONEED;};

  virtual int reserveSwapResSelf(int kind);
  virtual int reserveSwapResAsLower(int kind) {};

  virtual int doSwap();

  virtual int releaseSwapResSelf(int kind);
  virtual int releaseSwapResAsLower(int kind) {};

  virtual int writeSeq(ssize_t offs, void *buf, int bufkind, size_t size) {};
  virtual int readSeq(ssize_t offs, void *buf, int bufkind, size_t size) {};

  void *allocDevMem(size_t heapsize);
  void *hp_baseptr;
  dev *device;
};
#endif

class hostheap: public heap {
 public:
  hostheap();
  virtual int finalize();

  virtual int expandHeap(size_t reqsize);

  virtual int allocHeap();
  virtual int releaseHeap();
  virtual int restoreHeap();

  virtual int checkSwapResSelf(int kind, int *pline);
  virtual int checkSwapResAsLower(int kind, int *pline);

  virtual int reserveSwapResSelf(int kind);
  virtual int reserveSwapResAsLower(int kind);

  virtual int releaseSwapResSelf(int kind);
  virtual int releaseSwapResAsLower(int kind);

  virtual int writeSeq(ssize_t offs, void *buf, int bufkind, size_t size);
  virtual int readSeq(ssize_t offs, void *buf, int bufkind, size_t size);

  virtual void *allocCapacity(size_t offset, size_t size);

  int swapfd;
  int mmapflags;

#ifdef USE_CUDA
  cudaStream_t copystream;
#endif
};

class hostmmapheap: public hostheap {
 public:
  hostmmapheap(fsdir *fsd0);
  virtual void *allocCapacity(size_t offset, size_t size);
  virtual int restoreHeap();

  fsdir *fsd;
};

class fileheap: public heap {
 public:
  fileheap(int id, fsdir *fsd0);

  virtual int finalize();

  virtual int expandHeap(size_t reqsize);

  virtual int allocHeap() {return 0;};
  virtual int releaseHeap() {return 0;};
  virtual int restoreHeap() {return 0;};

  virtual int checkSwapResSelf(int kind, int *pline) {return HHSS_NONEED;};
  virtual int checkSwapResAsLower(int kind, int *pline);

  virtual int reserveSwapResSelf(int kind) {};
  virtual int reserveSwapResAsLower(int kind);

  virtual int releaseSwapResSelf(int kind) {};
  virtual int releaseSwapResAsLower(int kind);
  //virtual int releaseSwapRes() {return -1;};

  virtual int writeSeq(ssize_t offs, void *buf, int bufkind, size_t size);
  virtual int readSeq(ssize_t offs, void *buf, int bufkind, size_t size);

  int openSFileIfNotYet();
  int write_small(ssize_t offs, void *buf, int bufkind, size_t size);
  int read_small(ssize_t offs, void *buf, int bufkind, size_t size);

  virtual int swapOut();
  virtual int swapIn();

  size_t copyunit;
  void *copybufs[2];
#ifdef USE_CUDA
  cudaStream_t copystreams[2];
#endif
  int userid;
  char sfname[256];
  int sfd;
  fsdir *fsd;
};

// reqfin::mode
#define HHRF_SEND (1 << 0)
#define HHRF_RECV (1 << 1)

/*************/
/* MPI request finalizer */
struct reqfin {
  // copy mpi buf
  reqfin() {
    mode = 0;
    send.cptr = NULL; send.orgptr = NULL;
    send.packflag = 0;
    recv.cptr = NULL; recv.orgptr = NULL;
    recv.packflag = 0;
  };

  int mode; /* bitwise OR of HHRF_* */
  MPI_Comm comm;
  struct {
    void *cptr;
    int csize;
    MPI_Datatype ctype;
    void *orgptr;
    int orgsize;
    MPI_Datatype orgtype;
    int packflag;
  } send, recv;
};

/* user configuration */
struct hhconf {
  size_t devmem;
  int dh_slots; /* # of heap slots in a GPU */
  int maxrp; /* max runnable processes per device */
  int pin_hostbuf; /* if 1, pin down buffers allocated in host swapper */
  int nlphost; /* # of local procs that can share host memory simultaneously */
  int n_fileswap_dirs;
  char fileswap_dirs[MAX_FILESWAP_DIRS][CONFSTRLEN];
  int use_prof;
  char prof_dir[CONFSTRLEN];
};

/****************************************/
/* Process infomation */
/* Shared by multiple processes, so this should have flat and relocatable structure */
struct proc {
  int rank;
  int lrank;
  int pid;
  int curfsdirid; /* fileswap dir id now this process is using (constant) */
  int pmode; /* process mode: HHP_* */
  int devmode; /* device usage mode: specified by HH_devSetMode */
  int in_api; /* 0: usual, >=1: in API */

#ifdef USE_CUDA
  struct {
    int curdevid; /* CUDA device id now this process is using */
    int hpid; /* CUDA heap slot id on device */
  } cuda;
#endif

  int host_use;
  /* statistics */
  struct {
    ssize_t used[HHST_MAX];
  } hmstat;

  // debug
  char msg[1024];
};

/* Process information, Private structure */
struct proc2 {
  hhconf conf;

  int nheaps;
  heap *heaps[MAX_HEAPS]; // list of all heap structures

  heap *hostheap;
  heap *fileheap;

#ifdef USE_CUDA
  heap *devheaps[MAX_LDEVS]; // CUDA
#endif
#ifdef USE_SWAP_THREAD
  pthread_t swap_tid;
  heap *swapping_heap;
#endif

#if 0
  std::map<MPI_Request, reqfin> reqfins;
#endif

  char api_str[64];
#ifdef USE_MMAPSWAP
  int hswfd; // mmap fd for host swapped buffer
#endif

  struct {
    FILE *fp;
    char mode[64];
    double modest;
    char act[64];
    double actst;
  } prof;

};

/* Info about a node */
/* Initialized by leader (local rank=0) process in init_node() */
/* Shared by multiple processes, so this should have flat and relocatable structure */
struct shdata {
  int nprocs;
  int nlprocs;

  pthread_mutex_t sched_ml;

#ifdef USE_CUDA
  struct {
    int ndevs; /* # of physical devs */
    int ndh_slots; /* # of dev heap slots */
    struct dev devs[MAX_LDEVS]; /* physical device */
  } cuda;
#endif

  struct fsdir fsdirs[MAX_FILESWAP_DIRS];
  struct proc lprocs[MAX_LSIZE];
  char hostname[HOSTNAMELEN];

  double stime; // start time. mainly used for profiling and debug print
};

/************************************************/
// Global variables
extern struct proc *HHL;
extern struct proc2 *HHL2;
extern struct shdata *HHS;

/************************************************/
// Function declaration
// hhmain.cc
int HH_init();
void HH_finalize();

int HH_mutex_init(pthread_mutex_t *ml);
void HHstacktrace();

// hhmem.cc: memory management
int HH_finalizeHeaps();
heap *HH_findHeap(void *p);
int HH_accessRec(char rwtype, void *tgt, void *buf, int bufkind, size_t size);

// hhhostmem.cc: host memory layer
int HH_addHostMemStat(int kind, ssize_t incr);
int HH_printHostMemStat();
heap *HH_hostheapCreate();

// hhfilelayer.cc: file layer
int HH_fileInitNode(hhconf *confp);
int HH_fileInitProc();
fsdir *HH_curfsdir();
heap *HH_fileheapCreate(fsdir *fsd);

int HH_makeSFileName(fsdir *fsd, int id, char sfname[256]);
int HH_openSFile(char sfname[256]);

// hhsched.cc: scheduling
int HH_lockSched();
int HH_unlockSched();

int HH_progressSched();
int HH_sleepForMemory();
int HH_swapOutIfOver();

int HH_enterBlocking();
int HH_exitBlocking();
int HH_enterGComm(const char *str);
int HH_exitGComm();

#ifdef USE_CUDA
// hhcuda.cc: for CUDA
dev *HH_curdev();
heap *HH_curdevheap();
int HH_cudaInitNode(hhconf *confp);
int HH_cudaInitProc();
int HH_cudaCheckDev();

// hhcudamem.cc: CUDA device memory layer
heap *HH_devheapCreate(dev *d);
#endif

// hhaux.c
int HH_profInit();
int HH_profSetMode(const char *str);
int HH_profBeginAction(const char *str);
int HH_profEndAction(const char *str);


/************************************************/
/* aux definitions */
static double Wtime()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec * 0.000001;
}

/* Convert a time value to printable one */
static double Wtime_conv_prt(double t)
{
  //double td = (double)(((long)t/100000)*100000);
  double td = HHS->stime;
  return t-td;
}

static double Wtime_prt()
{
  return Wtime_conv_prt(Wtime());
}

#define piadd(p, i) (void*)((char*)(p) + (size_t)(i))
#define ppsub(p1, p2) (size_t)((char*)(p1) - (char*)(p2))

#define roundup(i, align) (((size_t)(i)+(size_t)(align)-1)/(size_t)(align)*(size_t)(align))

#define lock_log(lp) { \
double st = Wtime(), et; \
pthread_mutex_lock(lp); \
et = Wtime(); \
if (et-st > 0.1) { \
  fprintf(stderr, "[HH:mutex_lock@p%d] [%.2lf-%.2lf] %s:%d LOCK TOOK LONG\n", \
	  HH_MYID, Wtime_conv_prt(st), Wtime_conv_prt(et), __FILE__, __LINE__); \
 } \
} \

#endif /* HHRT_IMPL_H */
