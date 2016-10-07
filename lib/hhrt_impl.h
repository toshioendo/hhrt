#ifndef HHRT_IMPL_H
#define HHRT_IMPL_H

#include "hhrt_common.h"
extern "C" {
#include "ipsm.h"
}

#include <list>
#include <map>
using namespace std;

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

//#define USE_CUDA_MPS 1 // usually do not select this

// If you use hhview for debug, disable this
#define EAGER_IPSM_DESTROY 1

#define USE_SWAP_THREAD 1

// each process occupies this size even if it does nothing
#define DEVMEM_USED_BY_PROC (85L*1024*1024) // 74L

#define HOSTHEAP_PTR ((void*)0x700000000000)
#define HOSTHEAP_STEP (1L*1024*1024*1024)

#define FILEHEAP_STEP (256L*1024*1024)
#define FILEHEAP_PTR ((void*)512) // the pointer is not accessed

#define HHLOG_SCHED
#define HHLOG_SWAP
#define HHLOG_API

//#define USE_SHARED_HSC 1 // deprecated
//#define HSC_SIZE (1L*1024*1024*1024) // host swapper chunk size

#ifdef USE_SHARED_HSC
#define MAX_SHSC 32
#define SHSC_KEY(i) (HH_IPSM_KEY+1+(i))
#define SHSC_PTR0 ((void*)0x780000000000)
#endif

#define HH_MYID (HHL->rank)

enum {
  HHTAG_BARRIER = 29000,
};

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
};

static const char *hhsw_names[] = {
  "NONE",
  "SWOUT",
  "SWIN",
  "XXX",
  NULL,
};

// result of checkRes
enum {
  HHSS_OK = 0, // swapping can be started
  HHSS_EBUSY, // swapping must be suspended since resource is unavailable
  HHSS_NONEED, // no need for swapping
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
  HHST_HOSTSWAPPER,
  HHST_MPIBUFCOPY,
  HHST_ETC,
  //
  HHST_MAX,
};

static const char *hhst_names[] = {
  "HOSTHEAP",
  "HOSTSWAPPER",
  "MPIBUFCOPY",
  "ETC",
  NULL,
};

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

// info about fileswap dir */
struct fsdir {
  char dirname[CONFSTRLEN];

  int np_filein;
  int np_fileout;
};

struct membuf {
  membuf(ssize_t doffs0, size_t size0, size_t usersize0, int kind0) {
    doffs = doffs0; size = size0; usersize = usersize0;
    kind = kind0;
  }

  ssize_t doffs; /* offset is used instead of address */
  size_t size;
  size_t usersize; // valid if used by user. alignup(usersize) == size
  int kind; /* HHMADV* */

  ssize_t soffs; /* offset of swapepd out buffer */
};

//class swapper;
class heap;

// parent of swapper and heap
class memlayer {
 public:
  memlayer();

  virtual int swapOut() {};
  virtual int swapIn() {};
  virtual int addLower(heap *h);
  virtual int addUpper(heap *h);
  virtual int delLower(heap *h);
  virtual int delUpper(heap *h);
  virtual int finalize() {};
  virtual int finalizeRec(); // recursive finalize

  // description of memory layer tree
  //swapper *lower;
  heap *lower;
  heap *uppers[MAX_UPPERS];

  int swapped;
  char name[16]; /* for debug */

};


/*************/
class heap: public memlayer {
 public:
  heap(size_t size0);
  virtual int finalize();

  virtual void* alloc(size_t size);
  virtual int free(void *p);

  virtual list<membuf *>::iterator findMembufIter(ssize_t doffs);
  virtual membuf *findMembuf(void *p);

  virtual void* offs2ptr(ssize_t offs);
  virtual ssize_t ptr2offs(void* p);

  virtual int expandHeap(size_t reqsize);
  virtual int releaseHeap();
  virtual int allocHeap();
  virtual int restoreHeap();

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

  virtual int madvise(void *p, size_t size, int kind);

  virtual int dump();

  list <membuf *> membufs;
  void *heapptr;
  size_t heapsize;
  size_t align;
  int expandable;
  int memkind; // HHM_*
  int swapping_kind;
};

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
  //virtual int reserveSwapRes(int kind);

  virtual int releaseSwapResSelf(int kind);
  virtual int releaseSwapResAsLower(int kind) {};
  //virtual int releaseSwapRes();

  virtual int writeSeq(ssize_t offs, void *buf, int bufkind, size_t size) {};
  virtual int readSeq(ssize_t offs, void *buf, int bufkind, size_t size) {};

  void *allocDevMem(size_t heapsize);
  void *hp_baseptr;
  dev *device;
};

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
  //virtual int reserveSwapRes(int kind);

  virtual int releaseSwapResSelf(int kind);
  virtual int releaseSwapResAsLower(int kind);
  //virtual int releaseSwapRes();

  virtual int writeSeq(ssize_t offs, void *buf, int bufkind, size_t size);
  virtual int readSeq(ssize_t offs, void *buf, int bufkind, size_t size);

  virtual void *allocCapacity(size_t offset, size_t size);

  int swapfd;
  int mmapflags;

  size_t copyunit;
  void *copybufs[2];
  cudaStream_t copystream;
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
  cudaStream_t copystreams[2];
  int userid;
  char sfname[256];
  int sfd;
  fsdir *fsd;
};


#define HHRF_SEND (1 << 0)
#define HHRF_RECV (1 << 1)

/*************/
/* MPI request finalizer */
struct reqfin {
  // copy mpi buf
  int mode; /* bitwise OR of HHRF_* */
  MPI_Comm comm;
  struct {
    void *cptr;
    int csize;
    MPI_Datatype ctype;
    void *orgptr;
    int orgsize;
    MPI_Datatype orgtype;
  } send, recv;
};

/* user configuration */
struct hhconf {
  size_t devmem;
  int dh_slots; /* # of heap slots in a GPU */
  int maxrp; /* max runnable processes per device */
  int nlphost; /* if lrank < nlphost, host swapper is forcibly used */
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
  int curdevid; /* device id now this process is using */
  int curfsdirid; /* fileswap dir id now this process is using (constant) */
  int pmode; /* process mode: HHP_* */
  int devmode; /* device usage mode: specified by HH_devSetMode */
  int hpid; /* heap slot id on device */

  int in_api; /* 0: usual, >=1: in API */

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

  heap *devheaps[MAX_LDEVS];
  heap *hostheap;
  heap *fileheap;

#ifdef USE_SWAP_THREAD
  pthread_t swap_tid;
  heap *swapping_heap;
#endif

  std::map<MPI_Request, reqfin> reqfins;

  char api_str[64];
#ifdef USE_SHARED_HSC
  void *shsc_ptrs[MAX_SHSC];
#elif defined USE_MMAPSWAP
  int hswfd; // mmap fd for host swapper buffer
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
  int ndevs; /* # of physical devs */
  int ndh_slots; /* # of dev heap slots */

  pthread_mutex_t sched_ml;

  struct dev devs[MAX_LDEVS]; /* physical device */
  struct fsdir fsdirs[MAX_FILESWAP_DIRS];
  struct proc lprocs[MAX_LSIZE];
  char hostname[HOSTNAMELEN];

#ifdef USE_SHARED_HSC
  int nshsc;
  int shsc_users[MAX_SHSC];
  pthread_mutex_t shsc_ml;
#endif

  double stime; // start time. mainly used for profiling and debug print
};

extern struct proc *HHL;
extern struct proc2 *HHL2;
extern struct shdata *HHS;

/************************************************/
/* internal functions */
dev *HH_curdev();
heap *HH_curdevheap();
fsdir *HH_curfsdir();
int HH_mutex_init(pthread_mutex_t *ml);
void HHstacktrace();


/****************************************/
/* hhmem.cc: memory management */
int HH_finalizeHeaps();

/* hhXXXmem.cc: each file layer */
heap *HH_devheapCreate(dev *d);
heap *HH_hostheapCreate();
heap *HH_fileheapCreate(fsdir *fsd);

int HH_addHostMemStat(int kind, ssize_t incr);
int HH_printHostMemStat();

int HH_makeSFileName(fsdir *fsd, int id, char sfname[256]);
int HH_openSFile(char sfname[256]);

/****************************************/
/* hhsched.cc: scheduling */
int HH_lockSched();
int HH_unlockSched();

int HH_progressSched();
int HH_sleepForMemory();
int HH_swapOutIfOver();

int HH_enterAPI(const char *str);
int HH_exitAPI();
int HH_enterGComm(const char *str);
int HH_exitGComm();

/****************************************/
/* hhcuda.cc: for CUDA */
int HH_checkDev();

/* hhaux.c */
int HH_profInit();
int HH_profSetMode(const char *str);
int HH_profBeginAction(const char *str);
int HH_profEndAction(const char *str);



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
