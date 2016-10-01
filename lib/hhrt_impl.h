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
#define MAX_HEAPS (MAX_LDEVS+1) /* max # of heaps per proc */
#define MAX_DH_SLOTS 8 /* HH_DH_SLOTS (env var) cannot exceed this */
#define HOSTNAMELEN 64
#define CONFSTRLEN 128
#define HEAP_ALIGN (size_t)(1024*1024)
#define MAX_FILESWAP_DIRS 8

#define HH_IPSM_KEY ((key_t)0x1234)

//#define USE_CUDA_MPS 1 // usually do not select this

// If you use hhview for debug, disable this
#define EAGER_IPSM_DESTROY 1

#define USE_SWAP_THREAD 1

//#define USE_SHARED_HSC 1 // buggy

// each process occupies this size even if it does nothing
#define DEVMEM_USED_BY_PROC (85L*1024*1024) // 74L

#define HOSTHEAP_PTR ((void*)0x700000000000)
#define HOSTHEAP_STEP (1L*1024*1024*1024)

#define HHLOG_SCHED
#define HHLOG_SWAP
#define HHLOG_API

#define HSC_SIZE (1L*1024*1024*1024) // host swapper chunk size

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
  HHD_NONE = 0,
  HHD_ON_DEV,
  HHD_ON_HOST,
  HHD_ON_FILE,

  HHD_SO_ANY,
  HHD_SO_D2H,
  HHD_SO_H2F,
  HHD_SI_ANY,
  HHD_SI_F2H,
  HHD_SI_H2D,
  HHD_SWAP_NONE,
};

static const char *hhd_names[] = {
  "NONE",
  "ON_DEV",
  "ON_HOST",
  "ON_FILE",

  "SO_ANY",
  "SO_D2H",
  "SO_H2F",
  "SI_ANY",
  "SI_F2H",
  "SI_H2D",
  "S_NONE",
  "XXX",
  NULL,
};

static const char *hhd_snames[] = {
  "INVALID0",
  "INVALID1",
  "INVALID2",
  "INVALID3",

  "SO",
  "D2H",
  "H2F",
  "SI",
  "F2H",
  "H2D",
  "S_N",
  "XXX",
  NULL,
};

// result of checkRes
enum {
  HHSS_OK = 0, // swapping can be started
  HHSS_EBUSY, // swapping must be suspended since resource is unavailable
  HHSS_NONEED, // no need for swapping
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

class swapper;

// parent of swapper and heap
class mempool {
 public:
  mempool() {curswapper = NULL; swapped = 0;};

  virtual int swapOut() {};
  virtual int swapIn() {};
  virtual int setSwapper(swapper *swapper0);
  virtual int finalize() {};
  virtual int finalizeRec(); // recursive finalize

  swapper *curswapper;
  int swapped;
  char name[16]; /* for debug */

};

// swapper class. This is created per heap per memory hierarchy
class swapper: public mempool {
 public:
 swapper(): mempool() {
    align = 1; curswapper = NULL;
  }

  virtual int finalize() {};

  virtual int write1(ssize_t offs, void *buf, int bufkind, size_t size) {};
  virtual int read1(ssize_t offs, void *buf, int bufkind, size_t size) {};

  virtual int allocBuf() {};
  virtual int releaseBuf() {};

  virtual int swapOut() {};
  virtual int swapIn() {};

  /* sequential writer */
  size_t swcur;
  size_t align;
  virtual int beginSeqWrite();
  virtual size_t allocSeq(size_t size);
};

class hostswapper: public swapper {
 public:
  hostswapper();

  virtual int finalize();

  void *getNthChunk(int n);
  int write_small(ssize_t offs, void *buf, int bufkind, size_t size);
  virtual int write1(ssize_t offs, void *buf, int bufkind, size_t size);
  int read_small(ssize_t offs, void *buf, int bufkind, size_t size);
  virtual int read1(ssize_t offs, void *buf, int bufkind, size_t size);

  virtual int allocBuf();
  virtual int releaseBuf();

  virtual int swapOut();
  virtual int swapIn();

  size_t copyunit;
  void *copybufs[2];
  cudaStream_t copystream;
  int initing;

  // host swapper chunks
  list <void *> hscs;
};

class fileswapper: public swapper {
 public:
  fileswapper(int id, fsdir *fsd0);

  virtual int finalize();

  int openSFileIfNotYet();

  int write_small(ssize_t offs, void *buf, int bufkind, size_t size);
  virtual int write1(ssize_t offs, void *buf, int bufkind, size_t size);
  int read_small(ssize_t offs, void *buf, int bufkind, size_t size);
  virtual int read1(ssize_t offs, void *buf, int bufkind, size_t size);

  virtual int allocBuf() {};
  virtual int releaseBuf() {};

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

/*************/
class heap: public mempool {
 public:
  heap(size_t size0);
  virtual int finalize();

  virtual void* alloc(size_t size);
  virtual int free(void *p);

  virtual list<membuf *>::iterator findMembufIter(ssize_t doffs);
  virtual membuf *findMembuf(void *p);

  virtual int expandHeap(size_t reqsize);
  virtual int releaseHeap();
  virtual int allocHeap();
  virtual int restoreHeap();

  virtual int swapOut();
  virtual int swapIn();

  virtual int inferSwapMode(int kind) {};
  virtual int checkSwapRes(int kind) {};
  virtual int reserveSwapRes(int kind) {};
  virtual int doSwap() {};
  virtual int releaseSwapRes() {};

  virtual int madvise(void *p, size_t size, int kind);

  virtual int dump();

  list <membuf *> membufs;
  void *heapptr;
  size_t heapsize;
  size_t align;
  int expandable;
  int memkind; // HHM_*
  int swap_kind;
};

class devheap: public heap {
 public:
  devheap(size_t size0, dev *device0);
  virtual int finalize();

  virtual int releaseHeap();
  virtual int allocHeap();
  virtual int restoreHeap();

  virtual int inferSwapMode(int kind);
  virtual int checkSwapRes(int kind);
  virtual int reserveSwapRes(int kind);
  virtual int doSwap();
  virtual int releaseSwapRes();

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

  virtual int inferSwapMode(int kind);
  virtual int checkSwapRes(int kind);
  virtual int reserveSwapRes(int kind);
  virtual int doSwap();
  virtual int releaseSwapRes();

  virtual void *allocCapacity(size_t offset, size_t size);
  int swapfd;
  int mmapflags;
};

class hostmmapheap: public hostheap {
 public:
  hostmmapheap(fsdir *fsd0);
  virtual void *allocCapacity(size_t offset, size_t size);
  virtual int restoreHeap();

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
  
};

/* Process information, Private structure */
struct proc2 {
  hhconf conf;

  int nheaps;
  heap *heaps[MAX_HEAPS];

  heap *devheaps[MAX_LDEVS];
#ifdef USE_SWAPHOST
  heap *hostheap;
#endif

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
int HH_default_devid(int lrank);
dev *HH_curdev();
heap *HH_curdevheap();
fsdir *HH_curfsdir();
int HH_mutex_init(pthread_mutex_t *ml);

int HH_makeSFileName(fsdir *fsd, int id, char sfname[256]);
int HH_openSFile(char sfname[256]);


/****************************************/
/* hhmem.cc: memory management */
int HH_swapInH2D();
int HH_swapInF2H();
int HH_startSwapInF2H();
int HH_tryfinSwapInF2H();
int HH_swapOutD2H();
int HH_swapOutH2F();
int HH_startSwapOutH2F();
int HH_tryfinSwapOutH2F();

int HH_addHostMemStat(int kind, ssize_t incr);
int HH_printHostMemStat();

/****************************************/
/* hhheap.cc: heap structures */
heap *HH_devheapCreate(dev *d);
#ifdef USE_SWAPHOST
heap *HH_hostheapCreate();
#endif
int HH_finalizeHeaps();

/****************************************/
/* hhsched.cc: scheduling */
int HH_lockSched();
int HH_unlockSched();

int HH_progressSched();
int HH_enterAPI(const char *str);
int HH_exitAPI();
int HH_enterGComm(const char *str);
int HH_exitGComm();

int HH_sleepForMemory();
int HH_swapInIfOk();
int HH_swapOutIfBetter();
int HH_swapOutIfOver();

int HH_hsc_init_node();
int HH_hsc_init_proc();
int HH_hsc_fin_node();
void *HH_hsc_alloc(int id);
int HH_hsc_free(void *p);

/****************************************/
/* hhcuda.cc: for CUDA */
int HH_checkDev();

/* hhaux.c */
int HH_profInit();
int HH_profSetMode(const char *str);
int HH_profBeginAction(const char *str);
int HH_profEndAction(const char *str);


void HHstacktrace();

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
