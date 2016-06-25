#ifndef HHRT_H
#define HHRT_H

#include <mpi.h>
#include <cuda_runtime.h>

enum {
  HHMADV_FREED = 0,
  HHMADV_NORMAL,
  HHMADV_CANDISCARD,
};

enum {
  HHDEV_NORMAL = 0,
  HHDEV_NOTUSED,
};


/*** No-op version */

int HH_initHeap(size_t heapsize);
int HH_logNode();
int HH_logVp();

int HH_madvise(void *p, size_t size, int kind);

int HH_devLock();
int HH_devUnlock();
int HH_devSetMode(int kind);

#endif /* HHRT_H */
