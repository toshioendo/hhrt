#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "hhrt.h"

int HH_initHeap(size_t heapsize)
{
  /* ignored */
  return 0;
}

int HH_logNode()
{
  /* ignored */
  return 0;
}

int HH_logVp()
{
  /* ignored */
  return 0;
}

int HH_madvise(void *p, size_t size, int kind)
{
  return 0;
}

int HH_devLock()
{
  return 0;
}

int HH_devUnlock()
{
  return 0;
}

int HH_devSetMode(int kind)
{
  return 0;
}
