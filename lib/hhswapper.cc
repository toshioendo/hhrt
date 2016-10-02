#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <unistd.h>
#include <string.h>
#include <malloc.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

#include "hhrt_impl.h"


int swapper::beginSeqWrite()
{
  swcur = 0;
  return 0;
}

size_t swapper::allocSeq(size_t size)
{
  size_t cur = swcur;
  swcur += roundup(size, align);
  return cur;
}

