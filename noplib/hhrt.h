#ifndef HHRT_H
#define HHRT_H

enum {
  HHMADV_FREED = 0,
  HHMADV_NORMAL,
  HHMADV_CANDISCARD,
  HHMADV_READONLY,
};

enum {
  HHDEV_NORMAL = 0,
  HHDEV_NOTUSED,
};


/*** No-op version */

#define HH_madvise(p, size, kind)
#define HH_yield()

/* obsolete functions */
#define HH_devLock()
#define HH_devUnlock()
#define HH_devSetMode(kind)

#endif /* HHRT_H */
