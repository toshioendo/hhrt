#ifndef _IPSM_H_
#define _IPSM_H_

#include <time.h>
#include <sys/ipc.h>

#define IPSM_SHMSIZE_ENV     "IPSM_SHMSIZE"

typedef ssize_t ipsm_offset_t;

typedef enum {
    IPSM_OK                 = 0xE00,
    IPSM_EACCES,
    IPSM_ENOENT,
    IPSM_EEXIST,
    IPSM_EINVAL,
    IPSM_ENOMEM,
    IPSM_ETIMEDOUT,
    IPSM_EALREADY,

    IPSM_ENOTREADY,
    IPSM_EINTERNAL,
} ipsm_err;

#define IPSM_LOG_CRIT           1
#define IPSM_LOG_ERROR          2
#define IPSM_LOG_WARN           3
#define IPSM_LOG_NOTICE         4
#define IPSM_LOG_INFO           5
#define IPSM_LOG_DEBUG          6
#define IPSM_LOG_VERBOSE        7

void *ipsm_init(key_t shmkey, size_t reserve_size);
int ipsm_destroy(void);
int ipsm_share(void);

void *ipsm_join(key_t shmkey);
void *ipsm_tryjoin(key_t shmkey, int retry_interval_usec, int retry_times_max);

void *ipsm_alloc(size_t memsize);
int ipsm_free(void *ptr);

ipsm_offset_t ipsm_addr2offset(void *addr);
void *ipsm_offset2addr(ipsm_offset_t offset);

ipsm_err ipsm_getlasterror(void);
char *ipsm_strerror(ipsm_err err);

int ipsm_setloglevel(int level);

void ipsm_print_status(key_t shmkey);

#endif/*_IPSM_H_*/
