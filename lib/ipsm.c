#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/fcntl.h>
#include <time.h>
#include <pthread.h>
#include <string.h>
#include <limits.h>
#include <errno.h>
#include <assert.h>

#include "ipsm.h"

#define IPSM_MAGIC              0x20140210
#define IPSM_MAGIC_DEAD         0xdeadbeaf

#define IPSM_JOIN_RETRY_INTERVAL        (100*1000) /* usec */

#define SHM_ACCESSMODE          (S_IRWXU)

#define IPSM_ALIGNMENT  (0x10)
#define ROUNDUP(x)      ((((x)+IPSM_ALIGNMENT-1)/IPSM_ALIGNMENT)*IPSM_ALIGNMENT)

/*#define DEF_NUM_ENTRY   (64)*/
#define DEF_NUM_ENTRY   (4)

static int shmid = -1;

typedef struct {
    size_t size;
    ipsm_offset_t offset;
    bool used;
    int sorted_index;
} entry_st;

typedef struct {
    entry_st entry;
    int __fake_index;
} fake_entry_st;

typedef struct {
    int num_alloced;
    int num_entry;
    int num_free;
    fake_entry_st fake_entry_array[];
} share_ctl_st;

typedef struct {
    unsigned int magic;
    size_t shmsize;
    size_t reserved_size;
    bool shared;
    pthread_mutex_t mutex;
    size_t share_ctl_offset;
} ipsm_ctl_st;

static void *shmaddr_head;
static void *shmaddr_reserved;
static void *shmaddr_shared;
static void *shmaddr_tail;
static ipsm_ctl_st *ipsm_ctl;

#define ADDR2OFFSET(p)  ((ipsm_offset_t)(p)- (ipsm_offset_t)shmaddr_shared)
#define OFFSET2ADDR(o)  ((char*)shmaddr_shared + (o))
#define share_ctl       \
    ((share_ctl_st*)((char*)shmaddr_head + ipsm_ctl->share_ctl_offset))
#define entry_array     ((entry_st*)(share_ctl->fake_entry_array))
#define free_entries    ((int*)(entry_array + share_ctl->num_alloced))

#define SIZEOF_SHARE_CTL(num)   (sizeof(fake_entry_st)*(num) + sizeof(int)*3)

static pthread_key_t errno_key;
static pthread_once_t genkey_once = {PTHREAD_ONCE_INIT};
static int ipsm_loglevel = IPSM_LOG_CRIT;


#ifdef ABORT_ON_CRITICAL_ERROR
#define critical_error()        abort()
#else/*ABORT_ON_CRITICAL_ERROR*/
#define critical_error()
#endif/*ABORT_ON_CRITICAL_ERROR*/

#define RETCODE_FAILURE         (-1)
#define return_success()        return(0)
#define return_failure(_err)     do { \
    _set_errno(_err); \
    return(RETCODE_FAILURE); \
} while (/*CONSTCOND*/0)
#define return_null(_err)        do { \
    _set_errno(_err); \
    return(NULL); \
} while (/*CONSTCOND*/0)




static void _set_errno(ipsm_err err);
static void _printf(int level, const char *fmt, ...);
#define _critlog(fmt,...) \
    _printf(IPSM_LOG_CRIT,   "%s: " fmt "\n", __func__, ##__VA_ARGS__)
#define _errorlog(fmt,...) \
    _printf(IPSM_LOG_ERROR,  "%s: " fmt "\n", __func__, ##__VA_ARGS__)
#define _warnlog(fmt,...) \
    _printf(IPSM_LOG_WARN,   "%s: " fmt "\n", __func__, ##__VA_ARGS__)
#define _noticelog(fmt,...) \
    _printf(IPSM_LOG_NOTICE, "%s: " fmt "\n", __func__, ##__VA_ARGS__)
#define _infolog(fmt,...) \
    _printf(IPSM_LOG_INFO,   "%s: " fmt "\n", __func__, ##__VA_ARGS__)

#define _debuglog(fmt,...)      \
    _printf(IPSM_LOG_DEBUG,   "[DEBUG@%s] " fmt "\n", __func__, ##__VA_ARGS__)
#define _verboselog(fmt,...)      \
    _printf(IPSM_LOG_VERBOSE, "[VERB @%s] " fmt "\n", __func__, ##__VA_ARGS__)

#define USLEEP(usec)         do { \
    struct timespec ts = {usec/1000000, (usec%1000000)*1000}; \
    nanosleep(&ts, NULL); \
} while(/*CONSTCOND*/0)


static void
_printf(int level, const char *fmt, ...)
{
    if (ipsm_loglevel >= level) {
        va_list ap;
        va_start(ap, fmt);
        vfprintf(stderr, fmt, ap);
        va_end(ap);
    }
}

static void
_init_errno(void)
{
    int thr_err;

    thr_err = pthread_key_create(&errno_key, NULL);
    if (thr_err != 0) {
        _critlog("pthread_key_create() failed: %s", strerror(thr_err));
        critical_error();
        return;
    }
    thr_err = pthread_setspecific(errno_key, (void*)IPSM_OK);
    if (thr_err != 0) {
        _critlog("pthread_setspecific() failed: %s", strerror(thr_err));
        critical_error();
        return;
    }
}

static void
_set_errno(ipsm_err err)
{
    int thr_err;
    if ((thr_err = pthread_once(&genkey_once, _init_errno)) != 0) {
        _critlog("pthread_once() failed: %s", strerror(thr_err));
        critical_error();
        return;
    }
    if ((thr_err = pthread_setspecific(errno_key, (void*)err)) != 0) {
        _critlog("pthread_setspecific() failed: %s", strerror(thr_err));
        critical_error();
        return;
    }
}

static ipsm_err
_ipsm_lock(void)
{
    int thr_err;
    if (shmid < 0 || ipsm_ctl == NULL)
        return IPSM_ENOTREADY;

    thr_err = pthread_mutex_lock(&ipsm_ctl->mutex);
    if (thr_err != 0) {
        _critlog("pthread_mutex_lock() failed: %s", strerror(thr_err));
        critical_error();
        return IPSM_EINTERNAL;
    }
    return IPSM_OK;
}

static ipsm_err
_ipsm_unlock(void)
{
    int thr_err;
    if (shmid < 0 || ipsm_ctl == NULL)
        return IPSM_ENOTREADY;

    thr_err = pthread_mutex_unlock(&ipsm_ctl->mutex);
    if (thr_err != 0) {
        _critlog("pthread_mutex_unlock() failed: %s", strerror(thr_err));
        critical_error();
        return IPSM_EINTERNAL;
    }
    return IPSM_OK;
}

int
ipsm_setloglevel(int new)
{
    int old = ipsm_loglevel;
    ipsm_loglevel = new;
    return old;
}

ipsm_err
ipsm_getlasterror(void)
{
    int thr_err;
    void *val;
    if ((thr_err = pthread_once(&genkey_once, _init_errno)) != 0) {
        _critlog("pthread_once failed: %s", strerror(thr_err));
        critical_error();
        return IPSM_EINTERNAL;
    }
    val = pthread_getspecific(errno_key);
    if (val == NULL) {
        _set_errno(IPSM_OK);
        return IPSM_OK;
    }
    return (ipsm_err)val;
}

char *
ipsm_strerror(ipsm_err err)
{
#define _WRAP_ERRNO(_e)  case IPSM_##_e: return strerror(_e);
    switch(err) {
    case IPSM_OK:               return "Success";

    _WRAP_ERRNO(EACCES)
    _WRAP_ERRNO(ENOENT)
    _WRAP_ERRNO(EEXIST)
    _WRAP_ERRNO(EINVAL)
    _WRAP_ERRNO(ENOMEM)
    _WRAP_ERRNO(ETIMEDOUT)
    _WRAP_ERRNO(EALREADY)

    case IPSM_ENOTREADY:        return "Not ready";
    case IPSM_EINTERNAL:        return "Internal error";
    default:                    return "Unknown error";
    }
#undef _WRAP_ERRNO
}

static ipsm_err
_get_shmsize(size_t *sizep)
{
    char *str, *endp;
    long lval;
    if ((str = getenv(IPSM_SHMSIZE_ENV)) == NULL) {
        _errorlog("environment %s is not set", IPSM_SHMSIZE_ENV);
        return IPSM_EINVAL;
    }
    errno = 0;
    lval = strtoll(str, &endp, 16);
    if (errno != 0 || *endp != '\0' || lval < 0) {
        _errorlog("invalid shmsize: %s", str);
        return IPSM_EINVAL;
    }
    *sizep = (size_t)lval;
    return IPSM_OK;
}

static void
_init_share_ctl(void)
{
    assert(shmaddr_head && shmaddr_reserved && shmaddr_shared && ipsm_ctl);

    ipsm_ctl->share_ctl_offset = (size_t)shmaddr_shared - (size_t)shmaddr_head;
    share_ctl->num_alloced = DEF_NUM_ENTRY;

    /* initial share control data */
    entry_array[0].size = ROUNDUP(SIZEOF_SHARE_CTL(DEF_NUM_ENTRY));
    entry_array[0].offset = 0;
    entry_array[0].used = true;
    entry_array[0].sorted_index = -1;

    /* initial free space */
    entry_array[1].size = ipsm_ctl->shmsize 
        - (shmaddr_shared - shmaddr_head) - entry_array[0].size;
    entry_array[1].offset = (ipsm_offset_t)entry_array[0].size;
    entry_array[1].used = false;
    entry_array[1].sorted_index = 0;
    free_entries[0] = 1;

    share_ctl->num_entry = 2;
    share_ctl->num_free = 1;
}

void *
ipsm_init(key_t shmkey, size_t reserve_size)
{
    ipsm_err err;
    int thr_err;
    int tmp_shmid;
    size_t shmsize;
    pthread_mutexattr_t attr;

    _debuglog("shmkey: 0x%x, reserve_size: 0x%lx", shmkey, reserve_size);

    if (shmkey == IPC_PRIVATE) {
        _errorlog("ipsm_init() doesn't allow IPC_PRIVATE as shmkey");
        return_null(IPSM_EINVAL);
    }
    if (shmid != -1) {
        _errorlog("already initialized");
        return_null(IPSM_EALREADY);
    }

    if ((err = _get_shmsize(&shmsize)) != IPSM_OK)
        return_null(err);
    if (shmsize < ROUNDUP(reserve_size) + ROUNDUP(sizeof(ipsm_ctl_st))) {
        _errorlog("too large reserve size: 0x%lx", reserve_size);
        return_null(IPSM_EINVAL);
    }

    tmp_shmid = shmget(shmkey, shmsize, IPC_CREAT|IPC_EXCL|SHM_ACCESSMODE);
    if (tmp_shmid < 0) {
#define _WRAP_ERRNO(_e) do { \
        _errorlog("%s", strerror(_e)); \
        return_null(IPSM_##_e); \
    } while(/*CONSTCOND*/0)
        switch(errno) {
        case EEXIST:
            _errorlog("already initialized with key 0x%x", shmkey);
            return_null(IPSM_EEXIST); break;
        case ENOSPC:
            _errorlog("shmget() failed: %s", strerror(errno));
            return_null(IPSM_ENOMEM); break;
        case ENFILE:
            _errorlog("shmget() failed: %s", strerror(errno));
            return_null(IPSM_ENOMEM); break;
        case ENOMEM:
            _WRAP_ERRNO(ENOMEM); break;
        case EINVAL:
            _WRAP_ERRNO(EACCES); break;
        case EACCES: /* may not happen with IPC_CREAT */
            _WRAP_ERRNO(EACCES); break;
        case ENOENT: /* may not happen with IPC_CREAT */
            _WRAP_ERRNO(ENOENT); break;
        default:
            _errorlog("shmget() failed with unexpected reason: %s", 
                strerror(errno));
            return_null(IPSM_EINTERNAL);
        }
#undef _WRAP_ERRNO
        /*NOTREACHED*/
    }
    _infolog("created shm, key:0x%x id:0x%x size:0x%x", 
        shmkey, tmp_shmid, shmsize);

    /* created shm, now rollback needed on error. */
    err = IPSM_EINTERNAL;

    shmaddr_head = shmat(tmp_shmid, NULL, 0);
    if (shmaddr_head == (void*)-1) {
        switch(errno) {
        case ENOMEM:
            _errorlog("%s", strerror(errno));
            err = IPSM_ENOMEM;
            goto rollback;
        case EACCES: /* may not happen */
        case EINVAL: /* may not happen */
            _errorlog("%s", strerror(errno));
            goto rollback;
        default:
            _errorlog("shmat() failed with unexpected reason: %s",
                strerror(errno));
            goto rollback;
        }
        /*NOTREACHED*/
    }
    _infolog("attached shm at %p", shmaddr_head);
    
    ipsm_ctl = (ipsm_ctl_st*)shmaddr_head;
    ipsm_ctl->magic = IPSM_MAGIC;
    ipsm_ctl->shmsize = shmsize;
    ipsm_ctl->reserved_size = ROUNDUP(reserve_size);
    shmaddr_reserved = (void*)
        ((size_t)shmaddr_head + ROUNDUP(sizeof(ipsm_ctl_st)));
    shmaddr_shared = (void*)
        ((size_t)shmaddr_reserved + ipsm_ctl->reserved_size);
    shmaddr_tail = (char*)shmaddr_head + shmsize;
    _debuglog("ipsm_ctl: %p", ipsm_ctl);
    _debuglog("reserved: %p", shmaddr_reserved);
    _debuglog("shared:   %p", shmaddr_shared);
    _debuglog("tail:     %p", shmaddr_tail);

    if ((thr_err = pthread_mutexattr_init(&attr)) != 0) {
        _critlog("pthread_mutexattr_init() failed: %s", strerror(thr_err));
        goto rollback;
    }
    if ((thr_err = pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED))
            != 0) {
        _critlog("pthread_mutexattr_setpshared() failed: %s", strerror(thr_err));
        goto rollback;
    }
    if ((thr_err = pthread_mutex_init(&ipsm_ctl->mutex, &attr)) != 0) {
        _critlog("pthread_mutex_init() failed: %s", strerror(thr_err));
        goto rollback;
    }

    _init_share_ctl();

    shmid = tmp_shmid;

    return shmaddr_reserved;

rollback:
    if (shmaddr_head != (void*)-1) {
        if (shmdt(shmaddr_head) != 0) {
            _warnlog("cannot detache shm %p: %s", shmaddr_head, strerror(errno));
        } else {
            _noticelog("detached shm %p shmid: 0x%x", shmaddr_head, shmid);
        }
        shmaddr_head = (void*)-1;
    }
    if (tmp_shmid >= 0) {
        if (shmctl(tmp_shmid, IPC_RMID, NULL) < 0) {
            _errorlog("shmctl(IPC_RMID) failed for shmid 0x%x: %s", 
                shmid, strerror(errno));
        } else {
            _noticelog("deleted shm, id: 0x%x", tmp_shmid);
        }
    }
    return_null(err);
}

int
ipsm_destroy(void)
{
    if (shmid == -1)
        return_failure(IPSM_ENOTREADY);
    if (ipsm_ctl->magic == IPSM_MAGIC) {
        ipsm_ctl->magic = IPSM_MAGIC_DEAD;
        if (shmctl(shmid, IPC_RMID, NULL) != 0) {
            _critlog("shmctl(IPC_RMID) failed for shmid 0x%x: %s", 
                shmid, strerror(errno));
            return_failure(IPSM_EINTERNAL);
        }
        _infolog("destroyed shmid: 0x%x", shmid);
    } else if (ipsm_ctl->magic == IPSM_MAGIC_DEAD) {
        _debuglog("already destroyed");
    }
    shmid = -1;
    /*shm_shared = false;*/ /* Humm.... */
    return_success();
}


int
ipsm_share(void)
{
    if (shmid == -1 || ipsm_ctl->magic != IPSM_MAGIC)
        return_failure(IPSM_ENOTREADY);

    ipsm_ctl->shared = true;
    _infolog("start sharing shm, id: 0x%x", shmid);
    return_success();
}

void *
ipsm_tryjoin(key_t shmkey, int retry_interval_usec, int retry_times_max)
{
    ipsm_err err;
    int tmp_shmid;
    size_t shmsize;
    int retry = retry_times_max;

    _debuglog("shmkey: 0x%x", shmkey);

    if (shmkey == IPC_PRIVATE) {
        _errorlog("ipsm_join() doesn't allow IPC_PRIVATE as shmkey");
        return_null(IPSM_EINVAL);
    }
    if (shmid != -1) {
        _errorlog("already joined with shmid: 0x%x", shmid);
        return_null(IPSM_EALREADY);
    }

    if ((err = _get_shmsize(&shmsize)) != IPSM_OK)
        return_null(err);

retry:
    tmp_shmid = shmget(shmkey, shmsize, SHM_ACCESSMODE);
    if (tmp_shmid < 0) {
#define _WRAP_ERRNO(_e) do { \
        _errorlog("%s", strerror(_e)); \
        return_null(IPSM_##_e); \
    } while(/*CONSTCOND*/0)
        switch(errno) {
        case EEXIST: /* may not happen without IPC_CREAT */
            _errorlog("already initialized with key 0x%x", shmkey);
            return_null(IPSM_EEXIST); break;
        case ENOSPC: /* may not happen without IPC_CREAT */
            _errorlog("shmget() failed: %s", strerror(errno));
            return_null(IPSM_ENOMEM); break;
        case ENFILE: /* may not happen without IPC_CREAT */
            _errorlog("shmget() failed: %s", strerror(errno));
            return_null(IPSM_ENOMEM); break;
        case ENOMEM: /* may not happen without IPC_CREAT */
            _WRAP_ERRNO(ENOMEM); break;
        case EINVAL:
            _WRAP_ERRNO(EACCES); break;
        case EACCES:
            _WRAP_ERRNO(EACCES); break;
        case ENOENT:
            if (retry >= 0 && --retry < 0)
                return_null(IPSM_ENOTREADY);
            _verboselog("shmget() returned ENOENT, maybe not ready. retrying");
            USLEEP(retry_interval_usec);
            goto retry;
        default:
            _errorlog("shmget() failed with unexpected reason: %s", 
                strerror(errno));
            return_null(IPSM_EINTERNAL);
        }
#undef _WRAP_ERRNO
        /*NOTREACHED*/
    }
    _infolog("joined shm, key:0x%x id:0x%x size:0x%x", 
        shmkey, tmp_shmid, shmsize);

    /* joined shm, now rollback is needed on error. */
    err = IPSM_EINTERNAL;

    shmaddr_head = shmat(tmp_shmid, NULL, 0);
    if (shmaddr_head == (void*)-1) {
        switch(errno) {
        case ENOMEM:
            _errorlog("%s", strerror(errno));
            err = IPSM_ENOMEM;
            goto rollback;
        case EACCES: /* may not happen */
        case EINVAL: /* may not happen */
            _errorlog("%s", strerror(errno));
            goto rollback;
        default:
            _errorlog("shmat() failed with unexpected reason: %s",
                strerror(errno));
            goto rollback;
        }
        /*NOTREACHED*/
    }
    _infolog("attached shm at %p", shmaddr_head);
    
    ipsm_ctl = (ipsm_ctl_st*)shmaddr_head;
    if (ipsm_ctl->magic != IPSM_MAGIC) {
        _debuglog(
            "joined shm is invalid (magic=0x%x), maybe not ready. retrying", 
            ipsm_ctl->magic);
        err = IPSM_ENOTREADY;
        goto rollback;
    }
    if (ipsm_ctl->shared != true) {
        _verboselog("joined shm is not yet shared, retrying");
        err = IPSM_ENOTREADY;
        goto rollback;
    }
    if (ipsm_ctl->shmsize != shmsize) {
        _errorlog("joined shm with different shmsize, join: 0x%x, shm: 0x%x",
            shmsize, ipsm_ctl->shmsize);
        err = IPSM_EINVAL;
        goto rollback;
    }
    shmaddr_reserved = (void*)
        ((size_t)shmaddr_head + ROUNDUP(sizeof(ipsm_ctl_st)));
    shmaddr_shared = (void*)
        ((size_t)shmaddr_reserved + ipsm_ctl->reserved_size);
    shmaddr_tail = (char*)shmaddr_head + shmsize;
    _debuglog("ipsm_ctl: %p", ipsm_ctl);
    _debuglog("reserved: %p", shmaddr_reserved);
    _debuglog("shared:   %p", shmaddr_shared);
    _debuglog("tail:     %p", shmaddr_tail);

    shmid = tmp_shmid;
    err = IPSM_OK;

    return shmaddr_reserved;

rollback:
    if (shmaddr_head != (void*)-1) {
        if (shmdt(shmaddr_head) != 0) {
            _errorlog("cannot detache shm %p: %s", 
                shmaddr_head, strerror(errno));
            err = IPSM_EINTERNAL;
        } else {
            _noticelog("detached shm %p shmid: 0x%x", shmaddr_head, tmp_shmid);
        }
        shmaddr_head = (void*)-1;
    }
    tmp_shmid = -1;

    if (err == IPSM_ENOTREADY && retry >= 0 && --retry < 0) {
        USLEEP(retry_interval_usec);
        goto retry;
    }
    return_null(err);
}

void *
ipsm_join(key_t shmkey)
{
    return ipsm_tryjoin(shmkey, IPSM_JOIN_RETRY_INTERVAL, -1);
}


static void
_shift_entry_array(int src_idx, int num)
{
    if (num == 0)
        return;
    entry_st *dst, *src;
    size_t move_size;
    int idx;

    if (src_idx < share_ctl->num_entry ||
            (num < 0 && src_idx < share_ctl->num_entry + num)) {
        for (idx = src_idx; idx < share_ctl->num_entry; idx ++) {
            if (entry_array[idx].used == false) {
                assert(entry_array[idx].sorted_index >= 0);
                free_entries[entry_array[idx].sorted_index] += num;
            }
        }
        src = &entry_array[src_idx];
        dst = &entry_array[src_idx + num];
        move_size = sizeof(entry_st) * (share_ctl->num_entry - src_idx);
        memmove(dst, src, move_size);
    }
    share_ctl->num_entry += num;
}

static void
_shift_free_entries(int src_idx, int num)
{
    if (num == 0)
        return;
    int *dst, *src;
    size_t move_size;
    int idx;

    if (src_idx < share_ctl->num_free ||
            (num < 0 && src_idx < share_ctl->num_free + num)) {
        for (idx = src_idx; idx < share_ctl->num_free; idx ++) {
            entry_array[free_entries[idx]].sorted_index += num;
            assert(entry_array[free_entries[idx]].sorted_index >= 0);
        }
        src = &free_entries[src_idx];
        dst = &free_entries[src_idx + num];
        move_size = sizeof(int) * (share_ctl->num_free - src_idx);
        memmove(dst, src, move_size);
    }
    share_ctl->num_free += num;
}


static int
_free(void *addr, bool internal)
{
    /* internal free,
     * ipsm_ctl->mutex should be held.
     */
    int idx;
    ipsm_offset_t offset;
    int target;
    size_t new_size;

    if (addr == shmaddr_tail)
        return_success();
    if (addr < shmaddr_shared || shmaddr_tail < addr)
        return_failure(IPSM_EINVAL);
    offset = ADDR2OFFSET(addr);

    for (idx = 0; idx < share_ctl->num_entry; idx ++) {
        if (entry_array[idx].offset == offset)
            break;
    }
    if (idx >= share_ctl->num_entry) /* no such entry */
        return_failure(IPSM_EINVAL);
    if (entry_array[idx].used != true) /* already free'ed? */
        return_failure(IPSM_EINVAL);
    target = idx;

    _debuglog("free target: idx=%d", target);

    if (target > 0 && entry_array[target - 1].used == false) {
        /* target should be merged into previous entry */
        new_size = entry_array[target - 1].size + entry_array[target].size;

        if (target < share_ctl->num_entry - 1
                && entry_array[target + 1].used == false) {
            /* next also should be merged into previous */
            new_size += entry_array[target + 1].size;

            /* delete next from sorted */
            _shift_free_entries(entry_array[target+1].sorted_index + 1, -1);

            /* delete prev from sorted */
            _shift_free_entries(entry_array[target-1].sorted_index + 1, -1);

            /* delete target & next from entry_array */
            _shift_entry_array(target + 2, -2);

        } else {
            /* delete prev from sorted */
            _shift_free_entries(entry_array[target-1].sorted_index + 1, -1);

            /* delete target from entry_array */
            _shift_entry_array(target + 1, -1);

        }

        /* update prev and re-sort */
        entry_array[target - 1].size = new_size;
        for (idx = 0; idx < share_ctl->num_free; idx ++) {
            if (entry_array[free_entries[idx]].size > new_size)
                break;
        }
        _shift_free_entries(idx, 1);
        free_entries[idx] = target - 1;
        entry_array[target - 1].sorted_index = idx;

    } else {
        if (target < share_ctl->num_entry 
                && entry_array[target + 1].used == false) {
            /* next should be merged into target */
            entry_array[target].size += entry_array[target + 1].size;

            /* delete next from sorted */
            _shift_free_entries(entry_array[target+1].sorted_index + 1, -1);

            /* delete next from entry_array */
            _shift_entry_array(target + 2, -1);
        }

        /* put target(+next) into sorted */
        new_size = entry_array[target].size;
        for (idx = 0; idx < share_ctl->num_free; idx ++) {
            if (entry_array[free_entries[idx]].size > new_size)
                break;
        }
        _shift_free_entries(idx, 1);
        free_entries[idx] = target;
        entry_array[target].sorted_index = idx;
        entry_array[target].used = false;
    }

    return_success();
}

int
ipsm_free(void *addr)
{
    int ret = RETCODE_FAILURE;
    int ostat;
    ipsm_err err = IPSM_EINTERNAL;

    if (shmid < 0 || ipsm_ctl == NULL)
        return_failure(IPSM_ENOTREADY);
    if (addr == NULL)
        return_success();
    if (addr == shmaddr_tail) /* reserved addr for ipsm_alloc(0) */
        return_success();
    if (addr < shmaddr_shared || shmaddr_tail < addr)
        return_failure(IPSM_EINVAL);

    pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &ostat);
    if ((err = _ipsm_lock()) != IPSM_OK)
        goto finally;

    ret = _free(addr, false);

finally:
    if ((err = _ipsm_unlock()) != IPSM_OK) {
        _set_errno(err);
        /* cannot do anything for this error */
    }
    pthread_setcancelstate(ostat, NULL);
    return ret;
}


static void *
_alloc(size_t memsize, bool internal)
{
    /* internal alloc,
     * memsize should be rounded up.
     * ipsm_ctl->mutex should be held.
     */
    _debuglog("memsize: 0x%x internal: %d", memsize, internal);

    void *ptr = NULL;
    int idx, found, old_index, new_index;
    entry_st *victim_entry, *new_entry;
    size_t remnant;

    if (share_ctl->num_entry == share_ctl->num_alloced - 1 && !internal) {
        void *old_ctl = share_ctl;
        size_t new_ctl_size = SIZEOF_SHARE_CTL(share_ctl->num_alloced * 2);

        share_ctl_st *new_ctl = _alloc(ROUNDUP(new_ctl_size), true);

        if (new_ctl == NULL)
            return NULL;
        memset(new_ctl, 0, new_ctl_size);
        memcpy(new_ctl->fake_entry_array, entry_array,
            sizeof(entry_st) * share_ctl->num_entry);
        memcpy((entry_st*)new_ctl->fake_entry_array + share_ctl->num_alloced *2,
            free_entries, sizeof(int) * share_ctl->num_free);
        new_ctl->num_alloced = share_ctl->num_alloced * 2;
        new_ctl->num_entry = share_ctl->num_entry;
        new_ctl->num_free = share_ctl->num_free;
        ipsm_ctl->share_ctl_offset = (size_t)new_ctl - (size_t)shmaddr_head;
        if (_free(old_ctl, true) != 0) {
            _errorlog("internal freeing share_ctl failed");
            return_null(IPSM_EINTERNAL);
        }
    }

    /* search a free entry which has enough & smallest size */
    for (found = -1, idx = 0; idx < share_ctl->num_free; idx ++) {
        if (entry_array[free_entries[idx]].size >= memsize) {
            found = idx;
            break;
        }
    }
    if (found == -1) {
        return_null(IPSM_ENOMEM);
    }

    old_index = free_entries[found];
    victim_entry = &entry_array[old_index];

    /* delete found from sorted */
    _shift_free_entries(found + 1, -1);

    remnant = victim_entry->size - memsize;

    /* update victim as a used share */
    victim_entry->size = memsize;
    /* offset is not changed */
    victim_entry->used = true;
    victim_entry->sorted_index = -1;

    if (remnant > 0) {
        /* add a new free entry, remnant of victim */
        _shift_entry_array(old_index + 1, 1);

        new_index = old_index + 1;
        new_entry = &entry_array[new_index];
        new_entry->size = remnant;
        new_entry->offset = victim_entry->offset + memsize;
        new_entry->used = false;

        /* put new_entry into sorted */
        for (idx = 0; idx < share_ctl->num_free; idx ++) {
            if (entry_array[free_entries[idx]].size > new_entry->size)
                break;
        }
        _shift_free_entries(idx, 1);
        free_entries[idx] = new_index;
        new_entry->sorted_index = idx;
    }

    ptr = OFFSET2ADDR(victim_entry->offset);

    return ptr;
}

void *
ipsm_alloc(size_t required)
{
    int ostat;
    ipsm_err err;
    void *ptr = NULL;
    size_t memsize = ROUNDUP(required);

    if (shmid < 0 || ipsm_ctl == NULL)
        return_null(IPSM_ENOTREADY);
    if (required == 0)
        return shmaddr_tail;
    if (required > (size_t)shmaddr_tail - (size_t)shmaddr_shared)
        return_null(IPSM_ENOMEM);

    pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &ostat);
    if ((err = _ipsm_lock()) != IPSM_OK)
        goto finally;

    ptr = _alloc(memsize, false);

finally:
    if ((err = _ipsm_unlock()) != IPSM_OK) {
        _set_errno(err);
        /* cannot do anything for this error */
    }
    pthread_setcancelstate(ostat, NULL);

    return ptr;
}

ipsm_offset_t
ipsm_addr2offset(void *addr)
{
    if (shmid < 0 || ipsm_ctl == NULL) {
        _set_errno(IPSM_ENOTREADY);
        return (ipsm_offset_t)-1;
    }
    if (addr < shmaddr_shared || shmaddr_tail < addr) {
        _set_errno(IPSM_EINVAL);
        return(ipsm_offset_t)-1;
    }
    return (ipsm_offset_t)addr - (ipsm_offset_t)shmaddr_shared;
}

void *
ipsm_offset2addr(ipsm_offset_t offset)
{
    if (shmid < 0 || ipsm_ctl == NULL) {
        _set_errno(IPSM_ENOTREADY);
        return NULL;
    }
    void *addr = (char*)shmaddr_shared + offset;
    if (addr < shmaddr_shared || shmaddr_tail < addr) {
        _set_errno(IPSM_EINVAL);
        return NULL;
    }
    return addr;
}



void
ipsm_print_status(key_t shmkey)
{
    ipsm_err err;
    ipsm_ctl_st *tmp_ctl;
    size_t shmsize;
    int idx;
    entry_st *entry;

    if (shmkey == IPC_PRIVATE) {
        printf("ERROR: invalid shmkey: 0x%x (IPC_PRIVATE is not allowed).\n",
            shmkey);
        return;
    }

    _verboselog("trying temporary shmget().");
    shmid = shmget(shmkey, sizeof(ipsm_ctl_st), SHM_ACCESSMODE);
    if (shmid < 0) {
        switch(errno) {
        case ENOENT:
            _debuglog("shmget() returned ENOENT.");
            printf("Not ready.\n");
            return;
        default:
            printf("Internal error: shmget() returned '%s'.\n", strerror(errno));
            return;
        }
    }
    tmp_ctl = shmat(shmid, NULL, 0);
    if (tmp_ctl == (void*)-1) {
        printf("Internal error: shmat() returned '%s'.\n", strerror(errno));
        return;
    }
    if (tmp_ctl->magic != IPSM_MAGIC) {
        _debuglog("magic number mismatch (got 0x%x, expected 0x%x).",
            tmp_ctl->magic, IPSM_MAGIC);
        printf("Not ready.\n");
        return;
    }
    shmsize = tmp_ctl->shmsize;
    _verboselog("got stored shmsize, 0x%x.", shmsize);

    if (shmdt(tmp_ctl) != 0) {
        printf("Internal error: shmdt() returned '%s'.\n", strerror(errno));
        return;
    }

    _verboselog("calling shmget() again with stored shmsize.");
    shmid = shmget(shmkey, shmsize, SHM_ACCESSMODE);
    if (shmid < 0) {
        printf("Internal error: shmget() returned '%s'.\n", strerror(errno));
        return;
    }
    shmaddr_head = shmat(shmid, NULL, 0);
    if (shmaddr_head == (void*)-1) {
        printf("Internal error: shmat() returned '%s'.\n", strerror(errno));
        return;
    }
    ipsm_ctl = (ipsm_ctl_st*)shmaddr_head;
    if (ipsm_ctl->magic != IPSM_MAGIC) {
        _debuglog("magic number mismatch (got 0x%x, expected 0x%x).",
            ipsm_ctl->magic, IPSM_MAGIC);
        printf("Not ready.\n");
        return;
    }
    _verboselog("ok, ipsm looks ready to join. showing internal data.");

    if ((err = _ipsm_lock()) != IPSM_OK) {
        printf("Internal error: _ipsm_lock returned() '%s'.\n",
            ipsm_strerror(err));
        return;
    }

    printf("ipsm is ready to join with key '0x%x'.\n", shmkey);

    printf("----[ ipsm_ctl ]----\n");
    printf("magic:      0x%x\n",  ipsm_ctl->magic);
    printf("shmsize:    0x%lx\n", ipsm_ctl->shmsize);
    printf("reserve:    0x%lx\n", ipsm_ctl->reserved_size);
    printf("shared:     %x\n",    ipsm_ctl->shared);
    printf("share_ctl:  0x%lx\n", ipsm_ctl->share_ctl_offset);

    printf("----[ share_ctl ]----\n");
    printf("alloced:    %d\n", share_ctl->num_alloced);
    printf("num_entry:  %d\n", share_ctl->num_entry);
    printf("num_free:   %d\n", share_ctl->num_free);

    printf("----[ entry list ]----\n");
    printf("index\toffset\t\tsize\t\tused\tsorted_index\n");
    for (idx = 0; idx < share_ctl->num_entry; idx ++) {
        entry = &entry_array[idx];
        printf("%3d\t0x%08lx\t0x%08lx\t%d\t%3d\n",
            idx, entry->offset, entry->size, entry->used, entry->sorted_index);
    }
    printf("----[ sorted free entry ]----\n");
    printf("index\tsize\t\tentry_index\n");
    for (idx = 0; idx < share_ctl->num_free; idx ++) {
        entry = &entry_array[free_entries[idx]];
        printf("%3d\t0x%08lx\t%3d\n",
            idx, entry->size, free_entries[idx]);
    }
    printf("\n");

    if ((err = _ipsm_unlock()) != IPSM_OK) {
        printf("internal error, _ipsm_unlock() returned '%s'.\n",
            ipsm_strerror(err));
             return;
    }
    if (shmdt(shmaddr_head) != 0) {
        printf("internal error, shmdt() returned '%s'.\n", strerror(errno));
    }

}

