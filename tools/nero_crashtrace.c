/* NERO V4.6.6 NATIVE CRASH TRACE: loaded with LD_PRELOAD by start_nero.sh (built on first start with gcc).
 * On SIGSEGV / SIGABRT / SIGBUS / SIGFPE / SIGILL it appends the C-level stack (function + library names) to the file
 * in $NERO_NATIVE_CRASH_LOG, then lets the default action happen.  Python's faulthandler (crash.log) runs first and
 * hands the signal on to this handler, so both logs are written.  Nothing is done while the program runs normally. */
#define _GNU_SOURCE
#include <execinfo.h>
#include <fcntl.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int g_fd = -1;

static void put(const char *s) { if (g_fd >= 0) { ssize_t r = write(g_fd, s, strlen(s)); (void)r; } }

static void put_num(long v) {
    char t[24]; int k = 0;
    if (v < 0) { put("-"); v = -v; }
    do { t[k++] = (char)('0' + v % 10); v /= 10; } while (v && k < 23);
    while (k) { char c[2] = {t[--k], 0}; put(c); }
}

static void on_fatal(int sig, siginfo_t *si, void *uc) {
    (void)si; (void)uc;
    void *frames[128];
    int n = backtrace(frames, 128);
    put("\n===== NERO native crash: signal "); put_num(sig);
    put(" pid "); put_num((long)getpid());
    put(" =====\n");
    if (g_fd >= 0) backtrace_symbols_fd(frames, n, g_fd);
    put("===== end =====\n");
    signal(sig, SIG_DFL);
    raise(sig);
}

__attribute__((constructor)) static void nero_crashtrace_init(void) {
    const char *path = getenv("NERO_NATIVE_CRASH_LOG");
    if (!path || !*path) return;
    g_fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (g_fd < 0) return;
    void *warm[2]; backtrace(warm, 2);            /* load libgcc now, not inside the signal handler */
    static char altstack[1 << 16];
    stack_t ss; ss.ss_sp = altstack; ss.ss_size = sizeof altstack; ss.ss_flags = 0;
    sigaltstack(&ss, 0);
    struct sigaction sa; memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = on_fatal;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESETHAND;
    sigemptyset(&sa.sa_mask);
    int sigs[] = {SIGSEGV, SIGABRT, SIGBUS, SIGFPE, SIGILL};
    for (unsigned i = 0; i < sizeof sigs / sizeof sigs[0]; i++) sigaction(sigs[i], &sa, 0);
}
