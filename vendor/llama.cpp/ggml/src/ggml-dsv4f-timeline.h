#pragma once

// Optional CPU ranges on the profiler's own clock. Preload the ROCm ROCTx
// library and set DSV4F_TIMELINE=1; ordinary execution has no ROCTx dependency.
#include <cstdlib>
#include <cstring>
#if defined(__linux__)
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace dsv4f_timeline {
struct api {
    using push_t = int (*)(const char *);
    using pop_t = int (*)();
    using control_t = int (*)(unsigned long);
    push_t push = nullptr;
    pop_t pop = nullptr;
    control_t resume = nullptr, pause = nullptr;
    api() {
#if defined(__linux__)
        const char * enabled = std::getenv("DSV4F_TIMELINE");
        if (enabled && !std::strcmp(enabled, "1")) {
            push = reinterpret_cast<push_t>(dlsym(RTLD_DEFAULT, "roctxRangePushA"));
            pop = reinterpret_cast<pop_t>(dlsym(RTLD_DEFAULT, "roctxRangePop"));
            resume = reinterpret_cast<control_t>(dlsym(RTLD_DEFAULT, "roctxProfilerResume"));
            pause = reinterpret_cast<control_t>(dlsym(RTLD_DEFAULT, "roctxProfilerPause"));
            if (!push || !pop) push = nullptr, pop = nullptr;
        }
#endif
    }
};
inline const api & functions() { static const api value; return value; }
// Called by the single-sequence input thread only. A local control file lets
// the driver collect selected requests without tracing model load/prefill.
inline void update_collection() {
#if defined(__linux__)
    static const char * path = std::getenv("DSV4F_TIMELINE_CONTROL");
    static bool collecting = false;
    const auto & f = functions();
    if (!path || !f.resume || !f.pause) return;
    const int fd = open(path, O_RDONLY);
    if (fd < 0) return;
    char value = 0;
    const ssize_t bytes = read(fd, &value, 1);
    close(fd);
    if (bytes != 1 || (value != '0' && value != '1')) return;
    const bool wanted = value == '1';
    if (wanted != collecting) {
        if (wanted) f.resume(0); else f.pause(0);
        collecting = wanted;
    }
#endif
}
class scope {
    api::pop_t pop = nullptr;
public:
    explicit scope(const char * name) {
        const auto & f = functions();
        if (f.push) {
            f.push(name);
            pop = f.pop;
        }
    }
    ~scope() { if (pop) pop(); }
    scope(const scope &) = delete;
    scope & operator=(const scope &) = delete;
};
}
