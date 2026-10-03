## Executable memory for the JIT, and calls into / out of generated code.
##
## Code is written while its pages are read-write, then the pages are
## flipped to read-execute (W^X): a page is never writable and executable
## at the same time. `install` gives every function its own run of whole
## pages (bump-allocated from 64 KiB regions), so installing new code never
## has to make a page that already holds live code writable again - JIT code
## may be on the stack (calling back into the runtime) while more is compiled.
##
## nimony cannot `cast` between `pointer` and proc types, so both directions
## go through C (`{.emit.}` + `importc, nodecl`):
## * `jitCall*` call generated code with a fixed C signature;
## * `procAddr(p)` yields the address of a Nim proc (declare the proc
##   `{.cdecl.}`, plus `{.exportc: "name".}` if it should also be findable
##   by name) for `Assembler.callAbs`.
##
## Calling convention for JIT functions (the "trampoline"): generated code is
## an ordinary C-ABI function, SysV on Linux / Win64 on Windows:
##   int64 fn(void* vm, void* frame)            -> jitCall2
##   int64 fn(int64, int64, int64)              -> jitCall3
##   double fn(double, double)                  -> jitCallF2
## Arguments arrive in `argRegs` (x64asm.nim), the result goes in rax (xmm0
## for doubles). Callbacks from JIT code into the runtime are plain C-ABI
## calls to `{.cdecl.}` procs: the JIT keeps rsp 16-byte aligned at the call
## (x64asm `prologue` does this) and preserves rbx, rbp, r12-r15 (plus rdi,
## rsi, xmm6-15 on Windows).

{.emit: """
#include <stdint.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sys/mman.h>
#include <unistd.h>
#endif

static int64_t jit_page_size(void) {
#ifdef _WIN32
  SYSTEM_INFO si; GetSystemInfo(&si); return (int64_t)si.dwPageSize;
#else
  return (int64_t)sysconf(_SC_PAGESIZE);
#endif
}

static void* jit_map(int64_t size) {
#ifdef _WIN32
  return VirtualAlloc(NULL, (SIZE_T)size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
  void* p = mmap(NULL, (size_t)size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  return p == MAP_FAILED ? (void*)0 : p;
#endif
}

static int64_t jit_unmap(void* p, int64_t size) {
#ifdef _WIN32
  (void)size; return VirtualFree(p, 0, MEM_RELEASE) ? 0 : -1;
#else
  return munmap(p, (size_t)size);
#endif
}

/* exec != 0: read+execute; else read+write */
static int64_t jit_protect(void* p, int64_t size, int64_t exec) {
#ifdef _WIN32
  DWORD old;
  if (!VirtualProtect(p, (SIZE_T)size, exec ? PAGE_EXECUTE_READ : PAGE_READWRITE, &old)) return -1;
  if (exec) FlushInstructionCache(GetCurrentProcess(), p, (SIZE_T)size);
  return 0;
#else
  return mprotect(p, (size_t)size, exec ? (PROT_READ | PROT_EXEC) : (PROT_READ | PROT_WRITE));
#endif
}

/* A dual-mapped region (Linux): one memfd mapped read+write (where code is
   written) and read+execute (where it runs) - no page is ever writable and
   executable in the same mapping, yet code is packed densely and installing
   it needs no system call. *rx gets the executable view; the result is the
   writable one (0: not available, use per-install mprotect instead). */
static int64_t jit_map_dual(int64_t size, int64_t* rx) {
#if defined(__linux__)
  int fd = memfd_create("aowljs-jit", MFD_CLOEXEC);
  if (fd < 0) return 0;
  if (ftruncate(fd, (off_t)size) != 0) { close(fd); return 0; }
  void* w = mmap(NULL, (size_t)size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (w == MAP_FAILED) { close(fd); return 0; }
  void* x = mmap(NULL, (size_t)size, PROT_READ | PROT_EXEC, MAP_SHARED, fd, 0);
  close(fd);
  if (x == MAP_FAILED) { munmap(w, (size_t)size); return 0; }
  *rx = (int64_t)(intptr_t)x;
  return (int64_t)(intptr_t)w;
#else
  (void)size; (void)rx; return 0;
#endif
}

static void jit_copy(void* dst, const void* src, int64_t n) { memcpy(dst, src, (size_t)n); }
#include <stdlib.h>
static void* jit_calloc(int64_t n) { return calloc((size_t)n, 8); }

static void jit_poke(void* base, int64_t off, int64_t b) { ((uint8_t*)base)[off] = (uint8_t)b; }
static int64_t jit_peek(void* base, int64_t off) { return ((uint8_t*)base)[off]; }
static void* jit_offset(void* base, int64_t off) { return (void*)((uint8_t*)base + off); }
static int64_t jit_ptr_int(void* p) { return (int64_t)(intptr_t)p; }

static int64_t jit_call0(void* f) { return ((int64_t (*)(void))f)(); }
static int64_t jit_call1(void* f, int64_t a) { return ((int64_t (*)(int64_t))f)(a); }
static int64_t jit_call2(void* f, void* a, void* b) { return ((int64_t (*)(void*, void*))f)(a, b); }
static int64_t jit_call3(void* f, int64_t a, int64_t b, int64_t c) {
  return ((int64_t (*)(int64_t, int64_t, int64_t))f)(a, b, c);
}
static int64_t jit_call4(void* f, int64_t a, int64_t b, int64_t c, int64_t d) {
  return ((int64_t (*)(int64_t, int64_t, int64_t, int64_t))f)(a, b, c, d);
}
static double jit_callf1(void* f, double a) { return ((double (*)(double))f)(a); }
static double jit_callf2(void* f, double a, double b) { return ((double (*)(double, double))f)(a, b); }

static void* jit_load_ptr(void* slot) { return *(void**)slot; }
""".}

proc jit_page_size(): int64 {.importc, nodecl.}
proc jit_map(size: int64): pointer {.importc, nodecl.}
proc jit_unmap(p: pointer; size: int64): int64 {.importc, nodecl.}
proc jit_protect(p: pointer; size: int64; exec: int64): int64 {.importc, nodecl.}
proc jit_poke(base: pointer; off: int64; b: int64) {.importc, nodecl.}
proc jit_map_dual(size: int64; rx: ptr int64): int64 {.importc, nodecl.}
proc jit_copy(dst, src: pointer; n: int64) {.importc, nodecl.}
proc jit_calloc(n: int64): pointer {.importc, nodecl.}
proc callocWords*(n: int): pointer = jit_calloc(int64(n))
  ## n zeroed 8-byte words that never move (nil if they cannot be had)
proc jit_peek(base: pointer; off: int64): int64 {.importc, nodecl.}
proc jit_offset(base: pointer; off: int64): pointer {.importc, nodecl.}
proc jit_ptr_int(p: pointer): int64 {.importc, nodecl.}
proc jit_call0(f: pointer): int64 {.importc, nodecl.}
proc jit_call1(f: pointer; a: int64): int64 {.importc, nodecl.}
proc jit_call2(f: pointer; a, b: pointer): int64 {.importc, nodecl.}
proc jit_call3(f: pointer; a, b, c: int64): int64 {.importc, nodecl.}
proc jit_call4(f: pointer; a, b, c, d: int64): int64 {.importc, nodecl.}
proc jit_callf1(f: pointer; a: float): float {.importc, nodecl.}
proc jit_callf2(f: pointer; a, b: float): float {.importc, nodecl.}

proc jit_load_ptr(slot: pointer): pointer {.importc, nodecl.}

proc loadFnPtr*(slot: pointer): pointer = jit_load_ptr(slot)
  ## The code address stored in a proc-typed variable (pass `addr v`).

proc procAddr*[T](f: T): pointer =
  ## The machine address of a `{.cdecl.}` (or `{.nimcall.}` non-closure)
  ## proc, for embedding in generated code. (A proc variable of such a type
  ## is a bare C function pointer; its bits are read back through C.)
  var slot = f
  loadFnPtr(addr slot)

proc ptrToInt*(p: pointer): int64 = jit_ptr_int(p)
proc offsetPtr*(p: pointer; off: int): pointer = jit_offset(p, off)
proc peekByte*(p: pointer; off: int): int = int(jit_peek(p, off))

# calling generated code
proc jitCall0*(code: pointer): int64 = jit_call0(code)
proc jitCall1*(code: pointer; a: int64): int64 = jit_call1(code, a)
proc jitCall2*(code: pointer; vm, frame: pointer): int64 = jit_call2(code, vm, frame)
  ## the JIT entry convention: fn(vmStatePtr, framePtr) -> int64
proc jitCall3*(code: pointer; a, b, c: int64): int64 = jit_call3(code, a, b, c)
proc jitCall4*(code: pointer; a, b, c, d: int64): int64 = jit_call4(code, a, b, c, d)
proc jitCallF1*(code: pointer; a: float): float = jit_callf1(code, a)
proc jitCallF2*(code: pointer; a, b: float): float = jit_callf2(code, a, b)

type
  CodeRegion = object
    base: pointer
    size: int
    used: int          ## bytes handed out (page-aligned; 16-byte aligned when dual)
    wbase: int         ## dual-mapped: the writable view of `base` (else 0)

  JitMemory* = object
    regions: seq[CodeRegion]
    pageSize*: int
    regionSize*: int
    totalCode*: int    ## bytes of code installed

var jitDualMap* = true  ## --jit-mem=pages: every install on pages of its own (mprotect)

proc initJitMemory*(regionSize = 65536): JitMemory =
  let ps = int(jit_page_size())
  var rs = regionSize
  if rs < ps: rs = ps
  rs = ((rs + ps - 1) div ps) * ps
  JitMemory(regions: @[], pageSize: ps, regionSize: rs, totalCode: 0)

proc install*(jm: var JitMemory; code: seq[byte]): pointer =
  ## Copies `code` into executable memory and returns the entry address (nil
  ## if memory could not be mapped or protected). Dual-mapped regions (Linux)
  ## pack code densely: written through the read-write view, run from the
  ## read-execute one. Otherwise: fresh read-write pages, flipped to
  ## read-execute.
  let n = code.len
  if n == 0: return nil
  let ps = jm.pageSize
  let last = jm.regions.len - 1
  if last >= 0 and jm.regions[last].wbase != 0:
    let need = (n + 15) and not 15
    if jm.regions[last].size - jm.regions[last].used >= need:
      let off = jm.regions[last].used
      jit_copy(cast[pointer](jm.regions[last].wbase + off), addr code[0], n)
      jm.regions[last].used = off + need
      jm.totalCode = jm.totalCode + n
      return jit_offset(jm.regions[last].base, off)
  if jitDualMap:
    let size = max(jm.regionSize, ((n + ps - 1) div ps) * ps)
    var rx = 0'i64
    let w = jit_map_dual(size, addr rx)
    if w != 0:
      jm.regions.add CodeRegion(base: cast[pointer](rx), size: size, used: 0, wbase: int(w))
      return jm.install(code)
    jitDualMap = false
  let need = ((n + ps - 1) div ps) * ps
  var ri = -1
  if last >= 0 and jm.regions[last].wbase == 0:
    if jm.regions[last].size - jm.regions[last].used >= need: ri = last
  if ri < 0:
    let size = max(jm.regionSize, need)
    let base = jit_map(size)
    if base == nil: return nil
    jm.regions.add CodeRegion(base: base, size: size, used: 0, wbase: 0)
    ri = jm.regions.len - 1
  let dst = jit_offset(jm.regions[ri].base, jm.regions[ri].used)
  jit_copy(dst, addr code[0], n)
  if jit_protect(dst, need, 1) != 0: return nil
  jm.regions[ri].used = jm.regions[ri].used + need
  jm.totalCode = jm.totalCode + n
  dst

proc detachShared*(jm: var JitMemory) =
  ## In a forked child (agents.nim): dual-mapped regions are MAP_SHARED, so
  ## the parent and the child would both install code into the same free
  ## space. The child stops using every region it inherited (it still runs
  ## the code in them) and maps fresh ones of its own.
  for i in 0 ..< jm.regions.len: jm.regions[i].used = jm.regions[i].size

proc release*(jm: var JitMemory) =
  ## Unmaps every region; all code installed from `jm` becomes invalid.
  for r in jm.regions:
    discard jit_unmap(r.base, r.size)
    if r.wbase != 0: discard jit_unmap(cast[pointer](r.wbase), r.size)
  jm.regions = @[]
  jm.totalCode = 0
