# aowljit

The x86-64 JIT backend of the [aowljs](https://github.com/aoughwl/aowljs-engine) JavaScript
engine as a standalone [nimony](https://github.com/nim-lang/nimony) library, with no dependency
on any JS value model: an assembler that emits position-independent bytes, W^X executable
memory with fixed-signature calls into generated code, and linear-scan register assignment.

| module | |
|---|---|
| `aowljit/x64asm` | x86-64 assembler: `Assembler`, `Reg`, `Xmm`, `Cond`, `Mem`, `Label`; GPR, SSE2 and control-flow instructions appended to a `seq[byte]`, rel32 label fixups resolved by `finalize` |
| `aowljit/jitmem` | `JitMemory`: copy code into fresh pages, flip them read-execute (never W+X); `jitCall0..4`, `jitCallF1/F2` to call it, `procAddr` / `loadFnPtr` / `ptrToInt` for runtime helpers |
| `aowljit/linscan` | `linearScan` (GPR + xmm pools, spill slots, weight-per-lifetime eviction, two-address hints), `loopDepths`, `depthWeight` |
| `aowljit` | re-exports all three |

Generated code is an ordinary C-ABI function (SysV on Linux, Win64 on Windows); `argRegs`
and `shadowSpace` pick the right convention at compile time. Absolute addresses are embedded
with `mov r, imm64` + `call r`, so code needs no relocation after `install`.

## Usage

```nim
import aowljit

var a = initAssembler()
let done = a.newLabel()
a.movImm(rax, 0)              # sumTo(n) = 1 + ... + n
a.mov(rcx, argRegs[0])
a.test(rcx, rcx)
a.jcc(cLessEq, done)
let top = a.hereLabel()
a.add(rax, rcx)
a.decr(rcx)
a.jcc(cNotEqual, top)
a.bindLabel(done)
a.ret()
doAssert a.finalize()          # false if a referenced label was never bound

var jm = initJitMemory()
let f = jm.install(a.buf)      # nil if pages could not be mapped/protected
echo jitCall1(f, 10)           # 55
jm.release()                   # unmaps everything installed from jm
```

Register locations from `linearScan`: `0..15` a GPR, `100 + n` xmm n, `-(k+1)` spill slot k.
`loc` must be presized to the value count and `vals` sorted by start.

## Users

- **aowljs-engine**: the baseline JIT and the optimizing tier (`opt.nim`) are built on these
  three modules. The engine's build finds this checkout via `AOWL_JIT` (default `../jit`),
  like `AOWL_REGEX` / `AOWL_UNICODE`, and imports `aowljit/x64asm` etc.
- **aowli** (the nimony interpreter/runtime) has no assembler or executable-memory code of its
  own; its native FFI (`hostdyn.nim`) classifies SysV argument registers but calls through C.
  The API here is engine-neutral, so aowli can adopt it as-is if it grows a JIT.

## Test

```sh
nimony c -p:src tests/test_jit.nim   # then run the binary (x86-64 only)
```

## License

MIT. The assembler is modelled on Bali's amd64 assembler (BSD-3-Clause, Trayambak Rai), itself
derived from catnip's x64assembler (MIT, RSDuck); see the header of `src/aowljit/x64asm.nim`.
