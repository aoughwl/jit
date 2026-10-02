## Run: nimony c -p:src tests/test_jit.nim  (then the binary). x86-64 only.
import std/syncio
import aowljit

var failures = 0
var total = 0

proc check(name: string; got, want: int64) =
  inc total
  if got != want:
    echo "FAIL ", name, ": got ", got, " want ", want
    inc failures

var jm = initJitMemory()

# int64 add3(a, b, c) = a + b + c
block:
  var a = initAssembler()
  a.mov(rax, argRegs[0])
  a.add(rax, argRegs[1])
  a.add(rax, argRegs[2])
  a.ret()
  check "finalize add3", int64(ord(a.finalize())), 1
  let f = jm.install(a.buf)
  check "install add3", int64(ord(f != nil)), 1
  check "add3(1,2,3)", jitCall3(f, 1, 2, 3), 6
  check "add3(-10,4,100000000000)", jitCall3(f, -10, 4, 100000000000'i64), 99999999994'i64

# int64 sumTo(n) = 1 + 2 + ... + n   (backward branch, forward exit label)
block:
  var a = initAssembler()
  let done = a.newLabel()
  a.movImm(rax, 0)
  a.mov(rcx, argRegs[0])
  a.test(rcx, rcx)
  a.jcc(cLessEq, done)
  let top = a.hereLabel()
  a.add(rax, rcx)
  a.decr(rcx)
  a.jcc(cNotEqual, top)
  a.bindLabel(done)
  a.ret()
  check "finalize sumTo", int64(ord(a.finalize())), 1
  let f = jm.install(a.buf)
  check "sumTo(10)", jitCall1(f, 10), 55
  check "sumTo(100000)", jitCall1(f, 100000), 5000050000'i64
  check "sumTo(0)", jitCall1(f, 0), 0
  check "sumTo(-5)", jitCall1(f, -5), 0
  check "code bytes counted", int64(ord(jm.totalCode > 0)), 1

# an unbound label makes finalize fail
block:
  var a = initAssembler()
  let l = a.newLabel()
  a.jmp(l)
  check "unbound label", int64(ord(a.finalize())), 0

jm.release()

# linear scan: 3 overlapping values, 2 registers -> one spill (the lightest)
block:
  var loc = newSeq[int](3)
  var nslots = 0
  linearScan(@[0, 1, 2], @[0, 1, 2], @[10, 10, 10], @[1.0, 100.0, 50.0],
             @[false, false, false], @[false, false, false], @[3, 6], 8, loc, nslots)
  check "loc.len", int64(loc.len), 3
  check "v0 spilled", int64(ord(loc[0] < 0)), 1
  check "v1 in reg", int64(ord(loc[1] == 3 or loc[1] == 6)), 1
  check "v2 in reg", int64(ord(loc[2] == 3 or loc[2] == 6)), 1
  check "distinct regs", int64(ord(loc[1] != loc[2])), 1
  check "one slot", int64(nslots), 1

# linear scan: disjoint lifetimes share one register; float wants xmm
block:
  var loc = newSeq[int](3)
  var nslots = 0
  linearScan(@[0, 2, 1], @[0, 5, 0], @[4, 9, 9], @[1.0, 1.0, 1.0],
             @[false, false, true], @[false, false, false], @[3], 14, loc, nslots)
  check "reuse reg v0", int64(loc[0]), 3
  check "reuse reg v1", int64(loc[1]), 3
  check "xmm v2", int64(ord(loc[2] >= 114)), 1
  check "no slots", int64(nslots), 0

# loop depths: blocks 0 -> 1 -> 2 -> 1 (back edge), 2 -> 3
block:
  let d = loopDepths(@[0, 1, 2, 3], @[@[1], @[2], @[1, 3], @[]])
  check "depth b0", int64(d[0]), 0
  check "depth b1", int64(d[1]), 1
  check "depth b2", int64(d[2]), 1
  check "depth b3", int64(d[3]), 0

echo total - failures, "/", total, " passed"
if failures > 0: quit(1)
