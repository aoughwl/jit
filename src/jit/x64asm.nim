## An x86-64 assembler: instructions are encoded as bytes appended to a
## `seq[byte]`, with labels and rel32 fixups resolved by `finalize`.
##
## Modelled on Bali's amd64 assembler (src/bali/internal/assembler/amd64.nim,
## BSD-3-Clause, Copyright (c) 2024 Trayambak Rai), which itself derives from
## catnip's x64assembler (MIT, Copyright (c) 2021 RSDuck). Rewritten for
## nimony: no generics over operand kinds, no exceptions, no raw buffer - the
## output is plain position-independent bytes that jitmem.nim copies into
## executable memory. Absolute addresses (runtime helpers) are embedded with
## `mov r, imm64` + `call r`, so no relocation step is needed after copying.
##
## Conventions: two-operand instructions are `op(dst, src)` (Intel order).
## Memory operands are `Mem` values built with `mem(base, disp)` or
## `mem(base, index, scale, disp)`. All GPR operations are 64-bit unless the
## proc name says otherwise (`mov32`, `movzx8`, ...).

type
  Reg* = enum
    rax = 0, rcx, rdx, rbx, rsp, rbp, rsi, rdi, r8, r9, r10, r11, r12, r13, r14, r15

  Xmm* = enum
    xmm0 = 0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7,
    xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15

  Cond* = enum
    ## condition codes, in encoding order (the low nibble of Jcc/SETcc/CMOVcc)
    cOverflow = 0, cNoOverflow, cBelow, cAboveEq, cEqual, cNotEqual, cBelowEq,
    cAbove, cSign, cNoSign, cParity, cNoParity, cLess, cGreaterEq, cLessEq,
    cGreater

  Mem* = object
    ## [base + index*scale + disp]; index = -1 when there is none
    base*: Reg
    index*: int
    scale*: int
    disp*: int32

  Label* = distinct int

  Fixup = object
    pos: int        ## where the rel32 field starts
    label: int

  Assembler* = object
    buf*: seq[byte]      ## the code is its first `n` bytes; it is kept at full
    n: int               ## capacity (doubling) so emitting a byte never asks the
                         ## allocator for its size. `finalize` trims it to `n`.
    labelPos: seq[int]   ## bound position, or -1
    fixups: seq[Fixup]

  AluOp* = enum
    ## the /digit of the classic ALU group (and the opcode row)
    aluAdd = 0, aluOr, aluAdc, aluSbb, aluAnd, aluSub, aluXor, aluCmp

const
  # System V AMD64 calling convention (Linux, macOS)
  sysvArgRegs* = [rdi, rsi, rdx, rcx, r8, r9]
  sysvCalleeSaved* = [rbx, rbp, r12, r13, r14, r15]
  # Windows x64: four register args and 32 bytes of shadow space
  winArgRegs* = [rcx, rdx, r8, r9]

when defined(windows):
  const
    argRegs* = [rcx, rdx, r8, r9]
    shadowSpace* = 32
else:
  const
    argRegs* = [rdi, rsi, rdx, rcx, r8, r9]
    shadowSpace* = 0

proc `==`*(a, b: Label): bool {.borrow.}

proc invert*(c: Cond): Cond = Cond(ord(c) xor 1)

proc mem*(base: Reg; disp: int32 = 0): Mem =
  Mem(base: base, index: -1, scale: 1, disp: disp)

proc mem*(base: Reg; index: Reg; scale: int; disp: int32 = 0): Mem =
  ## `index` may not be rsp; `scale` is 1, 2, 4 or 8.
  Mem(base: base, index: ord(index), scale: scale, disp: disp)

proc initAssembler*(): Assembler =
  Assembler(buf: @[], n: 0, labelPos: @[], fixups: @[])

proc pos*(a: Assembler): int = a.n

# --- raw emission ----------------------------------------------------------------

proc growBuf(a: var Assembler; k: int) =
  var cap = max(256, a.buf.len * 2)
  while cap < a.n + k: cap = cap * 2
  # (not setLen: it zero-fills the new bytes one at a time, ~2% of wall on
  # JIT-heavy programs; every byte past `n` is written before it is read)
  var nb = newSeqUninit[byte](cap)
  if a.n > 0: copyMem(addr nb[0], addr a.buf[0], a.n)
  a.buf = ensureMove(nb)

template room(a: var Assembler; k: int) =
  if a.n + k > a.buf.len: growBuf(a, k)

# (bytes stored through the buffer's data pointer: `a.buf[i] = ..` is a call
# into another module per byte, a good part of every compile; x86-64 is
# little-endian, so the wider stores are plain unaligned stores)
type AsmRawSeq = object   # (seq[byte]'s layout: see fastutil.addCapped)
  len: int
  data: pointer

template bufAt(a: var Assembler; i: int): ptr byte =
  cast[ptr byte](cast[uint](cast[ptr AsmRawSeq](addr a.buf)[].data) + uint(i))

proc emit8*(a: var Assembler; b: int) {.inline.} =
  room(a, 1)
  bufAt(a, a.n)[] = byte(b and 0xFF)
  inc a.n

proc emit32*(a: var Assembler; v: int32) {.inline.} =
  room(a, 4)
  var u = cast[uint32](v)
  copyMem(bufAt(a, a.n), addr u, 4)
  a.n = a.n + 4

proc emit64*(a: var Assembler; v: uint64) {.inline.} =
  room(a, 8)
  var u = v
  copyMem(bufAt(a, a.n), addr u, 8)
  a.n = a.n + 8

proc patch32(a: var Assembler; at: int; v: int32) =
  var u = cast[uint32](v)
  copyMem(bufAt(a, at), addr u, 4)

proc fits8(v: int64): bool {.inline.} = v >= -128 and v <= 127
proc fits32(v: int64): bool {.inline.} = v >= -2147483648'i64 and v <= 2147483647'i64

proc rex(a: var Assembler; w: bool; r, x, b: int; force = false) =
  ## r/x/b are register numbers (only bit 3 matters).
  var v = 0x40
  if w: v = v or 8
  if (r and 8) != 0: v = v or 4
  if (x and 8) != 0: v = v or 2
  if (b and 8) != 0: v = v or 1
  if v != 0x40 or force: a.emit8 v

proc memRex(a: var Assembler; w: bool; reg: int; m: Mem; force = false) =
  a.rex(w, reg, (if m.index >= 0: m.index else: 0), ord(m.base), force)

proc modrmReg(a: var Assembler; reg, rm: int) =
  a.emit8 0xC0 or ((reg and 7) shl 3) or (rm and 7)

proc modrmMem(a: var Assembler; reg: int; m: Mem) =
  ## ModRM (+SIB) (+disp) for [base + index*scale + disp]. rsp/r12 as base
  ## need a SIB byte; rbp/r13 as base cannot use the no-displacement form.
  let b = ord(m.base) and 7
  var md = 2
  if m.disp == 0 and b != 5: md = 0
  elif fits8(m.disp): md = 1
  let needSib = m.index >= 0 or b == 4
  if needSib:
    a.emit8 (md shl 6) or ((reg and 7) shl 3) or 4
    var ss = 0
    case m.scale
    of 2: ss = 1
    of 4: ss = 2
    of 8: ss = 3
    else: ss = 0
    let idx = if m.index >= 0: m.index and 7 else: 4
    a.emit8 (ss shl 6) or (idx shl 3) or b
  else:
    a.emit8 (md shl 6) or ((reg and 7) shl 3) or b
  if md == 1: a.emit8 int(m.disp)
  elif md == 2: a.emit32 m.disp

# register-register / register-memory forms, 64-bit (w) or 32-bit
proc opRR(a: var Assembler; w: bool; op: openArray[int]; reg, rm: int) =
  a.rex(w, reg, 0, rm)
  for o in op: a.emit8 o
  a.modrmReg(reg, rm)

proc opRM(a: var Assembler; w: bool; op: openArray[int]; reg: int; m: Mem) =
  a.memRex(w, reg, m)
  for o in op: a.emit8 o
  a.modrmMem(reg, m)

# SSE: mandatory prefix, then REX, then 0F xx
proc sseRR(a: var Assembler; prefix: int; w: bool; op: int; reg, rm: int) =
  if prefix != 0: a.emit8 prefix
  a.rex(w, reg, 0, rm)
  a.emit8 0x0F
  a.emit8 op
  a.modrmReg(reg, rm)

proc sseRM(a: var Assembler; prefix: int; w: bool; op: int; reg: int; m: Mem) =
  if prefix != 0: a.emit8 prefix
  a.memRex(w, reg, m)
  a.emit8 0x0F
  a.emit8 op
  a.modrmMem(reg, m)

# --- labels ----------------------------------------------------------------------

proc newLabel*(a: var Assembler): Label =
  a.labelPos.add -1
  Label(a.labelPos.len - 1)

proc bindLabel*(a: var Assembler; l: Label) =
  ## Places `l` at the current position.
  a.labelPos[int(l)] = a.n

proc hereLabel*(a: var Assembler): Label =
  ## A new label bound at the current position (a backward-branch target).
  result = a.newLabel()
  a.bindLabel(result)

proc isBound*(a: Assembler; l: Label): bool = a.labelPos[int(l)] >= 0
proc labelOffset*(a: Assembler; l: Label): int = a.labelPos[int(l)]

proc rel32To(a: var Assembler; l: Label) =
  ## Emits a rel32 field (relative to its own end) aimed at `l`.
  let p = a.labelPos[int(l)]
  if p >= 0:
    a.emit32 int32(p - (a.n + 4))
  else:
    a.fixups.add Fixup(pos: a.n, label: int(l))
    a.emit32 0

proc pendingRefs*(a: Assembler; nLabels: int): seq[bool] =
  ## Per label (below nLabels): some jump emitted so far refers to it and it
  ## is not bound yet (a forward reference still to resolve).
  result = newSeq[bool](nLabels)
  for f in a.fixups:
    if f.label >= 0 and f.label < nLabels and a.labelPos[f.label] < 0: result[f.label] = true

proc finalize*(a: var Assembler): bool =
  ## Resolves forward references. false if some referenced label was never
  ## bound. The assembler may keep being used afterwards (fixups are kept
  ## only while unresolved).
  var rest: seq[Fixup] = @[]
  result = true
  for f in a.fixups:
    let p = a.labelPos[f.label]
    if p < 0:
      rest.add f
      result = false
    else:
      a.patch32(f.pos, int32(p - (f.pos + 4)))
  a.fixups = rest
  a.buf.shrink(a.n)

# --- data movement -----------------------------------------------------------------

proc mov*(a: var Assembler; dst, src: Reg) =
  a.opRR(true, [0x89], ord(src), ord(dst))

proc mov32*(a: var Assembler; dst, src: Reg) =
  ## 32-bit move (zero-extends into the full register)
  a.opRR(false, [0x89], ord(src), ord(dst))

proc movImm*(a: var Assembler; dst: Reg; imm: int64) =
  ## Shortest encoding: xor for 0, mov r32 for values that zero-extend,
  ## sign-extended imm32, else the full movabs imm64.
  ## NOTE: the xor form clobbers flags.
  if imm == 0:
    a.opRR(false, [0x31], ord(dst), ord(dst))
  elif imm > 0 and imm <= 0xFFFFFFFF'i64:
    a.rex(false, 0, 0, ord(dst))
    a.emit8 0xB8 + (ord(dst) and 7)
    a.emit32 cast[int32](uint32(imm))
  elif fits32(imm):
    a.rex(true, 0, 0, ord(dst))
    a.emit8 0xC7
    a.modrmReg(0, ord(dst))
    a.emit32 int32(imm)
  else:
    a.rex(true, 0, 0, ord(dst))
    a.emit8 0xB8 + (ord(dst) and 7)
    a.emit64 cast[uint64](imm)

proc movImm64*(a: var Assembler; dst: Reg; imm: uint64) =
  ## Always the 10-byte movabs form (patchable; flags untouched).
  a.rex(true, 0, 0, ord(dst))
  a.emit8 0xB8 + (ord(dst) and 7)
  a.emit64 imm

proc movPtr*(a: var Assembler; dst: Reg; p: pointer) =
  a.movImm64(dst, cast[uint64](p))

proc load*(a: var Assembler; dst: Reg; m: Mem) =
  ## mov dst, qword [m]
  a.opRM(true, [0x8B], ord(dst), m)

proc store*(a: var Assembler; m: Mem; src: Reg) =
  ## mov qword [m], src
  a.opRM(true, [0x89], ord(src), m)

proc load32*(a: var Assembler; dst: Reg; m: Mem) =
  ## mov dst32, dword [m] (zero-extends)
  a.opRM(false, [0x8B], ord(dst), m)

proc store32*(a: var Assembler; m: Mem; src: Reg) =
  a.opRM(false, [0x89], ord(src), m)

proc store8*(a: var Assembler; m: Mem; src: Reg) =
  ## mov byte [m], src8 (REX forced so spl/bpl/sil/dil are addressable)
  a.memRex(false, ord(src), m, force = ord(src) >= 4)
  a.emit8 0x88
  a.modrmMem(ord(src), m)

proc storeImm*(a: var Assembler; m: Mem; imm: int32) =
  ## mov qword [m], sign-extended imm32
  a.opRM(true, [0xC7], 0, m)
  a.emit32 imm

proc storeImm32*(a: var Assembler; m: Mem; imm: int32) =
  ## mov dword [m], imm32
  a.opRM(false, [0xC7], 0, m)
  a.emit32 imm

proc store16*(a: var Assembler; m: Mem; src: Reg) =
  ## mov word [m], src16
  a.emit8 0x66
  a.opRM(false, [0x89], ord(src), m)

proc storeImm16*(a: var Assembler; m: Mem; imm: int32) =
  ## mov word [m], imm16
  a.emit8 0x66
  a.opRM(false, [0xC7], 0, m)
  a.emit8 int(imm and 0xFF)
  a.emit8 int((imm shr 8) and 0xFF)

proc loadS32*(a: var Assembler; dst: Reg; m: Mem) =
  ## movsxd dst, dword [m]
  a.opRM(true, [0x63], ord(dst), m)

proc movzx8*(a: var Assembler; dst: Reg; m: Mem) =
  ## movzx dst32, byte [m]
  a.opRM(false, [0x0F, 0xB6], ord(dst), m)

proc movzx16*(a: var Assembler; dst: Reg; m: Mem) =
  a.opRM(false, [0x0F, 0xB7], ord(dst), m)

proc movzx8*(a: var Assembler; dst, src: Reg) =
  ## movzx dst32, src8 (REX forced so sil/dil/spl/bpl are meant)
  a.rex(false, ord(dst), 0, ord(src), force = ord(src) >= 4)
  a.emit8 0x0F
  a.emit8 0xB6
  a.modrmReg(ord(dst), ord(src))

proc movsx8*(a: var Assembler; dst: Reg; m: Mem) =
  a.opRM(true, [0x0F, 0xBE], ord(dst), m)

proc lea*(a: var Assembler; dst: Reg; m: Mem) =
  a.opRM(true, [0x8D], ord(dst), m)

proc leaLabel*(a: var Assembler; dst: Reg; l: Label) =
  ## lea dst, [rip + label] - the address of a label in the final code
  a.rex(true, ord(dst), 0, 0)
  a.emit8 0x8D
  a.emit8 ((ord(dst) and 7) shl 3) or 5
  a.rel32To(l)

proc xchg*(a: var Assembler; x, y: Reg) =
  a.opRR(true, [0x87], ord(y), ord(x))

proc push*(a: var Assembler; r: Reg) =
  a.rex(false, 0, 0, ord(r))
  a.emit8 0x50 + (ord(r) and 7)

proc pop*(a: var Assembler; r: Reg) =
  a.rex(false, 0, 0, ord(r))
  a.emit8 0x58 + (ord(r) and 7)

proc pushImm*(a: var Assembler; imm: int32) =
  a.emit8 0x68
  a.emit32 imm

# --- arithmetic ----------------------------------------------------------------------

proc alu*(a: var Assembler; op: AluOp; dst, src: Reg) =
  a.opRR(true, [ord(op) * 8 + 1], ord(src), ord(dst))

proc alu32*(a: var Assembler; op: AluOp; dst, src: Reg) =
  a.opRR(false, [ord(op) * 8 + 1], ord(src), ord(dst))

proc aluImm*(a: var Assembler; op: AluOp; dst: Reg; imm: int32) =
  if fits8(imm):
    a.opRR(true, [0x83], ord(op), ord(dst))
    a.emit8 int(imm)
  else:
    a.opRR(true, [0x81], ord(op), ord(dst))
    a.emit32 imm

proc aluMem*(a: var Assembler; op: AluOp; dst: Reg; m: Mem) =
  ## dst op= qword [m]
  a.opRM(true, [ord(op) * 8 + 3], ord(dst), m)

proc aluToMem*(a: var Assembler; op: AluOp; m: Mem; src: Reg) =
  ## qword [m] op= src
  a.opRM(true, [ord(op) * 8 + 1], ord(src), m)

proc aluMemImm*(a: var Assembler; op: AluOp; m: Mem; imm: int32) =
  if fits8(imm):
    a.opRM(true, [0x83], ord(op), m)
    a.emit8 int(imm)
  else:
    a.opRM(true, [0x81], ord(op), m)
    a.emit32 imm

proc add*(a: var Assembler; dst, src: Reg) = a.alu(aluAdd, dst, src)
proc sub*(a: var Assembler; dst, src: Reg) = a.alu(aluSub, dst, src)
proc andr*(a: var Assembler; dst, src: Reg) = a.alu(aluAnd, dst, src)
proc orr*(a: var Assembler; dst, src: Reg) = a.alu(aluOr, dst, src)
proc xorr*(a: var Assembler; dst, src: Reg) = a.alu(aluXor, dst, src)
proc cmp*(a: var Assembler; x, y: Reg) = a.alu(aluCmp, x, y)
proc addImm*(a: var Assembler; dst: Reg; imm: int32) = a.aluImm(aluAdd, dst, imm)
proc subImm*(a: var Assembler; dst: Reg; imm: int32) = a.aluImm(aluSub, dst, imm)
proc andImm*(a: var Assembler; dst: Reg; imm: int32) = a.aluImm(aluAnd, dst, imm)
proc orImm*(a: var Assembler; dst: Reg; imm: int32) = a.aluImm(aluOr, dst, imm)
proc xorImm*(a: var Assembler; dst: Reg; imm: int32) = a.aluImm(aluXor, dst, imm)
proc cmpImm*(a: var Assembler; x: Reg; imm: int32) = a.aluImm(aluCmp, x, imm)
proc cmpMem*(a: var Assembler; x: Reg; m: Mem) = a.aluMem(aluCmp, x, m)
proc cmpMem32*(a: var Assembler; x: Reg; m: Mem) =
  ## cmp x32, dword [m]
  a.opRM(false, [ord(aluCmp) * 8 + 3], ord(x), m)

proc test*(a: var Assembler; x, y: Reg) =
  a.opRR(true, [0x85], ord(y), ord(x))

proc testImm*(a: var Assembler; x: Reg; imm: int32) =
  a.opRR(true, [0xF7], 0, ord(x))
  a.emit32 imm

proc imul*(a: var Assembler; dst, src: Reg) =
  a.opRR(true, [0x0F, 0xAF], ord(dst), ord(src))

proc imul32*(a: var Assembler; dst, src: Reg) =
  ## dst32 *= src32 (OF on signed overflow)
  a.opRR(false, [0x0F, 0xAF], ord(dst), ord(src))

proc imulMem*(a: var Assembler; dst: Reg; m: Mem) =
  a.opRM(true, [0x0F, 0xAF], ord(dst), m)

proc imulImm*(a: var Assembler; dst, src: Reg; imm: int32) =
  ## dst = src * imm
  if fits8(imm):
    a.opRR(true, [0x6B], ord(dst), ord(src))
    a.emit8 int(imm)
  else:
    a.opRR(true, [0x69], ord(dst), ord(src))
    a.emit32 imm

proc neg*(a: var Assembler; r: Reg) = a.opRR(true, [0xF7], 3, ord(r))
proc notr*(a: var Assembler; r: Reg) = a.opRR(true, [0xF7], 2, ord(r))
proc incr*(a: var Assembler; r: Reg) = a.opRR(true, [0xFF], 0, ord(r))
proc decr*(a: var Assembler; r: Reg) = a.opRR(true, [0xFF], 1, ord(r))

proc cqo*(a: var Assembler) =
  ## sign-extend rax into rdx:rax (before idiv)
  a.emit8 0x48
  a.emit8 0x99

proc idiv*(a: var Assembler; r: Reg) =
  ## signed rdx:rax / r -> quotient rax, remainder rdx
  a.opRR(true, [0xF7], 7, ord(r))

proc divu*(a: var Assembler; r: Reg) = a.opRR(true, [0xF7], 6, ord(r))

proc div32*(a: var Assembler; r: Reg) =
  ## unsigned edx:eax / r32 -> quotient eax, remainder edx
  a.opRR(false, [0xF7], 6, ord(r))

type ShiftOp* = enum
  shRol = 0, shRor = 1, shShl = 4, shShr = 5, shSar = 7

proc shiftImm*(a: var Assembler; op: ShiftOp; r: Reg; count: int) =
  a.opRR(true, [0xC1], ord(op), ord(r))
  a.emit8 count and 63

proc shiftImm32*(a: var Assembler; op: ShiftOp; r: Reg; count: int) =
  ## A 32-bit shift / rotate by a constant (the upper half becomes zero).
  a.opRR(false, [0xC1], ord(op), ord(r))
  a.emit8 count and 31

proc shiftCl*(a: var Assembler; op: ShiftOp; r: Reg) =
  ## shift by cl
  a.opRR(true, [0xD3], ord(op), ord(r))

proc shlImm*(a: var Assembler; r: Reg; count: int) = a.shiftImm(shShl, r, count)
proc shrImm*(a: var Assembler; r: Reg; count: int) = a.shiftImm(shShr, r, count)
proc sarImm*(a: var Assembler; r: Reg; count: int) = a.shiftImm(shSar, r, count)
proc shlCl*(a: var Assembler; r: Reg) = a.shiftCl(shShl, r)
proc shrCl*(a: var Assembler; r: Reg) = a.shiftCl(shShr, r)
proc sarCl*(a: var Assembler; r: Reg) = a.shiftCl(shSar, r)

proc shlImm32*(a: var Assembler; r: Reg; count: int) =
  a.opRR(false, [0xC1], 4, ord(r))
  a.emit8 count and 31

proc sarImm32*(a: var Assembler; r: Reg; count: int) =
  a.opRR(false, [0xC1], 7, ord(r))
  a.emit8 count and 31

# --- flags consumers ----------------------------------------------------------------

proc setcc*(a: var Assembler; c: Cond; dst: Reg) =
  ## dst8 = c ? 1 : 0 (upper bits unchanged; follow with movzx8)
  a.rex(false, 0, 0, ord(dst), force = ord(dst) >= 4)
  a.emit8 0x0F
  a.emit8 0x90 + ord(c)
  a.modrmReg(0, ord(dst))

proc cmov*(a: var Assembler; c: Cond; dst, src: Reg) =
  a.opRR(true, [0x0F, 0x40 + ord(c)], ord(dst), ord(src))

# --- control flow --------------------------------------------------------------------

proc jmp*(a: var Assembler; l: Label) =
  ## jmp rel32 (rel8 when the target is bound and close)
  let p = a.labelPos[int(l)]
  if p >= 0 and fits8(p - (a.n + 2)):
    a.emit8 0xEB
    a.emit8 p - (a.n + 1)
    return
  a.emit8 0xE9
  a.rel32To(l)

proc jcc*(a: var Assembler; c: Cond; l: Label) =
  let p = a.labelPos[int(l)]
  if p >= 0 and fits8(p - (a.n + 2)):
    a.emit8 0x70 + ord(c)
    a.emit8 p - (a.n + 1)
    return
  a.emit8 0x0F
  a.emit8 0x80 + ord(c)
  a.rel32To(l)

proc jmpReg*(a: var Assembler; r: Reg) = a.opRR(false, [0xFF], 4, ord(r))
proc jmpMem*(a: var Assembler; m: Mem) = a.opRM(false, [0xFF], 4, m)

proc call*(a: var Assembler; l: Label) =
  ## call rel32 to a label inside this code
  a.emit8 0xE8
  a.rel32To(l)

proc callReg*(a: var Assembler; r: Reg) = a.opRR(false, [0xFF], 2, ord(r))
proc callMem*(a: var Assembler; m: Mem) = a.opRM(false, [0xFF], 2, m)

proc callAbs*(a: var Assembler; target: pointer; scratch: Reg = r11) =
  ## call an absolute address (a runtime helper): movabs scratch, target;
  ## call scratch. r11 is caller-saved and never an argument register.
  a.movPtr(scratch, target)
  a.callReg(scratch)

proc ret*(a: var Assembler) = a.emit8 0xC3
proc int3*(a: var Assembler) = a.emit8 0xCC
proc nop*(a: var Assembler) = a.emit8 0x90
proc ud2*(a: var Assembler) =
  a.emit8 0x0F
  a.emit8 0x0B

proc align*(a: var Assembler; n: int) =
  ## pad with nops to a multiple of n (a power of two)
  while (a.n and (n - 1)) != 0: a.nop()

# --- SSE2 scalar doubles -------------------------------------------------------------

proc movsd*(a: var Assembler; dst, src: Xmm) = a.sseRR(0xF2, false, 0x10, ord(dst), ord(src))
proc movsd*(a: var Assembler; dst: Xmm; m: Mem) = a.sseRM(0xF2, false, 0x10, ord(dst), m)
proc movsd*(a: var Assembler; m: Mem; src: Xmm) = a.sseRM(0xF2, false, 0x11, ord(src), m)

proc movq*(a: var Assembler; dst: Xmm; src: Reg) =
  ## movq xmm, r64 (bit copy)
  a.sseRR(0x66, true, 0x6E, ord(dst), ord(src))

proc movq*(a: var Assembler; dst: Reg; src: Xmm) =
  ## movq r64, xmm (bit copy)
  a.sseRR(0x66, true, 0x7E, ord(src), ord(dst))

proc addsd*(a: var Assembler; dst, src: Xmm) = a.sseRR(0xF2, false, 0x58, ord(dst), ord(src))
proc mulsd*(a: var Assembler; dst, src: Xmm) = a.sseRR(0xF2, false, 0x59, ord(dst), ord(src))
proc subsd*(a: var Assembler; dst, src: Xmm) = a.sseRR(0xF2, false, 0x5C, ord(dst), ord(src))
proc divsd*(a: var Assembler; dst, src: Xmm) = a.sseRR(0xF2, false, 0x5E, ord(dst), ord(src))
proc minsd*(a: var Assembler; dst, src: Xmm) = a.sseRR(0xF2, false, 0x5D, ord(dst), ord(src))
proc maxsd*(a: var Assembler; dst, src: Xmm) = a.sseRR(0xF2, false, 0x5F, ord(dst), ord(src))
proc sqrtsd*(a: var Assembler; dst, src: Xmm) = a.sseRR(0xF2, false, 0x51, ord(dst), ord(src))
proc addsd*(a: var Assembler; dst: Xmm; m: Mem) = a.sseRM(0xF2, false, 0x58, ord(dst), m)
proc mulsd*(a: var Assembler; dst: Xmm; m: Mem) = a.sseRM(0xF2, false, 0x59, ord(dst), m)
proc subsd*(a: var Assembler; dst: Xmm; m: Mem) = a.sseRM(0xF2, false, 0x5C, ord(dst), m)
proc divsd*(a: var Assembler; dst: Xmm; m: Mem) = a.sseRM(0xF2, false, 0x5E, ord(dst), m)

proc ucomisd*(a: var Assembler; x, y: Xmm) =
  ## compare doubles: ZF,PF,CF = unordered 111, less 001, equal 100, greater 000.
  ## Use cBelow/cAbove/cBelowEq/cAboveEq/cEqual and check cParity for NaN.
  a.sseRR(0x66, false, 0x2E, ord(x), ord(y))

proc ucomisd*(a: var Assembler; x: Xmm; m: Mem) = a.sseRM(0x66, false, 0x2E, ord(x), m)

proc cvtsi2sd*(a: var Assembler; dst: Xmm; src: Reg) =
  ## int64 -> double
  a.sseRR(0xF2, true, 0x2A, ord(dst), ord(src))

proc cvtsi2sd32*(a: var Assembler; dst: Xmm; src: Reg) =
  ## int32 -> double
  a.sseRR(0xF2, false, 0x2A, ord(dst), ord(src))

proc cvttsd2si*(a: var Assembler; dst: Reg; src: Xmm) =
  ## double -> int64, truncating (0x8000000000000000 when out of range/NaN)
  a.sseRR(0xF2, true, 0x2C, ord(dst), ord(src))

proc cvttsd2si32*(a: var Assembler; dst: Reg; src: Xmm) =
  ## double -> int32 in dst32 (0x80000000 when out of range/NaN)
  a.sseRR(0xF2, false, 0x2C, ord(dst), ord(src))

proc xorpd*(a: var Assembler; dst, src: Xmm) = a.sseRR(0x66, false, 0x57, ord(dst), ord(src))
proc andpd*(a: var Assembler; dst, src: Xmm) = a.sseRR(0x66, false, 0x54, ord(dst), ord(src))

# --- frames ----------------------------------------------------------------------------

proc prologue*(a: var Assembler; saved: openArray[Reg]; locals: int): int =
  ## push rbp; mov rbp, rsp; push each of `saved`; sub rsp to reserve
  ## `locals` bytes rounded so that rsp is 16-byte aligned afterwards (so
  ## a following `call` meets the ABI). Returns the byte count subtracted;
  ## pass it to `epilogue`. Locals live at [rbp - 8*(saved.len+1) - ...].
  a.push(rbp)
  a.mov(rbp, rsp)
  for r in saved: a.push(r)
  # after push rbp the stack is 16-aligned; each saved push adds 8
  var extra = ((locals + 7) div 8) * 8 + shadowSpace
  if ((saved.len * 8 + extra) and 15) != 0: extra += 8
  if extra > 0: a.subImm(rsp, int32(extra))
  extra

proc epilogue*(a: var Assembler; saved: openArray[Reg]; frameBytes: int) =
  ## undoes `prologue` and returns
  if frameBytes > 0: a.addImm(rsp, int32(frameBytes))
  var i = saved.len - 1
  while i >= 0:
    a.pop(saved[i])
    dec i
  a.pop(rbp)
  a.ret()
