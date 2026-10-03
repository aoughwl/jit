## Linear-scan register assignment for the optimizing tier (a separately
## imported module: the jsengine module is at nimony's size limit).
## Locations: >= 0 and < 100 a general register, >= 100 xmm (100 + n),
## < 0 a spill slot -(k+1).

proc loopDepths*(order: seq[int]; succs: seq[seq[int]]): seq[int] =
  ## Per block, its loop depth, approximated from the layout: a back edge
  ## s <- b spans the blocks laid out from s to b.
  result = newSeq[int](succs.len)
  var opos = newSeq[int](succs.len)
  for k in 0 ..< order.len: opos[order[k]] = k
  for b in order:
    for s in succs[b]:
      if opos[s] <= opos[b]:
        for k in opos[s] .. opos[b]: inc result[order[k]]

proc depthWeight*(d: int): float =
  result = 1.0
  for i in 0 ..< min(d, 6): result = result * 8.0

proc linearScan*(vals: seq[int]; start, stop: seq[int]; weight: seq[float];
                 wantX, crosses: seq[bool]; gprPool: seq[int]; xmmFirst: int;
                 loc: var seq[int]; nslots: var int; hint: seq[int] = @[];
                 ghint: seq[int] = @[]; calleeSavedFirst = false) =
  ## vals sorted by start. When no register is free, the value with the least
  ## weight per unit of lifetime lives in a slot (v itself when cheapest).
  var active: seq[int] = @[]
  var gprFree: seq[int] = @[]
  # (taken from the end: the pool's first registers first - the caller-saved
  # ones, which a region need not save - unless calleeSavedFirst: C helpers
  # called in the code keep those, and the caller-saved ones would be saved
  # around each)
  if calleeSavedFirst:
    for r in gprPool: gprFree.add r
  else:
    var ri = gprPool.len - 1
    while ri >= 0:
      gprFree.add gprPool[ri]
      dec ri
  var xmmFree: seq[int] = @[]
  for i in xmmFirst .. 15: xmmFree.add 100 + i
  for v in vals:
    var k = 0
    while k < active.len:
      let w = active[k]
      if stop[w] < start[v]:
        if loc[w] >= 100: xmmFree.add loc[w]
        elif loc[w] >= 0: gprFree.add loc[w]
        active.delete(k)
      else: inc k
    let wx = wantX[v]
    # a two-address op: take over the register of the operand that dies here
    var took = false
    if v < hint.len and hint[v] >= 0 and wx and not crosses[v]:
      let w = hint[v]
      let ka = active.find(w)
      if ka >= 0 and stop[w] == start[v] and loc[w] >= 100:
        loc[v] = loc[w]
        active.delete(ka)
        active.add v
        took = true
    if took: discard
    elif not crosses[v] and wx and xmmFree.len > 0:
      loc[v] = xmmFree.pop()
      active.add v
    elif not crosses[v] and not wx and gprFree.len > 0:
      # (a wanted register when free: a direct call's argument register)
      let gk = if v < ghint.len and ghint[v] >= 0: gprFree.find(ghint[v]) else: -1
      if gk >= 0:
        loc[v] = gprFree[gk]
        gprFree.delete(gk)
      else: loc[v] = gprFree.pop()
      active.add v
    else:
      var victim = -1
      if not crosses[v]:
        var best = weight[v] / float(stop[v] - start[v] + 2)
        for w in active:
          if loc[w] >= 0 and (loc[w] >= 100) == wx:
            let c = weight[w] / float(stop[w] - start[w] + 2)
            if c < best:
              victim = w
              best = c
      if victim >= 0:
        loc[v] = loc[victim]
        loc[victim] = -(nslots + 1)
        inc nslots
        active.delete(active.find(victim))
        active.add v
      else:
        loc[v] = -(nslots + 1)
        inc nslots
