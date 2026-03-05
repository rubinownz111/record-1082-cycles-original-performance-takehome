# Optimization Journal and Agent Instructions

## Objective
- Reduce cycle count below current baseline while preserving correctness on `python3 tests/submission_tests.py`.

## Hard Rules For Agents Working In This Repo
- Do not modify files under `tests/`.
- Every time you find something new and valuable, update this file immediately.
- Update this file again after each attempted path, whether it works or fails.
- Keep a chronological attempt log with concrete measurement numbers.
- "Something new and valuable" includes:
- A working improvement.
- A dead end that looked promising but regressed or broke correctness.
- A new bottleneck discovered from telemetry or measurement.
- A new architectural idea worth trying next.
- For each logged attempt include:
- What was changed/tested.
- Why it was attempted.
- Result (`cycles`, correctness pass/fail).
- Decision (`keep`, `revert`, `follow-up`).

## Current Baseline
- Command: `python3 tests/submission_tests.py`
- Result: `1065` cycles, correctness passing.
- Kernel stats: `12375 ops`, peak scratch `1528/1536`.
- Note: the currently checked-out worktree on `2026-03-05` re-validates at this `1065` baseline. Older `1078-1082` entries below are useful historical context from earlier architecture exploration, but they do not describe the present file state.

## Findings So Far
- Tail-only shallow stage-5 fusion is not a safe extension of the wrap-root idea:
- even though it drives `valu` much lower (`5922` tail-only, `5881` wrap+tail), both variants fail correctness.
- Implication: after round `11`, shallow rounds still depend on bias state in a way that is not handled by the simple “consecutive full-bias prefix” model; wrap-root remains the only currently safe narrow stage-5 fusion.
- A narrow overflow-to-root stage-5 fusion path is now the strongest architectural lead from this session:
- `ENABLE_WRAP_ROOT_STAGE5_FUSION=True` skips the stage-5 const only on round `10` (overflow), then repairs it in round `11`'s root XOR with no branch/index bias tracking.
- It is correctness-valid and reduces real slot pressure to `valu=5982` and `alu=12099` (`12345 ops`), but still measures `1067` cycles.
- Combining it with in-place scalar scatter-to-values removes even more graph scaffolding (`12101 ops`, `alu=12083`, `valu=5984`) but still remains at `1067`.
- Implication: this targeted fusion is directionally right, but it still needs a second architectural change that converts the better slot profile into actual overlap/cycle improvement.
- Deep scatter result storage can now be rewritten in place without breaking correctness:
- A new lane-precise partial-vector writer model allows scalar scatter XORs to write directly back into the live `values` vector.
- This removes `244` emitted ops from the `1065` kernel (`12375 -> 12131`) with correctness preserved, but cycle count and slot totals stay unchanged (`1065`, `load=2001`, `valu=6010`, `alu=12122`).
- Implication: a large fraction of remaining emitted ops are non-slot-bearing allocation/dataflow scaffolding; removing them alone is not enough unless the rewrite also changes engine-slot pressure or critical-path overlap.
- Current checked-out `1065` baseline re-measurement on `2026-03-05` confirms the same three hard bottlenecks:
- Final emitted bundle slot totals: `alu=12122`, `valu=6010`, `load=2001`, `flow=785`, `store=32`.
- Scheduler telemetry on this exact state still shows load pressure first:
- `defer_load_full: 30906`
- `defer_flow_full: 8090`
- `defer_valu_full: 6105`
- `defer_valu_full_offload_failed: 4184`
- Implication: reducing scratch lifetime alone is unlikely to help unless it also removes real `load` or `valu` work from the critical rounds.
- Engine pressure is dominated by `valu` and `load`:
- Approx utilization: `valu ~93.9%`, `load ~100%`, `alu ~101.1%`, `flow ~78.5%` (slot totals vs hard bounds).
- Current 1065-case scheduled slot totals / hard lower bounds:
- `valu: 6010` slots -> LB `1002` cycles
- `load: 2001` slots -> LB `1001` cycles
- `alu: 12130` slots -> LB `1011` cycles
- `flow: 785` slots -> LB `785` cycles
- Implication: sub-1000 now requires real reductions in at least three engines:
- `load` (`<=2000`), `alu` (`<=12000`), and `valu` (`<=6000`).
- Important correction on ALU accounting:
- Scheduler telemetry (`slot_alu`) undercounts ALU slots because retroactive ALU offload injects ALU ops into prior bundles, but telemetry records only the current cycle's `slot_counts`.
- Counting final emitted bundles (`kb.instrs`) gives true ALU total for current config:
- `alu: 12130` slots -> LB `1011` cycles
- `valu: 6010` slots -> LB `1002` cycles
- `load: 2001` slots -> LB `1001` cycles
- `flow: 785` slots -> LB `785` cycles
- Updated implication: we are back in a tri-bottleneck regime (`alu` + `load` + `valu`) despite lower flow pressure.
- External reference point:
- Community leaderboard (`kerneloptimization.fun`) currently shows a best of `1001` cycles for this challenge variant, which is consistent with our observation that current architecture families are converging near the low 1000s.
- Top defer reasons from scheduler telemetry:
- `defer_load_full: 31719`
- `defer_valu_full: 10534`
- `defer_flow_full: 4323`
- `defer_valu_full_offload_failed: 2917`
- Round-window token release cannot simply move to tree completion on the current `1065` branch:
- making rounds `7/9/10` release on `tree_values` instead of full round-tail completion lowers peak scratch materially (`1528 -> 1448`) and trims emitted ALU slots (`12122 -> 12066`), but it still regresses to `1072` cycles with `valu` slightly higher (`6017`).
- Implication: the current round gates are not just smoothing deep-load overlap; they are also protecting later hash/index overlap enough that earlier release loses more VALU critical-path slack than it gains from lower scratch/ALU pressure.
- A narrow setup rewrite can now hit the load lower-bound threshold without helping cycles:
- deriving `inp_values_ptr` from the loaded forest pointer removes one real load (`2001 -> 2000`) and, when combined with wrap-root stage-5 fusion, also drives `valu=5986` and `alu=12067`, but the kernel still measures `1067` cycles.
- Implication: simply nudging static slot totals below the obvious `load`/`valu` thresholds is not sufficient on this branch; the remaining gap is still dominated by schedule overlap and critical-path shape.
- New hash-direction lead from the current telemetry:
- stage 1 and stage 5 are the remaining temp-heavy hash combines on the critical path (`2 shifts + 5 xor-family vector ops` per group/round around the hash tail), and unlike fresh-temp reuse they can potentially write back into the already-mapped live `values` block.
- Implication: a narrow “stage1/stage5 combine-to-values” rewrite is the smallest hash dataflow change that could cut hash scratch/liveness without reintroducing the full temp-reuse allocator failure mode.
- New tree-direction lead now visible in existing scaffolding:
- `ENABLE_INPLACE_TREE_XOR_DATAFLOW` has not been exercised on the current `1065` branch, and unlike the failed full in-place vector chain it only overwrites `values` at the tree-output boundary while leaving hash SSA intact.
- Implication: combining shallow tree-xor in-place writes with the already-safe deep scalar scatter-to-values rewrite may be the cleanest remaining “all tree outputs write back to values” architecture to test without reopening the hash allocator alias problem.
- New constraint from the latest in-place writeback attempts:
- even correctness-valid writeback rewrites on offloadable vector ops (`hash` xor/shift combines, shallow `tree_xor`) tend to reduce emitted ALU slots but raise VALU slots enough to regress, which strongly suggests the current schedule benefits from retroactive/current-cycle ALU offload that these fixed-destination writes partially suppress.
- Implication: future dataflow rewrites should avoid `write_to` on offloadable vector ops unless they also preserve a comparable ALU-offload path.
- New scheduler/allocator lead from that constraint:
- retroactive ALU offload currently rejects full-vector `write_to` ops whenever the destination block is already mapped, even in cases where the overwritten block has no remaining readers beyond the current op and the past bundles do not touch that block.
- Implication: enabling a guarded “reuse existing mapped destination for retro-offload” path may be the key to making narrow in-place tree/hash rewrites competitive instead of VALU-regressive.
- Guarded retro-offload reuse for mapped `write_to` vectors is now validated:
- when the overwritten vector block has no remaining readers beyond the current op and the candidate past bundles do not touch that block, reusing the existing mapped destination for retroactive ALU offload restores the lost ALU-relief path without changing baseline behavior.
- Immediate impact on blocked writeback paths:
- `REUSE_HASH_COMBINE_TEMPS=True` is now correctness-valid again at `1065` cycles.
- `ENABLE_INPLACE_HASH_DATAFLOW=True` is also correctness-valid at `1065`.
- Prior regressive but correctness-valid writeback rewrites (`INPLACE_HASH_OUTPUT_STAGES={1,5}`, `ENABLE_INPLACE_TREE_XOR_DATAFLOW`, and the full-tree writeback combination) all snap back from `1070-1074` to baseline-neutral `1065`.
- Implication: the writeback-family blocker on this branch was primarily scheduler offload support, not the emitted mathematical graph. These paths are now usable scaffolding again, but they still have not produced a cycle win on their own.
- New strongest near-win after the scheduler fix:
- depth-5 vector scatter combined with wrap-root stage-5 fusion and the full writeback chain reaches `1067` with the best current emitted slot shape seen on this branch:
- `load=2001`, `alu=12011`, `valu=5993`, `flow=785`, `store=32`, `11909 ops`, peak scratch `1472/1536`.
- Implication: this family is now one real load and roughly one ALU bundle away from the hard lower-bound targets, making the previously regressive setup-side `load -> flow` rewrite worth re-testing only against this exact composition.
- Strongest current slot-shape after that follow-up:
- adding the narrow pointer-derivation rewrite on top of the same depth-5 vector-scatter family reaches `load=2000`, `alu=12003`, `valu=5994`, `flow=785`, `store=32`, `11908 ops`, but still measures `1067` cycles.
- Extending vector scatter to depths `{5,6}` can push ALU fully below the old `12000` target (`alu=11955`, `load=2000`, `valu=6000`), yet still remains at `1068`.
- Implication: this reopened vector-scatter family is no longer blocked by static slot counts alone; its remaining cap is schedule overlap / dependency shape, not just raw `load` or `alu` totals.
- New dependency-shape lead inside that family:
- each vector-scatter combine waits on an all-or-nothing bundle of eight scalar `_scat_load_*` ops, so interleaving those loads across unrelated groups/rounds can stretch the combine span even when total load-slot pressure is already near-optimal.
- Implication: a local scheduler heuristic that batches sibling scatter loads together, without globally biasing the ready queue toward loads, is worth testing before giving up on the reopened vector-scatter architecture.
- New fusion lead from that reopened family:
- the generic stage-5 fusion path paid for deep-round repair too late (`idx_unbias` + post-scatter bias-fix), but the specific `round 4 -> round 5` transition has only one flipped branch bit and depth-5 is a pure scatter round in the current near-win family.
- That enables a narrower critical-path rewrite: skip stage-5 const only on round `4`, then in round `5` repair `values` before the scatter combine (`values ^ const5`) so the repair can overlap with depth-5 loads, while correcting the depth-5 address with the known one-bit xor mask.
- Implication: if any new fusion can still help this branch, this is the most plausible one because it removes the repair op from the post-load critical path instead of only changing slot totals.
- The narrow `round 4 -> round 5` fusion did not survive contact with the scheduler:
- even with pointer-derivation, wrap-root, full writeback, and the reopened depth-5/6 vector-scatter family, the pre-bias repair path regressed badly (`1090-1092` cycles) and pushed `valu` back above `6000`.
- Implication: the remaining limiter in the reopened vector-scatter family is not just where the repair op sits on the depth-5 chain; skipping stage-5 const before the deep rounds still perturbs overlap enough to lose far more than it saves.
- The local scatter-load batching heuristic is also a dead end:
- forcing the scheduler to keep sibling `_scat_load_*` ops together regressed both the baseline (`1109`) and the reopened vector-scatter near-win (`1099`), despite leaving slot totals close to the original families.
- Implication: the scheduler’s existing global priority/interleave balance is more important than tightly finishing individual scatter batches; any future dependency-shape rewrite has to preserve that cross-engine ordering.
- New critical-path fusion lead after that failure:
- for the normal stage-5 hash (`out = (a ^ C5) ^ (a >> 16)`), the branch bit can be derived from stage-5 intermediates without waiting for the final output value:
- `out % 2 = 1` exactly when `lsb(a) == lsb(a >> 16)`.
- That means the next-round branch/index update can potentially start from the stage-4 output and the stage-5 shift result while the final stage-5 combine is still being scheduled, which is a real tail-shortening change rather than another slot-only rewrite.
- The selective stage-5 branch-hint path still loses:
- limiting the branch-hint rewrite to the exact reopened vector-scatter rounds (`round 5` alone, or `rounds 5/6`) reduced the blow-up versus the global version, but the best point still regressed to `1068`.
- Implication: the extra offloadable parity/equality ops cost more than the tail overlap they recover; this branch-bit fusion family is not competitive on the current scheduler/ISA.
- New strongest architecture on the branch after all of that:
- per-depth vector-scatter placement matters. The mixed layout `depth 5 -> write_to_values`, `depth 6 -> plain combine` is the first reopened vector-scatter family that gets back to the current `1065` baseline while keeping materially lower static pressure:
- `11699 ops`, `load=1999`, `flow=786`, `alu=11834`, `valu=6013`, `store=32`, peak scratch `1496/1536`.
- Implication: depth-5 and depth-6 do not want the same vector-scatter placement; the remaining gap to sub-`1065` is now a small follow-up around this mixed family rather than around the old uniform-placement variants.
- Narrow setup specialization interacts differently with the reopened families than it did on baseline:
- specializing `forest_values_ptr` and `inp_values_ptr` directly is still terrible on baseline (`1070`), but on the reopened vector-scatter families it consistently removes one more real load (`1999`) and can pull ALU well below `12000` without losing correctness.
- Implication: future work should continue to evaluate setup specialization only together with the reopened vector-scatter architecture, not in isolation.
- New tied family from this pass:
- guarded ALU offload on the depth-5 writeback combine plus the mixed depth-5/depth-6 vector-scatter layout still only ties at `1065`, but it shifts the static shape to `load=1999`, `flow=786`, `alu=11892-11894`, `valu=6006`.
- Implication: we now have multiple distinct `1065` architectures with materially lower `load` and `alu` than baseline; the missing win is still overlap, not slot-count blindness.
- The tied floor is now robust across several local variants of the same mixed family:
- `SPECIALIZE_FOREST_VALUES_PTR + SPECIALIZE_INPUT_VALUES_PTR`
- `SPECIALIZE_FOREST_VALUES_PTR + DERIVE_INPUT_VALUES_PTR_FROM_FOREST_PTR`
- `ROUND_GATE_WINDOW_BY_ROUND={5:28/29/30}`
- guarded scatter-writeback offload on or off
- and even broader `write_to_values` placement across depths `{5,6}`
- all repeatedly converge to correctness-valid `1065` rather than opening a new win.
- Implication: the current branch is no longer missing a tiny placement/gate/offload interaction around the mixed vector-scatter family; beating `1065` now likely needs a qualitatively different dependency graph rather than another local perturbation of this one.
- Chunked deep-round tiling is now ruled out for this branch:
- even when restricted to one deep round (`5` or `6`) or aimed at the public `group≈17 / round_tile≈13` style, round-major tiling inside concurrency-sized chunks regressed heavily (`1086+`, often `1090+`).
- Implication: the strongest remaining families still want group-major emission, even when the deep rounds themselves are rewritten.
- Deep tree traversal is the main load hotspot:
- Depths `5-10` use scatter path and contribute most scalar `load`+`alu` tree ops.
- New allocator/scheduler finding on the hash-temp reuse path:
- Retroactive ALU split-offload has a concrete vector-block leak for full-vector `write_to` ops.
- Root cause: `_alloc_safe_vector()` can pop/consume a free vector block before it notices `op.vbase` is already mapped (`op.vbase in alloc.vmap`), then returns `False` without restoring the block.
- Reproduction on `REUSE_HASH_COMBINE_TEMPS=True`: observed `68` leaked safe-vector attempts before the scheduler stuck at `421` cycles / `4471/12375` ops done.
- Implication: any write-to-based vector reuse path that allows retroactive ALU offload can deadlock from allocator leakage even if the dataflow itself is otherwise valid.
- Hash-temp reuse correctness is now narrowed further:
- After fixing the `_alloc_safe_vector()` leak, full hash-temp reuse no longer deadlocks, but it still fails correctness at `1070` cycles (`peak scratch 1521/1536`).
- The emitted graph/dataflow is not inherently wrong: a dense one-op-per-cycle execution of the same write-to graph is correct on a virtual-address machine.
- The remaining failure is specific to the tight physical allocator/scheduler regime:
- Re-running the same hash-temp reuse graph with the original scheduler and a very large scratch space (`2_000_000`) is correctness-valid at the same `53` cycles on the small repro case.
- Implication: the unresolved blocker is physical scratch mapping / alias scheduling under the `1536`-word layout, not the mathematical hash rewrite itself.

## Attempts and Outcomes
- Knob sweep on global/tuning constants (concurrency windows, recompute interval, offload policy, depth toggles, engine weights):
- No correctness-valid configuration beat `1082`.
- Several low-cycle outputs were invalid due to scheduler stuck states; these are dead ends.
- Decision: stop brute-force knob tuning and focus on architectural changes.

## Attempt Log
- 2026-03-06: Narrowing the tied mixed vector-scatter family.
- What changed/tested:
- Started from the tied mixed family:
- `depth 5 -> write_to_values`
- `depth 6 -> plain`
- specialized header pointers
- wrap-root fusion
- full tree/hash/scatter writeback
- Then tested:
- `SPECIALIZE_FOREST_VALUES_PTR + SPECIALIZE_INPUT_VALUES_PTR`
- `SPECIALIZE_FOREST_VALUES_PTR + DERIVE_INPUT_VALUES_PTR_FROM_FOREST_PTR`
- guarded scatter-writeback offload on / off
- round-5 gate widths `28`, `29`, `30`
- broader `write_to_values` placement across depths `{5,6}`
- Why: once the mixed family tied at `1065`, the last plausible path was to probe for a small local overlap/placement interaction around the same graph.
- Result:
- `SPECIALIZE_FOREST_VALUES_PTR + SPECIALIZE_INPUT_VALUES_PTR` with the mixed family -> correctness-valid `1065`, `11699` ops, `load=1999`, `alu=11834`, `valu=6013`.
- `SPECIALIZE_FOREST_VALUES_PTR + DERIVE_INPUT_VALUES_PTR_FROM_FOREST_PTR` + guarded writeback offload + `gate {5:29}` -> correctness-valid `1065`, `11702` ops, `load=1999`, `alu=11893`, `valu=6006`.
- `gate {5:28}` and `{5:30}` on the same family also tie at `1065`.
- Allowing `write_to_values` on both depths `{5,6}` with guarded offload also ties at `1065`.
- Decision: keep the scaffolding, but treat the mixed family as locally exhausted. Multiple nearby variants converge to the same `1065` floor.

- 2026-03-06: Chunked deep-round partial round-major tiling retest.
- What changed/tested:
- Added `ROUND_MAJOR_TILE_RANGE` and tested it on the strongest mixed family for:
- tile `5..6`
- tile `5`
- tile `6`
- external-inspired `MAX_CONCURRENT_GROUPS=17`, tile `0..12`
- Why: import the public tile/flat-stream idea in the narrowest way that preserves the current group-cap model.
- Result:
- tile `5..6` -> `1092`
- tile `5` -> `1086`
- tile `6` -> `1093`
- `17 / 0..12` style point -> `1146`
- Decision: keep disabled. This direction is decisively non-competitive on the current branch.

- 2026-03-06: Deep-round index-update decomposition.
- What changed/tested:
- Added `DECOMPOSE_INDEX_UPDATE_DEPTHS` so `new_index = 2*index + branch_bit` can be lowered as two offloadable vector adds on selected depths.
- Tested it on the strongest mixed family for depths `{5,6}` and `{5,6,7,8,9}`.
- Why: the tied family has lower ALU pressure than baseline, so moving deep-round tail work off VALU looked plausible.
- Result:
- depths `{5,6}` -> `1068`
- depths `{5,6,7,8,9}` -> `1076`
- Decision: keep disabled. This raises total tail work too much.

- 2026-03-06: Decomposed deep-round index update (`2*index + branch_bit`) into vector adds.
- What changed/tested:
- Added `DECOMPOSE_INDEX_UPDATE_DEPTHS` so selected depths replace the single VALU `multiply_add` in `_emit_index_update()` with two offloadable vector adds.
- Tested on the strongest mixed family for:
- depths `{5,6}`
- depths `{5,6,7,8,9}`
- Why: the mixed family finally has ALU headroom and slightly elevated VALU, so moving deep index-update work off VALU was a plausible tail optimization.
- Result:
- depths `{5,6}` -> `1068`, `11763` ops, `load=1999`, `flow=786`, `alu=11986`, `valu=6058`, `store=32`.
- depths `{5,6,7,8,9}` -> `1076`, `11859` ops, `alu=12274`, `valu=6118`.
- Decision: keep disabled. The extra vector-add chain costs more than it saves.

- 2026-03-06: Chunked deep-round partial round-major tiling.
- What changed/tested:
- Added `ROUND_MAJOR_TILE_RANGE` so selected rounds can be emitted round-major within concurrency-sized group chunks while the rest of the kernel remains group-major.
- Tested on the strongest mixed family for:
- tile `5..6`
- tile `5`
- tile `6`
- external-inspired point `MAX_CONCURRENT_GROUPS=17`, tile `0..12`
- Why: import the public “group tile + round tile” idea in the narrowest way that preserves current group-cap semantics.
- Result:
- tile `5..6` -> `1092`.
- tile `5` -> `1086`.
- tile `6` -> `1093`.
- `17 / 0..12` style point -> `1146`.
- Decision: keep the scaffold available but disabled. Tiling deep rounds inside chunks is non-competitive on this branch.

- 2026-03-06: Specialized setup + reopened vector-scatter family retests.
- What changed/tested:
- Added `SPECIALIZE_FOREST_VALUES_PTR` and `SPECIALIZE_INPUT_VALUES_PTR`.
- Re-tested reopened vector-scatter families with specialized pointers:
- depth `5` vector scatter
- depth `6` only
- depth `7` only
- depths `{6,7}`
- and one-group delayed depth-5-vselect hybrids on top of the strongest depth-5 vector-scatter family.
- Why: setup specialization had become harmful on the baseline but might be net-positive only on the lower-scratch reopened families.
- Result:
- strongest depth-5 family with both pointer specializations -> `1067`, `load=1999`, `alu=11978`, `valu=5997`.
- depth `6` only family with both specializations -> `1067`, `load=1999`, `alu=12010`, `valu=5991`.
- depth `7` only / `{6,7}` families -> `1072`.
- one-group delayed depth-5 hybrid re-test remained regressive (`1070+`).
- Decision: keep both setup-specialization scaffolds, but only as companions for the reopened vector-scatter family; they are still wrong for the plain baseline.

- 2026-03-06: Mixed depth-5/depth-6 vector-scatter family follow-ups.
- What changed/tested:
- Continued from the tied mixed family:
- `depth 5 -> write_to_values`, `depth 6 -> plain`
- specialized pointers
- wrap-root fusion
- full writeback chain
- hash temp reuse
- Tested nearby follow-ups:
- `ROUND_GATE_WINDOW_BY_ROUND={5:28}`, `{5:29}`, `{5:30}`
- `ROUND_GATE_WINDOW_BY_ROUND={7:28,9:24,10:28}`
- `ROUND_GATE_RELEASE_AFTER_HASH_ROUNDS={5}`
- guarded offload enabled for scatter writeback (`ALLOW_OFFLOAD_SCATTER_VEC_WRITEBACK=True`)
- Why: once the mixed family tied baseline, only a few local overlap variants remained worth checking.
- Result:
- base mixed family -> correctness pass, `1065`, `11699` ops, `load=1999`, `flow=786`, `alu=11834`, `valu=6013`, `store=32`.
- with guarded scatter-writeback offload and gate `{5:28}` -> `1065`, `11703` ops, `alu=11894`, `valu=6006`.
- same family with gates `{5:29}` and `{5:30}` -> both `1065`, `11702/11701` ops, `alu=11893/11892`, `valu=6006`.
- adding hash-release on round `5` stayed tied at `1065`.
- baseline gate map `{7:28,9:24,10:28}` stayed `1067`.
- Decision: keep the new per-depth placement and guarded scatter-writeback offload scaffolding, but none of the local overlap tweaks breaks below `1065`.

- 2026-03-05: Mixed depth-5/depth-6 vector-scatter placement.
- What changed/tested:
- Added depth-filtered vector-scatter placement controls so depth-5 and depth-6 can choose different combine destinations.
- Tested the two asymmetric layouts on the reopened vector-scatter family with:
- specialized header pointers
- wrap-root fusion
- full tree/hash/scatter writeback
- hash temp reuse
- no round gates
- Why: depth 5 sits on the first deep-round transition, while depth 6 is a later heavy-load round; forcing both to use the same vector-scatter placement was likely over-constraining the architecture.
- Result:
- `depth 5 -> write_to_values`, `depth 6 -> plain` -> correctness pass, `1065` cycles, `11699` ops, slots `load=1999`, `flow=786`, `alu=11834`, `valu=6013`, `store=32`, peak scratch `1496/1536`.
- `depth 5 -> plain`, `depth 6 -> write_to_values` -> `1070` cycles, `11699` ops, `alu=11866`, `valu=6009`.
- Decision: keep the per-depth placement scaffolding. The `d5->values / d6->plain` mix is now the strongest correctness-valid architecture on the branch, even though it only ties baseline so far.

- 2026-03-05: Follow-ups around the tied mixed-placement family.
- What changed/tested:
- Tested the strongest mixed family with:
- `ROUND_GATE_WINDOW_BY_ROUND={5:28}`
- `ROUND_GATE_WINDOW_BY_ROUND={7:28,9:24,10:28}`
- `SCATTER_VECTOR_XOR_INPLACE_DEPTHS={6}`
- slightly lower `MAX_CONCURRENT_GROUPS` (`18`, `16`)
- adjacent round-5 gate widths (`29`, `30`)
- guarded ALU offload enabled for scatter writeback variants
- `ROUND_GATE_RELEASE_AFTER_HASH_ROUNDS={5}`
- Why: once the mixed family tied baseline, the next step was a very small set of follow-ups around its overlap shape rather than a fresh broad rewrite.
- Result:
- `gate {5:28}` -> `1067`, `alu=11838`, `valu=6013`.
- baseline gate map `{7:28,9:24,10:28}` -> `1067`, `alu=11962`, `valu=5999`.
- `depth6 inplace` -> `1070`.
- `MAX_CONCURRENT_GROUPS=18` -> `1171`; `16` -> `1200`.
- `gate {5:29}` / `{5:30}` -> both `1067`.
- Allowing guarded offload on the depth-5 writeback combine:
- no gates -> `1067`, `alu=11970`, `valu=5996`
- with `gate {5:28}` -> `1065`, `alu=11894`, `valu=6006`
- adding `ROUND_GATE_RELEASE_AFTER_HASH_ROUNDS={5}` on top of that still stayed `1065`.
- Decision: keep the new scaffolding, but none of these local overlap tweaks beats the tied mixed family.

- 2026-03-05: Re-testing setup specialization and depth-local vector scatter around the reopened family.
- What changed/tested:
- Added `SPECIALIZE_FOREST_VALUES_PTR` and `SPECIALIZE_INPUT_VALUES_PTR` scaffolding and tested them only on reopened vector-scatter families.
- Also tested deeper-only vector-scatter activation (`depth 6` only, `depth 7` only, `depths 6+7`) and a one-group depth-5 vselect hybrid on top of the strongest depth-5 vector-scatter family.
- Why: once the reopened family got close, the next logical move was to remove one more setup load or shave a little more deep-round load locally without paying the full historical vselect penalty.
- Result:
- `SPECIALIZE_FOREST_VALUES_PTR=True` on the strongest depth-5 family -> `1067`, `load=1999`, `alu=11986`, `valu=5996`, `flow=786`.
- `SPECIALIZE_FOREST_VALUES_PTR=True` on plain baseline -> `1070`.
- `SPECIALIZE_FOREST_VALUES_PTR=True` + `SPECIALIZE_INPUT_VALUES_PTR=True` on the same depth-5 family -> still `1067`, `load=1999`, `alu=11978`, `valu=5997`.
- `depth 6` only vector scatter -> `1067`, `load=2000`, `alu=12051`, `valu=5986`.
- `depth 7` only -> `1072`; `depths 6+7` -> `1072`.
- One-group delayed depth-5 vselect hybrid on top of the strongest depth-5 vector-scatter family:
- `g=27` -> `1070`, `load=1996`, `flow=806`, `valu=6028`
- `g=30` -> `1071`.
- Decision: keep the pointer-specialization scaffolding because it improves the reopened families even though it hurts baseline; reject the one-group depth-5 hybrid and deeper-only vector-scatter variants as non-winning follow-ups.

- 2026-03-05: Selective early stage-5 branch-bit fusion.
- What changed/tested:
- Added a new branch-hint path that derives the next branch bit from stage-5 intermediates (`stage4_value` and `stage4_value >> 16`) instead of waiting for the final hash output.
- After the global version regressed badly, narrowed it with `EARLY_STAGE5_BRANCH_HINT_ROUNDS` and tested only:
- `round 5`
- `rounds 5 and 6`
- on the strongest reopened vector-scatter families.
- Why: this changes the round tail directly by allowing branch/index work to start before the final stage-5 combine completes.
- Result:
- Global path was catastrophic: baseline `1181` cycles, reopened vector-scatter family `1173`.
- Selective `round 5` on the strongest `depth {5,6}` vector-scatter family -> `1068`, `11764` ops, `load=2000`, `alu=12067`, `valu=6048`.
- Selective `rounds {5,6}` on the same family -> `1075`, `11828` ops, `valu=6112`.
- Selective `round 5` on the depth-5-only near-win -> `1068`, `11972` ops, `load=2000`, `alu=12123`, `valu=6043`.
- Decision: revert / keep disabled. The parity/equality branch-hint work is too expensive to beat the existing tail.

- 2026-03-05: Local scheduler batching of sibling scatter loads.
- What changed/tested:
- Added `PREFER_SCATTER_LOAD_BATCHING` so the scheduler can keep sibling `_scat_load_*` ops from the same scatter batch together when making a local load-vs-load choice, while leaving cross-engine heap order unchanged.
- Tested it on:
- the default `1065` baseline
- the strongest reopened vector-scatter family (`depths {5,6}`, no round gates, derived `inp_values_ptr`, wrap-root, full tree/hash/scatter writeback)
- Why: each vector-scatter combine waits on all eight scalar loads, so compressing those load windows looked like a plausible way to shorten the combine span without changing emitted slot totals.
- Result:
- `PREFER_SCATTER_LOAD_BATCHING=True` on baseline -> `1109` cycles, `12375` ops, slots `load=2001`, `flow=785`, `alu=12162`, `valu=6005`, `store=32`.
- `PREFER_SCATTER_LOAD_BATCHING=True` on the reopened vector-scatter family -> `1099` cycles, `11700` ops, slots `load=2000`, `flow=785`, `alu=12035`, `valu=5988`, `store=32`.
- Decision: revert / keep disabled. Tightening individual scatter batches harms the scheduler’s broader overlap badly enough that it is not competitive.

- 2026-03-05: Narrow round-4 -> round-5 stage-5 fusion for the reopened vector-scatter family.
- What changed/tested:
- Added a one-round fusion path that skips stage-5 const only on round `4`, then repairs round `5` by:
- correcting the known one flipped depth-5 index bit
- and pre-biasing `values` before the depth-5 scatter combine so the repair can overlap with the loads
- Tested it on the strongest reopened vector-scatter compositions:
- depth-5 vector scatter + wrap-root + full writeback + derived `inp_values_ptr`
- depth-5/6 vector scatter + wrap-root + full writeback + derived `inp_values_ptr` with no round gates
- Why: this was the cleanest remaining critical-path fusion idea after the generic stage-5 framework proved too heavy; the repair could happen before the scatter combine instead of after it.
- Result:
- `ENABLE_D5_SCATTER_STAGE5_FUSION=True` on the depth-5 vector-scatter near-win -> `1092` cycles, `11941` ops, `load=2000`, `alu=12099`, `valu=6015`, `flow=785`, peak scratch `1528/1536`.
- Same fusion on the depth-5/6 no-gate vector-scatter family -> `1090` cycles, `11733` ops, `load=2000`, `alu=11979`, `valu=6028`, `flow=785`, peak scratch `1526/1536`.
- Decision: revert / keep disabled. This fusion changes the critical path in the intended place, but the overall overlap disruption is severe enough that it is non-competitive.

- 2026-03-05: Reopened delayed depth-5 near-miss under the scheduler offload fix.
- What changed/tested:
- Re-tested the historical delayed depth-5 near-miss config:
- `DEPTH_5_VSELECT_GROUPS={27}`
- `DEPTH_5_VSELECT_DIFF_PAIRS=12`
- `DELAY_DEPTH_5_SETUP=True`
- `ROUND_GATE_WINDOW_BY_ROUND={}`
- `DEPTH_3_VSELECT_DIFF_PAIRS_BY_ROUND={3:4,14:3}`
- `DEPTH_4_VSELECT_DIFF_PAIRS_BY_ROUND={4:1,15:1}`
- Then combined it with:
- `ENABLE_WRAP_ROOT_STAGE5_FUSION=True`
- full writeback chain (`ENABLE_INPLACE_TREE_XOR_DATAFLOW=True`, `ENABLE_INPLACE_HASH_DATAFLOW=True`, `ENABLE_INPLACE_SCALAR_SCATTER_TO_VALUES=True`)
- and the two known best round-gate maps.
- Why: the scheduler fix made writeback paths allocator-safe, so this was the most likely previously blocked `1067` family to convert lower scratch into real cycles.
- Result:
- Historical config re-validates exactly at `1067` with `load=1997`, `flow=834`, `alu=12129`, `valu=6023`.
- Adding wrap-root alone regressed to `1070` (`alu=12114`, `valu=5994`).
- Adding full writeback chain recovered part of that loss to `1068` (`12152 ops`, peak scratch `1478/1536`) but still did not beat baseline.
- Re-introducing round gates made it worse (`1071-1072`).
- Decision: keep this family as a dead end for cycle reduction on the current branch. The load savings survive, but the added flow pressure still dominates.

- 2026-03-05: Depth-5 vector scatter + wrap-root + writeback family.
- What changed/tested:
- Combined:
- `SCATTER_VECTOR_XOR=True`
- depth filters on heavy rounds (`SCATTER_VECTOR_XOR_DEPTHS={5}`, then `{5,6}`, then `{5,6,7}`)
- wrap-root fusion (`ENABLE_WRAP_ROOT_STAGE5_FUSION=True`)
- full writeback chain (`ENABLE_INPLACE_TREE_XOR_DATAFLOW=True`, `ENABLE_INPLACE_HASH_DATAFLOW=True`, `ENABLE_INPLACE_SCALAR_SCATTER_TO_VALUES=True`)
- and re-tested the narrow pointer-derivation setup rewrite (`DERIVE_INPUT_VALUES_PTR_FROM_FOREST_PTR=True`) against this family only.
- Also checked combine placement variants (`plain`, `write_to_values`, `inplace`) and no-gate overlap variants where they were logically relevant.
- Why: this is the first current-branch family where the scheduler fix plus writeback support materially changes the viable slot tradeoff, so it became the best shot at a real sub-`1065` win.
- Result:
- `depth {5}`, wrap-root, write-to-values, full writeback -> `1067`, `11909 ops`, `load=2001`, `alu=12011`, `valu=5993`, peak scratch `1472/1536`.
- Adding pointer derivation on that exact point -> `1067`, `11908 ops`, `load=2000`, `alu=12003`, `valu=5994`, `flow=785`, peak scratch unchanged.
- `depth {5,6}`, write-to-values -> `1068`, `11716 ops`, `load=2000`, `alu=11955`, `valu=6000`.
- `depth {5,6}`, plain combine, no gates -> `1067`, `11700 ops`, `load=2000`, `alu=12011`, `valu=5991`, `flow=785`, peak scratch `1464/1536`.
- `depth {5,6,7}`, plain combine, no gates -> `1068`, `11508 ops`, `load=2000`, `alu=11907`, `valu=6004`.
- Targeted low-VALU rebalance on that last point (`DEPTH_4_VSELECT_DIFF_PAIRS_BY_ROUND={4:1,15:1}`) was catastrophic despite better slot totals: `1093` cycles, `flow=813`, `alu=11938`, `valu=5971`.
- Decision: keep the vector-scatter scaffolding disabled. This family now achieves the strongest slot shapes seen on the branch, but it still cannot convert them into a cycle win; the remaining limiter is overlap / dependency shape rather than emitted slot totals alone.

- 2026-03-05: Guarded retroactive ALU offload for fixed-destination vector writes.
- What changed/tested:
- Added op-level `write_to` ownership tracking and taught retroactive ALU offload to reuse an already-mapped destination block when:
- the op is a full-vector `write_to`
- the overwritten block has no remaining readers beyond the current op
- and the candidate past bundles do not touch that physical block.
- Re-tested the previously blocked writeback families:
- `REUSE_HASH_COMBINE_TEMPS=True`
- `ENABLE_INPLACE_HASH_DATAFLOW=True`
- `INPLACE_HASH_OUTPUT_STAGES={1,5}`
- `ENABLE_INPLACE_TREE_XOR_DATAFLOW=True` with and without `ENABLE_INPLACE_SCALAR_SCATTER_TO_VALUES=True`
- Why: the recent regressions strongly indicated that fixed-destination writebacks were losing the scheduler’s retroactive ALU-offload path; this change restores that path only when it is provably safe.
- Result:
- Baseline remains unchanged at `1065` cycles and passes correctness.
- `REUSE_HASH_COMBINE_TEMPS=True` -> correctness pass, `1065` cycles, `12375` ops, baseline slot totals.
- `ENABLE_INPLACE_HASH_DATAFLOW=True` -> correctness pass, `1065` cycles, `12375` ops, baseline slot totals.
- `INPLACE_HASH_OUTPUT_STAGES={1,5}` -> build/schedule returns to `1065` with baseline slot totals (previously `1070` / `alu=11946` / `valu=6032`).
- `ENABLE_INPLACE_TREE_XOR_DATAFLOW=True` + `ENABLE_INPLACE_SCALAR_SCATTER_TO_VALUES=True` -> build/schedule returns to `1065` with `12131` ops and baseline slot totals (previously `1074` / `alu=11498` / `valu=6088`).
- `ENABLE_WRAP_ROOT_STAGE5_FUSION=True` + full writeback chain still remains `1067` (`12101` ops, `alu=12083`, `valu=5984`), so the restored offload path does not by itself convert the best wrap-root architecture into a cycle win.
- Decision: keep the scheduler fix. It resolves a real allocator/offload limitation and reopens previously blocked hash/tree writeback scaffolds, even though none of the re-tested paths beat the `1065` baseline yet.

- 2026-03-05: Stage-1 / stage-5 hash combine writes back into `values`.
- What changed/tested:
- Added `INPLACE_HASH_OUTPUT_STAGES` so selected hash stages can overwrite the live `values` vector without enabling the full in-place hash chain.
- Tested the intended target `INPLACE_HASH_OUTPUT_STAGES={1,5}` and then split it into `{1}` and `{5}` to localize the effect.
- Why: stage 1 and stage 5 are the remaining temp-heavy xor/shift hash combines; this was the smallest hash dataflow rewrite that could cut scratch/liveness without re-entering the full temp-reuse alias failure.
- Result:
- `INPLACE_HASH_OUTPUT_STAGES={1,5}` -> correctness pass, `1070` cycles, `12375` ops, slots `load=2001`, `flow=785`, `alu=11946`, `valu=6032`, `store=32`, peak scratch `1521/1536`.
- `INPLACE_HASH_OUTPUT_STAGES={1}` -> `1070` cycles, build/schedule telemetry only, slots `alu=12002`, `valu=6025`.
- `INPLACE_HASH_OUTPUT_STAGES={5}` -> `1070` cycles, build/schedule telemetry only, slots `alu=12010`, `valu=6024`.
- Decision: keep disabled. The rewrite is correctness-valid, but it shifts too much work back onto VALU and loses `5` cycles.

- 2026-03-05: Tree-output writes back into `values`.
- What changed/tested:
- Exercised existing `ENABLE_INPLACE_TREE_XOR_DATAFLOW` scaffolding for shallow tree outputs.
- Also tested the natural full-tree follow-up with `ENABLE_INPLACE_SCALAR_SCATTER_TO_VALUES=True` so both shallow and deep tree outputs write back into `values`.
- Why: this is the narrowest remaining “all tree outputs in place” architecture that avoids the failed full in-place hash chain.
- Result:
- `ENABLE_INPLACE_TREE_XOR_DATAFLOW=True` -> correctness pass, `1074` cycles, `12375` ops, slots `load=2001`, `flow=785`, `alu=11498`, `valu=6088`, `store=32`, peak scratch `1527/1536`.
- `ENABLE_INPLACE_TREE_XOR_DATAFLOW=True` + `ENABLE_INPLACE_SCALAR_SCATTER_TO_VALUES=True` -> correctness pass, `1074` cycles, `12131` ops, identical slot totals `load=2001`, `flow=785`, `alu=11498`, `valu=6088`, `store=32`, peak scratch `1495/1536`.
- Decision: keep both disabled for this path. These rewrites remove graph scaffolding and lower ALU, but they block enough useful ALU offload that VALU rises sharply and the schedule regresses by `9` cycles.

- 2026-03-05: Header `inp_values_ptr` derivation from forest pointer.
- What changed/tested:
- Added `DERIVE_INPUT_VALUES_PTR_FROM_FOREST_PTR` so setup can compute `inp_values_ptr` as `forest_values_ptr + n_nodes + batch_size` via `flow/add_imm` instead of loading `mem[6]`.
- Also tested the most logical companions:
- with `ENABLE_WRAP_ROOT_STAGE5_FUSION=True`
- with `ENABLE_WRAP_ROOT_STAGE5_FUSION=True` and `ROUND_GATE_RELEASE_AFTER_HASH=True`
- Why: remove exactly one real load from the kernel while preserving the dynamic memory-layout anchor (`mem[4]`), avoiding the broader liveness cost of the older full compile-time header specialization.
- Result:
- `DERIVE_INPUT_VALUES_PTR_FROM_FOREST_PTR=True` -> `1070` cycles, pass, `12374` ops, slots `load=2000`, `flow=785`, `alu=12082`, `valu=6015`, `store=32`, peak scratch `1481/1536`.
- `DERIVE_INPUT_VALUES_PTR_FROM_FOREST_PTR=True` + `ENABLE_WRAP_ROOT_STAGE5_FUSION=True` -> `1067` cycles, pass, `12344` ops, slots `load=2000`, `flow=785`, `alu=12067`, `valu=5986`, `store=32`, peak scratch `1520/1536`.
- `DERIVE_INPUT_VALUES_PTR_FROM_FOREST_PTR=True` + `ENABLE_WRAP_ROOT_STAGE5_FUSION=True` + `ROUND_GATE_RELEASE_AFTER_HASH=True` -> `1067` cycles, pass, `12344` ops, same slot totals as the wrap-root companion.
- Decision: keep disabled / revert. This is the first current-branch path to hit `load=2000`, but it still does not convert into a cycle win even when paired with the best known overflow fusion.

- 2026-03-05: Round-window token release on hash completion.
- What changed/tested:
- Added `ROUND_GATE_RELEASE_AFTER_HASH` so the existing round-window gates (`7/9/10`) can release after the round hash finishes, instead of waiting for branch-bit/index tail work.
- Also tested the natural companion with `ENABLE_WRAP_ROOT_STAGE5_FUSION=True`.
- Why: the tree-completion release was too aggressive, so the next hypothesis was that gating only through the hash would preserve most of the overlap protection while still freeing the small index-update tail.
- Result:
- `ROUND_GATE_RELEASE_AFTER_HASH=True` -> `1068` cycles, pass, `12375` ops, slots `load=2001`, `flow=785`, `alu=12170`, `valu=6004`, `store=32`, peak scratch `1528/1536`.
- `ROUND_GATE_RELEASE_AFTER_HASH=True` + `ENABLE_WRAP_ROOT_STAGE5_FUSION=True` -> `1067` cycles, pass, `12345` ops, slots `load=2001`, `flow=785`, `alu=12091`, `valu=5983`, `store=32`, peak scratch `1456/1536`.
- Decision: keep disabled / revert. Releasing only the branch/index tail is much less harmful than full tree-completion release, but it still does not beat baseline.

- 2026-03-05: Round-window token release on tree completion.
- What changed/tested:
- Added `ROUND_GATE_RELEASE_ON_TREE_DONE` so the existing round-window gates (`7/9/10`) can release their scalar token immediately after `tree_values` instead of after the full round tail (`hash` + index-update).
- Kept the active round-window map unchanged and tested the feature in isolation.
- Why: the current gates exist to cap deep scatter-load overlap, so the hypothesis was that later groups could start their tree work sooner while prior groups finished the trailing hash/index work in parallel.
- Result:
- `ROUND_GATE_RELEASE_ON_TREE_DONE=True` -> `1072` cycles, pass, `12375` ops, slots `load=2001`, `flow=785`, `alu=12066`, `valu=6017`, `store=32`, peak scratch `1448/1536`.
- Decision: keep disabled. Earlier gate release improves scratch and ALU pressure, but it hurts overall overlap enough to lose `7` cycles.

- 2026-03-05: Tail-only shallow stage-5 fusion (`rounds 11..14`) and wrap+tail chain (`rounds 10..14`).
- What changed/tested:
- Added `ENABLE_TAIL_SHALLOW_STAGE5_FUSION` so the final shallow rounds can skip stage-5 const and rely only on shallow xor5/root fast paths (`depths 1..4`), with no scatter/mirror-base support.
- Also tested the chained version with `ENABLE_WRAP_ROOT_STAGE5_FUSION=True` so rounds `10..14` all skip consecutively.
- Why: wrap-root fusion improved real slot counts safely, so the next hypothesis was that the final shallow-only tail could support additional consecutive skips without reintroducing deep scatter complexity.
- Result:
- `ENABLE_TAIL_SHALLOW_STAGE5_FUSION=True` -> `1066` cycles, fail, `12327` ops, slots `load=2003`, `valu=5922`, `alu=12146`.
- `ENABLE_WRAP_ROOT_STAGE5_FUSION=True` + `ENABLE_TAIL_SHALLOW_STAGE5_FUSION=True` -> `1108` cycles, fail, `12295` ops, slots `load=2003`, `valu=5881`, `alu=12218`.
- Decision: keep the tail-fusion scaffolding disabled. The shallow tail still needs bias handling that this narrow fast-path model does not provide.

- 2026-03-05: Specialized overflow-to-root stage-5 fusion.
- What changed/tested:
- Added `ENABLE_WRAP_ROOT_STAGE5_FUSION`, which skips stage-5 const only on the overflow round (`depth == forest_height`) and absorbs that const into the next round's root XOR using `tree_root_vec_xor5`, without enabling the full generic fusion/bias framework.
- Also tested two targeted companion combinations:
- with `ENABLE_INPLACE_SCALAR_SCATTER_TO_VALUES=True`
- with the known cycle-tied gate map `ROUND_GATE_WINDOW_BY_ROUND={7:28,9:26,10:28}`
- Why: the generic fusion framework is too heavy, but the round-10 -> round-11 transition is special because the index always resets to the root, so this fuse can be done without branch-bit bias bookkeeping.
- Result:
- `ENABLE_WRAP_ROOT_STAGE5_FUSION=True` -> `1067` cycles, pass, `12345` ops, `valu=5982`, `alu=12099`.
- `ENABLE_WRAP_ROOT_STAGE5_FUSION=True` + `ENABLE_INPLACE_SCALAR_SCATTER_TO_VALUES=True` -> `1067` cycles, pass, `12101` ops, `valu=5984`, `alu=12083`.
- `ENABLE_WRAP_ROOT_STAGE5_FUSION=True` + `ROUND_GATE_WINDOW_BY_ROUND={7:28,9:26,10:28}` -> `1067` cycles, pass, `12343` ops, `valu=5982`, `alu=12097`.
- Decision: keep the new specialized fusion path as disabled scaffolding for now. It is the best new direction from this session, but it does not yet beat the `1065` baseline.

- 2026-03-05: Scheduler load-aware ordering narrowed to load-vs-load only.
- What changed/tested: refined the previous scheduler rewrite so global heap order stayed intact and only ready load ops were reordered against other ready load ops based on immediate unlock potential.
- Why: salvage the good part of the prior idea (better use of the two load slots) without globally biasing the whole schedule toward loads.
- Result: correctness pass but still `1075` cycles, matching the fully global rewrite. Peak scratch again dropped to `1481/1536`, but cycle time did not recover.
- Decision: reverted the scheduler queue rewrite entirely and restored the baseline scheduler. Load-aware selection in this form is not the missing piece.

- 2026-03-05: Scheduler ready-queue rewrite with global load-unlock preference.
- What changed/tested: changed the scheduler to scan the ready queue each cycle and globally prefer ready load ops that immediately unlock successor chains, instead of consuming the heap strictly in static-priority order.
- Why: load slots are still the dominant defer source, so the goal was to make the two load slots spend more time on dependency-critical deep scatter work.
- Result: correctness pass but regression to `1075` cycles. Peak scratch dropped materially to `1481/1536`, but the global load bias hurt overall overlap enough to lose `10` cycles.
- Decision: reject the global version. Any load-aware heuristic needs to stay local to load-vs-load choice rather than overriding cross-engine ordering.

- 2026-03-05: Lazy depth-4 setup on first use.
- What changed/tested: added `DELAY_DEPTH_4_SETUP` and a deferred `_emit_depth4_setup()` path so the main depth-4 broadcast/diff bundle can be emitted only when the first depth-4 vselect round reaches it.
- Why: stop paying the large depth-4 setup liveness cost from cycle 0; this was the next scratch/lifetime rewrite after the in-place deep-path work.
- Result: correctness pass but regression to `1075` cycles, with slot totals shifting to `load=2001`, `valu=6022`, `alu=12026`, `flow=785`.
- Decision: keep disabled. Moving depth-4 setup onto the first round-4 critical path is worse than carrying it live from the start.

- 2026-03-05: Scatter-vector combine writes directly into `values`.
- What changed/tested: added `SCATTER_VECTOR_XOR_WRITE_TO_VALUES` so the depth-5 vector-scatter path can load into a temporary scatter buffer but write the final vector XOR result back into the live `values` vector instead of a fresh result vector or the scatter buffer. Tested `SCATTER_VECTOR_XOR=True`, `SCATTER_VECTOR_XOR_WRITE_TO_VALUES=True`, `SCATTER_VECTOR_XOR_DEPTHS={5}`.
- Why: check whether the prior `1071` regression from vector scatter combine was caused by result-buffer placement/liveness rather than by the ALU->VALU engine trade itself.
- Result: correctness pass at `1071` cycles, `12151` ops, slot totals `load=2001`, `valu=6027`, `alu=11986`, `flow=785` — identical to the write-to-scatter variant.
- Decision: keep disabled. Result placement is not the problem; the vector-scatter architecture remains non-competitive on current engine bounds.

- 2026-03-05: In-place scatter-vector combine retry after allocator leak fix.
- What changed/tested: added `SCATTER_VECTOR_XOR_INPLACE` so the vector scatter combine can overwrite the loaded scatter buffer, and blocked ALU offload for that specific in-place op id. Tested the architecturally best prior placement: `SCATTER_VECTOR_XOR=True`, `SCATTER_VECTOR_XOR_INPLACE=True`, `SCATTER_VECTOR_XOR_DEPTHS={5}`.
- Why: the previous in-place scatter-vector path had deadlocked before the `_alloc_safe_vector()` leak fix; this re-test checks whether the repaired scheduler makes the lower-ALU path viable.
- Result: correctness pass at `1071` cycles, `12151` ops, slot totals `load=2001`, `valu=6027`, `alu=11986`, `flow=785`.
- Decision: keep the new scaffolding disabled; the path is now stable and correctness-valid, but still regresses because the ALU savings convert into extra VALU pressure instead of lowering total cycles.

- 2026-03-05: Narrow in-place hash rewrite for multiply-add-only stages.
- What changed/tested: added `ENABLE_INPLACE_HASH_MULADD_ONLY` so only hash stages `0` and `4` overwrite the live `values` vector, leaving the temp-heavy combine stages unchanged.
- Why: try to harvest scratch-lifetime benefits from the obviously safe single-op hash overwrites without re-entering the alias bugs of the full in-place hash path.
- Result: correctness pass, but no measurable schedule change: `1065` cycles, `12375` ops, peak scratch still `1528/1536`.
- Decision: keep disabled by default; this rewrite is semantically safe but too weak to matter on its own.

- 2026-03-05: Deep scalar scatter writes directly into `values`.
- What changed/tested:
- generalized partial-vector dependency tracking from coarse block-level writers to lane-precise writers.
- added `ENABLE_INPLACE_SCALAR_SCATTER_TO_VALUES` so deep scalar scatter XORs can write each lane back into the live `values` vector instead of allocating a separate output vector.
- also checked the natural follow-up with `ENABLE_INPLACE_HASH_MULADD_ONLY=True`.
- Why: remove one whole vector allocation from every deep scatter round, which is the highest-volume remaining SSA-style buffer in the kernel.
- Result:
- `ENABLE_INPLACE_SCALAR_SCATTER_TO_VALUES=True` -> `1065` cycles, pass, `12131` ops.
- `ENABLE_INPLACE_SCALAR_SCATTER_TO_VALUES=True` + `ENABLE_INPLACE_HASH_MULADD_ONLY=True` -> `1065` cycles, pass, `12131` ops.
- Peak scratch remained `1528/1536`, and final slot totals were unchanged from baseline.
- Decision: keep the rewrite available but disabled. It is a real graph simplification and now enables future in-place deep-path work, but by itself it does not reduce cycle count.

- 2026-03-05: Local round-gate retune around the current interleaving map.
- What changed/tested: searched a tight neighborhood around `ROUND_GATE_WINDOW_BY_ROUND={7:28,9:24,10:28}` using nearby windows on those same rounds only.
- Why: round gates change overlap without changing semantics or adding new ops, so this was the last configuration-only surface worth checking before returning to structural rewrites.
- Result:
- Best measured configs only tied baseline at `1065` cycles:
- `{7:28,9:24,10:28}` -> `1065` cycles, pass, `12375` ops.
- `{7:26,9:26,10:28}` -> `1065` cycles, pass, `12375` ops.
- `{7:28,9:26,10:28}` -> `1065` cycles, pass, `12373` ops.
- No neighborhood point improved below `1065`; most regressed to `1068+`.
- Decision: keep the existing gate map for now. The tied lower-op map `{7:28,9:26,10:28}` is interesting as scaffolding for future architectural changes, but not worth a baseline change by itself.

- 2026-03-05: Late shallow-round vselect mask single-flip neighborhood search.
- What changed/tested: evaluated all single-group flips around the current late shallow rounds:
- `DEPTH_3_VSELECT_GROUPS_BY_ROUND[14]` relative to the default depth-3 heuristic mask.
- `DEPTH_4_VSELECT_GROUPS_BY_ROUND[15]` relative to the default depth-4 heuristic mask.
- Why: current `1065` telemetry is load-bound, and late rounds `14/15` are the lowest-risk place to trade scatter loads for extra shallow `vselect` work without introducing long-lived setup vectors.
- Result:
- Baseline explicit late-round masks -> `1065` cycles, pass.
- Best depth-3 mutation: remove `g=1` from round-14 mask -> `1067` cycles, pass.
- Best depth-4 mutation: add `g=14` to round-15 mask -> `1071` cycles, pass.
- Several mutations reduced `load` from `2001` to `1993`, but the added flow/valu pressure still regressed overall schedule length.
- Decision: keep the current late-round masks unchanged; shallow single-group flips are not enough to beat `1065`.

- 2026-03-05: Delayed late shallow-tree setup duplication (`DELAY_LATE_D3D4_SETUP`).
- What changed/tested: enabled duplicate depth-3/depth-4 setup for late rounds so the early shallow setup could die after round 4; also confirmed the same result with `DELAY_DEPTH_5_SETUP=True` (which is effectively inert in the current baseline because `DEPTH_5_VSELECT_GROUPS=set()`).
- Why: the new scaffolding looked like a plausible way to lower long-lived scratch pressure and improve schedule freedom without changing tree semantics.
- Result:
- `DELAY_LATE_D3D4_SETUP=False` -> `1065` cycles, pass
- `DELAY_LATE_D3D4_SETUP=True` -> `1103` cycles, pass
- `DELAY_LATE_D3D4_SETUP=True`, `DELAY_DEPTH_5_SETUP=True` -> `1103` cycles, pass
- Ops increased from `12375` to `12420`.
- Decision: keep both delay flags disabled; duplicating shallow setup adds too much extra load/valu work to pay for the shorter lifetime.

- 2026-03-05: Re-validated the checked-out worktree baseline before new optimization work.
- What changed/tested: ran `python3 tests/submission_tests.py` on the current dirty worktree without changing code.
- Why: the journal contained older `1078-1082` history, so the first step was to confirm the actual baseline of the present file state before attempting new optimizations.
- Result: `1065` cycles, correctness pass, `12375 ops`, peak scratch `1528/1536`.
- Decision: keep this as the active baseline for all further measurements in this worktree.

- 2026-03-05: Fixed retroactive offload leak for full-vector `write_to` ops.
- What changed: moved the `_alloc_safe_vector()` early-return checks (`write_size == 0`, `vbase < 0`, and `op.vbase in alloc.vmap`) ahead of free-list / watermark allocation so failed safe-vector attempts no longer consume a block.
- Why: `REUSE_HASH_COMBINE_TEMPS=True` exposed a concrete allocator leak where split-offload tried to allocate a new safe vector for an op that already had a mapped destination.
- Result:
- Baseline behavior unchanged when the feature is off; full `python3 tests/submission_tests.py` re-validation still passes at `1065` cycles.
- Full hash-temp reuse no longer deadlocks, but still fails correctness at `1070` cycles (`peak scratch 1521/1536`).
- `ENABLE_INPLACE_HASH_DATAFLOW=True` also now reaches `1070` cycles but still fails correctness.
- Additional narrowing:
- Dense one-op-per-cycle execution of the emitted write-to graph is correctness-valid.
- The original scheduler with a huge scratch budget (`2_000_000`) is also correctness-valid on the small repro case, so the remaining problem is allocator/layout-specific.
- Decision: keep the scheduler leak fix (real correctness bug in experimental scaffolding); continue investigating the tight-layout alias issue before enabling any hash write-to path.

- 2026-03-05: Stage-specific hash-combine temp reuse sweep.
- What changed: added `REUSE_HASH_COMBINE_TEMP_STAGES` so hash temp reuse can be enabled per stage (`1`, `2/3 fused`, `5`) instead of all-or-nothing.
- Why: isolate whether only one hash combine site is unsafe under the `1536`-word allocator while others remain correctness-valid and potentially cycle-helpful.
- Result (all full-kernel runs, correctness-checked):
- `stage1 ({1})` -> `1070` cycles, fail
- `stage23 ({2})` -> `1072` cycles, fail
- `stage5 ({5})` -> `1070` cycles, fail
- `stage1_23 ({1,2})` -> `1069` cycles, fail
- `stage1_5 ({1,5})` -> `1070` cycles, fail
- `stage23_5 ({2,5})` -> `1070` cycles, fail
- Decision: no stage-local subset is safe under the current allocator; keep `REUSE_HASH_COMBINE_TEMPS=False` and `REUSE_HASH_COMBINE_TEMP_STAGES=set()`.

- 2026-02-24: Implemented round-aware deep-round gating in dependency graph.
- What changed: Added deep-round gate dependencies in `build_kernel`, plus `start_deps` plumbing through `_emit_tree_xor` and `_emit_vselect_tree`.
- Why: limit deep scatter-round pressure independently from global group window.
- Result: initial version used vector refs as round tokens and caused scheduler stuck states (`~384` reported cycles, incorrect output).
- Decision: dead end; do not keep vector-token version.

- 2026-02-24: Reworked deep-round tokens to scalar barrier refs.
- What changed: token emitted as scalar ALU op dependent on deep tree completion, so cross-group gating does not extend vector liveness.
- Why: preserve round gate semantics while preventing scratch liveness blow-up.
- Result (window sweep, correctness-checked):
- `MAX_CONCURRENT_DEEP_ROUNDS=12` -> `1110` cycles (pass)
- `MAX_CONCURRENT_DEEP_ROUNDS=16` -> `1087` cycles (pass)
- `MAX_CONCURRENT_DEEP_ROUNDS=20` -> `1090` cycles (pass)
- `MAX_CONCURRENT_DEEP_ROUNDS=24` -> `1083` cycles (pass)
- `MAX_CONCURRENT_DEEP_ROUNDS=0` -> `1082` cycles (pass)
- Decision: feature kept but disabled by default (`MAX_CONCURRENT_DEEP_ROUNDS=0`) because all tested active windows regress.
- Validation: `python3 tests/submission_tests.py` passes with final state at `1082` cycles.

- 2026-02-24: Attempted in-place vector dataflow for tree/hash outputs.
- What changed: added vector `write_to` support and rewired tree/hash to overwrite `values` buffers in place.
- Why: reduce transient vector allocations and lower scratch pressure.
- Result: scheduler stalled with heavy `defer_scratch_full`, incorrect output, and early stop (`~210` cycles reported, invalid).
- Diagnosis: in-place writes conflicted with current ownership/free semantics and dependency fanout.
- Decision: reverted in-place path; keep original SSA-like vector outputs.

- 2026-02-24: Implemented generalized dead-index elimination (future-use analysis).
- What changed: replaced penultimate-round heuristic with per-group/per-round future index-use analysis.
- Why: remove unnecessary index materialization ops whenever no future round uses `ts.index`.
- Result: correctness pass; ops reduced `12401 -> 12365`; peak scratch reduced `1390 -> 1326`; cycles unchanged at `1082`.
- Decision: keep this change (strict improvement in op count/scratch, no cycle regression).

- 2026-02-24: Re-tested deep-round windowing after dead-index elimination.
- What changed: swept `MAX_CONCURRENT_DEEP_ROUNDS` with new lower-scratch kernel.
- Why: verify whether previous regression was scratch-driven and now recoverable.
- Result: no win. Best active window remained slower (`24 -> 1083`, `16 -> 1087`, `12 -> 1110`), `0 -> 1082`.
- Decision: initially kept disabled; later superseded by selective thresholded gating win (entry below).

- 2026-02-24: Selective deep-round gating by deeper threshold.
- What changed: targeted deep gating to depths `>=6` with `MAX_CONCURRENT_DEEP_ROUNDS=20`.
- Why: reduce gate overhead from earlier full deep gating while still smoothing load-heavy scatter rounds.
- Result: correctness pass and cycle improvement to `1081` (`python3 tests/submission_tests.py` full pass).
- Decision: keep this as new default (`DEEP_ROUND_DEPTH_THRESHOLD=6`, `MAX_CONCURRENT_DEEP_ROUNDS=20`).

- 2026-02-24: Added per-round/depth deep-window override framework.
- What changed: introduced `DEEP_GATE_WINDOW_BY_ROUND` / `DEEP_GATE_WINDOW_BY_DEPTH` and helper `_deep_gate_window`.
- Why: enable true per-round gating experiments instead of only threshold-based gating.
- Result: framework validated; default behavior unchanged at `1081` and full correctness pass.
- Decision: keep framework (enables future structural search without code churn).

- 2026-02-24: Explicit per-round deep-gate map search (global fallback disabled).
- What changed: tested explicit round maps on rounds `6..10` with window `20` and nearby structured variants.
- Why: verify if selective round gating could beat thresholded global winner.
- Result: best remained exactly the current full map equivalent (`{6,7,8,9,10}:20`) at `1081`.
- Decision: no config change.

- 2026-02-24: Scheduler architecture attempt: defer offloadable VALU ops for later ALU offload.
- What changed: added temporary deferral path (`DEFER_OFFLOADABLE_VALU`) when split-offload fails.
- Why: attempt to reduce VALU saturation by giving ALU offload extra cycles to succeed.
- Result: severe regression (`1212` cycles), correctness still passed.
- Decision: keep code path off by default (`DEFER_OFFLOADABLE_VALU=False`); treat approach as dead end in current form.

- 2026-02-24: Guided per-round deep-window coordinate search.
- What changed: explored per-round windows around `{6..10:20}` with explicit round-map overrides.
- Why: check if interactions among deep rounds can beat `1081`.
- Result: no improvement; best remained `{6:20,7:20,8:20,9:20,10:20}` at `1081`.
- Decision: keep current map-equivalent default.

- 2026-02-24: Load-relief attempt via preferred flow const materialization.
- What changed: added `PREFER_FLOW_CONST` path to emit non-zero consts via `flow/add_imm` when possible.
- Why: reduce load-engine pressure (`defer_load_full` is dominant).
- Result: regression to `1092` cycles (correctness pass).
- Decision: keep feature flag off by default (`PREFER_FLOW_CONST=False`).

- 2026-02-24: Deep-gate token overhead reduction attempt (vector tokens).
- What changed: added `USE_VECTOR_DEEP_TOKENS` option to reuse deep tree result vectors as gating tokens instead of scalar ALU token ops.
- Why: remove token-op overhead while preserving selective deep gating.
- Result: deadlock/incorrect (`~385` cycles reported, invalid output) when enabled.
- Decision: keep disabled (`USE_VECTOR_DEEP_TOKENS=False`); scalar tokens remain required for correctness/stability.

- 2026-02-24: In-place vector dataflow retry with allocator ownership transfer.
- What changed: added `ENABLE_INPLACE_VECTOR_DATAFLOW` path plus allocator ownership-transfer tracking (`vbase_owner` / `owner_vbase`) for full-vector write_to chains.
- Why: try to reduce scratch pressure and improve schedule freedom without semantic alias bugs.
- Result: still deadlocked/incorrect (`~123` cycles reported, invalid output) when enabled.
- Decision: keep disabled (`ENABLE_INPLACE_VECTOR_DATAFLOW=False`); approach still unresolved.

- 2026-02-24: Structural neighborhood check for depth-3/depth-4 vselect masks.
- What changed: evaluated single-group flips around current depth-3 and depth-4 mask sets under the `1081` configuration.
- Why: verify local optimality of current scatter/vselect partition after new gating changes.
- Result: no single flip improved beyond `1081`.
- Decision: keep current mask sets unchanged.

- 2026-02-24: Attempted round-major emission architecture.
- What changed: emitted rounds across all groups (experimental path) while preserving dependency correctness.
- Why: reshape schedule critical path and improve engine overlap.
- Result: correctness pass but major regression (`1261` cycles).
- Decision: reverted round-major path; keep group-major emission.

- 2026-02-24: Targeted depth-3 vselect alignment experiment (group-specific).
- What changed: forced depth-3 vselect on groups that were scatter in baseline (`g=11`, then `{11,23}`) to trade deep scatter loads for flow/valu work.
- Why: reduce load pressure at depth-3/late rounds where load is the dominant defer reason.
- Result: regressions (`g=11 -> 1093`, `{11,23} -> 1118`), correctness passed.
- Decision: revert/keep original depth-3 selection mask.

- 2026-02-24: Strided `group_offset` dependency chain experiment.
- What changed: replaced fully serial `group_offset[g]=group_offset[g-1]+V` with a strided chain to reduce setup critical depth.
- Why: make high-group load addresses available earlier without changing semantics.
- Result: no cycle gain (`1082`), +1 op overhead in implementation variant.
- Decision: reverted to original chain for simplicity.

- 2026-02-24: Re-audited slot lower bounds using final emitted bundles (not cycle telemetry).
- What changed: compared scheduler telemetry totals against direct counts from `kb.instrs` slot tuples.
- Why: verify true lower bounds before selecting next architecture direction.
- Result: discovered telemetry undercounts ALU due retroactive split-offload insertion into past bundles; corrected ALU total from `11178` to `12151` in current 1081 config.
- Decision: keep current code, but treat prior ALU lower-bound analysis as superseded. Future planning now targets ALU+VALU+LOAD reductions together.

- 2026-02-24: Compile-time header-pointer specialization attempt (`forest_values_p` / `inp_values_p`).
- What changed: replaced setup header loads (`mem[4]`, `mem[6]`) with constants derived from `build_mem_image` layout.
- Why: reduce load slots by removing two scalar memory loads in setup.
- Result: correctness passed but large regression to `1094` cycles; peak scratch increased from `1380` to `1474`.
- Decision: reverted. Treat as dead end due liveness/scheduling side effects outweighing load-slot savings.

- 2026-02-24: Added depth-5 selective vselect architecture path (configurable group mask + diff pairs).
- What changed: introduced `DEPTH_5_VSELECT_GROUPS`, `DEPTH_5_VSELECT_DIFF_PAIRS`, depth-5 setup vectors/diffs, and depth-5 tree branch in `_emit_tree_xor`.
- Why: attempt a structural reduction of deep scatter-load pressure.
- Result:
- Framework is baseline-neutral when disabled (`1081`, pass).
- Single-group depth-5 vselect (`diff_pairs=0`) correctness/pacing is highly group-sensitive: many placements trigger scheduler-stuck invalid outputs (e.g. reported `~385-484` cycles), and all correctness-valid placements regress (`~1093-1143`).
- Decision: keep framework off by default (`DEPTH_5_VSELECT_GROUPS = set()`); treat current depth-5 vselect direction as a dead end for cycle reduction.

- 2026-02-24: Offload-policy architecture audit (global and op-class selective).
- What changed: evaluated proactive/deferred offload modes and selective ALU-offload eligibility classes (`hash-only`, `no_hash`, `no_branch_index`, etc.).
- Why: validate whether sub-1000 requires a different offload architecture rather than more tree changes.
- Result: current default remained best at `1081`; alternatives regressed (`1086-1279`).
- Decision: keep current offload policy (`PROACTIVE_VALU_OFFLOAD=True`, `DEFER_OFFLOADABLE_VALU=False`) and default op eligibility.

- 2026-02-24: Dead branch-bit emission analysis and implementation.
- What changed: added future bit-use analysis (`_tree_consumed_bits`, `compute_future_bit_needed`) to skip `%2` branch-bit ops when neither future tree selection nor index update needs them.
- Why: remove dead vector `%` work and reduce VALU/ALU slots.
- Result: no branch-bit ops were actually dead in current baseline shape; emitted ops and cycles unchanged (`12425 ops`, `1081`).
- Decision: keep the analysis scaffolding (correct and safe), but no immediate performance gain.

- 2026-02-24: Retroactive ALU split completion-cycle propagation.
- What changed: `_try_split_alu_offload` now returns effective completion cycle and successor earliest times use that completion cycle.
- Why: avoid conservative latency when split-offloaded work finishes in earlier bundles.
- Result: no measurable change (`1081`) because no offloaded op completed strictly in past-only bundles under current schedule (`past_only = 0` observed).
- Decision: keep as correctness-preserving scheduler improvement, but currently performance-neutral.

- 2026-02-24: Scatter combine redesign (`SCATTER_VECTOR_XOR`).
- What changed: experimental scatter path that loads 8 node values into a vector then applies one vector XOR instead of 8 scalar ALU XORs.
- Why: reduce hard-wired ALU pressure and make scatter combine schedulable on VALU/ALU-offload.
- Result:
- In-place overwrite variant (`write_to=scatter`) caused invalid/stuck schedules.
- Safe non-inplace variant was correct but regressed to `1093` cycles.
- Decision: keep feature flag off by default (`SCATTER_VECTOR_XOR=False`); treat current form as dead end.

- 2026-02-24: Full depth-5 vselect conversion sweep (all groups, varied diff pairs and bit order).
- What changed: enabled `DEPTH_5_VSELECT_GROUPS = all groups`, swept `DEPTH_5_VSELECT_DIFF_PAIRS` and `REVERSE_TREE_BIT_ORDER_DEPTH_5`.
- Why: check whether a true deep architectural swap could break into sub-1000-like regime.
- Result: all tested configurations were invalid/stuck (very low bogus cycle counts such as `~53-199`, correctness failed).
- Decision: treat full depth-5 vselect conversion as an unstable dead end with current dependency/allocation model.

- 2026-02-24: Partial `SCATTER_VECTOR_XOR` by depth/group.
- What changed: added `SCATTER_VECTOR_XOR_DEPTHS` and `SCATTER_VECTOR_XOR_GROUPS` filters and tested targeted activation.
- Why: see if selective ALU->VALU trade helps without global regression.
- Result:
- Best selective cases were neutral or regressive (`depth=5` all groups `1083`; many single-group depth5 masks `1081`; no case beat `1081`).
- Wider depth sets regressed (`1089-1096` range).
- Decision: keep filters as experiment scaffolding but leave feature off by default.

- 2026-02-24: Deep-gate token reuse from existing scatter lane op.
- What changed: replaced explicit scalar deep-round token op with dependency on existing `*_scat_xor_7` op for depth>=6.
- Why: remove 60 token ops while preserving deep-round gating semantics.
- Result: correctness passed but regressed to `1082` cycles despite lower op count (`12365` ops), indicating token no longer represented full deep-tree completion strongly enough for load smoothing.
- Decision: reverted to explicit scalar token op path; restored `1081`.

- 2026-02-24: Group interleaving architecture attempt (`GROUP_EMIT_ORDER_MODE`).
- What changed: decoupled logical emit order from physical memory group index, with modes `identity`, `light_first`, `heavy_first`.
- Why: test fundamentally different group interleaving to reshape bottleneck overlap (especially depth3/4 scatter pressure).
- Result:
- `identity`: `1081` (pass, baseline).
- `light_first`: `1094` (pass).
- `heavy_first`: `1091` (pass).
- Decision: keep feature only as disabled scaffolding (`identity` default), treat tested reorderings as dead ends.

- 2026-02-24: Web-research guided architecture intake.
- What changed: reviewed public discussion hints (HN thread on this challenge) for sub-1000 strategy patterns.
- Why: identify fundamentally different approaches rather than knob tuning.
- Result: extracted concrete candidate families: stage-5/tree fusion, round-class specialized bundles, speculative preloading from earlier hash stages, and collision-aware/broadcast-oriented depth strategy.
- Decision: follow-up; use these as architecture hypotheses and test in code.

- 2026-02-24: Depth-3/4 leaf-lowering architecture (VALU<->FLOW rebalance).
- What changed: introduced `DEPTH_3_VSELECT_DIFF_PAIRS` and `DEPTH_4_VSELECT_DIFF_PAIRS` to control how many vselect leaves use `multiply_add` vs pure `flow/vselect`.
- Why: trade VALU pressure into FLOW headroom to reduce hard VALU lower bound while preserving correctness.
- Result:
- Broad sweep (`d3 pairs 0..4`, `d4 pairs 0..8`) found multiple correctness-valid points.
- New best: `DEPTH_3_VSELECT_DIFF_PAIRS=3`, `DEPTH_4_VSELECT_DIFF_PAIRS=2` -> `1080` cycles (pass), `12422 ops`.
- Slot totals at new best: `load=2017`, `alu=12108`, `valu=5981`, `flow=794`.
- Some low-pair settings caused scheduler-stuck invalid outputs (bogus ~`438-503` cycle reports).
- Decision: keep new default (`DEPTH_4_VSELECT_DIFF_PAIRS=2`), establishing `1080` as the new baseline.

- 2026-02-24: Selective stage-5 fusion windowing (round-aware skip set).
- What changed: added `STAGE5_FUSION_SKIP_ROUNDS` and rewired build logic so `biased_prev` and `include_stage5_const` are driven by per-round skip decisions rather than all non-final rounds.
- Why: preserve only the potentially beneficial subset of stage-5 fusion without paying full global-fusion cost.
- Result:
- Late-round skip test (`[10,11,12,13,14]`) was correctness-valid only at reduced `MAX_CONCURRENT_GROUPS` and regressed badly (`1256-1303`).
- At default group concurrency it often became scheduler-stuck/invalid (bogus ~`456-580` cycle outputs).
- Decision: keep framework for future targeted experiments, but keep `ENABLE_STAGE5_TREE_FUSION=False` in baseline.

- 2026-02-24: Constant materialization engine experiment.
- What changed: added `CONST_NONZERO_ENGINE` with `flow` mode (`add_imm` from zero) to shift scalar const generation from load to flow.
- Why: cut load slots toward the `load <= 2000` lower-bound requirement.
- Result: load dropped `2017 -> 2006`, flow increased `794 -> 805`, but cycle regressed to `1092`.
- Decision: keep feature flag as scaffolding (`CONST_NONZERO_ENGINE='load'` baseline).

- 2026-02-24: Depth-3 all-group vselect architecture test.
- What changed: added explicit depth mask overrides (`DEPTH_3_VSELECT_GROUPS_OVERRIDE`, `DEPTH_4_VSELECT_GROUPS_OVERRIDE`) and tested full depth-3 vselect activation (all groups) with varied windows/concurrency.
- Why: push deeper load reduction in line with slot-bound model.
- Result:
- Many configurations became scheduler-stuck at default concurrency.
- Correctness-valid configurations required reduced concurrency and regressed heavily (`~1288-1338` cycles best).
- Decision: dead end in current scheduler/allocation regime.

- 2026-02-24: Offload policy for scatter vector-xor combine.
- What changed: added `DISABLE_OFFLOAD_SCATTER_VEC_XOR` to keep `*_scat_xor_vec` on VALU when desired.
- Why: prevent proactive ALU split from canceling intended ALU relief of selective scatter-vector combine.
- Result:
- Confirmed expected static-op delta exists (`-144 ALU`, `+18 VALU` in op graph for one selective case).
- In scheduled bundles, dynamic offload rebalancing still erased most gains; best measured cycle remained regressive (`1088+`).
- Decision: keep knob as analysis/control scaffolding; not baseline-worthy yet.

- 2026-02-24: Round-specific depth-4 vselect overrides.
- What changed: added `DEPTH_3_VSELECT_GROUPS_BY_ROUND` / `DEPTH_4_VSELECT_GROUPS_BY_ROUND` and plumbed round-aware checks through future-use analysis + tree emission.
- Why: enable per-round structural shifts (e.g., round-4 vs round-15 asymmetry) without globally changing depth masks.
- Result:
- Several single-group round-specific flips were correctness-valid.
- Best remained a tie at `1080` (e.g., round-4 add `g=17`), while most variants regressed (`1082-1138`).
- Decision: keep round-specific override framework; no baseline change.

- 2026-02-24: Selective offload-family sweeps (tokenized blocklists).
- What changed: profiled offloaded op families and tested selective ALU-offload blocking across hash/tree op categories (`_hash5_combine`, `_hash1_b`, `tree_xor`, `branch_bit`, etc.).
- Why: reduce ALU hard lower bound without reintroducing load pressure.
- Result:
- Multiple policies reduced ALU materially (for example, blocking `_hash5_combine` gave `alu=11788`) but all correctness-valid runs stayed `>=1080`.
- Best remained a tie at `1080` (e.g., blocking `_hash1_b`).
- Decision: keep as dead-end for now; no default policy change.

- 2026-02-24: Deep-round window stochastic retune on 1080 baseline.
- What changed: randomized search over explicit round windows for rounds `5..10` with threshold fallback disabled.
- Why: verify if revised architecture changed the old local optimum of deep-round gating.
- Result: many maps tied at `1080` but none improved below it.
- Decision: keep current simple default behavior; no map override promoted.

- 2026-02-24: Exhaustive single-group depth-5 vselect re-evaluation.
- What changed: swept all depth-5 single-group placements across `DEPTH_5_VSELECT_DIFF_PAIRS` and bit-order modes under updated baseline.
- Why: re-check prior dead-end after new 1080 architecture changes.
- Result: best correctness-valid case regressed to `1086`; no placement beat baseline.
- Decision: keep depth-5 vselect disabled by default.

- 2026-02-24: Round-specific depth-3/depth-4 diff-pair architecture.
- What changed:
- Added `DEPTH_3_VSELECT_DIFF_PAIRS_BY_ROUND` / `DEPTH_4_VSELECT_DIFF_PAIRS_BY_ROUND`.
- Extended setup/emission so depth-3/4 trees can use asymmetric per-round leaf-lowering while preserving correctness.
- Why: depth `3` and `4` appear in two distinct rounds each (`3/14` and `4/15`) with different schedule pressure; global diff-pair constants were too rigid.
- Result:
- Exhaustive sweep over `(d3_r3,d3_r14,d4_r4,d4_r15)` found a new best:
- `(4,3,3,1)` -> `1079` cycles (full submission tests pass), `12431 ops`.
- New slot totals: `load=2017`, `alu=12142`, `valu=6008`, `flow=770`.
- Decision: keep as new default (`DEPTH_3_VSELECT_DIFF_PAIRS_BY_ROUND={3:4,14:3}`, `DEPTH_4_VSELECT_DIFF_PAIRS_BY_ROUND={4:3,15:1}`), establishing `1079` baseline.

- 2026-02-24: Full round-aware interleaving windowing framework (round end-gates).
- What changed:
- Added `ROUND_GATE_WINDOW_BY_ROUND` / `ROUND_GATE_WINDOW_BY_DEPTH`.
- Added `_round_gate_window()` and integrated round-start deps with round-end scalar tokens (`*_round_gate_token`) so selected rounds can cap in-flight groups independently of deep-tree-only gating.
- `_emit_index_update()` now returns emitted refs so round-tail token deps can include branch/index updates when present.
- Why: deep-only round gating was too narrow; this adds a true round-aware interleaving architecture to test round/depth-specific windows.
- Result: framework is baseline-neutral when disabled; `python3 tests/submission_tests.py` still passes at `1079` cycles (`12431 ops`, peak scratch `1499/1536`).
- Decision: keep as experimentation scaffolding and run targeted round-window searches.

- 2026-02-24: Depth-targeted round-window sweeps (full-round gates).
- What changed: swept `ROUND_GATE_WINDOW_BY_DEPTH` across depth-3/4/5 and multi-depth sets with varying windows.
- Why: test whether broad depth-class interleaving caps can improve bottleneck overlap.
- Result:
- Many aggressive windows were correctness-invalid/stuck (bogus sub-1000 reports).
- Correctness-valid cases were mostly regressive (`1080+`) and best tied baseline (`1079`) only when windows were effectively loose/no-op (`32`).
- Decision: treat broad depth-window gating as dead end for cycle reduction.

- 2026-02-24: Sparse round-map interleaving search with deep-gate interaction.
- What changed: sampled round-specific maps over rounds `5..10` under both `MAX_CONCURRENT_DEEP_ROUNDS=20` and `0`.
- Why: discover asymmetric interleaving windows impossible with depth-only gating.
- Result:
- With deep gating on (`20`): best remained `1079` (empty map).
- With deep gating off (`0`): found new best `1078` at `ROUND_GATE_WINDOW_BY_ROUND={7:28,9:24,10:28}`.
- New-best stats: `12387 ops`, peak scratch `1456/1536`, slot totals `load=2017`, `alu=12170`, `valu=5999`, `flow=770`.
- Decision: promote sparse round-map gating + deep-gate disable as new baseline architecture.

- 2026-02-24: Local neighborhood sweeps around new `1078` round-map.
- What changed:
- Sampled and partial exhaustive searches around rounds `7/9/10` windows, plus targeted additions on rounds `5/6/8`.
- Compared alternate tied maps (e.g. `{7:24,9:20,10:26}`) and deep-window reintroductions.
- Why: verify whether `1078` is a local optimum and attempt `1077`.
- Result:
- Multiple `1078` ties, no correctness-valid `1077` found.
- Re-enabling deep gating (`>0`) with these maps regressed (`1080+`) or became invalid at aggressive settings.
- Decision: keep `MAX_CONCURRENT_DEEP_ROUNDS=0` and `ROUND_GATE_WINDOW_BY_ROUND={7:28,9:24,10:28}` as default.

- 2026-02-24: Baseline promotion and full validation.
- What changed: set defaults to sparse round-map gating baseline (`MAX_CONCURRENT_DEEP_ROUNDS=0`, `ROUND_GATE_WINDOW_BY_ROUND={7:28,9:24,10:28}`).
- Why: operationalize the new best configuration.
- Result: `python3 tests/submission_tests.py` full pass at `1078` cycles (`12387 ops`).
- Decision: keep as current baseline.

- 2026-02-24: `SCATTER_VECTOR_XOR` retest on 1078 baseline.
- What changed: re-ran selective scatter-vector combine variants with new round-map baseline, including offload suppression.
- Why: verify whether previous ALU-relief path becomes beneficial after round-aware interleaving win.
- Result:
- Best case remained regressive (`1080` at depth-5 only; baseline `1078` unchanged).
- Wider depth activation further regressed (`1081-1084`) despite lower ALU slot totals in some cases.
- Decision: keep `SCATTER_VECTOR_XOR=False`; still a dead end for cycle reduction.

- 2026-02-24: Stage-5 fusion retest on 1078 baseline.
- What changed: re-tested `ENABLE_STAGE5_TREE_FUSION=True` with several sparse late-round skip sets.
- Why: check if new interleaving architecture makes stage-5 fusion viable.
- Result:
- Correctness-valid variants regressed (`1087-1089`), and several skip sets were correctness-invalid.
- Decision: keep `ENABLE_STAGE5_TREE_FUSION=False`; treat as dead end under current architecture.

- 2026-02-24: Joint depth-3/4 diff-pair retune under new round-map baseline (partial exhaustive).
- What changed: began exhaustive sweep of `(d3_r3,d3_r14,d4_r4,d4_r15)` while fixing round-map baseline `{7:28,9:24,10:28}`.
- Why: test whether previous leaf-fusion optimum shifted after interleaving architecture change.
- Result:
- First 1000/2025 combos completed; best reached only `1080` (no improvement vs `1078` baseline).
- Search was stopped after this signal to prioritize larger-architecture paths.
- Decision: currently a likely dead end; revisit only if coupled with a new architecture that changes round pressure distribution.

- 2026-02-24: Single-group depth-5 vselect retest on 1078 baseline.
- What changed: swept all `DEPTH_5_VSELECT_GROUPS={g}` (`g=0..31`) with round-map baseline.
- Why: target the remaining `load` lower-bound gap using minimal depth-5 vselect substitution.
- Result:
- Most groups were correctness-invalid/stuck.
- Correctness-valid groups regressed (`best = 1087` at `g=12`) even though load dropped slightly (`2017 -> 2013` in best valid case).
- Decision: single-group depth-5 vselect remains a dead end.

- 2026-02-24: Multi-group depth-5 vselect (stable-group subset) retest.
- What changed: tested pair/triple combinations among the only single-group-valid set `{1,3,6,7,8,12,15,18,22,25,27}`.
- Why: see if small multi-group conversion can cross the `load <= 2000` threshold while preserving cycle wins.
- Result:
- Best correctness-valid case was heavily regressive (`1114` cycles at groups `(1,3)`), despite larger load drop (`load=2006`).
- Decision: reject this direction for current scheduler/dataflow model.

- 2026-02-24: Full depth-5 vselect with explicit round-5 windowing (new round-gate architecture).
- What changed: enabled `DEPTH_5_VSELECT_GROUPS=all` and swept `ROUND_GATE_WINDOW_BY_ROUND` (including round-5 caps), `DEPTH_5_VSELECT_DIFF_PAIRS`, and depth-5 bit-order.
- Why: test whether the new round-aware end-gate architecture can stabilize previously invalid full depth-5 conversion.
- Result:
- All tested configurations remained correctness-invalid/stuck (198/198 invalid in broad sweep), including aggressive round-5 throttling.
- Very low bogus cycle reports persisted.
- Decision: full depth-5 vselect remains an unstable dead end under current allocator/dependency model.

- 2026-02-24: Global scatter-vector combine + fusion retune.
- What changed: enabled `SCATTER_VECTOR_XOR` globally and retuned depth-3/4 diff-pair settings + scatter offload suppression.
- Why: aggressively cut ALU pressure from scalar scatter XOR path while rebalancing flow/VALU with leaf fusion.
- Result:
- ALU dropped substantially in best variant (`alu=11787`) but VALU rose (`valu=6074`) and best cycle stayed regressive (`1084`).
- Decision: keep `SCATTER_VECTOR_XOR=False` baseline; architecture is not winning in current form.

- 2026-02-24: Immediate group load-address generation architecture.
- What changed:
- Added `USE_IMMEDIATE_GROUP_LOAD_ADDR` to replace serial `group_offset` ALU chain with per-group `flow/add_imm` load-address generation.
- Fixed latent IR bug uncovered by this path: flow scalar writers now infer write-size correctly (`add_imm`/`select` -> scalar write).
- Why: remove setup/startup ALU chain depth and trade into non-bottleneck flow slots.
- Result:
- Correctness-valid but large regression when enabled (`1109` cycles, `12355 ops`).
- Decision: keep feature disabled by default (`USE_IMMEDIATE_GROUP_LOAD_ADDR=False`); keep flow write-size inference fix as correctness/IR hygiene.

- 2026-02-24: Group/path-selective stage-5 fusion architecture.
- What changed:
- Added `STAGE5_FUSION_SELECTIVE_BY_PATH`.
- Reworked fusion control to allow per-group stage-5 const skipping based on next-round scatter/vselect path (instead of only round-global skip sets).
- Why: retain stage-5 fusion wins where next round can absorb const cheaply, avoid scatter compensation overhead elsewhere.
- Result:
- Baseline-neutral with fusion disabled.
- With fusion enabled, selective path produced correctness-invalid outputs (example: `1084` cycles, fail); explicit late-round skip sets remained correctness-valid but regressive (`1089`).
- Decision: keep as disabled experimentation framework only (`ENABLE_STAGE5_TREE_FUSION=False` baseline).

- 2026-02-24: Tail store-address recomputation architecture.
- What changed: added `RECOMPUTE_STORE_ADDR_AT_TAIL` to recompute `vstore` address near group tail instead of keeping initial load address live through all rounds.
- Why: reduce long-lived scalar liveness pressure and allocator contention near scratch limit.
- Result: correctness-valid but regressive (`1081` cycles, `12419 ops`) when enabled.
- Decision: keep disabled by default.

- 2026-02-24: Late-round stage-5 fusion + round-window co-search.
- What changed: sampled round-window maps (`rounds 7..10`) under the only correctness-valid fusion schedule (`STAGE5_FUSION_SKIP_ROUNDS={10,11,12,13,14}`).
- Why: verify whether round-aware interleaving can rescue stage-5 fusion once fusion correctness is constrained.
- Result: best remained regressive (`1084` cycles at empty round-window map).
- Decision: treat current stage-5 fusion path as non-competitive with 1078 baseline.

- 2026-02-25: Interrupted pairwise additive mask search completed.
- What changed: completed exhaustive `(d3@r3 + one excluded group) x (d4@r4 + one excluded group)` (126 combos) around the `1065` baseline masks.
- Why: finish the interrupted structural neighborhood search and verify no missed win from two-step additive interactions.
- Result:
- All 126 combos correctness-valid.
- Best was regressive at `1069` cycles (`g3=11`, `g4=31`).
- Decision: keep existing masks unchanged.

- 2026-02-25: In-place dataflow + tree-bit-order architecture retest.
- What changed: swept `ENABLE_INPLACE_*` vector/tree/hash flags with `REVERSE_TREE_BIT_ORDER_DEPTH_{3,4,5}`.
- Why: check whether dependency-shape/dataflow rewrite can reduce critical path without mask tuning.
- Result:
- Only configurations with all in-place flags disabled remained valid.
- Reversing depth-3/depth-4 bit order regressed heavily (`1129+` / `1163+`); baseline remained best at `1065`.
- Decision: keep in-place vector/tree/hash disabled and bit-order defaults unchanged.

- 2026-02-25: Stage-5 selective fusion correctness root cause and generalized fix attempt.
- What changed:
- Diagnosed original selective-fusion invalidity: partial branch-bit bias history corrupted later scatter/vselect selection.
- Added index-bias tracking and scatter index unbiasing (`idx ^ reversed_mask`) plus a vselect de-bias fallback path under fusion.
- Added bit-mask safety checks to selective skip logic for next-round vselect compatibility.
- Why: recover the previously promising selective-fusion slot profile while preserving correctness.
- Result:
- Correctness was recovered for selective fusion (`ok=True`) but cycle regressed to `1124` with higher op/slot pressure (`12530 ops`).
- Structured/global skip-set sweeps under the corrected fusion framework remained regressive; best seen was `1070` (e.g. skip `{14}`).
- Decision: keep fusion framework as experimental scaffolding only; baseline remains `ENABLE_STAGE5_TREE_FUSION=False`.

- 2026-02-25: Stage-5 fusion schedule sweep (structured windows/prefix/suffix).
- What changed: evaluated curated global `STAGE5_FUSION_SKIP_ROUNDS` sets (prefixes, suffixes, contiguous windows, hand-picked mixes).
- Why: verify whether corrected fusion can beat baseline with a different round schedule.
- Result:
- Best correctness-valid configuration remained regressive (`1070` at skip `{14}`).
- No schedule beat `1065`.
- Decision: reject stage-5 fusion as a current optimization path.

- 2026-02-25: Scheduler offload-policy architecture sweep.
- What changed: swept `CRITICALITY_AWARE_OFFLOAD`, priority cutoff, and deferred-offload knobs around current baseline.
- Why: attempt a structural ALU/VALU rebalance without changing kernel math.
- Result:
- No improvement; best tied baseline (`1065`).
- Most variants regressed (`1069-1268`), with deferral especially harmful.
- Decision: keep offload policy defaults unchanged.

- 2026-02-25: Scheduler priority-weight heuristic sweep.
- What changed: tested broad and focused ranges of `CRITICAL_PATH_SCALE`, `EMIT_ORDER_SCALE`, late-flow cost/threshold, and base flow/load engine costs.
- Why: check if current 1065 is heuristic-limited rather than algorithm-limited.
- Result:
- Best tied baseline (`1065`) at/near current default regime.
- Many alternative weightings regressed materially (`1072-1116+`).
- Decision: keep current scheduler heuristic defaults.

- 2026-02-25: Depth-5 vselect sparse-path retest on 1065 baseline.
- What changed:
- Re-tested full depth-5 activation (`all groups`, varied diff-pairs/reverse order) and sparse single-group activations.
- Why: revisit load-bottleneck reduction after mask/round-window architecture improvements.
- Result:
- Full depth-5 remained invalid/stuck.
- Single-group valid points were all regressive; best `1072` (`group=27`, `diff_pairs=8`).
- Decision: depth-5 vselect remains a dead end under current allocator/scheduler model.

- 2026-02-25: Round-window neighborhood/random map retest at 1065 baseline.
- What changed: random and neighborhood sampling over `ROUND_GATE_WINDOW_BY_ROUND` for rounds `6..11`.
- Why: validate whether 1065 masks changed the round-gating optimum.
- Result:
- No map beat current default; repeated best stayed exactly `{7:28,9:24,10:28}` at `1065`.
- Decision: keep current round-window map unchanged.

- 2026-02-25: Constant-materialization engine path retest.
- What changed: evaluated `PREFER_FLOW_CONST` and `CONST_NONZERO_ENGINE=flow`, including co-tuning depth-3/4 diff-pairs.
- Why: reduce load-engine pressure (currently `2001`) to push below `2000` without deep-path changes.
- Result:
- `PREFER_FLOW_CONST=True` reduced load to `2000` but regressed (`1070` best local `1069` with diff retune).
- `CONST_NONZERO_ENGINE=flow` reduced load to `1990` but still regressed (`1070` best after diff retune).
- Decision: keep baseline constant strategy (`CONST_NONZERO_ENGINE=\"load\"`, `PREFER_FLOW_CONST=False`).

- 2026-02-25: Diff-pair architecture re-validation on new baseline.
- What changed:
- Exhaustive local sweeps for `(d3@r3, d4@r4)` and `(d3@r14, d4@r15)` under current 1065 setup.
- Why: ensure no hidden drift after broader architectural changes.
- Result:
- Baseline settings remained optimal:
- `DEPTH_3_VSELECT_DIFF_PAIRS_BY_ROUND[3]=4`
- `DEPTH_4_VSELECT_DIFF_PAIRS_BY_ROUND[4]=3`
- `DEPTH_3_VSELECT_DIFF_PAIRS_BY_ROUND[14]=3`
- `DEPTH_4_VSELECT_DIFF_PAIRS_BY_ROUND[15]=1`
- Decision: keep diff-pair defaults unchanged.

## Active Architectural Plan
- Round-aware interleaving windowing is implemented and enabled selectively:
- `DEEP_ROUND_DEPTH_THRESHOLD=6`
- `MAX_CONCURRENT_DEEP_ROUNDS=0`
- `ROUND_GATE_WINDOW_BY_ROUND={7:28,9:24,10:28}`
- Current validated baseline from submission tests is `1065` cycles (`12375 ops`).
- Follow-up: major slot-count reduction path is needed now (especially VALU/load), not additional gating micro-tuning.
- Priority focus: reduce deep scatter load footprint or reduce hash-stage VALU count with a mathematically valid fusion.

## Next Candidates After Round-Aware Interleaving
- Next High-Value Path (immediate): `Depth-5+ Scatter Pair-Fusion (load+ALU collapse)`.
- Why this is highest value now: current floor is still dominated by deep scatter rounds where we pay `8 scalar loads + 8 scalar XORs` per group/round; scheduler-only and mask-only tuning has plateaued.
- Core idea: replace scalar scatter XOR emission with a pair-fused scatter micro-kernel that computes two lanes together from shared address structure, so deep rounds consume fewer scalar load/alu slots without requiring full depth-5 vselect.
- Implementation direction: introduce a new deep-scatter path that emits lane-pair operations (`(0,1)`, `(2,3)`, `(4,5)`, `(6,7)`) with shared address prep and reduced per-lane scalar XOR fanout, gated by depth/round so only heavy rounds (`5..10`) use it.
- Guardrails: keep exact math/branch semantics unchanged, keep stage-5 fusion disabled, and keep round-window map fixed while validating this path in isolation.
- Success criteria for keeping the path: correctness pass with slot movement in the right direction (`load < 2001` and `alu < 12130`) and cycle below `1065`.
- Dependency-aware ownership model for safe in-place vector writes (current allocator/dependency model was a blocker).
- Hash-stage algebraic reductions beyond existing stage2-3 fusion (primary remaining path for meaningful VALU cuts).
- Deep-depth hybrid selection that does not inflate flow critical path (current depth-5 vselect variants regressed).
- Rework VALU offload from runtime split heuristic into finer IR-level lane ops for better scheduler control.

- 2026-03-05: Current baseline validation on synced `main`.
- What changed: no code changes; ran `python3 tests/submission_tests.py` on the current checked-out tree after confirming local `main` matched `origin/main`.
- Why: measure the real current cycle count before further optimization work.
- Result: correctness pass; `1065` cycles, `12375 ops`, peak scratch `1528/1536`.
- Decision: baseline unchanged; keep current configuration.

- 2026-03-05: Current slot-budget audit and hot-path decomposition.
- What changed: no code changes; profiled the emitted bundles and op families from the current baseline kernel.
- Why: quantify exactly which structural regions still dominate cycle lower bounds before attempting more tuning.
- Result:
- Current final bundle slot totals are `load=2001`, `valu=6010`, `alu=12122`, `flow=785`, so the kernel is still simultaneously load-, VALU-, and ALU-bound.
- Tree scatter remains the largest reducible pool: `1952` scalar scatter loads and `1952` scalar scatter XOR ALU ops total.
- Deep scatter depths `5..10` account for `1536` of those loads and `1536` of those ALU XORs (32 groups across 6 rounds), while residual depth-3/4 scatter accounts for the remaining `416 + 416`.
- Hash remains the other major structural pressure source with `5632` hash-family ops across rounds, so materially lower cycles likely still require a real hash VALU reduction in addition to scatter relief.
- Round-window gating on the current `1065` baseline is still tied to full round-tail completion (`hash` + index-update), even on the load-heavy deep rounds `7/9/10` where the actual intent is to cap only deep-tree/scatter overlap.
- New architectural lead worth testing next: release round-window tokens at tree-completion instead of full round-tail completion so later groups can overlap the trailing hash/index work while keeping the same deep-load concurrency cap.
- Decision: treat scheduler/gating retunes as exhausted for now; next winning paths should target deep-scatter structural reduction first, with hash-stage fusion as the parallel second candidate.

- 2026-03-05: Delayed depth-5 vselect setup architecture.
- What changed:
- Added `DELAY_DEPTH_5_SETUP` and moved depth-5 setup emission into a helper so depth-5 vectors/diffs can be materialized only when the first depth-5 round reaches them.
- Baseline remains neutral when the flag is off (`python3 tests/submission_tests.py` still passes at `1065`).
- Why: prior full depth-5 vselect path looked allocator/scratch-limited because all one-round-only setup refs were ready at cycle 0 and stayed live too early.
- Result:
- Full all-group depth-5 vselect remained correctness-invalid/stuck even with delayed setup.
- Single-group variants became broadly correctness-valid under delayed setup, but all were still regressive.
- Exhaustive single-group sweep best valid point was `1071` cycles at `DEPTH_5_VSELECT_GROUPS={30}`, `DEPTH_5_VSELECT_DIFF_PAIRS=12`, `REVERSE_TREE_BIT_ORDER_DEPTH_5=False`, `DELAY_DEPTH_5_SETUP=True`.
- Additional cross-tuning around the strongest group (`g=27`) still bottomed out at `1071-1072`.
- Decision: keep delayed setup only as disabled scaffolding (`DELAY_DEPTH_5_SETUP=False`); depth-5 vselect remains non-competitive on this branch.

- 2026-03-05: Late shallow setup split for depth-3/depth-4.
- What changed:
- Added `DELAY_LATE_D3D4_SETUP` plus a late-copy setup helper so rounds `14/15` can use duplicated depth-3/depth-4 setup refs emitted near first late use, allowing the early copy to die after round `4`.
- Baseline remains neutral when the flag is off (`1065` cycles, correctness passing).
- Why: current peak scratch is near the limit, and d3/d4 setup refs otherwise stay live across deep rounds `5..10` even though late shallow rounds are far away.
- Result:
- Enabling the split regressed materially instead of helping: `1103` cycles on the plain baseline, `1094` with round-window map removed, and `1090` in the best tested variant with `PREFER_FLOW_CONST=True`.
- Slot pressure also moved the wrong way (`load=2004`, `valu=6031`, `alu=12212` in the direct-on case).
- Decision: keep the feature disabled (`DELAY_LATE_D3D4_SETUP=False`); the added duplicate setup overhead outweighs any liveness benefit.

- 2026-03-05: Bounded mixed config search over surviving scaffolds.
- What changed: ran a 120-candidate bounded search combining nearby round-window maps, constant-materialization modes, selective scatter-vector XOR depths, updated depth-3/4 diff-pair maps, delayed depth-5 single-group variants, and the new late shallow setup split.
- Why: verify that no obvious cross-term among the remaining scaffolds beats baseline before committing to another larger architectural rewrite.
- Result:
- Best valid configuration remained the empty/default config at `1065` cycles (`12375 ops`, slots `load=2001`, `flow=785`, `alu=12122`, `valu=6010`, `store=32`).
- No candidate beat baseline; no new correctness-valid improvement surfaced from this neighborhood.
- Decision: treat current scaffold combinations as exhausted on this branch. Next required path is a genuinely new slot-reducing architecture, most likely a deep-scatter load reduction that does not explode flow/VALU, or a new algebraic hash reduction.

- 2026-03-05: Delayed depth-5 multi-group amortization search.
- What changed: searched delayed depth-5 vselect subset combinations (up to 8 candidate groups, sizes `2..6`) to see whether shared setup amortization can make the new depth-5 rewrite competitive.
- Why: single-group delayed depth-5 results were clearly setup-amortization-limited; multi-group subsets are the first place the rewrite could plausibly beat baseline.
- Result:
- Shared setup did improve the rewrite versus the earlier single-group numbers, but not enough.
- Best found subset was `DEPTH_5_VSELECT_GROUPS={22,30}`, `DEPTH_5_VSELECT_DIFF_PAIRS=16`, `DELAY_DEPTH_5_SETUP=True` with the baseline round-window map, reaching `1076` cycles (correctness passing).
- Best slot profile at that point: `load=1989`, `flow=817`, `alu=12174`, `valu=6068`, `store=32`.
- Decision: keep delayed depth-5 setup as disabled scaffolding only; the amortized rewrite still loses because load savings are overpaid by flow/VALU growth.

- 2026-03-05: Delayed depth-5 + low-VALU d3/d4 rebalance search.
- What changed: searched the new delayed depth-5 rewrite jointly with aggressive round-specific d3/d4 diff-pair reductions and loose round-window maps, focusing on configurations that can move `load` below `2000` and pull `valu` back down.
- Why: the multi-group delayed depth-5 rewrite proved load-saving but VALU/flow-heavy; low-fusion d3/d4 settings are the one existing mechanism that can materially reduce VALU without touching deep scatter semantics.
- Result:
- This family came closest to a new win but still did not beat baseline.
- Best seen configuration was:
- `DEPTH_5_VSELECT_GROUPS={27}`
- `DEPTH_5_VSELECT_DIFF_PAIRS=12`
- `DELAY_DEPTH_5_SETUP=True`
- `ROUND_GATE_WINDOW_BY_ROUND={}`
- `DEPTH_3_VSELECT_DIFF_PAIRS_BY_ROUND={3:4,14:3}`
- `DEPTH_4_VSELECT_DIFF_PAIRS_BY_ROUND={4:1,15:1}`
- Result at that point: `1067` cycles (correctness passing), `12425 ops`, slots `load=1997`, `flow=834`, `alu=12129`, `valu=6023`, `store=32`.
- A focused local search around this point did not improve further; best nearby follow-up seen was still regressive (`1075+`).
- Decision: keep the rewrite disabled by default. This path is now a near miss rather than a winner, but still not baseline-worthy.

- 2026-03-05: Hash temp-reuse rewrite (`REUSE_HASH_COMBINE_TEMPS`).
- What changed: added a new local hash rewrite that reuses freshly-created hash temp vectors (`ha` / `hb`) for the final combine output instead of allocating a third output vector, while leaving full input-buffer in-place overwrites disabled.
- Why: reduce transient vector allocation pressure in the hash hot path without invoking the much riskier full in-place hash dataflow model.
- Result:
- Baseline remains neutral when disabled.
- When enabled, the scheduler still deadlocks/halts incorrectly (`~419` cycles reported, incorrect output), so the current alias/ownership model still cannot support this reuse path safely.
- Decision: keep `REUSE_HASH_COMBINE_TEMPS=False`; treat it as another allocator-model blocker, not a promotable optimization.
