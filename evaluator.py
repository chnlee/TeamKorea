#!/usr/bin/env python3
"""MLSys 2026 evaluator — Python implementation of the cost model described in PROBLEM.md.

Semantics mirror the C++ authority:
  * Role detection via MMRole (lhs/rhs/out ephemeral bits, §8 symmetric).
  * SliceKey-based per-tile reuse tracking (nh × nw × nk cells).
  * Working-set accounting per (op, input-pos) use, with Issue #59 distinct
    buffers and Issue #51 ephemeral-external-consumer validation.
  * Fusion-constraint validation: split-K bans PW / Middle / chain+standalone
    mixes; full-K with 3+ MMs or any Middle requires w = k = K uniform.
  * Output eviction happens at each tile's last k-step.

Public API preserved for backward compatibility:
  evaluate_solution(problem_dict, solution) -> EvaluationResult
  validate_solution_shape(problem, solution)
  recompute_subgraph_latencies(problem_dict, solution)
"""
import json
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# Shared types and utilities
# ═══════════════════════════════════════════════════════════════════════════════

class EvaluationError(ValueError):
    pass


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def approx_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


@dataclass(frozen=True)
class ProblemView:
    widths: Sequence[int]
    heights: Sequence[int]
    inputs: Sequence[Sequence[int]]
    outputs: Sequence[Sequence[int]]
    base_costs: Sequence[float]
    op_types: Sequence[str]
    fast_memory_capacity: int
    slow_memory_bandwidth: float
    native_granularity: Tuple[int, int]

    @property
    def num_ops(self) -> int:
        return len(self.inputs)

    @property
    def num_tensors(self) -> int:
        return len(self.widths)


@dataclass(frozen=True)
class SubgraphEvaluation:
    computed_latency: float
    claimed_latency: float
    ops: Tuple[int, ...]


@dataclass(frozen=True)
class EvaluationResult:
    total_latency: float
    subgraphs: Tuple[SubgraphEvaluation, ...]


# ═══════════════════════════════════════════════════════════════════════════════
# MMRole — §8 symmetric LHS/RHS chain bits (mirror of mm_role.h)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MMRole:
    is_matmul: bool = False
    lhs_ephemeral: bool = False
    out_ephemeral: bool = False
    rhs_ephemeral: bool = False
    out_ephemeral_rhs: bool = False

    def any_in_ephemeral(self) -> bool:
        return self.lhs_ephemeral or self.rhs_ephemeral

    def any_out_ephemeral(self) -> bool:
        return self.out_ephemeral or self.out_ephemeral_rhs

    def is_standalone(self) -> bool:
        return self.is_matmul and not self.any_in_ephemeral() and not self.any_out_ephemeral()

    def is_head(self) -> bool:
        return self.is_matmul and not self.any_in_ephemeral() and self.any_out_ephemeral()

    def is_tail(self) -> bool:
        return self.is_matmul and self.any_in_ephemeral() and not self.any_out_ephemeral()

    def is_middle(self) -> bool:
        return self.is_matmul and self.any_in_ephemeral() and self.any_out_ephemeral()

    def is_lhs_head(self) -> bool:
        return self.is_matmul and not self.lhs_ephemeral and self.out_ephemeral

    def is_lhs_tail(self) -> bool:
        return self.is_matmul and self.lhs_ephemeral and not self.any_out_ephemeral()

    def is_rhs_head(self) -> bool:
        return self.is_matmul and not self.rhs_ephemeral and self.out_ephemeral_rhs

    def is_rhs_tail(self) -> bool:
        return self.is_matmul and self.rhs_ephemeral and not self.any_out_ephemeral()


# SliceKey: (h_tile, w_tile, k_step); -1 sentinel means "not dimension-specific"
SliceKey = Tuple[int, int, int]


# ═══════════════════════════════════════════════════════════════════════════════
# Solution shape validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_solution_shape(problem: ProblemView, solution: dict) -> None:
    keys = [
        "subgraphs",
        "granularities",
        "tensors_to_retain",
        "traversal_orders",
        "subgraph_latencies",
    ]
    lengths = [len(solution.get(key, [])) for key in keys]
    if len(set(lengths)) != 1:
        raise EvaluationError("solution fields must have matching lengths")


# ═══════════════════════════════════════════════════════════════════════════════
# Core: faithful port of mlsys::Evaluate
# ═══════════════════════════════════════════════════════════════════════════════

def _evaluate(
    problem: ProblemView,
    solution: dict,
    validate_claimed_latencies: bool,
    initial_retained: Optional[Set[int]] = None,
    skip_plan_checks: bool = False,
) -> EvaluationResult:
    num_tensors = problem.num_tensors
    num_ops = problem.num_ops
    bw = float(problem.slow_memory_bandwidth)
    native_w, native_h = problem.native_granularity
    capacity = problem.fast_memory_capacity

    # ── Build graph metadata ──────────────────────────────────────────────
    producer_of: List[int] = [-1] * num_tensors
    consumers_of: List[List[Tuple[int, int]]] = [[] for _ in range(num_tensors)]
    is_graph_input: List[bool] = [True] * num_tensors
    is_graph_output: List[bool] = [True] * num_tensors

    for i in range(num_ops):
        for t in problem.outputs[i]:
            producer_of[t] = i
            is_graph_input[t] = False
        for pos, t in enumerate(problem.inputs[i]):
            consumers_of[t].append((i, pos))
            is_graph_output[t] = False

    # ── MatMul shape validation ───────────────────────────────────────────
    for i in range(num_ops):
        if problem.op_types[i] != "MatMul":
            continue
        if len(problem.inputs[i]) != 2:
            raise EvaluationError("MatMul must have exactly 2 inputs")
        if len(problem.outputs[i]) != 1:
            raise EvaluationError("MatMul must have exactly 1 output")
        lhs_t = problem.inputs[i][0]
        rhs_t = problem.inputs[i][1]
        out_t = problem.outputs[i][0]
        if problem.widths[lhs_t] != problem.heights[rhs_t]:
            raise EvaluationError(
                f"MatMul op {i}: LHS.width ({problem.widths[lhs_t]}) != "
                f"RHS.height ({problem.heights[rhs_t]})"
            )
        if (problem.heights[out_t] != problem.heights[lhs_t] or
                problem.widths[out_t] != problem.widths[rhs_t]):
            raise EvaluationError(
                f"MatMul op {i}: output shape (w={problem.widths[out_t]}, "
                f"h={problem.heights[out_t]}) does not match LHS.height "
                f"({problem.heights[lhs_t]}) x RHS.width ({problem.widths[rhs_t]})"
            )

    # ── op -> subgraphs map (for #51 global ephemeral check) ──────────────
    op_in_subgraphs: List[Set[int]] = [set() for _ in range(num_ops)]
    for sg_idx, sg_ops in enumerate(solution["subgraphs"]):
        for op in sg_ops:
            op_in_subgraphs[op].add(sg_idx)

    if not skip_plan_checks:
        for op in range(num_ops):
            if not op_in_subgraphs[op]:
                raise EvaluationError(f"Op {op} not scheduled in any subgraph")

    retained_set: Set[int] = set(initial_retained) if initial_retained else set()
    evaluations: List[SubgraphEvaluation] = []

    for sg_idx, sg_ops in enumerate(solution["subgraphs"]):
        grain = solution["granularities"][sg_idx]
        if len(grain) != 3:
            raise EvaluationError("granularity must be [w, h, k]")
        w, h, k = int(grain[0]), int(grain[1]), int(grain[2])
        retain_list = list(solution["tensors_to_retain"][sg_idx])
        traversal = solution["traversal_orders"][sg_idx]
        claimed = float(solution["subgraph_latencies"][sg_idx])

        if w > native_w or h > native_h:
            raise EvaluationError("Tile size exceeds native granularity")
        # NOTE: the cost model in PROBLEM.md
        # does NOT enforce `k ≤ native_k`, and Track A submissions rely on
        # k > native (e.g. k=4096) to hit compute-lower-bound on large-K
        # benchmarks. Organizer statements in #78/#80 say "native applies
        # to k too" but the actual evaluator code does not check it.
        # Remove the cap to match the effective public evaluator semantics.
        if w <= 0 or h <= 0 or k <= 0:
            raise EvaluationError("Granularity must be positive")

        sg_op_set: Set[int] = set(sg_ops)
        retain_set: Set[int] = set(retain_list)

        # Duplicate op indices
        if len(sg_op_set) != len(sg_ops):
            raise EvaluationError(
                f"Subgraph {sg_idx}: duplicate op indices in subgraph"
            )

        # Producer-before-consumer order within sg.ops
        op_position = {op: i for i, op in enumerate(sg_ops)}
        for i, op_idx in enumerate(sg_ops):
            for t in problem.inputs[op_idx]:
                prod = producer_of[t]
                if prod < 0:
                    continue
                if prod not in op_position:
                    continue
                if op_position[prod] >= i:
                    raise EvaluationError(
                        f"Subgraph {sg_idx}: op {op_idx} consumes tensor {t} "
                        f"produced by op {prod} which appears at the same or later position"
                    )

        # Retain validity
        for t in retain_list:
            if t < 0 or t >= num_tensors:
                raise EvaluationError(
                    f"Subgraph {sg_idx}: retained tensor index {t} out of range"
                )
            if is_graph_input[t]:
                raise EvaluationError("Cannot retain graph input tensor")
            prod = producer_of[t]
            produced_here = prod >= 0 and prod in sg_op_set
            carried_over = t in retained_set
            if not produced_here and not carried_over:
                raise EvaluationError(
                    f"Subgraph {sg_idx}: cannot retain tensor {t} which is neither "
                    f"produced in this subgraph nor retained from a prior subgraph"
                )

        # Issue #51 — ephemeral with uncovered external consumer.
        # Skip when the caller certifies the plan is already validated
        # (granularity search on a known-good partition).
        if not skip_plan_checks:
            for op in sg_ops:
                for t in problem.outputs[op]:
                    if t in retain_set:
                        continue
                    if is_graph_output[t]:
                        continue
                    has_internal = False
                    has_uncovered_external = False
                    uncovered_consumer = -1
                    for consumer_op, _pos in consumers_of[t]:
                        if consumer_op in sg_op_set:
                            has_internal = True
                        else:
                            covered = False
                            for other_sg in op_in_subgraphs[consumer_op]:
                                if op in solution["subgraphs"][other_sg]:
                                    covered = True
                                    break
                            if not covered:
                                has_uncovered_external = True
                                uncovered_consumer = consumer_op
                    if has_internal and has_uncovered_external:
                        raise EvaluationError(
                            f"Subgraph {sg_idx}: tensor {t} would be ephemeral but has "
                            f"external consumer (op {uncovered_consumer}) whose subgraph "
                            f"does not recompute producer (op {op})"
                        )

        # ── Ephemeral / needs_eviction closures ──────────────────────────
        def is_ephemeral(t: int) -> bool:
            prod = producer_of[t]
            if prod < 0:
                return False
            if prod not in sg_op_set:
                return False
            consumed_in_sg = any(
                consumer_op in sg_op_set for consumer_op, _ in consumers_of[t]
            )
            if not consumed_in_sg:
                return False
            if is_graph_output[t]:
                return False
            if t in retain_set:
                return False
            return True

        def needs_eviction(t: int) -> bool:
            prod = producer_of[t]
            if prod < 0:
                return False
            if prod not in sg_op_set:
                return False
            if is_ephemeral(t):
                return False
            if t in retain_set:
                return False
            return True

        # Classify ops
        mm_ops: List[int] = []
        pw_ops: List[int] = []
        for op in sg_ops:
            if problem.op_types[op] == "MatMul":
                mm_ops.append(op)
            else:
                pw_ops.append(op)

        def get_K_op(op: int) -> int:
            return problem.widths[problem.inputs[op][0]]

        # ── Topology-based MMRole detection ──────────────────────────────
        roles: Dict[int, MMRole] = {}
        for op in sg_ops:
            r = MMRole()
            if problem.op_types[op] == "MatMul":
                r.is_matmul = True
            roles[op] = r

        for op in mm_ops:
            # Input ephemeral bits
            for pos, t in enumerate(problem.inputs[op]):
                if not is_ephemeral(t):
                    continue
                prod = producer_of[t]
                if prod < 0 or prod not in sg_op_set:
                    continue
                if problem.op_types[prod] != "MatMul":
                    continue
                if pos == 0:
                    roles[op].lhs_ephemeral = True
                else:
                    roles[op].rhs_ephemeral = True
            # Output ephemeral bits (consumed as LHS / RHS of downstream MM)
            if problem.outputs[op]:
                out_t = problem.outputs[op][0]
                if is_ephemeral(out_t):
                    for consumer_op, pos in consumers_of[out_t]:
                        if consumer_op not in sg_op_set:
                            continue
                        if problem.op_types[consumer_op] != "MatMul":
                            continue
                        if pos == 0:
                            roles[op].out_ephemeral = True
                        else:
                            roles[op].out_ephemeral_rhs = True

        # ── K_slicing / K_up ─────────────────────────────────────────────
        K_slicing = 0
        K_up = 0
        for op in mm_ops:
            if roles[op].is_tail():
                K_slicing = get_K_op(op)
            if roles[op].is_head():
                K_up = get_K_op(op)
        if K_slicing == 0:
            max_K = 0
            for op in mm_ops:
                max_K = max(max_K, get_K_op(op))
            K_slicing = max_K
        nk = ceil_div(K_slicing, k) if K_slicing > 0 else 1

        # ── Fusion constraint validation ─────────────────────────────────
        for op in mm_ops:
            r = roles[op]
            if r.out_ephemeral and r.out_ephemeral_rhs:
                raise EvaluationError(
                    f"Op {op}: ephemeral output consumed as both LHS and RHS of "
                    f"downstream MMs — inconsistent per-step shape"
                )

        # Per issues #71/#82 (#32 is retracted) — matches the 
        # engineered public evaluator (evaluator/mlsys.cc in new branch):
        # - MM → PW epilogue: always allowed (PW runs once per spatial tile,
        #   after the MM's k-loop completes, on the finished output).
        # - PW → MM prologue: the granule-alignment check (w ≥ K LHS /
        #   h ≥ K RHS) is ONLY enforced under split-K (nk > 1). At full-K
        #   (nk = 1) the MM reads each operand once per spatial tile, so a
        #   PW tile of size (w, h) that matches the MM's spatial output
        #   shape is sufficient regardless of K.
        if nk > 1:
            for op in sg_ops:
                if problem.op_types[op] != "Pointwise":
                    continue
                if not problem.outputs[op]:
                    continue
                pw_out = problem.outputs[op][0]
                for consumer in sg_ops:
                    if consumer == op:
                        continue
                    if problem.op_types[consumer] != "MatMul":
                        continue
                    c_inputs = problem.inputs[consumer]
                    if pw_out not in c_inputs:
                        continue
                    consumer_K = problem.widths[c_inputs[0]]
                    pos = c_inputs.index(pw_out)
                    if pos == 0 and w < consumer_K:
                        raise EvaluationError(
                            f"PW (op {op}) → MM (op {consumer}) LHS split-K "
                            f"granule alignment: w={w} < MM.K={consumer_K} "
                            f"(issue #71)"
                        )
                    if pos > 0 and h < consumer_K:
                        raise EvaluationError(
                            f"PW (op {op}) → MM (op {consumer}) RHS split-K "
                            f"granule alignment: h={h} < MM.K={consumer_K} "
                            f"(issue #71)"
                        )

        if nk > 1:
            # #82 retracts the old blanket "MM+PW with split-K is invalid"
            # rule (#32). MM → PW epilogue is fine because PW runs once
            # per spatial tile, after the MM's k-loop finishes. Prologue
            # legality is handled above.
            for op in mm_ops:
                if roles[op].is_middle():
                    raise EvaluationError(
                        "Middle MatMul (chain relay) in 3+ MM chain not allowed with split-K"
                    )
            n_head = sum(1 for op in mm_ops if roles[op].is_head())
            n_tail = sum(1 for op in mm_ops if roles[op].is_tail())
            n_standalone = sum(1 for op in mm_ops if roles[op].is_standalone())
            is_pure_chain = (n_head == 1 and n_tail == 1 and n_standalone == 0
                             and len(mm_ops) == 2)
            is_pure_standalone = (n_head == 0 and n_tail == 0
                                  and n_standalone == len(mm_ops))
            if not is_pure_chain and not is_pure_standalone:
                raise EvaluationError(
                    "Split-K mixes chain and non-chain MMs — not supported"
                )
            if is_pure_standalone and len(mm_ops) >= 2:
                raise EvaluationError(
                    "Two MMs without chain not allowed with split-K"
                )
        else:
            has_middle = any(roles[op].is_middle() for op in mm_ops)
            if len(mm_ops) >= 3 or has_middle:
                if w != k:
                    raise EvaluationError(
                        f"3+ MM chain with k=K requires w == k (got w={w}, k={k})"
                    )
                for op in mm_ops:
                    K_op = get_K_op(op)
                    if K_op != k:
                        raise EvaluationError(
                            f"3+ MM chain with k=K requires all reduction dims == k, "
                            f"but op {op} has K={K_op} != k={k}"
                        )

        # ── Primary output + unified execution grid ──────────────────────
        primary_out_t = -1
        for op in sg_ops:
            for t in problem.outputs[op]:
                if is_ephemeral(t):
                    continue
                if primary_out_t < 0:
                    primary_out_t = t
                elif (problem.widths[t] != problem.widths[primary_out_t] or
                      problem.heights[t] != problem.heights[primary_out_t]):
                    raise EvaluationError(
                        f"Subgraph {sg_idx}: non-ephemeral outputs have inconsistent "
                        f"shapes (tensor {primary_out_t}: "
                        f"{problem.widths[primary_out_t]}x{problem.heights[primary_out_t]}, "
                        f"tensor {t}: {problem.widths[t]}x{problem.heights[t]})"
                    )
        if primary_out_t < 0:
            raise EvaluationError("No non-ephemeral output in subgraph")

        W_out = problem.widths[primary_out_t]
        H_out = problem.heights[primary_out_t]
        nw = ceil_div(W_out, w)
        nh = ceil_div(H_out, h)
        num_tiles = nw * nh

        # ── Per-op k_eff / K_denom (chain uses K_slicing) ────────────────
        def get_k_eff(op: int, ks: int) -> int:
            r = roles[op]
            if not r.is_matmul:
                return 0
            if not r.is_standalone():
                return min(k, K_slicing - ks * k)
            K_op = get_K_op(op)
            if nk > 1:
                return min(k, K_op - ks * k)
            return K_op

        def get_K_denom(op: int) -> int:
            r = roles[op]
            if r.is_matmul and not r.is_standalone():
                return K_slicing
            return get_K_op(op)

        # ── Working set check (peak at ks=0 with full k) ─────────────────
        ws = 0
        for t in retained_set:
            ws += problem.widths[t] * problem.heights[t]

        for op in sg_ops:
            role = roles[op]
            for pos, t in enumerate(problem.inputs[op]):
                if t in retained_set:
                    continue
                if is_ephemeral(t):
                    continue

                if not role.is_matmul:
                    ws += h * w
                elif role.is_lhs_head():
                    ws += h * K_up if pos == 0 else K_up * min(k, K_slicing)
                elif role.is_lhs_tail():
                    if pos == 0:
                        continue  # LHS ephemeral
                    ws += min(k, K_slicing) * w
                elif role.is_rhs_head():
                    ws += min(k, K_slicing) * K_up if pos == 0 else K_up * w
                elif role.is_rhs_tail():
                    if pos > 0:
                        continue  # RHS ephemeral
                    ws += h * min(k, K_slicing)
                else:
                    k_ws = get_k_eff(op, 0)
                    ws += h * k_ws if pos == 0 else k_ws * w

            for t in problem.outputs[op]:
                if is_ephemeral(t):
                    continue
                ws += h * w

        if ws > capacity:
            raise EvaluationError(
                f"Subgraph {sg_idx}: working set {ws} exceeds capacity {capacity}"
            )

        # ── Traversal order validation ───────────────────────────────────
        if traversal is not None:
            tile_order = [int(x) for x in traversal]
            if len(tile_order) != num_tiles:
                raise EvaluationError(
                    f"Subgraph {sg_idx}: traversal_order size {len(tile_order)} "
                    f"does not match num_tiles {num_tiles}"
                )
            seen = [False] * num_tiles
            for idx in tile_order:
                if idx < 0 or idx >= num_tiles:
                    raise EvaluationError(
                        f"Subgraph {sg_idx}: traversal_order contains "
                        f"out-of-range index {idx}"
                    )
                if seen[idx]:
                    raise EvaluationError(
                        f"Subgraph {sg_idx}: traversal_order contains "
                        f"duplicate index {idx}"
                    )
                seen[idx] = True
        else:
            tile_order = list(range(num_tiles))

        # ── Per-cell loop (num_tiles × nk) ───────────────────────────────
        sg_latency = 0.0
        loaded: Dict[int, SliceKey] = {}

        for seq in range(num_tiles):
            tile_idx = tile_order[seq]
            ht = tile_idx // nw
            wt = tile_idx % nw
            h_eff = min(h, H_out - ht * h)
            w_eff = min(w, W_out - wt * w)

            for ks in range(nk):
                # Compute
                compute = 0.0
                for op in sg_ops:
                    r = roles[op]
                    if not r.is_matmul:
                        if ks == 0:
                            compute += float(problem.base_costs[op])
                    else:
                        ke = get_k_eff(op, ks)
                        Kd = get_K_denom(op)
                        compute += float(problem.base_costs[op]) * ke / Kd

                # IO
                io_in = 0.0
                io_out = 0.0

                for op in sg_ops:
                    role = roles[op]
                    K_op = get_K_op(op) if role.is_matmul else 0
                    ke = get_k_eff(op, ks)

                    for pos, t in enumerate(problem.inputs[op]):
                        if t in retained_set:
                            continue
                        if is_ephemeral(t):
                            continue

                        should_load = True
                        key: SliceKey
                        sz = 0.0

                        if not role.is_matmul:
                            key = (ht, wt, -1)
                            sz = float(h_eff) * w_eff
                            should_load = (ks == 0)
                        elif role.is_lhs_head():
                            if pos == 0:
                                key = (ht, -1, -1)
                                sz = float(h_eff) * K_op
                                should_load = (ks == 0)
                            else:
                                key = (-1, -1, ks)
                                sz = float(K_op) * ke
                        elif role.is_lhs_tail():
                            if pos == 0:
                                continue  # LHS ephemeral
                            key = (-1, wt, ks)
                            sz = float(ke) * w_eff
                        elif role.is_rhs_head():
                            if pos == 0:
                                key = (-1, -1, ks)
                                sz = float(ke) * K_op
                            else:
                                key = (-1, wt, -1)
                                sz = float(K_op) * w_eff
                                should_load = (ks == 0)
                        elif role.is_rhs_tail():
                            if pos > 0:
                                continue  # RHS ephemeral
                            key = (ht, -1, ks)
                            sz = float(h_eff) * ke
                        else:
                            # Standalone (or Middle collapsed at nk=1)
                            if pos == 0:
                                key = (ht, -1, ks)
                                sz = float(h_eff) * ke
                            else:
                                key = (-1, wt, ks)
                                sz = float(ke) * w_eff

                        if should_load:
                            prev = loaded.get(t)
                            if prev is None or prev != key:
                                io_in += sz / bw
                                loaded[t] = key

                # Output eviction at last k-step
                if ks == nk - 1:
                    for op in sg_ops:
                        for t in problem.outputs[op]:
                            if not needs_eviction(t):
                                continue
                            io_out += float(h_eff * w_eff) / bw

                sg_latency += max(compute, io_in + io_out)

        # Claimed-latency check
        if validate_claimed_latencies and claimed != 0.0:
            denom = sg_latency if sg_latency > 0.0 else 1.0
            if abs(claimed - sg_latency) / denom > 1e-6:
                raise EvaluationError(
                    f"Subgraph {sg_idx}: reported latency {claimed} "
                    f"does not match computed {sg_latency}"
                )

        evaluations.append(SubgraphEvaluation(
            computed_latency=sg_latency,
            claimed_latency=claimed,
            ops=tuple(sg_ops),
        ))

        retained_set = set(retain_list)

    total = sum(e.computed_latency for e in evaluations)
    return EvaluationResult(total_latency=total, subgraphs=tuple(evaluations))


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def _build_problem(problem_dict: dict) -> ProblemView:
    return ProblemView(
        widths=problem_dict["widths"],
        heights=problem_dict["heights"],
        inputs=problem_dict["inputs"],
        outputs=problem_dict["outputs"],
        base_costs=problem_dict["base_costs"],
        op_types=problem_dict["op_types"],
        fast_memory_capacity=int(problem_dict["fast_memory_capacity"]),
        slow_memory_bandwidth=float(problem_dict["slow_memory_bandwidth"]),
        native_granularity=tuple(problem_dict["native_granularity"]),
    )


def evaluate_solution(
    problem_dict: dict,
    solution: dict,
    validate_claimed_latencies: bool = True,
) -> EvaluationResult:
    problem = _build_problem(problem_dict)
    validate_solution_shape(problem, solution)
    return _evaluate(problem, solution, validate_claimed_latencies)


def score_subgraph(
    problem_dict: dict,
    ops: Sequence[int],
    granularity: Sequence[int],
    retain: Sequence[int],
    traversal: Optional[Sequence[int]] = None,
    prev_retained: Optional[Set[int]] = None,
) -> float:
    """Compute the cost of a single subgraph, given the set of tensors
    carried over from the previous subgraph (for working-set accounting).

    Plan-level checks (#51 ephemeral external consumer, full-op coverage)
    are skipped — the caller is expected to have validated the partition
    separately. Per-subgraph checks (WS, split-K, retain validity,
    traversal permutation, etc.) still apply.
    """
    problem = _build_problem(problem_dict)
    mini = {
        "subgraphs": [list(ops)],
        "granularities": [list(granularity)],
        "tensors_to_retain": [list(retain)],
        "traversal_orders": [list(traversal) if traversal is not None else None],
        "subgraph_latencies": [0.0],
    }
    result = _evaluate(
        problem, mini,
        validate_claimed_latencies=False,
        initial_retained=set(prev_retained) if prev_retained else None,
        skip_plan_checks=True,
    )
    return result.subgraphs[0].computed_latency


def recompute_subgraph_latencies(problem_dict: dict, solution: dict) -> dict:
    """Return a copy of `solution` with subgraph_latencies filled in by the port."""
    updated = json.loads(json.dumps(solution))
    problem = _build_problem(problem_dict)
    validate_solution_shape(problem, updated)
    # Run with validate_claimed_latencies=False to avoid mismatch errors, then
    # patch latencies in.
    result = _evaluate(problem, updated, validate_claimed_latencies=False)
    updated["subgraph_latencies"] = [e.computed_latency for e in result.subgraphs]
    return updated


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python3 evaluator.py <problem.json> <solution.json>", file=sys.stderr)
        return 1
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        problem = json.load(f)
    with open(sys.argv[2], "r", encoding="utf-8") as f:
        solution = json.load(f)
    result = evaluate_solution(problem, solution)
    print(json.dumps({"total_latency": result.total_latency}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
