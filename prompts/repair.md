The evaluator rejected your previous partition. Read the exact error
string below and fix ONLY the specific cause — do not redesign the
whole partition unless necessary.

```
{error}
```

# Diagnostic guide (match the error string to the right fix)

- **"working set {N} exceeds capacity {M}"** → some subgraph is too
  ambitious. Easiest fix: split the offending subgraph. If it contains
  MM + PW together, separate the Pointwise into its own subgraph. If
  it contains 2+ MMs, break the chain. Also: shrink
  `tensors_to_retain[i]` if the retained tensor is the one blowing the
  budget.

- **"Grain search could not find a feasible (w, h, k)"** → same root
  cause as "working set exceeds capacity" but from the optimizer's
  side. Same fix: split the subgraph into smaller pieces.

- **"3+ MM chain with k=K requires w == k"** → a subgraph has 3 or
  more MatMuls. Even if they form a legal chain, uniform-K mode
  forces `w = k = K_op` for every MM. Almost always wrong choice —
  break the chain into at most 2 MMs per subgraph.

- **"3+ MM chain with k=K requires all reduction dims == k, but op N
  has K=X != k=Y"** → MMs in the same subgraph have different `K`
  dimensions. Under 3+ MM chain they must all share K. Break into
  smaller subgraphs.

- **"Middle MatMul ... not allowed with split-K"** → you fused three
  or more MMs such that one is both consuming an ephemeral MM output
  AND producing an ephemeral MM input. Middle MMs are only legal at
  full-K, which is almost never feasible. Split the chain.

- **"k={k} exceeds native granularity"** → the granule picker tried a
  k-value larger than the native granule (128). Native applies to all
  three axes (w/h/k). This is usually a symptom of a subgraph whose
  MatMul's K is too large for any fusion fitting the WS budget — split
  the subgraph into smaller pieces.

- **"PW ... → MM ... LHS prologue requires w≥K_consumer"** or
  **"... RHS prologue requires h≥K_consumer"** → you fused a Pointwise
  whose output feeds a MatMul, but the MatMul's K dimension exceeds
  the native granule (max w=h=128). The Pointwise tile cannot cover
  the MatMul's K axis in one spatial tile. Split the Pointwise into
  its own subgraph — the MatMul has to split-K and PW cannot ride the
  per-k-step loop.

- **(legacy — now retracted)** The old "MM+PW fusion not allowed with
  split-K" rule from issue #32 was retracted in #82. MM→PW epilogue
  fusion is legal. You may see legacy caches; ignore that specific
  message if encountered.

- **"Two MMs without chain not allowed with split-K"** → two
  standalone MMs in one subgraph under split-K regime. Split into two
  subgraphs, one MM each.

- **"Split-K mixes chain and non-chain MMs"** → a subgraph has a
  chain (head+tail) AND a standalone MM. The only legal configurations
  under split-K are {1 standalone}, {head + tail chain}, or
  {pointwise-only}. Remove the standalone MM from the chain's
  subgraph.

- **"ephemeral output consumed as both LHS and RHS of downstream
  MMs"** → you fused an MM whose output is simultaneously the LHS
  input of one downstream MM and the RHS input of another, inside the
  same subgraph. Remove one of the consumers from this subgraph.

- **"cannot retain tensor ... neither produced in this subgraph nor
  retained from a prior"** → `tensors_to_retain[i]` lists a tensor id
  that isn't available. Either the subgraph doesn't produce it, or
  you forgot to retain it in the chain of prior subgraphs.

- **"Cannot retain graph input tensor"** → the tensor you tried to
  retain is a graph input (no producer op). Drop it from the retain
  list.

- **"op N consumes tensor T produced by op M which appears at the same
  or later position"** → producer/consumer ordering is wrong inside
  the subgraph. Reorder so every producer comes before its consumer.

- **"non-ephemeral outputs have inconsistent shapes"** → the subgraph
  produces two or more non-ephemeral outputs of different `(W, H)`.
  Split so that each subgraph has a single output shape.

- **"tensor ... would be ephemeral but has external consumer"** →
  you marked a tensor as ephemeral (consumed inside this subgraph,
  not retained, not graph output) but another subgraph also needs it.
  Fix by either (a) adding the tensor to `tensors_to_retain[i]`, or
  (b) including the producer op in every consuming subgraph
  (recomputation).

- **"Op N not scheduled in any subgraph"** → coverage violation. Add
  op N to some subgraph.

- **"subgraphs[i] contains invalid op id X"** → you used an op id
  that does not exist. Op ids are strictly `0..n-1` where `n` is the
  number of operators stated in the user prompt. A common mistake is
  using `n` itself as an id — that is INVALID. Re-verify every integer
  in `subgraphs` is `< n`.

- **"tensors_to_retain must have length N"** or **"length mismatch"**
  → `len(subgraphs)` must equal `len(tensors_to_retain)`. Count both.
  If `subgraphs` has 132 entries, `tensors_to_retain` must also have
  132 entries — pad with `[]` for boundaries where you do not retain
  anything.

- **"tensors_to_retain[i] must be a list"** → every entry of
  `tensors_to_retain` is a list of tensor ids (possibly empty `[]`),
  never a single integer or null.

- **"retained tensor index X out of range"** → you put an id
  `>= num_tensors` (not op count!) in `tensors_to_retain`. Tensor ids
  and op ids are separate — a problem with 103 ops can have many
  more tensors. Check the problem JSON's `widths`/`heights` array
  length for the valid tensor-id range.

- **JSON / schema problems not listed above** → your reply is likely
  structurally broken. Return a single top-level object with exactly
  two keys `subgraphs` and `tensors_to_retain`, both lists of lists of
  integers.

# Output

Return ONLY the corrected JSON object, no markdown fences, no prose:

```json
{
  "subgraphs":         [[op_ids], ...],
  "tensors_to_retain": [[tensor_ids], ...]
}
```
