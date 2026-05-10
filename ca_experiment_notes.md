# CA Transformer Experiment: Questions, Methods, and Notes

---

## What We Are Trying To Do

The core goal is to move from observing that a model's loss decreases to understanding **what specific computation each training update strengthens**. Given a model at step `t` and a batch `B`, we want to say something precise about whether the update improved routing, payload extraction, or local computation — and whether it generalized or memorized.

Cellular automata are the first testbed because the ground truth dependency structure is completely known. For a radius-1 binary CA, every output cell is determined by exactly three input cells:

```
y[i] = f(x[i-1], x[i], x[i+1])
```

This means we can define a "correct" internal structure for any trained model and measure the gap between what the model learned and what it should have learned, at the level of attention routing, value extraction, and MLP computation — not just final accuracy.

The three-part decomposition is:

```
QK   →   where to look      (routing)
V    →   what to read       (payload extraction)
MLP  →   what to compute    (local rule application)
```

Each of these can succeed or fail independently, and each leaves a distinct diagnostic signature.

---

## Why Cellular Automata

- The full local rule is known for every neighborhood
- There are only 8 possible local neighborhoods for binary radius-1 CA, so rule learning can be fully audited per-pattern
- The correct attention structure is known exactly: output cell `i` should depend on input offsets `{-1, 0, +1}`, or a strict subset for copy rules
- Data can be controlled precisely along rule type, neighborhood frequency, row length, boundary condition, and curriculum order
- Length generalization is a clean binary test: a model that learned relative-position routing should transfer to longer rows; one that memorized absolute positions should not

---

## Current Model and Format

Architecture: small decoder-only transformer (1–2 layers, 1–4 heads, d_model 64–128), custom implementation with explicit Q/K/V caches exposed at every layer.

Token format:
```
<BOS> <X> x[0] x[1] ... x[L-1] <SEP> <Y> y[0] y[1] ... y[L-1] <EOS>
```

Loss is applied only on the y-token positions. The model is trained next-token LM style; the y-bit logit at position `<Y>+i` predicts `y[i]`.

Key position mappings (row length L):
- x-token positions: `[2, ..., L+1]` → used as attention key positions
- y-prediction positions (pred_pos): `[L+3, ..., 2L+2]` → used as attention query positions
- Cell offset in QK analysis: `key_cell_index - query_cell_index`, so offset 0 = same cell, −1 = left neighbor, +1 = right neighbor

Rules currently in scope: `LEFT_COPY`, `CENTER_COPY`, `RIGHT_COPY`, `MAJORITY`, `XOR`, `RULE_110`.

---

## Questions and Hypotheses

### A. Routing (QK)

**Q-R1. Does QK learn relative offsets or absolute positions?**
A generalizing model should learn compatibility as a function of `j - i`, not of `(i, j)` separately. The test: train on length 16, evaluate on lengths 32 and 64. Relative-position QK should transfer; absolute-position QK should not. The cleaner test is to extract the QK score matrix over position-embedding pairs alone and check whether it is Toeplitz (diagonal-constant), which would confirm offset-based routing directly without relying on attention patterns.

**Q-R2. Does QK locality form before full rule competence?**
The model may first learn to attend to the local neighborhood before learning the Boolean function over it. Observable signature: QK true-offset margin rises while per-neighborhood output accuracy is still low. This would establish a temporal ordering — routing before computation — and explain why early training sometimes shows correct attention but wrong outputs.

**Q-R3. For copy rules, does QK become strictly offset-specific?**
`LEFT_COPY` should produce a single sharp peak at offset −1. `CENTER_COPY` at offset 0. `RIGHT_COPY` at offset +1. Any attention mass at other offsets is dead weight. Testing this across multiple seeds and checking that the offset peak sharpens over training is the cleanest possible QK experiment.

**Q-R4. Do heads specialize to different offsets in multi-head models?**
For a 3-neighbor rule with 2 or 3 heads, the heads may divide offset coverage (specialization) or redundantly attend to the same cells (redundancy). These are qualitatively different failure modes for capacity. The diagnostic: cross-head KL divergence between attention distributions. High divergence = specialization. Near-zero divergence = redundancy or dominance. The prediction: redundancy is more likely when `n_heads < 3` for 3-neighbor rules, and specialization emerges more cleanly with `n_heads = 3`.

**Q-R5. Does skewed data produce center-cell over-attention?**
Training on smooth rows (long constant regions) creates many `000` and `111` neighborhoods where center-cell prediction is correct by persistence. The model may learn to over-attend to offset 0 as a shortcut. This should be detectable as high center-offset attention mass on balanced evaluation sets even when it produces wrong outputs on minority neighborhoods.

**Q-R6. What does the attention sink at structural tokens look like, and does it compete with useful routing?**
Structural tokens (`<X>`, `<SEP>`, `<Y>`) are always present and carry no CA-relevant information. Attention mass allocated to them is unavailable for the true neighbors. Track structural-token attention mass separately from x-token attention over training. The specific question: does structural-token sink attention correlate with routing failure, and does it behave differently at boundary y-positions (where one true neighbor doesn't exist) vs interior positions? A model that parks residual attention at `<SEP>` specifically for boundary cells may be handling the missing neighbor correctly or may be failing silently.

**Q-R7. Is head specialization to offsets stable once formed, or does it drift?**
Once one head locks onto offset −1 and another onto offset +1, does this specialization persist through further training, or do heads drift and reassign roles? For copy rules the answer is probably stable, but for complex rules like RULE_110 where different heads may need to cooperate on an asymmetric function, role stability is less obvious. Track cross-head KL divergence over the full training run, not just at the end.

---

### B. Payload Extraction (V)

**Q-V1. Is V payload failure separable from routing failure?**
A model may attend correctly (QK points at the right cells) while still failing because V does not cleanly extract the cell state. The signature: high true-offset attention mass, low V bit-state probe accuracy, poor output accuracy. This three-way dissociation — attention correct, probe poor, output poor — is the clearest evidence for a V-level bottleneck.

**Q-V2. Does V bit-state decodability rise before, during, or after routing formation?**
The ordering relative to QK margin improvement is the key question. One plausible story: V learns to encode cell state early (because it's a simple function of the token embedding) and QK then learns to select the right V. Another: QK sharpens first, creating pressure for V to carry more useful content. The training curves of V probe accuracy vs QK true-margin tell this story.

**Q-V3. Do V vectors encode cell role separately from cell state?**
Cell state = 0 or 1. Cell role = which neighbor position this cell serves as for nearby predictions (left neighbor for `i+1`, center for `i`, right for `i-1`). A fully general circuit should encode state and role independently. A memorizing circuit might encode entangled representations like "left-neighbor-that-is-1." The probe: at x-token positions, probe V for state (binary) and for which y-prediction position is most likely to use this cell. A model with clean disentanglement should encode state but not role in V.

**Q-V4. Does V quality predict output accuracy once routing is solved?**
After QK true-offset margin saturates, further accuracy improvements should be more correlated with V probe accuracy than with QK metrics. This is a testable temporal claim: compute the partial correlation of accuracy gain with V probe gain vs QK gain, separately in the early and late training phases.

---

### C. Local Computation (MLP)

**Q-M1. Does MLP learn the full rule table or a partial version?**
With 8 possible neighborhoods, the model could learn all 8 mappings uniformly or learn frequent ones first and rare ones last (or never). Per-neighborhood accuracy over training reveals this directly. The shape of the learning curve — whether all 8 neighborhoods improve together or sequentially — distinguishes a rule-table learner from a frequency-weighted approximator.

**Q-M2. Does rule complexity determine MLP learning difficulty in a predictable order?**
Expected rough order from easiest to hardest MLP task: `constant < CENTER_COPY < LEFT_COPY/RIGHT_COPY < MAJORITY < XOR < RULE_110`. MAJORITY is smoother than XOR because the output function is monotone (more 1s → more likely output 1) while XOR has sharp parity structure. RULE_110 is asymmetric and requires learning all 8 cases without simplification.

**Q-M3. Which neighborhoods are learned first for each rule?**
For MAJORITY: `000→0` and `111→1` are learnable from center-only statistics and should be acquired early. The "minority wins" neighborhoods (`001→0`, `110→1`) are harder and may transition later. For XOR: all neighborhoods are equally complex, so transition order should follow frequency rather than computational difficulty. Comparing these two rules directly tests whether the MLP bottleneck is statistical or structural.

**Q-M4. Do rare-neighborhood batches produce localized or generalized updates?**
A batch containing mostly `101` neighborhoods may improve accuracy on `101` specifically. Whether it also improves accuracy on structurally related neighborhoods (same output, different bit pattern) would indicate whether the model has an abstract rule or only a pattern cache. This tests the difference between memorization and rule internalization at the MLP level.

---

### D. Update Semantics

**Q-U1. Can we classify each update as routing-strengthening, payload-strengthening, or computation-strengthening?**
The behavioral delta vector — per-neighborhood, per-position, per-length accuracy change after a single update — is the empirical fingerprint of the update's semantic type. A routing update should show correlated improvement across all neighborhoods at longer evaluation lengths. A computation update should show improvement at specific neighborhoods without length generalization. A payload update should show improvement correlated with V probe accuracy gain.

**Q-U2. Can two updates with the same loss decrease have different semantic effects?**
A batch of smooth rows might reduce loss by reinforcing center-cell prediction (memorization). A batch of balanced neighborhoods might reduce loss by strengthening the local rule (generalization). Both produce similar scalar loss drops. The delta vector and internal metric changes should distinguish them, even when loss curves look identical.

**Q-U3. Is the gradient routing-aligned or computation-aligned?**
For each update, compute the cosine similarity between the actual gradient on W_Q/W_K and the gradient that would maximally increase the QK true-offset margin (computed by backpropping through the margin metric rather than the loss). A high routing-alignment score means this batch is primarily correcting where the model looks. Low alignment with routing and high alignment with an analogous MLP-gradient direction means the batch is primarily correcting what the model computes. This turns "what is this update doing" from a post-hoc behavioral question into a real-time signal.

**Q-U4. What does a shortcut-forming update look like vs a shortcut-breaking update?**
In a two-phase curriculum — skewed data first, balanced data second — there should be a detectable transition where the shortcut dissolves. The signature of dissolution: QK attention redistributes away from the center offset, gradient flows primarily through W_Q and W_K rather than MLP, and V probes for side-neighbor states improve. The sign reversal of the delta vector across the phase transition is the cleanest evidence.

**Q-U5. Do phase transitions exist in per-neighborhood accuracy, or is learning gradual?**
For rules with computational complexity (XOR, RULE_110), individual neighborhoods may transition from near-chance to near-perfect accuracy within a small number of steps rather than improving smoothly. If this happens asynchronously across neighborhoods, it would suggest the MLP is learning the rule table entry by entry rather than as a smooth function. Dense logging in early training (every 10–20 steps) is needed to see this. The transition order is itself a diagnostic: for MAJORITY the easy neighborhoods (`000→0`, `111→1`) should transition first; for XOR, order should follow frequency rather than computational difficulty.

**Q-U6. Does improving one neighborhood cause regression on another?**
When a rare-neighborhood batch fires and improves accuracy on `101`, does accuracy on `010` or `110` drop slightly? Negative interference between neighborhoods would indicate that the model is using overlapping weight directions for different rule-table entries, and that learning is not cleanly compositional. Positive transfer (improving `101` also helps `100` and `110`) would indicate the model has generalized the underlying rule structure. Track the full 8×8 cross-neighborhood correlation matrix of accuracy changes across updates.

---

### E. Generalization and Transfer

**Q-G1. Does routing generalization (length transfer) precede or follow rule accuracy on seen lengths?**
A model that has learned relative-position QK structure should generalize to longer rows before it achieves perfect accuracy on the training length. This seems counterintuitive but is plausible: routing structure can be established early while MLP computation is still being refined, and if the routing is relative, longer rows come for free once the structure exists.

**Q-G2. What transfers between structurally related rules?**
Train fully on `LEFT_COPY`, then fine-tune on `MAJORITY`. The prediction: W_Q/W_K should change minimally (MAJORITY also uses the left-neighbor offset), while W_V and MLP change substantially (the output function is completely different). Then reverse: train on MAJORITY first (requires all three offsets), fine-tune on LEFT_COPY. Does the model prune the now-unused center and right offsets, and do QK matrices show larger updates than in the forward direction? This tests whether circuit reuse is asymmetric.

**Q-G3. Is the minimum head count for generalization rule-dependent?**
For copy rules, one head should suffice for perfect routing. For 3-neighbor rules, the prediction is that one head creates V superposition failure — the weighted average of three neighbor V vectors is an ambiguous signal for the MLP. The specific prediction: with one head, accuracy should be worst on neighborhoods where all three bits differ (e.g., `010`, `101`) rather than on constant neighborhoods (`000`, `111`), because the V superposition is most destructive when all three inputs are distinct.

---

## Core Experimental Methods

### 1. Behavioral Delta Vector
Run the controlled evaluation suite before and after each update (or every N steps). Store the vector of per-class accuracy changes:
```
Δ[rule_name, neighborhood_id, position, eval_length, QK_margin, V_probe]
```
This is the primary unit of analysis for update semantics. It enables distinguishing generalization from memorization and classifying updates by type.

### 2. QK Toeplitz Structure Test
For each head, extract QK scores using positional embeddings only (factoring out token content):
```
score(i, j) = (W_Q @ pos_embed[i]) · (W_K @ pos_embed[j]) / sqrt(d_head)
```
Compute how Toeplitz the resulting matrix is (variance of values along each diagonal vs across diagonals). A high Toeplitz index means routing depends on relative position. A low index means absolute-position encoding. This is a zero-cost diagnostic requiring no new data.

### 3. Gradient Routing Alignment
For each update, compute the cosine similarity between the actual gradient on W_Q/W_K and the gradient of the QK true-offset margin with respect to those matrices. Call this `routing_alignment ∈ [−1, 1]`. Track this alongside loss and accuracy. Positive values indicate the batch is correcting routing. Near-zero values indicate the batch is acting on something else.

### 4. Linear Probes (existing + extensions)
Current: V bit-state, V position, head-output rule-output, MLP neighborhood.
Add: V cell-role probe (which query position is this x-token currently serving), cross-head divergence metric (KL between head attention distributions over training), structural-token attention mass as a separate tracked quantity alongside x-token attention mass.

### 4a. Cross-Head Specialization Index
At each logging step, compute KL divergence between head 0 and head 1 attention distributions (averaged over query positions and batch):
```
specialization = mean_KL(head_0_attn || head_1_attn)
```
Track this over training. Rising specialization = heads are differentiating. Flat near-zero = redundancy or one head dominating. Compare across rules: specialization should rise faster and higher for 3-neighbor rules than copy rules.

### 5. Two-Phase Shortcut Curriculum
Phase 1: train on smooth rows until center-offset attention dominates and shortcut is measurable. Phase 2: switch to balanced data without resetting. Track per-step gradient norms by matrix and delta-vector through the transition. This is the cleanest possible experiment for studying shortcut dissolution.

### 6. Per-Neighborhood Dense Logging
Log per-neighborhood accuracy every 10–20 steps during the first 500 steps of training. Look for non-monotone transitions (sudden jumps from ~50% to ~90% on specific neighborhoods). Compare transition order across rules to test whether it follows computational difficulty (MAJORITY prediction) or frequency (XOR prediction).

### 7. Head Ablation
For multi-head models, zero out one head at a time during evaluation and measure the drop in per-neighborhood accuracy. This reveals which head carries which computation and whether heads are specialized or redundant. Cheap to implement: multiply one head's output by zero before the W_O projection.

### 8. Rule-to-Rule Transfer
Checkpoint a fully trained single-rule model. Fine-tune on a different rule. Track gradient norms by matrix during fine-tuning. Compare against training from scratch to measure which components need the most updating. Repeat for multiple source-target rule pairs to build a transfer matrix. Key asymmetry to test: `LEFT_COPY → MAJORITY` (QK should change little, MLP a lot) vs `MAJORITY → LEFT_COPY` (QK may need to prune unused offsets, which is a different kind of update).

### 9. Cross-Neighborhood Interference Matrix
After each update, record the full vector of per-neighborhood accuracy changes. Over many updates, compute the correlation matrix: when neighborhood A improves, does neighborhood B tend to improve or regress? A positive correlation matrix indicates compositional learning (shared structure). Negative off-diagonal entries indicate interference — the weights serving one neighborhood are hurting another. This reveals whether the MLP is learning a clean rule table or a set of competing patterns.

---

## Priority Order

**Do first (foundations):**
1. Dense delta-vector logging — every downstream analysis depends on having pre/post update measurements, not just checkpoint snapshots.
2. QK Toeplitz test — zero cost, directly tests the most important structural hypothesis (Q-R1), should be run on every completed training run.
3. Two-phase shortcut curriculum — single cleanest experiment for the update semantics question.

**Do second (mechanism):**
4. Gradient routing alignment metric — requires one additional backward pass through the margin metric per logging step.
5. V cell-role probe — extends existing probe infrastructure to test the entanglement hypothesis (Q-V3).
6. Phase-transition detection via dense early-training logging (Q-U5).
7. Cross-head specialization index — cheap to add alongside existing per-head attention metrics (Q-R4, Q-R7).
8. Structural-token sink tracking — one additional attention mass computation per logging step (Q-R6).

**Do later (capacity and transfer):**
9. Cross-neighborhood interference matrix — requires accumulating delta vectors over many steps (Q-U6).
10. Head ablation experiments — cheap per run but needs systematic sweep across rules and head counts (Q-G3).
11. Rule-to-rule fine-tuning transfer matrix — requires multiple training runs with checkpointing (Q-G2).

---

## What Success Looks Like

The experiment succeeds if, given a model checkpoint and a batch description, we can predict in advance which of the following will happen after the update:
- Accuracy improves on all neighborhoods at all lengths (general rule learning)
- Accuracy improves on seen lengths only (position memorization)
- Accuracy improves on specific neighborhoods only (pattern memorization)
- QK true-offset margin increases without accuracy change (routing before competence)
- V probe accuracy increases without QK change (payload refinement)
- Specific neighborhoods regress while others improve (interference)

If the behavioral delta vector, gradient alignment score, and internal metric changes together allow us to classify the update correctly before evaluating the post-update model on held-out data, then we have the beginnings of a predictive theory of training dynamics — which is the actual long-term goal.
