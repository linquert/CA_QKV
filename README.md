# CA QKV Experiments

A first-pass codebase for observing how a tiny transformer learns cellular automaton rules.

Core decomposition:

- **QK routing:** which input cells an output position attends to.
- **V payload:** whether attended cells carry bit-state information.
- **MLP computation:** whether the local CA rule `(left, center, right) -> next bit` is learned.

## Colab setup

```bash
pip install -r requirements.txt
wandb login  # optional
python run_experiment.py --config configs/base_left_copy.yaml
```

For a local smoke test without W&B:

```bash
python run_experiment.py --config configs/debug_left_copy.yaml
```

## Important outputs

- `outputs/checkpoints/`: model/optimizer checkpoints
- `outputs/figures/`: attention matrices, QK matrices, prediction grids, neighborhood bars
- `outputs/samples/`: fixed sample predictions across epochs
- W&B metrics: loss, accuracy by length/distribution/neighborhood/position, per-query QK/attention offset mass, QK margin, V probes, MLP probes, gradient/update norms

## First experiment ladder

1. `base_left_copy.yaml` — observe offset -1 QK routing.
2. `left_copy_skewed.yaml` — compare shortcut/skew effects.
3. `left_copy_single_cell_balanced.yaml` — exact supervised-neighborhood balance; best for clean per-update attribution.
4. `majority_balanced.yaml` — observe QK + V + simple nonlinear MLP.
5. `xor_balanced.yaml` — stress parity computation.
6. `rule110_balanced.yaml` — nontrivial asymmetric local rule.

## Dataset note

`balanced` is approximate for full-row prediction: it implants target neighborhoods into otherwise random rows. Use `single_cell_balanced` when exact balancing of the supervised training signal is needed. That mode still provides the full CA row in context, but masks the loss to one queried output cell whose neighborhood cycles exactly through `000` to `111`.

## Patch 1 diagnostics

This version adds the first diagnostics patch for observing early learning dynamics:

- **Dense single-cell neighborhood eval**: `dense_eval` builds a small fixed `single_cell_balanced` diagnostic set and logs masked per-neighborhood and per-position accuracy during early training. This is intended to reveal asynchronous phase transitions across the 8 local CA neighborhoods.
- **Structural-token attention mass**: attention diagnostics now log how much y-query attention goes to token groups such as `x_cells`, `<BOS>`, `<X>`, `<SEP>`, `<Y>`, `structural`, and previous y-bit tokens. This detects attention sinks outside the true CA input cells.
- **Head competition / specialization metrics**: attention diagnostics now log pairwise full-attention Jensen-Shannon divergence and offset-profile distances between heads, plus each head's top offset.

Example config section:

```yaml
dense_eval:
  enabled: true
  every_steps: 1
  until_step: 300
  then_every_steps: 100
  examples_per_neighborhood: 16
  distribution: single_cell_balanced
  log_at_epoch_end: true
```

Key W&B metric families added:

```text
dense/neighborhood/000_acc ... dense/neighborhood/111_acc
dense/position/<i>_acc
diagnostic/attn/head_<h>/group/x_cells_mass
diagnostic/attn/head_<h>/group/structural_mass
diagnostic/attn/head_pair_<h1>_<h2>/js_full
diagnostic/attn/head_pair_<h1>_<h2>/offset_cosine_distance
diagnostic/attn/head_<h>/top_offset
```

## Patch 2: update fingerprints and delta vectors

Patch 2 adds the update-level object needed for studying what a single batch update strengthened.
When `delta.enabled: true`, the trainer evaluates a small fixed diagnostic fingerprint before and after selected optimizer steps, then logs and saves the difference:

```text
delta/fingerprint/neighborhood/000_acc
...
delta/fingerprint/position/0_acc
delta/fingerprint/attn/head_0/true_offset_mass
delta/fingerprint/qk/head_0/true_false_margin
```

The raw before/after/delta records are appended to:

```text
outputs/deltas/update_deltas.jsonl
```

Each JSONL record also includes lightweight batch features, such as supervised neighborhood counts and supervised position counts. This makes it possible to later train regressors or other auxiliary models on `(batch_features, update_delta)` pairs.

Useful config block:

```yaml
delta:
  enabled: true
  every_steps: 1
  until_step: 300
  then_every_steps: 100
  include_attention_qk: true
  max_attention_batches: 1
  log_before_after: false
  save_jsonl: true
```

For larger runs, keep dense delta logging only for early training and use `then_every_steps` for sparse later sampling.

## Patch 4: gradient-alignment diagnostics

Patch 4 adds routing-gradient alignment metrics. These ask whether the actual
training update induced by a batch points in the same direction as an idealized
update that would increase the QK true-offset margin.

The key metric is:

```text
cosine(-∇loss, ∇QK_true_false_margin)
```

where `-∇loss` approximates the first-order optimizer update direction and
`∇QK_true_false_margin` is the ideal routing-improvement direction. Positive
values mean the batch update is routing-aligned.

Config:

```yaml
gradient_alignment:
  enabled: false
  every_steps: 50
  until_step: null
  then_every_steps: 50
  layer_idx: 0
  heads: all
  include_wq: true
  include_wk: true
  use_eval_mode: false
```

Logged metrics include:

```text
grad_alignment/routing/layer_0/all_heads/update_cos
grad_alignment/routing/layer_0/head_0/update_cos
grad_alignment/routing/layer_0/head_0/actual_update_norm
grad_alignment/routing/layer_0/head_0/ideal_norm
grad_alignment/routing/layer_0/head_0/objective_margin
```

`update_cos` is the main quantity. `loss_grad_cos` is also logged as a sanity
check and should often have the opposite sign.

## Config status after config-fix patch

The real-run configs have been updated so they are executable with the current tokenizer and model constraints.

Important fixes:

- All non-debug configs now use `model.max_seq_len: 160`, which supports `eval_lengths: [16, 32, 64]`. Length 64 requires a full sequence length of `2*64+5 = 133`, so `128` was too small.
- The 3-neighbor diagnostic configs now use `d_model: 192` with `n_heads: 3`, because `d_model` must be divisible by `n_heads`.
- `left_copy_single_cell_balanced.yaml` is now the recommended first real diagnostic run. It enables dense eval, delta vectors, and sparse gradient alignment.
- `majority_balanced.yaml`, `xor_balanced.yaml`, and `rule110_balanced.yaml` are now diagnostic single-cell-balanced configs with 3 heads, intended for observing offset specialization and local-rule computation.

Recommended first run:

```bash
python run_experiment.py --config configs/left_copy_single_cell_balanced.yaml
```

Recommended second run:

```bash
python run_experiment.py --config configs/majority_balanced.yaml
```
