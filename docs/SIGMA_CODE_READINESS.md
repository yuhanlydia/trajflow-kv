# SIGMA formal experiment code: delivered scope and remaining boundaries

Canonical repository: `yuhanlydia/trajflow-kv`. Audit base: `47e8c91`.

This delivery imports the previously separate experimental Python package into the
repository. It preserves the legacy code/results and does not run GPU training or
GUI experiments. `tango_iclr` is a compatibility package name for SIGMA, not another paper.

## What is now executable code

- Clean-prefix teacher labels from matched decoder K/V replacements: `python -m tango_iclr.teacher`.
- Credit-controller pretraining and policy refinement: `python -m tango_iclr.train`.
- MLP/GRU/Transformer, K/V/K+V, selected layers/rank, signed/unsigned/uniform,
  shuffled/sign-flipped credit, no energy, no credit auxiliary, and donor controls.
- Actual action-token SFT (`action_sft`), separately named candidate CE (`ce`),
  successful candidate CE, logged-return surrogate, and reference-weighted action advantage.
- Free-action native browser/Android evaluation: `python -m tango_iclr.online`.
- Official MemGUI model endpoint and wrapper: `tango_iclr.serve`, `scripts/run_tango_memgui.sh`.
- Paired one-decision patch followed by rollout: `tango_iclr.rollout_credit`.
- LoRA finite-candidate KL/residual/CE distillation and no-hook student evaluation.
- Paired native success summaries: `tango_iclr.report`.

## Run the actual primary comparison first

Install a CUDA-compatible PyTorch wheel and `requirements-tango-iclr.txt`.
Use the root of this checkout (`PYTHONPATH` must include it). Prepare installed
model weights and healthy native environments separately.

Export real successful demonstrations; repeat for independently grouped validation/test:

```bash
python -m tango_iclr.real_data --input /path/to/native_train_logs \
  --split train --output data/sigma_real/train.jsonl --image-root /path/to/images \
  --min-history 2 --history-limit 5
python -m tango_iclr.prepare \
  --train data/sigma_real/train.jsonl --validation data/sigma_real/validation.jsonl \
  --test data/sigma_real/test.jsonl --output data/sigma_checked
```

The exporter accepts this project's `episode.json` and legacy trajectory JSONL,
not arbitrary AITW/MemGUI vendor dumps. Preserve a true `task_id` and task success.
It excludes failed/infrastructure-corrupted trajectories and never invents Q values.
Screenshots after a decision are excluded from its input. Prefixes with fewer than
two history screens are omitted from this training-label extraction, NOT from online evaluation.
Equal-length donors are still required; unmatched history spans stop label generation.

Configure paths and real `case_id/task_id/backend/seed` manifests in
`configs/sigma_real_gui.yaml`. The native manifest examples and environment contract
are in `docs/TANGO_RUN_AGENT.md` and `tango_iclr/online.py`. Test tasks must not enter
training/selection. Do not fill a real-GUI manifest with MiniWoB/synthetic tasks and
label it AndroidWorld, MemGUI or WorkArena.

```bash
python -m pytest -q tests_iclr
python -m tango_iclr.preflight --output outputs/sigma_real_gui/preflight.json
bash scripts/run_sigma_iclr.sh --profile primary                 # inspect jobs
bash scripts/run_sigma_iclr.sh --profile smoke --execute         # actual local GPU smoke
bash scripts/run_sigma_iclr.sh --profile primary --execute --continue-on-error
bash scripts/run_sigma_iclr.sh --profile all --execute --continue-on-error
```

`primary` means Base + capacity-matched action SFT + SIGMA, not a complete ICLR
result package. `all` enumerates implemented controls/ablations from this config;
it is NOT an assertion that every E01--E17 experiment is implemented. Default
`all` does not run policy distillation before a teacher benefit is established.
Use separate output roots for configuration changes. Offline single-demonstration
rows report action NLL, not 100% candidate accuracy. End-to-end success comes only
from native execution/evaluation.

The old `configs/tango_iclr_release.yaml` remains a multi-candidate reproduction
profile. Global-return requires actual mixed logged returns and logged action IDs;
action advantage requires independently obtained continuation values. The new
single-demonstration SFT profile cannot silently manufacture those controls.

## Delivery audit by paper experiment

| Plan | Implemented entrypoint / boundary |
|---|---|
| S00 | Code integrated and CPU tested; GPU/runtime/model validation still required. |
| E01 | `real_data`, `prepare`: real-log export and group split validation. External vendor adapters/task manifests remain inputs. |
| E02 | `preflight`, `evaluate`, `online`, `serve`; real Qwen/cache/native smoke NOT RUN. |
| E03 | `teacher`: local action score and candidate-value proxy, clean same-prefix donors. Cross-instance donor pool/missing-credit masks NOT IMPLEMENTED. |
| E04 | `train`, `suite`: two-stage learned controller and residual; actual token SFT and finite-candidate controls. Formal GPU run NOT RUN. |
| E05 | Official MemGUI endpoint/wrapper provided. Runtime, judge credentials, task manifests, native log conversion must be validated; not automatically in the generic suite. |
| E06 | `online`: AndroidWorld HTTP client integration. Need healthy server and fixed task/instance manifest; do not conflate decoding seed with task_idx. |
| E07 | `online`: BrowserGym coordinate-action bridge. Need WorkArena service and official registered task IDs; MiniWoB is not this benchmark. |
| E08 | Signed/unsigned/shuffle/flip/uniform implemented. Magnitude-only, recency and designer-oracle full protocols remain to implement. |
| E09 | MLP/one-layer GRU/one-layer Transformer and removal of energy/credit auxiliary implemented. Explicit sign-classification loss and all paper loss controls remain to implement. |
| E10 | K/V/K+V, early/middle/late, ranks 4/8/16 implemented for the specified backbone layer map. |
| E11 | Single/multiple same-prefix donor labels and per-donor effects implemented; comprehensive cross-instance robustness not implemented. |
| E12 | Evaluate any supplied independent OOD split. New task-family generators and the complete OOD grid are not supplied here. |
| E13 | `rollout_credit`: reset/replay, one-decision intervention then base continuation. Pixel-hash equality alone is not proof of full hidden environment-state identity. |
| E14 | Energy/latency/invalid actions and paired summaries implemented; full retention benchmark and failure taxonomy need separate protocols. |
| E15 | `distill`: optional candidate-policy LoRA KL/centered residual/CE. Multiple proposals required. Run only after independently validated teacher gain. |
| E16 | Engine supports Qwen2.5-VL / Qwen3-VL class loading; new-model layer maps/quantization still require GPU smoke. |
| E17 | Pairwise native JSON/CSV summaries implemented. Official MemGUI metrics import and all publication plots are not auto-completed. |

Full official FocusMem/SPD reproductions are NOT included; the fixed-gradient
`spd_control` is only an adapted GUI control. Do not relabel it as a published SOTA.

## Important resolved differences

The prior baseline code used candidate CE. This release adds action-token SFT
without candidate normalization. SIGMA and action SFT use the same signed
controller/residual capacity; only measured-credit supervision is added for SIGMA.
Baselines no longer depend on costly memory teacher labels they do not consume.
Base pixel/token/revision settings now follow the same configured budget as learned
arms. Donor cache IDs include the model revision and visual/context budgets.

This config aligns hidden size 256, training seeds 11/23/37, five credit plus five
policy epochs and accumulation eight. It uses a COMMON 100352 pixel cap. Separate
historical/current image caps, the paper's explicit sign loss, all gate-temperature
controls, and native-validation checkpoint selection are not implemented by changing
YAML keys. Current checkpoint selection uses candidate accuracy when informative,
otherwise held-out action NLL. Record this difference before claiming manuscript parity.

Passing CPU tests proves software behavior in the tested contracts, not that Qwen
training or real GUI performance succeeded. No new task-success numbers are included.
