# Experiment status, 2026-09-17

Repository revision before this update: `fda89ac5789e715572d032f201c6c2554bf0961a`.

## Local preparation

- Qwen2.5-VL 3B and 7B checkpoints were downloaded, checksum-verified, and passed GPU smoke tests.
- Main Python environment, AndroidWorld environment, and MemGUI environment passed dependency checks.
- Legacy tests passed: 78/78.
- ICLR tests passed: 46/46.
- Mind2Web public files are present locally; AITW was kept to a small sample by request.
- BrowserGym MiniWoB native Chromium smoke passed. WorkArena environments are registered locally, but real WorkArena evaluation still needs a ServiceNow instance and credentials.

## Controlled synthetic run

The controlled interference-chain run completed on Qwen2.5-VL-3B-Instruct with:

- train: 200 template-A prefixes
- validation: 50 template-B prefixes
- sealed test: 200 template-C prefixes
- intervention: middle-layer K, layers 12-23, rank 8
- teacher protocol: local action log-probability credit with one matched donor

Validation selected alpha 20:

| Arm | Accuracy | Mean action NLL |
|---|---:|---:|
| Base validation | 0.16 | 0.402201 |
| Memory validation, alpha 20 | 0.18 | 0.402121 |

Final sealed test:

| Arm | Correct | Accuracy | Mean action NLL |
|---|---:|---:|---:|
| Base | 24/200 | 0.120 | 0.412617 |
| Memory, alpha 20 | 23/200 | 0.115 | 0.412004 |

This is a no-go for a discrete accuracy improvement claim. The memory arm slightly improved mean action NLL but regressed accuracy by one decision. The sealed template-C test must not be reused for hyperparameter selection.

## Positive evidence still available

Earlier pilot evidence remains useful as preliminary paper material, but it should be framed as pilot evidence rather than broad benchmark proof:

- return-conditioned KV improved held-out return margin from 0.1965 to 0.2598 in the prior go/no-go ledger;
- the Bluetooth online probe improved from 0/4 to 4/4 under the structured policy, but this is single-task evidence;
- internalized KV residuals preserved hooked-teacher ranking metrics after folding into model weights;
- low-energy transport reduced measured transport energy by 4.47x while retaining most of the unregularized return margin;
- MiniWoB click-color showed a weak paired gain from 12/30 to 14/30.

## External blockers

Native mobile/browser-service benchmarks remain blocked by resources outside this repository:

- AndroidWorld and MemGUI require a usable `/dev/kvm` device on the environment host. The software emulator was tried for 15 minutes but did not bring up Activity Manager reliably.
- MemGUI also needs `MEMGUI_API_KEY` and judge endpoint access.
- WorkArena needs ServiceNow instance credentials and access to the gated WorkArena instance resources.
- Real GUI training/validation/test demonstrations and screenshots are still missing.

## Code fixes in this update

- Kept controller and memory bank in training mode during controlled-gradient replay to avoid cuDNN GRU backward failures.
- Optimized the local action-logprob teacher to score only the labelled action instead of all candidates.
- Made the AndroidWorld software setup script idempotent and install official requirements before the package.
- Added regression tests for the GRU training-mode path and the labelled-action teacher path.
