# LONGGOAL — SIGMA / *Memory Is Signed* (ICLR 2027)

**Status:** active · **Opened:** 2026-09-18 · **Scope confirmed by the author:** this
paper only (`paper_ICLR27/trajflow-kv/`).

> Standing directive that governs this goal: rewrite the paper from the *problem and
> the problematic experiment*, aim at the contribution no one else has made, and
> confine negatives to a single main-text sentence with full detail in the appendix.
> **Existing methods' limitations may be stated freely; our own must stay honest.**
> The headline is the *phenomenon/diagnostic*, not the method. Binding answers from
> the author's 2026-09-18 review: headline = phenomenon-led; negatives = one
> main-text sentence + appendix detail; blocked work = do the independently
> completable items first.

---

## 1. Scope boundary

In scope: `paper_ICLR27/trajflow-kv/paper/` — the single-source manuscript, its
plotting/extraction pipeline, and the anonymised supplementary snapshot.

**Out of scope** (other sessions own these; do not edit, do not fold in):
`paper_ICLR27/animal/` (AKR), `paper_ICLR27/PBPF/`, `paper_ICLR27/improving/`
(SPECTRUM), and the LACES / S0 line. If a task appears to require touching those
repositories, it is a mis-scoped task.

## 2. Definition of done

The goal is complete when **all four** hold:

1. **D1 — Manuscript.** `main.tex` compiles clean (two `pdflatex` passes, zero
   overfull boxes, zero undefined citations/references), main text ends on or
   before **page 9**, and every number in Tables 1–6 and Figures 1–5 traces to a
   record read by `plotting/extract_evidence.py`.
2. **D2 — No placeholders.** The manuscript contains no `[RESULT NEEDED]`, no `NR`
   cell presented as a result, and no bracketed placeholder in visible text.
   Unmeasured quantities are stated as prose limitations.
3. **D3 — Anonymity.** The supplementary snapshot builds, its anonymity scan is
   clean, and its staged extraction runs standalone.
4. **D4 — Ledger closed.** Every open issue in `DELIVERY_PACKAGE.md` §12 is either
   resolved or explicitly parked as *author action* / *needs new experiment*, with
   the reason recorded.

## 3. Resume protocol

Read in this order before doing anything; do not re-derive context from scratch:

1. `paper/DELIVERY_PACKAGE.md` — the 12-deliverable contract, claim–evidence
   matrix (C1–C11), change map, and the open-issue ledger (§12).
2. `paper/longgoal_status.json` — machine-readable step status (this goal).
3. `paper/VERIFICATION.json` — the last verified build record.
4. `paper/EVIDENCE_AND_REVISIONS.md` — the provenance ledger for every number.

Then reproduce the build before editing anything:

```bash
cd paper
python3 plotting/extract_evidence.py    # prints the verification report
pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```

A `MISSING SOURCE` error from the extraction is a hard stop: a figure or table
input has lost its source record, and the manuscript numbers are no longer
verified. Fix the source, never the figure.

## 4. Done ledger

| # | Item | Evidence |
|---|---|---|
| 1 | Full rewrite to the phenomenon-led structure, §1–§8 + App. A–G | `main.tex`; headline is the measurement, not the method |
| 2 | Built the measurement pipeline as the single verified path | `plotting/extract_evidence.py` — hard-fails on a missing source; 9 CSVs in `data/` |
| 3 | Five production figures | `fig1_intervention.pdf`, `fig2_signed_credit.pdf`, `fig3_per_prefix.pdf`, `fig4_divergence.pdf`, plus inline TikZ pipeline `figures/fig5_method.tex` |
| 4 | Six tables | T1 pilot, T2 signed credit, T3 method comparison, T4 evidence paths, T5 return-vs-ranking, T6 transport/gate |
| 5 | Page limit met | main text ends p. 9 (`\newlabel{main-text-end}` → page 9); 16 pp total |
| 6 | Clean build | exit 0, 0 overfull, 0 undefined |
| 7 | Placeholders eliminated (O5) | last three converted to prose limitation statements; the only remaining `RESULT NEEDED` string is a source comment |
| 8 | Anonymised snapshot (O4) | `sigma_supplementary_anonymous.tar.gz`, 137 KiB, 24 files; anonymity scan clean; staged extraction runs standalone |
| 9 | Venue rules re-verified (O1) | 9-page main-text limit confirmed; abstract **Sep 18 2026 23:59 AOE**, full paper **Sep 25 2026**; `iclr2027_conference.sty` byte-identical to the official release (`sha256 797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6`) |
| 10 | AI use statement rewritten (O3) | present and specific; only the authors' own sign-off sentence is outstanding |
| 11 | Bibliography defects fixed | FocusMem arXiv ID corrected (2606.04530 → 2608.04530); the defined-but-uncited entry now wired into the memory-focused evaluation sentence |
| 12 | All 12 bibliography entries re-verified (O2) | Resolved against primary arXiv abs pages 2026-09-18: title + author list + ID agree 12/12. The arXiv API returned HTTP 406, so the abs-page route was used |

## 5. Parked — needs new experiments (cannot be closed from this session)

These are recorded, not hidden. Each is stated as a boundary in the manuscript.

**A ready-to-fire job package for both exists at
`experiments/closure_20260918/`** (read its README first). It is written and its
CPU half is validated; nothing has been submitted.

| # | Item | Why parked | What would close it |
|---|---|---|---|
| O6 | The hidden-memory gate family is non-monotone (0.704 → 0.039 → 0.290) | Reported descriptively in App. F; a single seed cannot distinguish a real non-monotonicity from seed noise | One additional seed for that family — prepared as `experiments/closure_20260918/o6_seed17.sh` (retrain seed 7→17, re-evaluate) |
| O7 | C5's AUROC is measured within a single generator family | The 200-prefix extraction has not been repeated on a held-out template | Re-run the 200-prefix extraction on a held-out template — prepared as `experiments/closure_20260918/o7_heldout_credit.sh` (templates B/C/D) |

Both are blocked at the same place: **`Qwen2.5-VL-3B-Instruct` is not on the
filesystem**, and the entire `data/` and `outputs/` trees are gone, so the inputs
must be regenerated (CPU, offline) and the weights fetched on the login node
first. See the package README for the run order and for the one provenance
caveat that applies to O6 but not O7.

## 6. Parked — author action (not mine to take)

- **The AI Use Statement's verification sentence** must be written and owned by the
  authors. The manuscript names the categories; it cannot attest on their behalf.
- **Deadlines.** Abstract: Sep 18 2026 23:59 AOE. Full paper: Sep 25 2026. Both are
  final, with no accommodations. Authors cannot be added or removed after the
  abstract deadline, and **every author must register as a reviewer or the paper is
  desk-rejected.** This is the author's call, and it is time-critical.
- **Anonymity at submission.** The repository is on branch `latentcom-only` with a
  `yunbo` remote; confirm the submitted artifact carries no identifying history.

## 7. Red lines (unchanged)

- Never fabricate a number. A missing measurement is stated as missing; `NR` never
  becomes a value.
- No bracketed placeholders in final art or visible prose.
- Do not leak internal artifacts, absolute paths, or author identity into the
  manuscript or the snapshot. The snapshot's anonymity scan is the enforcement
  point — if it fails, the archive is not written.
- Negatives get one main-text sentence and appendix detail. They do not get
  promoted, and they do not get deleted either.
