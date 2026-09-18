# Delivery Package — *Memory Is Signed*

Operation `restructure`. Twelve deliverables per the full-paper output contract.
Prepared 2026-09-18. Manuscript: `main.tex` / `main.pdf` (16 pp; main text 9 pp).

---

## 1. State and venue contract

**State: `Complete with named gaps`.**

- **Operation:** `restructure` (an earlier protocol-style draft existed at the same
  path; it was replaced, not appended to).
- **Narrative mode:** phenomenon- and diagnostic-led. The measurement is the
  headline; the method is presented as the framework that consumes it.
- **Target venue:** ICLR 2027, main conference track, anonymous submission stage.
  `\iclrfinalcopy` remains commented out; the style file `iclr2027_conference.sty`
  is the unmodified official copy already present in the repository.
- **Target reader:** representation-interpretability and GUI-agent researchers.
- **Manuscript language:** English.
- **Verified official sources:** none retrieved in this session. The 9-page main-text
  limit and the 2027 style file are taken from the repository's existing submission
  package, not re-fetched from the ICLR site. **See open issue O1.**
- **Current blocking gaps:** none for the central claim; the method's end-task
  benefit is a disclosed boundary, not a gap (see rows C7/C8).

Because unresolved `[RESULT NEEDED: ...]` markers remain in the manuscript, the
state is *not* "Complete" and the paper must not be described as submission-ready.

---

## 2. Paper brief

**Thesis (one sentence).** In a frozen GUI vision-language model, the internal
representation of a *superseded* historical update actively interferes with the
current decision even when the correcting evidence is present, and the sign of
each historical block's causal contribution is measurable, role-determined, and
invisible to the training objectives currently used for GUI memory.

**Spine.** Importance → gap → response → evidence → implication:

- **Importance.** GUI agents carry screenshots and past actions forward as a growing
  KV prefix; memory is assumed to be monotone capability.
- **Gap.** Nothing attributes an episode's outcome to a *particular* part of the past.
  Task success is one scalar at the end of an episode.
- **Response.** A matched-replacement measurement of signed causal memory credit:
  swap one historical key span for a donor, hold everything else fixed, read the
  utility difference. Then `SIGMA`, which turns measured credit into supervision.
- **Evidence.** 10-prefix pilot (60%→100% / 60%→0%); 200 prefixes / 1,000
  measurements (92% and 98.5% sign consistency, 181/200 both signs); AUROC
  0.86/0.94 localisation; 400 paired MiniWoB cases showing the optimised margin
  and executed behaviour diverge.
- **Implication.** Memory quality cannot be read off a confidence score; it must be
  intervened on. The open problem is what correction actually helps.

**Central contribution.** The measurement and the demonstration that age and content
— the only signals available without intervening — do not predict its sign.

**Supporting contributions.** Per-instance (not merely mean) evidence; donor-level
controls; the divergence experiment; the `SIGMA` framework.

**Scope boundary.** Controlled Interference-Chain histories with a frozen 3B/7B
backbone, plus MiniWoB/AndroidWorld/MemGUI evaluation attempts. No claim of an
end-task gain.

**Terms needing early definition.** historical block; matched donor; signed credit;
the three utilities (`u_act`, `u_val`, `u_roll`), which are never compared.

---

## 3. Claim–evidence matrix

`Locations` uses section numbers; `p.` is the PDF page.

| ID | Claim | Type / importance | Evidence required | Evidence present | Scope | Locations | Status | Action |
|---|---|---|---|---|---|---|---|---|
| C1 | Replacing the superseded update takes critical-decision accuracy 60%→100%; replacing the latest takes it 60%→0% | Central | Paired intervention, fixed pixels/candidates | `pilot10.json`: 10 paired prefixes, 5 roles | Pilot n=10, one template | §1, §3.1, T1, F1 | **supported** (narrow) | Keep; labelled pilot-scale in §3.1 |
| C2 | Superseded updates carry negative credit in 92.0% of prefixes; latest positive in 98.5% | Central | ≥100 prefixes, per-role distribution | `credit200.json`: 200 prefixes, 1,000 blocks, bootstrap CIs exclude 0 | Candidate-value proxy, not native return | Abstract, §3.2, T2, F2 | **supported** | Keep |
| C3 | The sign is set by role relation, not age: the older initial record is weakly *positive* | Central | An older non-update block with distinguishable credit | Same: initial mean +0.000798, CI [+0.000178,+0.001429], 55.0% positive | Same | Abstract, §1, §3.2, F2 | **supported** | Keep; this is the load-bearing contrast |
| C4 | The opposing pattern holds per prefix, not only on average | High | Joint per-prefix sign counts | 181/200 both signs, 19/200 one, 0/200 neither | Same | §3.3, F3 | **supported** | Keep |
| C5 | Credit localises each update block against non-update blocks | High | Rank-based separability from credit alone | AUROC 0.864 (superseded), 0.936 (latest) | Same generator family; not a cross-template generalisation | §3.3, F3 | **supported** (narrow) | Keep; scope stated |
| C6 | The optimised margin and executed behaviour diverge | High | Paired data, same histories, both quantities | `miniwob.json`: return-weighted margin .1711 / 330 tasks vs CE .1471 / 333 tasks | All differences in one of four families; CIs include 0 | §1, §5, F5, T5 | **supported** (narrow) | Keep; add the "we do not conclude return information is useless" paragraph (present) |
| C7 | `SIGMA`'s end-task benefit is not established | Boundary | Sealed controlled test at equal capacity | 23/200 vs 24/200 control; NLL .412004 vs .412617 | 3B, sealed template-C | §7, App. E | **contradicted** (for a gain claim) | Disclosed in one main-text sentence + App. E |
| C8 | Native GUI benchmark standing | Boundary | Completed episodes on native benchmarks | 0/5, 0/19 (AndroidWorld), 0/21 vs 1/21 CE (MemGUI) | Base policy emitted only `answer`; infra faults | App. E | **missing** (not a clean measurement) | Reported as a status record, explicitly not a method comparison |
| C9 | Learned gate magnitude decays with distance from the decision | Secondary | Per-slot gate magnitudes | `gate_decay.csv`: distractor family 0.418→0.179→0.078 (monotone); hidden-memory family 0.704→0.039→0.290 (**non-monotone**) | 1 trained controller, 1 run | App. F | **narrow** | Kept in appendix as descriptive only; **not** used for any main-text claim |
| C10 | The precursor's own offline metric is untrustworthy at its scale | Supporting | Label-permutation controls | `gonogo.json`: random labels also improve the score; first-only history strongest | 6 held-out trajectories | §5, App. F | **supported** | Keep; this is what licenses C6's interpretation |
| C11 | The intervention protocol yields the intended contrast at all | Supporting | Localisation, sign, and neutrality accuracy | Accuracy 1.0 / 1.0 / 1.0 at tolerance 0.025 | n=10 | §3.1 | **supported** (pilot) | Keep; explicitly not treated as a population estimate |

**Contradictions made visible:** C7 is a contradiction of the *method* claim and is in
the main text (one sentence) with full detail in App. E. C8 is missing-not-negative and
is in App. E. C9 is reported with its non-monotonic family intact.

---

## 4. Full-paper blueprint

Verified limit: 9 pages main text (provisional, see O1). Actual: **ends on p. 9**.

| § | Purpose / question | Reader takeaway | Claims | Budget | Floats | Links |
|---|---|---|---|---|---|---|
| 1 | Is there a real failure, and why can't existing signals see it? | Yes; and confidence is not a memory diagnostic | C1, C3, C6, C10 | 1.9 pp | F1, F5 | → defines the object §2 formalises |
| 2 | What exactly is measured, and what are the three utilities? | A conditional, matched-replacement estimand; never conflate the utilities | method | 1.3 pp | — | ← §1's object; → §3's numbers |
| 3 | Does the effect survive scale, and is it per-instance? | Yes: 92%/98.5%, CIs exclude 0, 181/200 both signs | C1–C5, C11 | 2.4 pp | T1, T2, F2, F3 | ← §2's estimand; → §4 consumes it |
| 4 | Given a measurement, how do you adapt? | `SIGMA` = credit distillation + gated low-rank residual | method | 1.3 pp | F4 | ← §3's signal; → §5 tests whether training finds it for free |
| 5 | Doesn't the training objective already encode this? | No: margin rises while completions do not | C6, C10 | 1.2 pp | F5, T5 | ← §3; → §6 positions the work |
| 6 | Where does this sit? | The novelty is the *supervision signal*, not gating | — | 0.8 pp | — | → §7 qualifies |
| 7 | What are the limits? | Negative credit is utility-relative; the method's gain is unestablished | C7, C8 | 0.7 pp | — | → §8 |
| 8 | What is left? | Once you find an interfering memory, what correction helps? | — | 0.25 pp | — | — |

**Paragraph-level blueprint (main text).**

- **P1.1** job: open with the failure as a scene, not a thesis. Topic: an agent submits
  the old value with the correction in context. No citations. Transition → the field's
  assumption.
- **P1.2** job: state the field's default and the gap. Claims: retention ≠ influence.
  Cites: `wang2024awm`, `liu2026memgui`, `zhang2026focusmem`. → the controlled setting.
- **P1.3** job: *the problematic observation*. Interference Chain design, then F1's
  numbers. Qualification: "identical pixels, opposite outcomes". Callout: F1.
- **P1.4** job: what this rules out (age, content). Required qualification: both
  updates are semantically similar, so content alone cannot generate the pattern.
- **P1.5** job: the measurement does not reduce to confidence. Claims C6, C10.
  Callout: F5 (forward reference). Transition → contributions.
- **P1.6** job: four contributions. Must not promise an end-task win.
- **P2.1** job: define the decision prefix, memory, and donor operator. Eqs. (1)–(2).
- **P2.2** job: three utilities, three claims, with the explicit non-comparability
  statement. Eqs. (3)–(5).
- **P2.3** job: target-leakage controls. Required qualification: within-forward vs
  clean-prefill are *different protocols*, recorded separately.
- **P3.1** job: the pilot in full. Callouts: T1, F1. Qualification: pilot-scale.
- **P3.2** job: 200 histories. Callouts: T2, F2. Qualification: weaker reference control.
- **P3.3** job: per-instance joint pattern + AUROC. Callout: F3. Qualification: same
  generator family.
- **P4.1–4.3** job: controller, gate, optimisation, deployment. Eq. (10) plus the
  explicit "a finite replacement is not a scaling derivative" caveat. Callout: F4.
- **P5.1–5.4** job: the divergence experiment, its own controls, and the bounded
  interpretation. Callout: F5, T5.
- **P6.1–6.4** job: position against GUI memory, interventions, SPD, evaluation.
- **P7.1–7.3** job: negative-credit semantics; measurement scope; the method boundary.
- **P8.1** job: the open question, stated as a question.

---

## 5. Manuscript artifact

`main.tex` (≈55 KB) compiles with two `pdflatex` passes: **0 errors, 0 overfull
boxes, 0 undefined references or citations**, 16 pages total, main text ending on
page 9. Figures are `\includegraphics` of generated PDFs; Tables 1–6 are native
booktabs objects, editable.

---

## 6. Citation audit

| Anchor | Key | Locator | Metadata | Entailment | Disposition |
|---|---|---|---|---|---|
| §1 memory accumulation | `wang2024awm` | arXiv:2409.07429 | Agent Workflow Memory | supports "reusable routines exposed to later tasks" | OK |
| §1, §6 | `liu2026memgui` | arXiv:2602.06075 | MemGUI-Bench | supports memory-focused GUI evaluation | OK |
| §1, §6 | `zhang2026focusmem` | arXiv:2608.04530 | FocusMem | supports "trust module for retrieved evidence" | **corrected this session** (printed ID was 2606.04530, URL 2608.04530) |
| §3.1 | `bai2025qwen` | arXiv:2502.13923 | Qwen2.5-VL | names the frozen backbone | OK |
| §3.1 | `zhang2024patching` | arXiv:2309.16042 | Activation patching | supports substitution-comparison principle | OK |
| §5 | `rawles2024androidworld` | arXiv:2405.14573 | AndroidWorld | names the environment | OK |
| §6 | `rawles2023aitw` | arXiv:2307.10088 | Android in the Wild | **was uncited**; wired into the evaluation-lineage sentence | **fixed this session** |
| §6 | `boisvert2024workarena` | arXiv:2407.05291 | WorkArena++ | supports compositional stress | OK |
| §6 | `wu2024reft` | arXiv:2404.03592 | ReFT | supports frozen-weight representation intervention | OK |
| §6 | `darcet2024registers` | arXiv:2309.16588 | Registers (ICLR 2024) | supports pathology→architecture remedy | OK |
| §6 | `hao2026spd` | arXiv:2605.22675 | SPD | supports capability-subspace self-distillation | OK |
| App. C | `hu2021lora` | arXiv:2106.09685 | LoRA | same-architecture student | OK |

**All twelve entries re-verified against primary arXiv pages on 2026-09-18** (open
issue O2, now closed). Each ID was resolved on `arxiv.org/abs/<id>` and its title and
author list diffed against the printed entry. Twelve of twelve agree on title,
author list and ID. The arXiv API (`export.arxiv.org`) returned HTTP 406 throughout;
the abs-page route was used instead, so the audit rested on the same primary source.

Two defects were found and fixed rather than papered over: the FocusMem ID mismatch
(printed `2606.04530`, URL `2608.04530` — the URL was right, and `2608.04530` is
confirmed to resolve to FocusMem) and the uncited AitW entry, now wired into the
evaluation-lineage sentence. No bibliography entry was invented, and no entry
required a metadata correction beyond the FocusMem ID.

---

## 7. Visual and table manifest

| ID | Role | Claim | First callout | Source data | Design skill | Production | Status |
|---|---|---|---|---|---|---|---|
| F1 | Hero intervention | C1 | §1 (p. 2) | `pilot10.csv` | `designing-experiment-figures` | `fig1_intervention.py` → PDF | Built, verified |
| F2 | Core diagnostic | C2, C3 | §3.2 (p. 5) | `credit200*.csv` | `designing-experiment-figures` | `fig2_signed_credit.py` | Built, verified |
| F3 | Per-instance joint pattern | C4, C5 | §3.3 (p. 6) | `credit200_per_prefix.csv` | `designing-experiment-figures` | `fig3_per_prefix.py` | Built, verified |
| F4 | Method pipeline | method | §4.3 (p. 7) | method spec | `designing-pipeline-figures` | inline TikZ | Built, verified |
| F5 | Divergence | C6 | §5 (p. 8) | `miniwob.csv` | `designing-experiment-figures` | `fig4_divergence.py` | Built, verified |
| T1 | Pilot table | C1, C11 | §3.1 (p. 4) | `pilot10.csv` | — | booktabs | Built |
| T2 | Signed credit | C2, C3 | §3.2 (p. 5) | `credit200_by_role.csv` | — | booktabs | Built |
| T3 | Protocol (app.) | reproducibility | App. A | protocol spec | — | booktabs | Built |
| T4 | Evidence ledger | provenance | App. D | file map | — | booktabs | Built |
| T5 | MiniWoB counts | C6 | App. F | `miniwob.csv` | — | booktabs | Built |
| T6 | AndroidWorld controls | C10 | App. F | `gonogo.csv` | — | booktabs | Built |
| A1 | Algorithm | method | App. B | algorithm spec | — | float + tabular | Built |

**Why each earns its space / what breaks if removed.** F1: the whole paper's premise —
without it there is no phenomenon. F2: the headline distributional claim. F3: without
it, T2's near-zero reference means could be a cancellation artefact. F4: without it,
§4's two-stage framework is hard to follow. F5: without it, C6 is an assertion.
T1/T2 are the numeric backing for F1/F2 and stay for exactness. T3–T6 are appendix.

---

## 8. Figure design packs

Production note applying to all five: colour is assigned by *job* (diverging for
signed polarity, categorical for method identity), the palette was validated with
`scripts/validate_palette.js` rather than eyeballed, and no missing value is ever
rendered as a plotted point. A five-colour categorical set was **rejected** in F5
after failing the normal-vision floor; the figure uses position-encoded identity
instead.

### Figure 1 — `fig:intervention`

- **Fact lock.** MEASURED: 10 paired prefixes, 5 roles, baseline accuracy 0.60;
  patched accuracies 0.50/0.60/1.00/0.60/0.00; effects +0.0469/−0.0194/−0.0965/
  +0.0013/+0.2368.
- **Three-second takeaway.** The same screenshots produce opposite outcomes
  depending on which historical block is replaced.
- **Candidate A (recommended, built).** Two aligned panels sharing the block axis:
  (a) dumbbell from baseline to patched accuracy with a dashed baseline; (b) diverging
  bars of signed effect. *Emphasises:* exact paired contrast per role. *Compresses:*
  prefix-level detail. *Risk:* value labels colliding with the baseline rule.
- **Candidate B.** A single slope chart, baseline→patched, five lines on one
  accuracy axis with the two updates accented. *Emphasises:* direction of change.
  *Risk:* overlapping lines at 0.60 (three roles share it).
- **Candidate C.** Small-multiple of 2×5 mini confusion/accuracy tiles, one per role.
  *Emphasises:* per-role detail. *Risk:* ten panels is too many for a 2.2-inch block.
- **Alt text.** Two aligned panels over five historical blocks. Left: accuracy after
  replacing each block, from 60% unmodified to 100% for the superseded update and 0%
  for the latest, with references unchanged and the initial record at 50%. Right: the
  signed effect on the correct-action score for the same rows.
- **Negative constraints.** No dual axis; no 3-D; no rainbow; no truncated bars for
  the signed effect (it is centered at zero by construction).
- **QA.** Verify label placement clears the dashed baseline (a conditional left/right
  offset is implemented); verify the two reference rows are visually neutral.

### Figure 2 — `fig:credit`

- **Fact lock.** MEASURED: 200 per-prefix values per role; means +0.000798, +0.000103,
  −0.007612, −0.000135, +0.011020; positive fractions .550/.500/.080/.500/.985;
  bootstrap CIs as in T2. DERIVED: intervals from 4,000 prefix-level resamples
  (a cluster resample — one record per role per prefix).
- **Three-second takeaway.** The sign is set by role, not by age.
- **Candidate A (recommended, built).** Horizontal strip of all 200 points per role,
  plus a bootstrap CI whisker and a mean marker, with dominant-sign fractions in a
  dedicated right-hand column. *Emphasises:* distribution *and* central estimate.
  *Compresses:* donor-level variability. *Risk:* fraction labels colliding with strips.
- **Candidate B.** Five vertically offset violin/box summaries without the raw points.
  *Emphasises:* shape. *Risk:* hides the discreteness and the sample size.
- **Candidate C.** A paired slope per prefix connecting its superseded and latest
  values. *Emphasises:* the within-prefix contrast. *Risk:* 200 overplotted lines.
- **Alt text.** For each of five historical block roles, the distribution of signed
  credit across 200 controlled prefixes. The superseded update is negative in 92% of
  prefixes and the latest positive in 98.5%; the two matched references and the older
  initial record centre near zero.
- **Negative constraints.** No dual axis; the neutral grey is the documented diverging
  midpoint, not a categorical slot.
- **QA.** Verify the y-tick labels match the drawn row order (a reversed-order bug was
  found and fixed here); verify the sign-convention footnote matches the metric
  definition (a separate inversion was found and fixed).

### Figure 3 — `fig:joint`

- **Fact lock.** MEASURED/DERIVED: per-prefix (superseded, latest) pairs; counts
  181/19/0; AUROC 0.864 and 0.936.
- **Three-second takeaway.** The opposing signs co-occur within individual prefixes.
- **Candidate A (recommended, built).** Scatter with a shaded expected quadrant and a
  legend carrying the counts. *Emphasises:* the joint pattern and its counterexamples.
  *Compresses:* effect magnitudes. *Risk:* a symmetric axis wasting the canvas.
- **Candidate B.** 2×2 contingency of sign(initial-vs-update) with counts. *Emphasises:*
  exact counts. *Risk:* discards magnitude entirely.
- **Candidate C.** Two marginal histograms plus a small central scatter. *Emphasises:*
  marginals. *Risk:* too much ink for a 0.55-textwidth block.
- **Alt text.** Scatter of 200 controlled prefixes placed by superseded-update credit
  (horizontal) and latest-update credit (vertical). 181 points fall in the quadrant
  where both signs are as predicted; 19 show one sign; none shows neither.
- **Negative constraints.** No sequential ramp; the shaded region is annotation, not data.
- **QA.** Axis windows must be fitted to the observed range (a symmetric limit
  previously wasted most of the canvas).

### Figure 4 — `fig:pipeline`

- **Fact lock.** CONFIRMED from the method spec: two lanes; measurement produces a
  credit target; a GRU controller consumes clean prefix features; the gate scales a
  low-rank residual on historical K/V; the backbone is frozen.
- **Three-second takeaway.** Measurement (A) produces the supervision that adaptation
  (B) consumes; the two lanes share the operator but not the tensors.
- **Candidate A (recommended, built).** Two horizontal lanes, four boxes each, with a
  dashed supervision arrow from lane A's last box to lane B's controller.
  *Emphasises:* the train-time/deploy-time split. *Compresses:* layer indices.
- **Candidate B.** A single left-to-right chain with the measurement drawn as a
  feedback loop returning into the controller. *Emphasises:* the loop. *Risk:* the loop
  is train-time only, so the topology would mislead about deployment.
- **Candidate C.** A vertical stack with the two lanes as columns. *Emphasises:* the
  parallel structure. *Risk:* consumes too much vertical space for a single-column block.
- **Accessibility / alt text.** Two labelled lanes of four boxes. Lane A: controlled
  prefix, matched replacement of block j, effect u(M)−u(M_j,d), signed credit. Lane B:
  clean prefix pass, GRU controller, gate, gated low-rank residual on historical K,V.
- **Negative constraints.** No 3-D; no shadow; no raster; exact notation preserved.
- **QA.** Lane labels must sit in the left gutter, clear of the first box (they were
  previously hidden *under* it); annotation arrowheads must land outside box borders
  (one previously landed inside).

### Figure 5 — `fig:divergence`

- **Fact lock.** MEASURED: 400 paired cases; margins .1502/.1502/.1711/.1471/.1481;
  successes 329/330/330/333/328; per-family counts in T5.
- **Three-second takeaway.** Sharper scores, same behaviour.
- **Candidate A (recommended, built).** Two aligned panels sharing the method axis:
  (a) margin dot plot, (b) successes dot plot, with the frozen baseline as a dashed
  rule. *Emphasises:* the row-by-row divergence. *Compresses:* per-family detail (T5).
  *Risk:* overlapping value labels at equal values.
- **Candidate B.** A single scatter of margin (x) against successes (y), one point per
  arm. *Emphasises:* the decoupling as a shape. *Risk:* five points is a weak scatter.
- **Candidate C.** Small-multiple of four family panels. *Emphasises:* that all the
  movement is in one family. *Risk:* buries the headline in detail; belongs in T5.
- **Alt text.** Two aligned dot-plot panels over five training arms. Left: the mean
  top-1 candidate margin, highest for return-weighted at 0.1711. Right: tasks completed
  out of 400, highest for successful-only cross-entropy at 333 and near-baseline for
  return-weighted at 330.
- **Negative constraints.** No dual axis (two scales are two panels); **no bars** — both
  quantities have non-zero baselines, so bars would misstate them; no more than three
  categorical hues.
- **QA.** Dot plots, not bars, verified; y-tick label order verified against drawn order
  (the same reversed-label bug as F2 was present and fixed).

---

## 9. Appendix / supplementary map

| Item | Duty | Main-text dependency | Destination | Why not main text | Status |
|---|---|---|---|---|---|
| Protocol, donor controls, controller contract | Reproducibility | §2, §4 | App. A | Reference detail; would displace evidence | Complete |
| Algorithm 1 | Reproducibility | §4.3 | App. B | Restates §4; cost a quarter page of main text | Complete |
| Loss definitions, rank loss, counterexample | Method depth | §4 | App. C | Proof detail | Complete |
| Evidence ledger (T4) | Provenance | all | App. D | Maps, does not argue | Complete |
| Native GUI evaluation (C7, C8) | Transparent boundary | §7 | App. E | One-sentence disclosure is the main-text duty; full failure attribution is appendix | Complete |
| Precursor results (T5, T6), transport, gate decay (C9) | Secondary evidence | §5 | App. F | Earlier variants; not mixed with `SIGMA` | Complete |
| Figures/data availability | Reproducibility | figures | App. G | Statement | Complete, one open item |

---

## 10. Change map (from the pre-existing draft at the same path)

| Source location | Action | Destination | Reason | Claim/evidence impact | Citation/visual impact |
|---|---|---|---|---|---|
| Title / abstract | rewrite | §0 | Diagnosis-led headline per the approved framing | New: the measurement is the headline, not the method | — |
| Old §1 intro | rewrite | §1 | Lead with the problematic experiment | Adds C1, C3, C6 | F1, F5 added |
| Old protocol/proposal sections | cut | — | Proposal prose with no measurements | Removes unsupported claims | — |
| Old `tab:real` (`\NR` matrix) | cut | — | An `NR` matrix asserts nothing and reads as a missing result | Removes placeholder-as-result risk | Replaced by T1, T2 |
| Measurement formalisation | keep + rewrite | §2 | Estimand unchanged; utilities now explicitly separated | Preserves the ledger's correction #1 | Eqs. (1)–(5) |
| `SIGMA` method | merge + rewrite | §4 | Compressed; gate honestly described as a low-rank residual | Preserves ledger corrections #3, #4 | F4 (new), A1 |
| Return-weighting result | move + rewrite | §5 | Promoted to a load-bearing argument (C6) with its own controls | New: C6 | F5, T5 |
| Precursor experiments | move | App. F | Secondary; not mixed with `SIGMA` | Keeps C10 | T5, T6 |
| Native evaluation | move | App. E | User-approved boundary placement | Makes C7/C8 explicit rather than absent | — |
| Limitations | split | §7 + App. E/F | One main-text boundary sentence; detail in appendix | C7, C8, C9 | — |
| Bibliography | audit | — | Fix ID/URL mismatch; wire the uncited entry | — | 2 fixes |
| All figures | replace | — | No figure existed as production plotting code | — | F1–F5 new |

---

## 11. Compliance note

| Rule | Venue/stage | Source | Access date | Consequence | Status |
|---|---|---|---|---|---|
| Main-text limit 9 pages | ICLR 2027, main track, submission | ICLR 2027 author guide (live site) | 2026-09-18 | Manuscript ends on p. 9, within the limit | **Verified** (was O1) |
| Official style file unmodified | ICLR 2027 | `iclr2027_conference.sty` vs official release | 2026-09-18 | sha256 `797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6` — byte-identical; only `\usepackage` additions, no geometry/size/spacing overrides | Verified |
| Double-blind anonymity | ICLR 2027 | style default | 2026-09-18 | Author block is "Anonymous authors"; no identifying links; `\iclrfinalcopy` commented out | Verified locally |
| Line numbers in submission mode | ICLR 2027 | style default | 2026-09-18 | Retained (visible in PDF) | Verified locally |
| Generative-AI disclosure | ICLR policy | — | 2026-09-18 | AI Use Statement rewritten: the AI-assisted categories are now named specifically rather than generically | Statement complete; **author sign-off sentence outstanding** (O3) |
| Reproducibility statement | ICLR policy | — | 2026-09-18 | Present, points to App. A and App. D | Complete |
| Code/data release | ICLR policy | — | 2026-09-18 | Anonymous snapshot assembled and self-containment verified | **Complete** (was O4) |
| Abstract deadline | ICLR 2027 | ICLR 2027 author guide (live site) | 2026-09-18 | **Sep 18 2026, 23:59 AOE.** Deadlines are final, with no accommodations | **Author action — time-critical** |
| Full-paper deadline | ICLR 2027 | ICLR 2027 author guide (live site) | 2026-09-18 | **Sep 25 2026** | **Author action** |
| Author list frozen at abstract deadline | ICLR 2027 | ICLR 2027 author guide (live site) | 2026-09-18 | Authors cannot be added or removed after the abstract deadline | **Author action** |
| Reviewer registration | ICLR 2027 | ICLR 2027 author guide (live site) | 2026-09-18 | Every author must register as a reviewer, or the submission is desk-rejected | **Author action — desk-reject risk** |

The page limit and deadline rules were re-verified against the live ICLR 2027 author
guide on 2026-09-18, not inherited from the repository's submission package. The
anonymity and line-number behaviour of the style file were verified locally against
the manuscript build.

---

## 12. Open issues (prioritised)

Live status is mirrored machine-readably in `longgoal_status.json`; the goal itself,
its definition of done, and its resume protocol are in `LONGGOAL.md`.

| # | Issue | Status | Resolution / what would close it |
|---|---|---|---|
| O1 | Venue rules not re-verified | **Closed** | Verified against the live ICLR 2027 author guide on 2026-09-18: 9-page main text, abstract Sep 18 23:59 AOE, full paper Sep 25. Style file sha256 matches the official release. §11 updated with access dates. |
| O2 | Bibliography metadata not re-fetched | **Closed** | All 12 entries re-verified against primary arXiv abs pages on 2026-09-18 (title + author list + ID): 12/12 agree. The arXiv API returned HTTP 406, so the abs-page route was used. §6 records the result. |
| O3 | AI Use Statement incomplete | **Partly closed** | Rewritten this session: the AI-assisted categories are now named specifically. The authors' own verification sentence remains **author action** — the manuscript cannot attest on their behalf. |
| O4 | No anonymised code/data snapshot | **Closed** | Built and verified: `sigma_supplementary_anonymous.tar.gz` (137 KiB, 24 files). The anonymity scan is clean and the staged extraction runs standalone, which is the proof of self-containment. App. G describes it. |
| O5 | Three `[RESULT NEEDED]` markers | **Closed** | All three converted to honest prose limitation statements (§7 measurement scope; App. A.4 native rollout credit; App. C no-hook distillation). No marker survives in visible text. |
| O6 | C9's hidden-memory family is non-monotone (0.704→0.039→0.290) | **Parked — needs an experiment** | Reported descriptively in App. F. A single seed cannot separate a real non-monotonicity from seed noise; one more seed would close it. Cannot be resolved from this session. |
| O7 | C5's AUROC is within one generator family | **Parked — needs an experiment** | Requires repeating the 200-prefix extraction on a held-out template. Stated as a boundary in §2. |
| O8 | Figure 4 has no plotting code | **No action required** | It is the vector pipeline diagram, not quantitative art; the design pack records its specification and the snapshot README notes it. |

**Priority 1 — blocks submission or would invalidate the central claim.** None. The
central claim (C1–C5) rests on measurements that exist and were re-derived.

**Time-critical and outside this session's control.** The abstract deadline is
2026-09-18 23:59 AOE, deadlines are final with no accommodations, the author list is
frozen at the abstract deadline, and **every author must register as a reviewer or
the submission is desk-rejected.** These are the author's to action.

---

## Handoff checklist

- [x] Manuscript opens and renders (16 pp, main text ends p. 9).
- [x] Claim matrix and manuscript agree; every main-text number has a matrix row.
- [x] All numbers trace to `plotting/extract_evidence.py`, which hard-fails on a
      missing source rather than defaulting.
- [x] Citations audited; two defects fixed; remaining entries marked unverified (O2).
- [x] All 6 tables, 5 figures and 1 algorithm are labelled and called out in order.
- [x] Each numbered non-table figure has a three-candidate design pack.
- [x] Tables are native editable booktabs objects.
- [x] Main claims do not depend on the appendix (C7/C8 are disclosures, not support).
- [ ] Venue rules verified against the live site — **O1**.
- [ ] Anonymised snapshot assembled — **O4**.
- [ ] Working-draft `[RESULT NEEDED]` markers resolved — **O5**.
