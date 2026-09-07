# Decision record — training round budget

**Date of decision: 7 September 2026**
**Author: Charan Rajashekar**
**Status at time of writing: M1 fitted at the registered budget. M2 HAS NOT BEEN FITTED.
No M1-vs-M2 contrast has been computed or observed.**

Pre-registration: DOI 10.17605/OSF.IO/37QJK, registered 25 August 2026 (pre-analysis).

---

## 1. The defect

The registration fixes two stopping rules for both M1 and M2:

```yaml
n_estimators: 500
early_stopping_rounds: 50      # on validation PR-AUC
```

It does not specify which rule governs **when the round cap binds before the patience
does**. That omission is the defect being resolved here. The registered numbers
themselves are not in question and are not being changed.

## 2. The observation that exposed it

M1 fitted 4 September 2026, 500 rounds, 5m 10.9s:

| Quantity | Value |
|---|---|
| `best_iteration` | 481 of a 500 cap |
| Rounds elapsed after the peak | 18, against a patience of 50 |
| Early stopping fired? | **No** |
| Validation aucpr at round 500 | 0.0516, still rising |
| Validation aucpr trajectory (per 50 rounds) | 0.0094 · 0.0231 · 0.0301 · 0.0362 · 0.0407 · 0.0438 · 0.0458 · 0.0474 · 0.0486 · 0.0509 · 0.0516 |
| Train aucpr at round 499 | 0.10083 |

The model reached the round cap. It did not converge. The per-50-round increment is
decelerating (final 100 rounds: +0.0030) but is not zero, so the registered budget is
a binding constraint on both arms rather than an inactive ceiling.

Leakage is not indicated: validation PR-AUC of 0.052 sits approximately 52x above the
validation base rate of 0.0998%, far below the registered label-permutation trigger of
PR-AUC > 0.90.

## 3. Decision

**The registered 500-round budget remains the confirmatory analysis.** The verdicts on
H1, H2 and H4 are taken from models fitted at `n_estimators = 500`,
`early_stopping_rounds = 50`. This is not altered by anything below.

**A supplementary higher-budget refit is added, pre-specified here, before M2 exists.**

### 3.1 Supplementary specification

| Parameter | Value |
|---|---|
| `n_estimators` | 2000 |
| `early_stopping_rounds` | 50 (unchanged) |
| Every other hyperparameter | Unchanged from the registered configuration |
| `random_state` / `seed` | 42 (unchanged) |
| Split index | Unchanged |
| Applies to | M1 and M2, identically |
| Only difference between arms | M2's feature list includes the 16 graph features |

### 3.2 Rationale for 2000

Stated honestly as a resource-bounded operationalisation, not a derived optimum:

The governing rule is *raise the cap until early stopping, rather than the ceiling,
is the binding constraint*, subject to a declared compute budget of two hours on the
project machine. At the measured 0.62 s/round for M1, a 2000-round cap gives
approximately 21 minutes for M1 and an estimated 35-40 minutes for M2 (26 features
against 10), which fits that budget. A larger cap does not.

2000 is also approximately 4x the observed `best_iteration` of 481, which on a
decelerating validation curve is a reasonable expectation for the patience to bind.

### 3.3 Fallback rule if early stopping still does not fire at 2000

No further increase. The non-firing is reported as a finding in its own right: that
under the registered configuration, gradient-boosted models on HI-Small had not
converged at four times the registered budget. The cap is not raised a second time,
because a second raise after a second observation is exactly the iterative,
data-dependent budget selection this record exists to prevent.

### 3.4 Pre-committed rule for disagreement

**Fixed before the supplementary is run.**

If the supplementary result disagrees with the registered 500-round result on any of
H1, H2 or H4:

- the **registered 500-round result stands as the answer** to that hypothesis;
- the disagreement is reported in full, in the findings section, as a limitation on
  the robustness of that answer rather than as a competing result;
- no reconciliation, averaging, or selection between the two is performed.

If the two agree, the supplementary is reported as a robustness confirmation.

### 3.5 SHAP at the higher budget — DEFERRED, not omitted

TreeSHAP cost scales with the number of fitted trees, so a 2000-round model costs
roughly 4x the SHAP time of a 500-round model. The SHAP timing pilot has not yet been
run, so the feasibility of a higher-budget SHAP run is unknown.

**Deferred decision, to be recorded in a dated addendum to this file immediately after
the SHAP timing pilot completes.** If it is not feasible within the declared budget,
H2 and H4 are reported at the registered budget only, and the absence of a
higher-budget explanation comparison is stated as a limitation. It is not silently
dropped.

## 4. Why this is defensible

Three properties, all checkable by a third party:

1. **Chronology.** This record is committed to the repository before M2 is fitted. The
   commit timestamp is the evidence. At the moment of the decision only the baseline
   arm existed, so no budget choice can have been made to favour either arm — the
   contrast had not been seen.
2. **Symmetry.** The budget is identical across arms. A shared budget cannot
   preferentially advantage the arm with more features.
3. **Subordination.** The registered analysis remains confirmatory. The supplementary
   cannot alter a verdict; §3.4 removes all discretion in advance.

This is a **pre-specified supplementary analysis**, not a deviation from the
registration. The registered parameters are unchanged and the registered analysis is
still run and still decides. No addendum to the OSF record is required; this file and
its commit constitute the internal record, and the report will describe it in these
terms.

## 5. Consequential code change (applies to both notebooks)

Prediction must be truncated identically for both models. Where `best_iteration`
differs between M1 and M2, relying on library default behaviour risks scoring the two
models on different numbers of trees, which would breach Audit 3 of
`07_Validity_Audit_Playbook.md` (identical protocol, single manipulated variable).

Set the range explicitly in Notebook 06 and Notebook 07:

```python
p_test = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
```

## 6. Reporting commitments

| Where | What is reported |
|---|---|
| §2 Methodology | One sentence: budget registered, cap bound first, supplementary pre-specified on this date |
| §3B Findings | Registered verdicts; supplementary values in the same table, labelled supplementary |
| §4 Discussion | Whether the H1/H4 conclusions are budget-sensitive |
| §5 Recommendations | Model documentation should record training budget and convergence status |
| §6 Critical reflection | The under-specification, how it was found, how it was resolved |

## 7. Completion tests

A test is met by execution and evidence, not by mention.

- [ ] This file committed to the repository, on a commit whose timestamp precedes the
      first M2 fit. **Test: the commit is visible in the GitHub web UI and its
      timestamp precedes the Notebook 07 execution.**
- [ ] `iteration_range` set explicitly in Notebook 06 and Notebook 07.
      **Test: the string appears in both notebooks.**
- [ ] M1 supplementary fit executed at 2000 rounds; `best_iteration` and whether early
      stopping fired both recorded. **Test: both values printed in notebook output.**
- [ ] M2 fitted at 500 (registered) and 2000 (supplementary), same seed, same split.
      **Test: four fits exist with recorded metrics.**
- [ ] SHAP addendum written after the timing pilot. **Test: a dated §3.5 addendum
      exists in this file, stating feasible or not feasible.**
- [ ] `00_CURRENT_STATE.md` updated with this decision.

---

*Recorded 7 September 2026, before M2 was fitted.*
