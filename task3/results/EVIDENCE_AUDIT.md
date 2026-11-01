# ATML PA3 — Task 3 Evidence Audit Report

**Overall Status:** `COMPLETE`
**Repository Isolation (task1 & task2 preserved):** `PASS`

## 1. Required Tables (T00 – T17)

| ID | Status | File | Rows |
|---|---|---|---|
| T00 | ✅ Complete | T00_source_data_preflight.csv | 2 |
| T01 | ✅ Complete | T01_split_manifest_summary.csv | 2 |
| T02 | ✅ Complete | T02_scorer_normalization_audit.csv | 2 |
| T03 | ✅ Complete | T03_firstans_oracle_diversity.csv | 12 |
| T04 | ✅ Complete | T04_offline_method_comparison.csv | 84 |
| T05 | ✅ Complete | T05_paired_accuracy_differences.csv | 8 |
| T06 | ✅ Complete | T06_signal_length_correctness.csv | 9 |
| T07 | ✅ Complete | T07_confidence_vision_relationship.csv | 1 |
| T08 | ✅ Complete | T08_fusion_comparison.csv | 4 |
| T09 | ✅ Complete | T09_live_question_results.csv | 24 |
| T10 | ✅ Complete | T10_live_strategy_summary.csv | 3 |
| T11 | ✅ Complete | T11_refinement_transition_matrix.csv | 4 |
| T12 | ✅ Complete | T12_adaptive_dev_grid.csv | 52 |
| T13 | ✅ Complete | T13_adaptive_heldout_results.csv | 4 |
| T14 | ✅ Complete | T14_premature_stop_recovery.csv | 12 |
| T15 | ✅ Complete | T15_qualitative_case_index.csv | 4 |
| T16 | ✅ Complete | T16_resource_run_inventory.csv | 9 |
| T17 | ✅ Complete | T17_evidence_completeness_audit.csv | 34 |

## 2. Required Figures (P01 – P16)

| ID | PNG Status | PDF Status | File |
|---|---|---|---|
| P01 | ✅ | ✅ | P01_math_accuracy_oracle.png |
| P02 | ✅ | ✅ | P02_mathvista_accuracy_oracle.png |
| P03 | ✅ | ✅ | P03_answer_diversity.png |
| P04 | ✅ | ✅ | P04_failure_decomposition.png |
| P05 | ✅ | ✅ | P05_math_cost_tradeoff.png |
| P06 | ✅ | ✅ | P06_mathvista_cost_tradeoff.png |
| P07 | ✅ | ✅ | P07_quartile_analysis.png |
| P08 | ✅ | ✅ | P08_confidence_vision.png |
| P09 | ✅ | ✅ | P09_fusion_comparison.png |
| P10 | ✅ | ✅ | P10_live_accuracy_tokens.png |
| P11 | ✅ | ✅ | P11_live_runtime_breakdown.png |
| P12 | ✅ | ✅ | P12_refinement_transitions.png |
| P13 | ✅ | ✅ | P13_adaptive_dev_grid.png |
| P14 | ✅ | ✅ | P14_adaptive_heldout_comparison.png |
| P15 | ✅ | ✅ | P15_stopping_distributions.png |
| P16 | ✅ | ✅ | P16_premature_stops.png |

## 3. Manifests & Verification

- MATH Split Manifest: `Found`
- MathVista Split Manifest: `Found`
- Adaptive Selected Policies: `Found`
- Qualitative Cases Gallery: `Found`

## 4. Integrity and Compliance
- All seedings strictly anchored to assignment seed `6304`.
- No gold answer leakage into selection, voting, RWC, or adaptive decisions.
- FirstAns cost proxy correctly accounts for single candidate only.
- Parallel and sequential refinement live generation share the first 1,024-token response.
