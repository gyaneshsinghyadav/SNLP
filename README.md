# RC-DCLP — Final Research Codebase

**Retention-Constrained DQN Continual Link Predictor**  
**Final selected model = E4 retention-aware checkpoint selection ON TOP OF E3 representative replay + historical-logit distillation.**

This package restores the **original measured implementation**, including the unchanged model, DQN, replay and retention code, together with **all five authentic archived frozen GNN checkpoints, all five DQN checkpoints, the DBLP experimental CSV logs, manifests, matched E3 reference logs, and a reproducible train / verify / evaluate command interface**.

**Not part of the final model:** later multi-trajectory DQN (E5), encoder learning-rate variants (E6b). Neither is used in training or inference here.

## Architecture

Historical, undirected DBLP collaboration graph → 12 structural pair features and bounded two-hop enclosing subgraphs → gated two-layer GNN (8-D per-node inputs, 48-D states, at most 32 subgraph nodes) + structural MLP + prequential logistic-regression logit → residual link-scoring decoder → sigmoid output. TGAT forward features are **not active** in `structural_subgraph` mode.

The separate Double-DQN controller receives 25 state features and chooses one of six replay/EWC/gradient-projection interventions. The continual learner uses representative memory (capacity 256) and historical-logit MSE (weight 0.1). After each training epoch, retention-constrained checkpoint selection admits only models with no more than 2 percentage points *incremental* loss of old-task validation AP, and can roll back **both** GNN weights and AdamW optimizer state.

## Quick start: inspect original trained model, without DBLP CSV

Python 3.10+ (Google Colab recommended for training):

```bash
python -m pip install -e '.[test]'
python scripts/rc_dclp.py verify
python scripts/rc_dclp.py compare --output outputs/paired_comparison
python -m pytest -q tests/test_final_release.py
```

`verify` checks the weight files, seed/manifests, saved results and leakage audit. `compare` regenerates the five-seed comparison using the included matched reference. These commands do **not** retrain.

## Use the recorded final models to score frozen test candidates

Your **original DBLP file** is required. It is intentionally not included because it was not supplied in these archives. Put it at e.g. `data/dblp_tgat_5000.csv` and verify:

```
SHA256: 3cade711abbe2452f1bf22b1ed938ed46640d58651114676e85aea4198578744
```

Then:

```bash
python scripts/rc_dclp.py predict --csv data/dblp_tgat_5000.csv --seed 7 --years 2015,2016 --evaluate --output outputs/seed7_frozen_scores.csv
```

This faithfully rebuilds historical graph features, candidate splits, bounded enclosing subgraphs and the **prequential LR anchor** from the exact CSV; it checks candidate fingerprints against the saved run and loads the seed's real frozen predictor weights. Outputs are scores for the frozen *sampled evaluation candidates* under the archived DBLP protocol. **It is not a generic arbitrary-node production API.** Label use with `--evaluate` is post-hoc metric reporting only.

The sklearn LR anchor is rebuilt from the original chronological training-fit subsets; it was not serialized as part of the GNN `.pt` file. The replay buffer and optimizer state from 2014 were not archived, so **resuming training from a frozen checkpoint is not supported**. For continued training, rerun the chronological trainer.

## Retrain the exact final E4-on-E3 configuration

```bash
python scripts/rc_dclp.py train --csv data/dblp_tgat_5000.csv --seeds 7 --device cpu --results-dir runs/final
python scripts/rc_dclp.py train --csv data/dblp_tgat_5000.csv --seeds 7,13,42,101,202 --device cpu --results-dir runs/final
python scripts/rc_dclp.py compare --results-dir runs/final --output outputs/new_comparison
```

`train` is a constrained wrapper around the **original hash-locked E4 training runner**, with `e4_retention` forced. It verifies the CSV's SHA256, default temporal split, and candidate hashes against `docs/e6e_e6d_reference`. The original final configuration: six epochs maximum per year, patience 3, 256 replay items, full/light replay 64/16, DQN state 25/actions 6, gamma 0.9, distillation 0.1, checkpoint tolerance 2 pp. Years **2003–2014** are training; **2015 and 2016** are strictly frozen-parameter evaluations.

Do not reuse a run directory for altered source code, seeds, or hyperparameters; experiment locks reject incompatible configurations. Training the five seeds is computationally substantial and has **not** been rerun in this release.

## Verified existing findings (archived DBLP experiments)

| Metric | DQN–RRD without retention gate | **Final RC-DCLP** |
|---|---:|---:|
| Test Average Precision | 90.585% | **90.529%** |
| ROC-AUC | ~89.167% | **~89.172%** |
| Mean historical validation AP drop | 2.694 pp | **0.450 pp** |
| Mean peak historical drop | 21.177 pp | **3.173 pp** |
| Oldest-task final AP | ~61.685% | **~79.096%** |

The five random seeds were **7, 13, 42, 101, 202**. The recorded forgetting metric uses *past training-year controller-validation tasks*, and these same tasks influence checkpoint selection. Thus it is a **selection-conditioned retention measure**, not an independent held-out forgetting test. The final model rolled back many updates, especially in later years. AP is measured on sampled balanced candidates, **not classification accuracy or full-graph production AP**. The main paper should disclose these limitations.

## Project layout

```
RC_DCLP_FINAL/
├── scripts/rc_dclp.py              # Final supported CLI: verify/train/predict/compare
├── scripts/e6g_run_suite.py         # Preserved exact E4-on-E3 trainer
├── scripts/e6g_compare.py          # Preserved five-seed statistics and plots
├── scripts/e6c_run_suite.py         # Causal task construction and historical LR
├── scripts/e6d_run_suite.py         # Subgraph packet extraction and model scoring
├── src/adaptive_graph/             # Full original research model dependencies
├── docs/e6g_e3_reference/          # Original matched RRD reference, read-only
├── docs/e6e_e6d_reference/         # Candidate fingerprint reference, read-only
├── recorded/e6g_checkpoint/        # Five E4-on-E3 predictors, DQNs, CSV logs
├── recorded/e6g_checkpoint_analysis/ # Five-seed results and comparisons
├── tests/test_final_release.py     # Actual bundled weights/results verification
├── tests/test_e6g_checkpoint.py    # Original synthetic retention/rollback tests
└── RC_DCLP_Final_Colab.ipynb       # Colab setup, inspect, predict, optional retrain
```

The older internal module filenames remain unchanged as dependencies to avoid breaking the **validated original implementation**; they do not mean E5 or E6b is included in the final active training path.

## How to work with it

For a new machine or a clean Google Colab runtime, install requirements and use the provided notebook. Start with `verify` which needs no dataset, then add the original CSV for frozen predictions or reproducible retraining. `--device cuda` is available with CUDA-compatible PyTorch.

## Scientific provenance

- Preserved source: `E6G_GNN_DQN_E4_Complete.zip` (pre-E5/E6b implementation)
- Preserved training outcomes: `E6G_E4_DBLP_FiveSeed_Results(1).zip`
- Preserved comparison outcomes: `E6G_E4_DBLP_Analysis(1).zip`
- E3 matched reference: archived `docs/e6g_e3_reference`
- No new real-data results are claimed in the packaged release.
