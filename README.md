# Closure Challenge velocity correction

a small model that predicts velocity corrections from RANS via teacher/student learning.
Phase 1 trains a truth (in this case LES) encoder, predictor and field decoder. Phase 2 trains a
RANS encoder/predictor against the frozen truth representation and decoder.
This is a direct velocity correction and does not enforce any governing equations. I dabbled a bit with divergence-free projections and other constraints but they all make it worse.


## Use the ten-seed ensemble

The equal-weight ten-seed ensemble uses a 9k/3k training schedule and scores
**0.0425321534** with the official evaluator. Individual seeds score between
0.0443 and 0.0582.

| Case | RANS baseline | Ensemble |
|---|---|---|
| `alpha_15_13929_4048` | 0.1320 | 0.0244 |
| `alpha_15_13929_2024` | 0.2049 | 0.0204 |
| `alpha_05_4071_4048` | 0.0461 | 0.0306 |
| `alpha_05_4071_2024` | 0.0719 | 0.0451 |
| `AR_1_Ret_360` | 0.1288 | 0.0571 |
| `AR_3_Ret_360` | 0.1243 | 0.0557 |
| `AR_14_Ret_180` | 0.0590 | 0.0215 |
| `NASA_2DWMH` | 0.0621 | 0.0854 |
| Overall | 0.1036 | 0.0425 |

Things to know before trusting that number:
- On the NASA hump the model is worse than doing nothing (the plain RANS baseline).
- No test case was used to fit weights. Learning rate, weight decay, latent
  weight, dropout, the regularizer variant and the 9k/3k schedule were compared
  on five held-out non-test cases.
- I did preview evaluator scores on the test cases during development (training
  length, with/without the regularizer, single seeds vs ensemble), and I tried
  the ensemble after seeing how much single-seed test scores vary.
- The hump comes in the x-z plane while all 2D training cases are x-y, and the
  pipeline does not realign it. I tried realigning it afterwards; that scored
  0.0426940638, and the released predictions are from the original pipeline.

- [Download the full ensemble bundle](releases/ensemble-9k3k-v1/closurechallenge-ensemble-10seed-9k3k-v1.zip): all ten checkpoints, exact source, prediction CSVs, and reproduction instructions.
- [Download the submission CSVs](releases/ensemble-9k3k-v1/closurechallenge-ensemble-10seed-9k3k-v1-predictions.zip): eight CSVs under `test/`.
- [Recipe and evaluation details](releases/ensemble-9k3k-v1/README.md).

Extract the full bundle and follow its README to regenerate the ensemble with
`sh predict-ensemble.sh`.

## Use the included single model

```sh
python predict.py
python score.py
```

These commands use `cache.npz` (from running prep.py), `checkpoints/seed1234.pt`, and `predictions/`.
The evaluator score for this checkpoint is **0.0489819221**, using seed 1234.
It was retrained from the current code with 9k/3k steps and all 32 non-test cases.
The recipe, dependency versions, and source hashes are recorded in
[checkpoints/seed1234.json](checkpoints/seed1234.json).

## Install dependencies
```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use the pinned dependencies for both preprocessing and training. Different NumPy
versions can produce slightly different cached array values.


## Train a new model

```sh
python -u train.py --out runs/my_run
python predict.py runs/my_run/seed1234.pt --out runs/my_run/predictions
python score.py runs/my_run/predictions --out runs/my_run/scores.json
```

Use a new output directory for each run. Without `--out`, training writes to
`runs/train/`.

Defaults use: 9,000 truth-phase steps, 3,000
RANS-phase steps, seed 1234, and it trains on all 32 non-test cases.
CPU execution uses one PyTorch thread by default. I ran all of this on a m2 macbook and it runs fairly quick (about 6 mins).

The schedule was selected from an exploratory one-seed comparison on five held-out validation cases, where 9k/3k had slightly lower validation error than 20k/10k.

There is one objective in both phases: normalized velocity error + `0.1` *
latent alignment error + `0.05` * a SIGReg-inspired Gaussian penalty.
Point errors are averaged in each case, normalized by its sampled mean LES/DNS speed, and
then averaged across cases.

heres some override options:

```sh
python train.py --steps 9000 3000 --seeds 1234 --threads 1 --out runs/another_run
python train.py --config runs/my_run/config.json --out runs/reproduction
```


## holding out validation for hyperparameters optimization
```sh
python train.py --holdout-validation --out runs/validation_run
python validate.py runs/validation_run/seed1234.pt
```

## preparing data

```sh
python prep.py /path/to/closure-challenge-benchmark/data --out cache.npz
```

 data directory needs to contain `Parm_PH_29`, `DUCT`, `CBFS`, `PH_Breuer`,
`NASA_2DWMH`, and `evaluation_points`.
please refer to [this repo](https://github.com/rmcconke/closure-challenge-benchmark) for the data.

file overview
- `prep.py`, `data.py`: field reading, geometry, case splits and normalization.
- `model.py`, `train.py`: model architecture and two-phase training.
- `predict.py`, `score.py`: CSV export and official test scoring.
- `validate.py`: held-out evaluation.
- `experiments/`: optional parameter-search utilities, separate from the main method.
- `releases/ensemble-9k3k-v1/`: downloadable ensemble and submission bundles.

