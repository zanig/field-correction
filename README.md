# Closure Challenge velocity correction

a small model that predicts velocity corrections from RANS via teacher/student learning.
Phase 1 trains a truth (in this case LES) encoder, predictor and field decoder. Phase 2 trains a
RANS encoder/predictor against the frozen truth representation and decoder.
This is a direct velocity correction and does not enforce any governing equations. I dabbled a bit with divergence-free projections and other constraints but they all make it worse.


## Use the ten-seed ensemble

The equal-weight ten-seed ensemble uses a 9k/3k training schedule and scores
**0.0406128562** with the official evaluator. Individual seeds score between
0.0407 and 0.0487. Each member has 87,523 parameters.

| Case | RANS baseline | Ensemble | Mean member | Worst member |
|---|---|---|---|---|
| `alpha_15_13929_4048` | 0.1320 | 0.0238 | 0.0251 | 0.0294 |
| `alpha_15_13929_2024` | 0.2049 | 0.0151 | 0.0199 | 0.0236 |
| `alpha_05_4071_4048` | 0.0461 | 0.0298 | 0.0320 | 0.0375 |
| `alpha_05_4071_2024` | 0.0719 | 0.0466 | 0.0502 | 0.0597 |
| `AR_1_Ret_360` | 0.1288 | 0.0576 | 0.0593 | 0.0658 |
| `AR_3_Ret_360` | 0.1243 | 0.0491 | 0.0506 | 0.0618 |
| `AR_14_Ret_180` | 0.0590 | 0.0202 | 0.0211 | 0.0256 |
| `NASA_2DWMH` | 0.0621 | 0.0828 | 0.1001 | 0.1260 |
| Overall | 0.1036 | 0.0406 | 0.0448 | 0.0487 |

Mean and worst member are taken per case over the ten single models, so the
worst entry can come from a different seed in each row. The Overall row uses
each member's overall score.

Things to know before trusting that number:
- On the NASA hump the model is worse than doing nothing (the plain RANS baseline).
- No test case was used to fit weights.
- Every setting I tuned was picked on five held-out non-test cases: learning
  rate, weight decay, latent weight, dropout and the regularizer variant
  (27 Oct), the 9k/3k training length (6 Nov), and the model size (18 Nov,
  [script](experiments/size_sweep.py), [results](experiments/size_sweep_results.json)).
  For the size, the largest one had the lowest validation error, though the
  gaps between neighbouring sizes are about as big as the gap between two seeds.
- I also ran the evaluator on the test cases along the way to see where I
  stood: single seeds, a five-seed and a ten-seed ensemble, one run without the
  regularizer, and both training lengths. I did not pick settings from those
  scores. For example, the shorter training length that validation preferred
  scored worse on the test cases for the one seed I checked (0.0490 against
  0.0427), and I kept it.
- Two things did follow from looking at test scores. I went for a ten-seed
  ensemble after seeing on the test cases how much single seeds vary (0.043 to
  0.051 over five seeds) and that averaging two and then five seeds helped; the
  ten seeds and equal weights were fixed beforehand and no member is dropped. And the model size of my first ensemble
  ([version 1](releases/ensemble-9k3k-v1/README.md), 24,691 parameters,
  0.0425321534) was never checked on validation. That is why I redid that
  choice: this version's size comes from the validation sweep, was written down
  together with the seeds and weights before training, and was scored once.
- The hump comes in the x-z plane while all 2D training cases are x-y, and the
  pipeline does not realign it. I tried realigning it with version 1; that
  scored slightly worse (0.0426940638 instead of 0.0425321534). I did not try
  it with this version.

- [Download the full ensemble bundle](releases/ensemble-9k3k-v2/closurechallenge-ensemble-10seed-9k3k-v2.zip): all ten checkpoints, exact source, prediction CSVs, and reproduction instructions.
- [Download the submission CSVs](releases/ensemble-9k3k-v2/closurechallenge-ensemble-10seed-9k3k-v2-predictions.zip): eight CSVs under `test/`.
- [Recipe and evaluation details](releases/ensemble-9k3k-v2/README.md).

Extract the full bundle and follow its README to regenerate the ensemble with
`sh predict-ensemble.sh`.

## Use the included single model

```sh
python predict.py
python score.py
```

These commands use `cache.npz` (from running prep.py), `checkpoints/seed1234.pt`, and `predictions/`.
The evaluator score for this checkpoint is **0.0462587786**, using seed 1234.
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
CPU execution uses one PyTorch thread by default. I ran all of this on a m2 macbook and it runs fairly quick (about 15 mins for one seed).

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
- `experiments/`: optional parameter-search utilities, separate from the main method,
  and the validation-only model-size sweep.
- `releases/ensemble-9k3k-v2/`: downloadable ensemble and submission bundles.
- `releases/ensemble-9k3k-v1/`: the earlier, smaller ensemble, kept for the record.

