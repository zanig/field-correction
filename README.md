# Closure Challenge velocity correction

a small model that predicts velocity corrections from RANS via teacher/student learning.
Phase 1 trains a truth (in this case LES) encoder, predictor and field decoder. Phase 2 trains a
RANS encoder/predictor against the frozen truth representation and decoder.
This is a direct velocity correction and does not enforce any governing equations. I dabbled a bit with divergence-free projections and other constraints but they all make it worse.


## Use the included model

```sh
python predict.py
python score.py
```

These commands use `cache.npz` (from running prep.py), `checkpoints/seed1234.pt`, and `predictions/`.
The evaluator score for this checkpoint is **0.0426686240** with seed 1234 (I swear I didnt cherry pick this).
A 5-seed ensemble (1234, 4321, 1337, 1338, 777) scores **0.0436807166**.
Individual seed scores range from **0.0426686240** to **0.0510671909**; seed 1234
has the lowest  error. All seeds used 20k/10k training steps.


## Install dependencies
```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```


## Train a new model

```sh
python -u train.py --out runs/my_run
python predict.py runs/my_run/seed1234.pt --out runs/my_run/predictions
python score.py runs/my_run/predictions --out runs/my_run/scores.json
```

Use a new output directory for each run. Without `--out`, training writes to
`runs/train/`.

Defaults use: 20,000 truth-phase steps, 10,000
RANS-phase steps, seed 1234, and it trains on all 32 non-test cases.
CPU execution uses one PyTorch thread by default. I ran all of this on a m2 macbook and it runs fairly quick (about 15 mins).

There is one objective in both phases: normalized velocity error + `0.1` times
latent alignment error + `0.05` times a SIGReg-inspired Gaussian penalty.
The teacher target is computed without gradients. Point errors are averaged
in each case, normalized by its sampled mean LES/DNS speed, and
then averaged across cases.

heres some override options:

```sh
python train.py --steps 20000 10000 --seeds 1234 --threads 1 --out runs/another_run
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


