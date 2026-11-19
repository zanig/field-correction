# Ten-seed velocity-correction ensemble (9k/3k), version 2

An equal-weight ensemble of ten teacher/student RANS velocity-correction models.
Each member has 87,523 parameters and uses 9,000 reference-phase and 3,000
RANS-phase training steps on all 32 non-test cases. Inference uses RANS fields
and geometry.

- Official evaluator error: **0.0406128562** (`closure-challenge==0.3.1`).
- Individual members: 0.0407 to 0.0487 (mean 0.0448).
- Seeds: `1234 4321 1337 1338 777 9153 17072 2024 2025 42`.
- `closurechallenge-ensemble-10seed-9k3k-v2.zip`: exact source, pinned requirements, ten checkpoints, eight CSVs, scores, hashes, and reproduction instructions.
- `closurechallenge-ensemble-10seed-9k3k-v2-predictions.zip`: the eight submission CSVs under `test/`.

The only change from [version 1](../ensemble-9k3k-v1/README.md) is the model
size (8 latent tokens, latent dimension 32, hidden width 64 instead of 4, 16,
32). It was chosen from a sweep on five held-out non-test validation cases that
never loaded the test split or the evaluator; the script and results are in
`experiments/size_sweep.py` and `experiments/size_sweep_results.json`. The size,
seeds and equal weights were written down before this ensemble was trained, and
it was scored once.

Version 1 scored 0.0425321534 with the same recipe, seeds and pipeline. Public
evaluator scores were previewed while developing it, for training-length,
regularization, ensemble, and frame-alignment experiments, and the decision to
try an ensemble followed single-model test results. Everything here other than
the model size is inherited from that work. All model fitting used non-test
cases. Every ensemble member is included with equal weight.

On `NASA_2DWMH` the model is worse than the uncorrected RANS baseline (0.0828
against 0.0621). That case is supplied in the x-z plane while every 2D training
case is in the x-y plane, and the pipeline does not realign it. Realigning it at
inference was tried with version 1 and scored slightly worse there; it was not
tried with this version.

After extracting the full bundle, follow its README. Scoring the supplied
CSVs needs no raw CFD data; regenerating predictions requires the benchmark
data and preprocessing step.
