# Ten-seed velocity-correction ensemble (9k/3k)

An equal-weight ensemble of ten teacher/student RANS velocity-correction models.
Each member uses 9,000 reference-phase and 3,000 RANS-phase training steps on
all 32 non-test cases. Inference uses RANS fields and geometry.

- Official evaluator error: **0.0425321534** (`closure-challenge==0.3.1`).
- Seeds: `1234 4321 1337 1338 777 9153 17072 2024 2025 42`.
- `closurechallenge-ensemble-10seed-9k3k-v1.zip`: exact source, pinned requirements, ten checkpoints, eight CSVs, scores, hashes, and reproduction instructions.
- `closurechallenge-ensemble-10seed-9k3k-v1-predictions.zip`: the eight submission CSVs under `test/`.

The schedule followed an exploratory one-seed held-out validation comparison.
Public evaluator scores were also previewed during development for training-
length, regularization, and ensemble experiments. All model fitting used
non-test cases. Every ensemble member is included with equal weight.

After extracting the full bundle, follow its README. Scoring the supplied
CSVs needs no raw CFD data; regenerating predictions requires the benchmark
data and preprocessing step.
