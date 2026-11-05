# Cross-path direct-FIR experiment

This branch replaces residual-around-prior generation with a direct FIR generator and adds path-disjoint training/validation/test splits.

## Model output

```text
observation -> Encoder -> Q
scene/path Wiener prior -> frequency tokens -> K/V
cross-attention Decoder -> pooled context
MLP -> normalized complete FIR
w_hat = RMS(w_prior) * normalized FIR
```

The prior supplies environment conditioning and coefficient scale. The network generates all FIR coefficients; it does not compute `w_prior + residual` in direct mode.

For the controlled residual diagnostic, keep the data, bank, seed, architecture,
optimizer, and acoustic loss fixed and change only the output parameterization:

```text
residual mode: w_hat = w_prior + delta_w
initial state: delta_w = 0, therefore w_hat = w_prior
```

The direct and residual runs use separate artifact paths. This diagnostic
separates failures in direct full-FIR generation from failures in prior use:
a useful residual run should start near the direct prior baseline and learn a
small correction, rather than searching for all 1967 coefficients from random
initialization.

## Path protocol

```text
training paths:   1-7
validation paths: 8-9
test path:        10
```

For every scene and path, a Wiener prior is fitted only from that scene/path pair's training interval. The network never trains on target paths 8-10. Validation and test samples receive their own path prior and secondary path.

Window start positions are shared across paths within a split, so scene/path comparisons use identical reference time ranges.

## Run

```bash
source /home/zqbai/.venvs/anc-transformer/bin/activate
cd /home/zqbai/Project/ANC_Transformer

python -m pytest tests -q
python -m data.inspect_data --config config.cross_path.yaml
python -m wiener.build_bank --config config.cross_path.yaml

# One-epoch-style smoke check; inspect its loss and metrics before full training.
python -m experiments.train --config config.cross_path.smoke.yaml
python -m experiments.train --config config.cross_path.residual.smoke.yaml

# Full path-disjoint training.
python -m experiments.train --config config.cross_path.yaml

# Controlled residual-mode comparison; uses the same bank and split.
python -m experiments.train --config config.cross_path.residual.yaml

# Development comparison on validation paths 8-9 only.
python -m experiments.compare --config config.cross_path.yaml --split validation
python -m experiments.compare --config config.cross_path.residual.yaml --split validation

# Paired root-cause audit against each episode's unchanged Wiener prior.
python -m experiments.diagnose_residual --config config.cross_path.residual.yaml --split validation
```

The residual diagnostic writes episode-level NR/MSE deltas, path summaries,
one-third-octave band deltas from 50 Hz through Nyquist, and the episodes contributing
the largest positive MSE change. Positive `residual_mse_delta` and negative
`primary_nr_delta_db` mean the model is worse than the paired prior.

The episode audit reports both the raw MSE change that contributes to the
overall mean and the change relative to each episode's prior residual. The
extended band table covers 50 Hz through Nyquist; `maximum_rebound_db` keeps
the original 1-8 kHz project definition.

Do not run the test comparison until the validation experiment and hyperparameters are frozen:

```bash
python -m experiments.compare --config config.cross_path.yaml --split test
```

## Expected split counts

```text
train:      8 scenes x 7 paths x 128 windows = 7168 episodes
validation: 8 scenes x 2 paths x  16 windows = 256 episodes
test:       8 scenes x 1 path  x  32 windows = 256 episodes
```

## Interpretation limits

- This is cross-path generalisation with the same eight source-noise recordings.
- It is not true cross-recording generalisation.
- Paths 8-10 are unseen by network optimization, but their priors are calibrated from their own training intervals. This assumes an environment calibration stage, which must be stated explicitly.
- No target-window future disturbance is used to construct its prior.
