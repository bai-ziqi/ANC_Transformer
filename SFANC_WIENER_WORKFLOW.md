# SFANC clip labels and independent environment priors

This experiment uses 16-kHz WAV files and independent lengths for PP (512), SP
(256), and the control FIR (1024 coefficients, order 1023). Each environment has
one independent broadband prior, and every clip has its own target FIR.

## Server inputs

`config.sfanc.yaml` and `config.sfanc.smoke.yaml` use:

```text
/home/zqbai_1/Dataset/Synthesized_Dataset/
    Training_data/    80000 WAV files, recursive discovery
    Validate_data/     2000 WAV files
    Testing_data/      2000 WAV files
/home/zqbai_1/Dataset/Path/path_1/
    Primary_path.mat     512 coefficients
    Secondary_path.mat   256 coefficients
```

WAV files must be mono, finite, 16 kHz, and have the same clip length. There is no
implicit normalization, resampling, random bandpass, cropping, or padding.
The original corpus splits are retained. Source identities use path, size, and
mtime; copied audio or overlap in the original generator cannot be proven absent
from filenames alone. A MAT file with one numeric vector is selected automatically;
if multiple vectors exist, set `primary_key` or `secondary_key` to the actual
variable name. MATLAB v7.3/HDF5 paths require conversion to a supported format.

The supplied path sample rate is declared as 16 kHz. If MAT rate metadata exists,
it must agree. A declaration does not establish that an undocumented impulse
response was measured at 16 kHz; confirm that with the path source. Gain, leading
delay, and coefficient order are preserved.

## Two distinct artifacts

1. `wiener.build_targets` pairs each clip with its configured environment, forms
   `d=P*x` and `z=S*x`, then solves the pure finite-window least-squares problem
   `min_w ||d - causal_conv(z,w)||^2` with matrix-free LSMR and no ridge term or
   prior anchor. The fixed 1024-tap FIR is fitted and scored over the complete clip,
   including startup with causal zero initial states. A label is accepted only when
   LSMR reports a successful least-squares stop and an independently recomputed
   `A.T(d-Aw)` satisfies the configured scaled tolerance. Iteration-limit and
   condition-limit stops fail generation rather than silently producing a label.
   Every row records LSMR status, residual and normal-equation diagnostics, the
   float64 and float32 residual, and comparisons against the independent prior and
   the legacy Toeplitz candidate. The legacy Toeplitz path remains diagnostic only.
2. `wiener.build_priors` never reads corpus clips. For each environment it generates
   independent 20-7980-Hz broadband fit and selection streams using named random
   seeds and causal Butterworth SOS filtering. Path/filter warmup is retained, then
   excluded from calibration scoring. It writes one 1024-tap prior per environment.
   The band edges are Butterworth cutoff frequencies, not ideal brick-wall edges.

Targets are float32 compressed chunks plus `index.json` and least-squares diagnostics.
Computation and optimum checks use float64; stored-target MSE after float32
quantization is recorded separately. A configured non-null quantization threshold
can make an excessive increase fail generation.
Priors are exported as PT, NPZ, and JSON. Checksums bind the labels, source manifest,
paths, configuration, and checkpoints. A configuration change requires compatible
artifacts or a new output directory. `--resume` verifies existing chunks before
continuing; it does not silently reuse labels from another experiment.

## Inspect and run a small pilot

Run inside the server checkout with its Python environment activated:

```bash
python -m data.sfanc --config config.sfanc.smoke.yaml --limit 8
python -m wiener.build_priors --config config.sfanc.smoke.yaml
python -m wiener.build_targets --config config.sfanc.smoke.yaml --limit 8 --resume
CUDA_VISIBLE_DEVICES=0 python -m experiments.train --config config.sfanc.smoke.yaml
CUDA_VISIBLE_DEVICES=0 python -m experiments.compare --config config.sfanc.smoke.yaml --split validation
```

`--limit` is per original split, before environment pairing. Expected original
source counts are still checked; the subset manifest/cache is explicitly marked
partial. Only the smoke configuration opts into partial training. It uses separate
artifact paths so it cannot be mistaken for the full 84000-clip run.

The initial inspection reports MAT/vector, sample-rate, split-count, and clip-length
errors before expensive computation. Review these results and least-squares diagnostics
before launching the full run. An all-zero/invalid sample is not silently skipped.

## Full generation and training

```bash
python -m data.sfanc --config config.sfanc.yaml
python -m wiener.build_priors --config config.sfanc.yaml
python -m wiener.build_targets --config config.sfanc.yaml --resume
CUDA_VISIBLE_DEVICES=0 python -m experiments.train --config config.sfanc.yaml
CUDA_VISIBLE_DEVICES=0 python -m experiments.compare --config config.sfanc.yaml --split validation
```

Generation runs on CPU; training uses the selected GPU. Interrupted target generation
can be continued with the same command. The first baseline is epoch 0, followed by
validation-only checkpoint selection. Test labels can be generated offline with the
predefined solver, but test results must not select epochs or alter the rule. After
the method and hyperparameters are frozen, the final command is:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.compare --config config.sfanc.yaml --split test
```

## Training and interpretation

The model entry keeps physical paths separate from the network input. PP512 and
SP256 are used by label construction and acoustic simulation; neither is fed into
the Encoder/Decoder as a waveform or a prior. The prior tokenizer input and FIR
head output both use `filter_length=1024`. With the default 1-second, 16-kHz input:

```text
observation       [B,16000]
Encoder tokens    [B,122,64]
environment prior [B,1024] -> prior tokens [B,32,64]
output/target FIR [B,1024]
```

Check these dimensions without reading the server corpus or MAT files:

```bash
python -m models.inspect_model --config config.sfanc.yaml --samples 16000 --batch-size 2
```

This command uses synthetic inputs and dummy normalization to check shapes, not
trained weights or calibrated filters. Actual training still loads train-derived
normalization and the generated artifacts. The model checks path/prior dimensions,
rejects malformed waveform inputs, and converts floating input tensors to its
parameter dtype/device. Normalization and target tensors cannot silently broadcast.
Train this experiment with `config.sfanc.yaml`; checkpoints must match its
1024-tap tokenizer, head, and artifact signatures.

The dataset supplies separate `prior_filter` and `target_filter` tensors. Only the
current noise and its environment prior enter the Transformer. The main training
loss is normalized FIR error relative to `target_filter`; no acoustic simulation
is required for a filter-only loss. Feature normalization is fitted on train clips
only. The baseline methods are `zero_filter`, `environment_prior`, `sample_wiener`,
and `transformer`. Results include residual MSE, broadband energy NR, target-filter
error, and change relative to the prior. The 50-5000-Hz equal-band NR is
an auxiliary statistic, not the full-window least-squares label objective.

The default observes the whole WAV and fits/scores the same whole clip, including
startup. This is an offline same-clip experiment, not a real-time promise that a
filter generated after observation could have controlled the past. A causal
observe-then-control experiment needs its own future scoring protocol and labels.

The initial configuration has only `path_1` in every split. It measures adaptation
to unseen noise clips in that environment; cross-path claims require additional
environments and disjoint path sets. No fixed six- or fifteen-band labels are used.

The complete 84000 x 1024 float32 target array is about 0.32 GiB. By default a dataset
caches chunks without eviction to avoid repeated decompression during random
training. `sfanc.target_cache_chunks` can cap this per-worker cache. Do not blindly
increase `num_workers`, because each worker has an independent cache.
