# EEG + EMG → Finger Force Prediction

Predicting continuous finger force from non-invasive neural and muscular signals, using
classical gradient boosting and recurrent deep learning, with a real-time inference path.

Research internship project, 2025.

## Overview

The pipeline takes simultaneously recorded **EEG** (31 channels, BrainVision), **EMG**
(40 channels, high-density grid), and **force** (2048 Hz, 5 digits) from a grasping task,
aligns the three streams trial-by-trial, extracts spectral and time-domain features, and
regresses continuous per-digit force.

Two model families are compared, each with Optuna hyperparameter search:

| Model | Input | Notes |
|---|---|---|
| **XGBoost** | Windowed feature vectors | Baseline; grid search, random search, and Optuna variants |
| **LSTM** | Sequences of 5 feature windows | Best performer; also a bidirectional variant |

Both are run in **EEG-only** and **EEG+EMG fusion** configurations, which is the central
experimental question — how much the muscular signal adds over the cortical signal alone.

## Pipeline

```
raw EEG (.vhdr) ─┐
raw EMG (.pkl)  ─┼─► data_loader ─► preprocessing ─► features ─► [sequencer] ─► model ─► results
force (.pkl)    ─┘   align trials    filter/CSD/ICA   spectral      windows      XGB/LSTM
```

1. **`src/data_loader.py`** — loads BrainVision EEG via MNE, EMG and force from pickled
   DataFrames, resamples force from 2048 Hz to the EEG rate (polyphase), and aligns all three
   per trial, truncating to a common length.
2. **`src/preprocessing.py`** — bandpass filtering (0.1–48 Hz, 4th-order Butterworth IIR),
   current source density transform, and ICA artifact removal.
3. **`src/features.py`** — per 200 ms window: alpha (8–12 Hz), beta (13–30 Hz), and gamma
   (30–48 Hz) band power via multitaper PSD, plus RMS and variance. EMG contributes RMS per
   channel. 80 EEG + 40 EMG = 120 features.
4. **`src/data_sequencer.py`** — stacks windows into length-5 sequences for the LSTM.
5. **`src/model.py`** / **`src/train.py`** — XGBoost and `LSTMRegressor` (120 in → 10 out),
   both wrapped in Optuna studies.
6. **`src/visualize.py`** — loss curves, per-signal prediction overlays, Optuna history,
   XGBoost feature importance.

### Channel selection

`analysis/01_find_best_channels.py` and `analysis/02_find_channels_with_lasso.py` reduce the
31 EEG channels to the most predictive subset (correlation-based and Lasso-based respectively)
before feature extraction.

## Running it

Dependencies are managed with [uv](https://github.com/astral-sh/uv); Python 3.13.

```bash
uv sync
```

Configure the experiment with the two flags at the top of `main.py`:

```python
USE_EMG_FEATURES    = True   # True = EEG+EMG fusion, False = EEG only
RUN_LSTM_EXPERIMENT = True   # True = LSTM, False = XGBoost
```

```bash
uv run main.py
```

The pipeline caches each stage to `.pkl`, so re-runs skip straight to the expensive part.
Delete the caches to force a full recompute.

### Real-time inference

`real_time_inference.py` connects to a Lab Streaming Layer (LSL) stream from the BrainVision
connector, applies the same preprocessing to 50 ms chunks, and emits a rolling force
prediction. `test_inference_speed.py` benchmarks the per-window latency.

## Repository layout

```
├── main.py                    experiment driver
├── real_time_inference.py     LSL streaming inference
├── test_inference_speed.py    latency benchmark
├── analysis/                  channel-selection studies
├── src/                       pipeline modules
└── experiments/               result figures and logs per configuration
```

`experiments/` holds the output of each configuration that was run — LSTM fusion variants
(35-epoch, early-stopping, bidirectional, sequence-length sweeps), the four XGBoost tuning
strategies (default, 200-iteration, grid search, Optuna), and Optuna post-processing plots.

## Data availability

**Raw data is not included.** The recordings are human-subject EEG/EMG from a pilot session
and are not mine to redistribute; `data/` is empty by design. Trained model weights and
cached intermediates (~3 GB of `.pkl`/`.pth`) are likewise excluded — they exceed GitHub's
file size limits and are reproducible from the pipeline.

To run against your own recordings, place them under `data/<session>/` matching the paths at
the top of `src/data_loader.py`.

## Known issues

- `real_time_inference.py` references `montage` and `ica` (lines 108/111) that the elided
  "load or generate preprocessing tools" block at line 28 was meant to define. It will raise
  `NameError` until that block is restored from `real_time_preprocessing_tools.pkl`.
- Hyperparameters in `real_time_inference.py` are hardcoded from one specific Optuna run
  rather than loaded alongside the checkpoint.
