# depression-detection-daic-woz
Depression detection from speech &amp; transcripts using the DAIC-WOZ dataset

## Project Overview

This project builds a multimodal pipeline for **depression detection** using the **DAIC-WOZ** interview dataset.  
The goal is to predict a binary depression label (PHQ-based) from **speech audio** and **transcripts**.

The workflow includes:
- Data preparation: organizing raw files, creating unified labels, generating participant-only audio, and segmenting audio into fixed windows.
- Feature extraction:
  - Voice activity / silence features (VAD + energy-based metrics)
  - Classical acoustic features (OpenSMILE eGeMAPS)
  - Deep audio embeddings (WavLM-Large)
  - transcript-based features via LLM text baselines
- Modeling and evaluation:
  - audio-only models
  - multimodal fusion experiments (audio + text)
  - threshold calibration and final model packaging for deployment

Outputs include structured metadata, extracted features, evaluation reports, and a deployment-ready model package (model + scaler + threshold + feature list).


# Notebook 00 — Preparing DAIC-WOZ Files (Documented)

This notebook prepares the **DAIC-WOZ** dataset for downstream feature extraction and modeling (e.g., VAD, OpenSMILE, WavLM, fusion).

## What this notebook does (high level)
1. **Environment setup**
   - Installs required audio-processing packages (e.g., `noisereduce`).

2. **Project folder setup (Google Drive)**
   - Mounts Google Drive and defines the project paths.
   - Creates a consistent folder structure for raw inputs and generated outputs.

3. **Download DAIC-WOZ participant archives (IDs 300–492)**
   - Downloads participant `.zip` files (if not already present).
   - Logs successful downloads, skipped files, and failures.

4. **Extract & organize files from each participant archive**
   - Unzips each archive and routes files into dedicated folders:
     - `audio_raw/` (WAV interview audio)
     - `transcripts_raw/` (TRANSCRIPT CSV files)
     - `covarep_features_raw/` (COVAREP CSV features)
     - `formant_features_raw/` (Formant CSV features)

5. **Audio metadata scan**
   - Iterates over `audio_raw/` and computes basic metadata per file:
     - sample rate, duration, channels, file size, etc.
   - Saves a metadata table (e.g., `audio_metadata.csv`).

6. **Create “participant-only” audio (remove interviewer)**
   - Uses transcript timestamps to keep only segments where the **Participant** speaks.
   - Concatenates these segments into a single waveform per participant.
   - Output example: `audio_raw_only_patient/{pid}_PARTICIPANT.wav`

7. **Audio cleaning / preprocessing**
   - Applies preprocessing such as:
     - optional noise reduction (`noisereduce`)
     - normalization to a consistent amplitude range
     - Output example: `clean_audio/{pid}_..._clean.wav`

8. **Build a unified labels file**
   - Loads train/dev/test split label files and merges them into one table:
     - `Participant_ID`, `PHQ_Binary`, `PHQ_Score`, `split`
   - Output example: `metadata/labels_all.csv`

9. **Segment cleaned audio into fixed windows**
   - Cuts cleaned participant-only audio into short `.wav` segments
     - Example naming: `{pid}_seg000.wav`, `{pid}_seg001.wav`, ...
     - Typical setup: 10-second windows (and may include overlap depending on config).
   - Saves segment metadata such as number of segments per participant.
   - Output example: `segments/audio/segments_metadata.csv`

---

## Key outputs (artifacts)
- `RAW_DATA/audio_raw/audio_metadata.csv`
- `RAW_DATA/audio_raw_only_patient/{pid}_PARTICIPANT.wav`
- `clean_audio/{pid}_..._clean.wav`
- `metadata/labels_all.csv`
- `segments/audio/{pid}_segXXX.wav`
- `segments/audio/segments_metadata.csv`

---

## Notes / assumptions
- Downstream notebooks expect:
  - segment files named `{pid}_segXXX.wav`
  - `labels_all.csv` for train/dev/test splits and PHQ labels
  - transcripts available under `transcripts_raw/`

---

## How this notebook fits the pipeline
This notebook is the **data preparation foundation**. It creates the cleaned, segmented audio and unified labels needed for:
- VAD/silence analysis
- acoustic feature extraction (OpenSMILE, COVAREP, Formants)
- deep audio embeddings (e.g., WavLM)
- multimodal fusion with transcript-derived features

---

# Notebook: Feature Extraction & Final Model (`feature extract and final model.ipynb`)

This notebook runs the main **feature extraction + modeling pipeline** on pre-segmented DAIC-WOZ data and produces the final evaluation and deployment artifacts.

### What this notebook does
1. **Load segment inventory and transcript mapping**
   - Assumes audio segments exist as WAV files named `{pid}_segXXX.wav` (created earlier in the preparation notebook).
   - Maps transcript text to each segment using a fixed window setup (e.g., 10s window, 5s hop).
   - Saves a segment-level table (e.g., `all_transcriptions.csv`) containing:
     - `participant_id`, `segment_num`, `filename`, `text`, `word_count`

2. **Compute speech/silence (VAD) features**
   - Uses either a pretrained VAD model (Silero) and/or an energy-based method (RMS via librosa).
   - Extracts segment-level metrics such as:
     - `speech_fraction`
     - `max_silence_duration`
   - Uses checkpointing to avoid recomputation for long runs.

3. **Filter and select a fixed number of segments per participant**
   - Filters low-quality segments (e.g., no text / very low speech activity).
   - Selects a consistent set of **50 segments per participant** using the **10–30–10 strategy**:
     - 10 from the beginning, 30 from the middle, 10 from the end
   - This creates a standardized input size per participant for feature extraction and modeling.

4. **Extract deep audio embeddings (WavLM-Large)**
   - Loads a pretrained WavLM-Large model and extracts embeddings from multiple hidden layers.
   - Summarizes embeddings (e.g., mean/std pooling) into participant-level features.
   - Saves intermediate checkpoints (e.g., `.pt`) for robust long-running extraction.

5. **Extract classical acoustic features (OpenSMILE eGeMAPS)**
   - Runs OpenSMILE with the `eGeMAPSv02` feature set, producing interpretable acoustic features
     (pitch, jitter, shimmer, loudness, spectral measures, etc.).
   - Produces feature tables suitable for statistical analysis and ML baselines.

6. **Combine feature sets (COVAREP / Formants / Silence / WavLM / OpenSMILE)**
   - Merges available acoustic feature sources into a unified participant-level table
     (e.g., `combined_acoustic_features.csv`), joined with PHQ labels and splits.

7. **Train and evaluate models**
   - Trains multiple configurations:
     - "Focused" acoustic model (selected high-signal feature subsets)
     - "Deployable" model (only features feasible to compute in production)
     - transcript-derived features via LLM/text baselines
     - Fusion models (audio + text)
   - Performs threshold calibration to optimize metrics such as F1 on the target class.

8. **Save final artifacts for deployment**
   - Stores the best model package (model + scaler + threshold + metadata)
   - Outputs evaluation reports (metrics + confusion matrix) and summary CSVs for reproducibility.

### Key outputs (artifacts)
- Segment transcription mapping:
  - `all_transcriptions.csv`
- VAD results / checkpoints:
  - `vad_results/*.csv` (and/or cached results)
- Selected segments:
  - `filtered_segments/selected_segments.csv`
- Feature checkpoints:
  - `wavlm_embeddings/*.pt`
  - `opensmile_features/*.pt`
- Combined feature tables:
  - `combined_acoustic_features.csv`
- Final evaluation:
  - `results/final_metrics.csv`
  - `results/model_leaderboard.csv` (if generated)
- Deployment package:
  - `results/best_model_package/` (model + scaler + threshold + feature list)


