
---

# Notebooks Documentation

## Notebook 00 — Preparing DAIC-WOZ Files (`00_preparing_DAIC-WOZ_files_documented.ipynb`)
This notebook prepares the DAIC-WOZ dataset for downstream feature extraction and modeling.

### What it does
1. Environment setup (audio-processing dependencies).
2. Google Drive project structure setup.
3. Download DAIC-WOZ participant archives (IDs 300–492).
4. Extract and organize files into:
   - `audio_raw/` (WAV interview audio)
   - `transcripts_raw/` (TRANSCRIPT CSVs)
   - `covarep_features_raw/` (COVAREP CSVs)
   - `formant_features_raw/` (Formants CSVs)
5. Audio metadata scan (duration, sample rate, file size).
6. Create **participant-only** audio (remove interviewer via transcript timestamps).
7. Clean audio (optional noise reduction + normalization).
8. Build unified labels: `Participant_ID`, `PHQ_Binary`, `PHQ_Score`, `split` → `labels_all.csv`.
9. Segment cleaned participant-only audio into fixed windows:
   - `{pid}_seg000.wav`, `{pid}_seg001.wav`, ...
   - Saves `segments_metadata.csv`.

### Key outputs
- `RAW_DATA/audio_raw/audio_metadata.csv`
- `RAW_DATA/audio_raw_only_patient/{pid}_PARTICIPANT.wav`
- `clean_audio/{pid}_..._clean.wav`
- `metadata/labels_all.csv`
- `segments/audio/{pid}_segXXX.wav`
- `segments/audio/segments_metadata.csv`

---

## Notebook — Feature Extraction & Final Model (`feature extract and final model.ipynb`)
This notebook runs the main feature extraction and modeling pipeline on pre-segmented audio.

### What it does
1. Transcript-to-segment mapping (windowing) → `all_transcriptions.csv`
2. VAD / silence features (Silero and/or RMS energy-based) + checkpointing
3. Segment filtering + fixed selection of **50 segments per participant** (10–30–10)
4. WavLM-Large embeddings extraction (multi-layer hidden states → pooled features) + checkpoints
5. OpenSMILE eGeMAPS extraction (interpretable acoustic features) + checkpoints
6. Feature merging (Silence / COVAREP / Formants / WavLM / OpenSMILE) → `combined_acoustic_features.csv`
7. Modeling:
   - focused acoustic models
   - deployable feature set models
   - transcript-derived text features (LLM/text baseline)
   - fusion experiments (audio + text)
   - threshold calibration
8. Save final evaluation + deployment artifacts (model + scaler + threshold + feature list)

### Key outputs
- `all_transcriptions.csv`
- `filtered_segments/selected_segments.csv`
- `wavlm_embeddings/*.pt`
- `opensmile_features/*.pt`
- `combined_acoustic_features.csv`
- `results/final_metrics.csv`
- `results/model_leaderboard.csv` (optional)
- `results/best_model_package/`
