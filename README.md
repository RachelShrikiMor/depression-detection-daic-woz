# depression-detection-daic-woz
Depression detection from speech &amp; transcripts using the DAIC-WOZ dataset

## Project Overview

This project builds a multimodal pipeline for **binary depression detection** using the **DAIC-WOZ** interview dataset.  
The goal is to predict a binary depression label (PHQ-based) from **speech audio** and **transcripts**.

Main components:
- Data preparation (participant-only audio, segmentation, unified labels)
- Feature extraction (VAD/silence, OpenSMILE eGeMAPS, WavLM embeddings, transcript features)
- Modeling (audio-only baselines, multimodal fusion, threshold calibration)
- Deployment artifacts (model package: model + scaler + threshold + feature list)

## Repository Structure (high level)
```text
notebooks/
  00_preparing_DAIC-WOZ_files_documented.ipynb
  feature extract and final model.ipynb
docs/
  NOTEBOOKS.md
  ```
---

## How to Run (Google Colab + GPU)

### Enable GPU in Colab
In Colab:
1. `Runtime` → `Change runtime type`
2. Set `Hardware accelerator` to **GPU**
3. Save

### Install dependencies
run 
`!pip install -r requirements.txt`

### Mount Google Drive
`from google.colab import drive
 drive.mount("/content/drive")`

### Run notebooks in order
1. 00_preparing_DAIC-WOZ_files_documented.ipynb
   * Downloads/extracts DAIC-WOZ files, creates participant-only audio, cleans audio, builds labels_all.csv, generates {pid}_segXXX.wav segments.
2. feature extract and final model.ipynb
   * Maps transcripts to segments, extracts VAD/silence + OpenSMILE + WavLM features, trains models, evaluates, and saves deployment artifacts.

## Notes
The DAIC-WOZ dataset files are not included in this repository.

## Documentation
for detailed notebook explanations and generated artifacts See docs/NOTEBOOKS.md.


