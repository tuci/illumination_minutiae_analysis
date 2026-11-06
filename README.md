# Illumination Direction and Minutiae Detection

Code accompanying the research manuscript:

**The Influence of Illumination Direction on Minutiae Detection in Human Impressions: A Case Study of Terracotta Sculptures from the Rijksmuseum Collection**

## Overview

This repository contains Python code developed to investigate the influence of illumination direction on the detection and analysis of minutiae in human impressions preserved on terracotta sculptures from the Rijksmuseum collection.

The scripts support image preprocessing, minutiae detection, analysis of multiple illumination directions, and evaluation and visualization of results.

## Repository Contents

The repository includes the following Python scripts and Jupyter notebooks:

- `pre_processing.py` — Image preprocessing.
- `minutiae_detection.py` — Minutiae detection.
- `minutiae_detection_multiview_evaluation.py` — Evaluation involving multiple illumination directions.
- `alignment_gui.py` — Graphical interface for image alignment.
- `NFIQ2Features.py` — NFIQ2-related feature analysis.
- `read_images.py` — Image loading utilities.
- `utils.py` — Supporting functions.
- `evaluation.ipynb` — Evaluation and analysis notebook.
- `visualize_minutiae.ipynb` — Minutiae visualization notebook.

## Data Availability

The dataset will be made publicly available following manuscript acceptance.

## Requirements

**Python version:** 3.11.15

**Main Python libraries:**

```text
numpy==2.4.4
pandas==3.0.3
scipy==1.16.3
opencv-python==4.13.0.92
scikit-image==0.26.0
matplotlib==3.10.9
seaborn==0.13.2
pillow==12.2.0
tensorflow==2.20.0
jupyterlab==4.5.7
pyfing==0.6
```

**Additional dependency:**
- Tkinter (Tk 8.6) — Used for the graphical interface.

## Citation

Citation information will be updated following manuscript publication.
