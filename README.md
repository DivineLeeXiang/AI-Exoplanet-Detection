# 🌌 AI-Enabled Detection of Exoplanets

[![📄 Theory & Methodology](https://img.shields.io/badge/📄-Theory_&_Methodology-blue?style=for-the-badge)](docs/AI_Exoplanet_Detection_Theory_and_Methodology.pdf)

[![📘 Full Technical Report](https://img.shields.io/badge/📘-Full_Technical_Report-green?style=for-the-badge)]
.(docs\AI Enabled Detection of Exoplanets By Light Curve AI.pdf)


# 🌌 AI-Enabled Detection of Exoplanets from NASA TESS Light Curves

An end-to-end AI-assisted pipeline for detecting exoplanet transit candidates from NASA TESS light curve data using **Box Least Squares (BLS)**, astrophysical parameter estimation, machine learning classification, and an interactive Streamlit dashboard.

---

## 📌 Overview

This project automatically analyzes NASA TESS FITS light curve files to detect periodic transit signals that may indicate the presence of exoplanets.

The system combines classical astronomical signal processing with machine learning to provide an intuitive and scientifically meaningful exoplanet detection workflow.

---

## ✨ Features

- 📂 Read NASA TESS FITS light curve files
- 🧹 Automatic preprocessing and data cleaning
- 📉 Transit detection using Box Least Squares (BLS)
- 🔄 Phase folding of detected transit signals
- 📊 Orbital parameter estimation
- 🤖 AI-based candidate classification
- 📈 Interactive Streamlit dashboard
- 📄 CSV export of detected candidates
- 🖼 Automatic visualization of light curves and phase-folded plots
- 🏆 Automatic ranking of multiple uploaded candidates

---

# Project Workflow

```
NASA TESS FITS
        │
        ▼
Load FITS Data
        │
        ▼
Preprocessing
(Remove bad data, Normalize)
        │
        ▼
Transit Detection
(Box Least Squares)
        │
        ▼
Parameter Estimation
        │
        ▼
AI Classification
(Random Forest)
        │
        ▼
Visualization
        │
        ▼
CSV Export
        │
        ▼
Interactive Dashboard
```

---

# Folder Structure

```
AI Exoplanet Detection
│
├── app.py
├── main.py
│
├── src
│   ├── load_data.py
│   ├── preprocess.py
│   ├── detect.py
│   ├── estimate.py
│   ├── classify.py
│   ├── pipeline.py
│   ├── export.py
│   └── visualize.py
│
├── detection
│   ├── models
│   │      exoplanet_model.pkl
│   │
│   └── training
│          train_model.py
│          cumulative_2020.12.30_14.14.11.csv
│
├── outputs
│   ├── exoplanet_candidates.csv
│   └── plots
│
├── data
│   └── raw
│
└── requirements.txt
```

---

# Technologies Used

- Python
- Streamlit
- Astropy
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Joblib

---

# Dataset

### Light Curve Data

- NASA TESS Mission
- FITS Files
- SPOC Light Curves

### AI Training Dataset

NASA Kepler Exoplanet Candidate Catalog

Training Features

- Orbital Period
- Transit Depth
- Transit Duration

---

# Mathematical Background

## Transit Depth

```
Depth = Median Flux − Minimum Flux
```

Approximate relation

```
δ ≈ (Rp / R★)²
```

where

- δ = transit depth
- Rp = planet radius
- R★ = stellar radius

---

## Phase Folding

```
Phase = (Time mod Period) / Period
```

Phase folding aligns repeated transits into a single orbital cycle.

---

## Box Least Squares (BLS)

The Box Least Squares algorithm searches thousands of trial orbital periods and transit durations to identify periodic box-shaped brightness dips.

Unlike Fourier methods, BLS is specifically designed for planetary transit detection.

---

# Parameter Estimation

The pipeline estimates

- Orbital Period
- Transit Depth
- Transit Duration
- Signal-to-Noise Ratio
- Estimated Planet Radius
- Semi-major Axis

---

# Machine Learning

A Random Forest classifier predicts whether the detected signal resembles a genuine exoplanet candidate or a false positive.

### Features Used

- Orbital Period
- Transit Depth
- Transit Duration

### Output

- Planet Candidate
- False Positive

along with prediction confidence.

---

# Computational Complexity

| Module | Complexity |
|----------|------------|
| FITS Loading | O(N) |
| Preprocessing | O(N) |
| Phase Folding | O(N) |
| BLS Detection | O(N × M × D) |
| Parameter Estimation | O(N) |
| Random Forest Prediction | O(T × Depth) |
| Candidate Ranking | O(K log K) |

where

- N = number of observations
- M = number of trial periods
- D = number of transit durations
- T = number of decision trees
- K = number of uploaded candidates

---

# Current Pipeline

```
Load FITS

↓

Remove Bad Observations

↓

Remove Missing Values

↓

Normalize Flux

↓

Box Least Squares

↓

Find Best Period

↓

Estimate Parameters

↓

AI Classification

↓

Visualization

↓

CSV Export
```

---

# Visualizations

The application generates

- Raw Light Curve
- Phase Folded Light Curve
- BLS Detection Results
- Candidate Ranking

---

# Streamlit Dashboard

The dashboard provides

- Upload one or multiple FITS files
- Automatic candidate ranking
- Scientific parameter cards
- AI prediction
- Interactive plots
- Raw parameter viewer

---

# Real-World Challenges

The project addresses several practical issues encountered in astronomical observations.

- Instrumental noise
- Stellar variability
- Detector artifacts
- Cosmic rays
- Missing observations
- False positives
- Extremely shallow transit signals
- Large observational datasets

---

# Why Box Least Squares?

Planetary transits produce short, box-shaped reductions in stellar brightness.

BLS is specifically designed to detect these signals, making it the preferred method over FFT or standard Fourier analysis for transit searches.

---

# Current Limitations

- Simplified planetary radius estimation
- Approximate semi-major axis calculation
- Prototype AI classifier
- Limited astrophysical feature set
- Requires light curve FITS files

---

# Future Improvements

- Support additional FITS formats
- Automatic mission detection
- Better detrending algorithms
- CNN-based transit classification
- Uncertainty estimation
- PDF report generation
- Automatic discovery report
- Dark-mode scientific dashboard
- Interactive candidate explorer
- NASA Exoplanet Archive integration

---

# Installation

Clone the repository

```bash
git clone <repository_url>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the dashboard

```bash
streamlit run app.py
```

Run the command-line pipeline

```bash
python main.py
```

---

# Example Output

```
Detected Candidate

Period          5.6595 days

Depth           0.005862

Duration        0.0640 days

SNR             3.81

Planet Radius   8.36 Earth Radii

Orbit           0.062 AU

Prediction      False Positive

Confidence      77.33%
```

---

# References

- NASA TESS Mission
- NASA Kepler Mission
- Astropy Documentation
- Scikit-learn Documentation
- Box Least Squares Transit Detection Algorithm
- Kepler Exoplanet Candidate Catalog

---

# License

This project is intended for educational, research, and proof-of-concept purposes.

---

# Authors

Developed as an undergraduate Electronics and Communication Engineering project focused on combining astronomical signal processing and machine learning for automated exoplanet detection.
