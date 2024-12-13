# Reference-Free Evaluation Framework for Stem Separation Accuracy and Quality

This project provides a novel reference-free evaluation framework for audio stem separation tasks, introducing two key metrics:
- **Frequency Isolation Score (FIS):** Measures frequency isolation and harmonic preservation.
- **Dynamic Stability Score (DSS):** Assesses temporal energy stability and spectral flux.

The framework is designed to evaluate stem separation quality without relying on reference ground truth stems, making it suitable for scenarios such as AI-generated stems or real-world applications where references are unavailable.

## Project Structure

```plaintext
code/
  analysis/
    Analysis.ipynb         # Jupyter notebook for exploratory analysis
    evaluate.py            # Main script to run the evaluation process
    sap.py                 # Preprocessing script for source separation outputs
  scores/
    Dynamic_Stability_Score.py  # Implementation of DSS
    Frequency_Isolation_Score.py # Implementation of FIS
    score.py               # Central module integrating both scores
evaluation_results/
  evaluation_graphs/
    Linear Regresion SAR vs DSS.png  # Regression plot for SAR and DSS
    Linear Regresion SAR vs FIS.png  # Regression plot for SAR and FIS
    Pearson Correlation Metrics.png # Correlation matrix visualization
  CSV/
    evaluation_results.csv          # Consolidated evaluation scores
logs/
  evaluation_log.txt                # Log of the evaluation process
  sap_log.txt                       # Log of SAP processing
  umx_log.txt                       # Log of UMX processing
README.md                           # Project documentation


```

## Features
- **Reference-Free Evaluation**: No need for ground truth stems.
- **Signal Processing-Based Metrics**: FIS and DSS focus on frequency isolation and temporal stability, providing objective measures of separation quality.
- **Visualization Tools**: Graphs for regression and correlation analysis.
- **Extensibility**: Modular design for adding new metrics or adapting to new datasets.

## Installation

### Prerequisites
- Python 3.8 or higher
- Required packages: 
  - `librosa`
  - `numpy`
  - `matplotlib`
  - `mir_eval`
  - `openunmix`
  - `demucs`
- FFmpeg (optional, for audio format conversion)


## Contact
For questions or issues, please reach out to:

Author: Jiayi Wang
Email: jwang2@oxy.edu
