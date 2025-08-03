# Forecasting Dengue Virus Genomic Evolution Based on Environmental Variables Using Machine Learning Models

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Data Preprocessing and Cleaning](#data-preprocessing-and-cleaning)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Usage Instructions](#usage-instructions)
- [Conclusion](#conclusion)
- [Acknowledgements](#acknowledgements)

## Overview

This repository contains code and resources for a research project that uses machine learning to forecast likely genomic changes in the Dengue virus (DENV) based on environmental trends. Focusing on key genomic regions (Envelope (E) protein, 5′UTR, and 3′UTR), the project combines historical DENV sequences from Delhi, India, with environmental data (average temperature and CO₂ emissions) to explore how climate factors may influence viral evolution. The pipeline covers data collection, preprocessing, exploratory analysis, and predictive modeling using five machine learning models:
- Random Forest (RF)
- Artificial Neural Networks (ANN)
- Long Short-Term Memory networks (LSTM)
- XGBoost (XGB)
- LightGBM (LGB)

The goal is to support vaccine design and epidemic preparedness by enabling more accurate predictions of dominant viral genotypes in response to environmental change.

## Project Structure
```plaintext
denv-genome-forecasting/
│
├── data/
│ ├── metadata.csv                    # NCBI metadata file
│ ├── indian_denv_genomes.gb          # Full GenBank records (downloaded)
│ ├── extracted_sequences.csv         # DENV regions (E, 5′UTR, 3′UTR) with accession/year
│ ├── delhi_env_1960_2024.csv         # Yearly Delhi temperature and CO₂ data
│ ├── combined_dataset.csv            # Merged genomic and environmental data
│ └── processed_genomic_dataset.csv   # Final, processed input for ML modeling
│
├── notebooks/
│ ├── data_eda.ipynb                    # Exploratory data analysis
│ ├── data_preprocessing.ipynb          # Data cleaning and preparation
│ ├── download_genbank.ipynb            # Scripts to download GenBank records
│ ├── extract_regions.ipynb             # Extract E, 5′UTR, 3′UTR regions from genomes
│ ├── merge_environment_sequences.ipynb # Merge genomic and environmental data
│ ├── ann_model.ipynb                   # Train and evaluate a Multi-Layer Perceptron (MLP) model
│ ├── lightgbm_model.ipynb              # Train and evaluate a LightGBM classifier
│ ├── lstm_model.ipynb                  # Train and evaluate a LSTM model
│ ├── random_forest_model.ipynb         # Train and evaluate a Random Forest classifier
│ ├── xgboost_model.ipynb               # Train and evaluate a XGBoost classifier
│ └── model_comparison.ipynb            # Analyze and visualize the comparative performance
│
├── models/
│ ├── rnn_lstm_model.joblib               # Trained LSTM model
│ ├── rf_multioutput_model.joblib         # Trained Random Forest classifier
│ ├── xgb_multioutput_model.joblib        # Trained XGBoost classifier
│ ├── lgb_multioutput_model.joblib        # Trained LightGBM classifier
│ └── ann_mlp_model.joblib                # Trained ANN model
│
├── results/
│ ├── ann_per_position_accuracy.csv                # ANN model performance
│ ├── ann_per_position_classification_report.csv   # ANN model performance
│ ├── ann_per_position_confusion_matrices.json     # ANN model performance
│ ├── lgb_per_position_accuracy.csv                # LightGBM classifier performance
│ ├── lgb_per_position_classification_report.csv   # LightGBM classifier performance
│ ├── lgb_per_position_confusion_matrices.json     # LightGBM classifier performance
│ ├── lstm_per_position_accuracy.csv               # LSTM model performance
│ ├── lstm_per_position_classification_report.csv  # LSTM model performance
│ ├── lstm_per_position_confusion_matrices.json    # LSTM model performance
│ ├── rf_per_position_accuracy.csv                 # Random Forest classifier performance
│ ├── rf_per_position_classification_report.csv    # Random Forest classifier performance
│ ├── rf_per_position_confusion_matrices.json      # Random Forest classifier performance
│ ├── xgb_per_position_accuracy.csv                # XGBoost classifier performance
│ ├── xgb_per_position_classification_report.csv   # XGBoost classifier performance
│ ├── xgb_per_position_confusion_matrices.json     # XGBoost classifier performance
│ └── model_comparison_summary.csv                 # Overall performance comparison
│
├── README.md            # Project overview and usage instructions
├── requirements.txt     # Python dependencies for running code
├── .gitignore           # Patterns for files/folders to exclude from Git
```

## Dataset

- **Location:** Delhi, India
- **Content:** Dengue virus genomic sequences (Envelope, 5′UTR, 3′UTR) with associated environmental variables (average temperature, CO₂ emissions)
- **Sources:** 
    - Genomic data: [NCBI GenBank](https://www.ncbi.nlm.nih.gov/genbank/)  
      (filtered for human-sampled DENV sequences from Delhi, India)
    - Temperature: [World Bank Climate Knowledge Portal](https://climateknowledgeportal.worldbank.org/)
    - CO₂: [Our World in Data](https://ourworldindata.org/co2-emissions)

### Data Ingestion

- **Genomic data:**  
  Downloaded DENV records sampled from humans in Delhi (411 records).  
  Extracted metadata and downloaded annotated GenBank files using Biopython (Entrez).  
  Extracted only the E, 5′UTR, and 3′UTR regions.  
  Final: 404 sequences with all three regions present.

- **Environmental data:**  
  Extracted Delhi grid cell temperature data (World Bank) and India CO₂ emissions by year (Our World in Data).

- **Merging:**  
  Combined genomic and environmental data by collection year.  
  After cleaning (removing rows with missing year), the merged dataset has **395 samples**.

### Data Schema

| Column Name           | Description                                                              |
|-----------------------|--------------------------------------------------------------------------|
| `Accession`           | Unique GenBank accession ID for each DENV record                         |
| `Collection_Year`     | Year the DENV sample was collected                                       |
| `Combined_Sequence`   | Concatenated nucleotide sequence of E, 5′UTR, and 3′UTR regions          |
| `Avg_Temp_C`          | Average annual temperature (°C) in Delhi for the corresponding year       |
| `CO2_Emission_Mt`     | Annual CO₂ emissions for India (metric tons) for the corresponding year  |

### Access Instructions

The merged dataset (**combined_dataset.csv**) is available on Kaggle.
**Download link:**  
  [Kaggle Dataset: Delhi DENV Genomic & Environmental Data](https://kaggle.com/datasets/a65efae08c47f91a4219ded7ab83a92a27eb75cb136b1c04495344a00a123a10)

## Data Preprocessing and Cleaning

The raw genomic and environmental data underwent a systematic preprocessing pipeline to ensure consistency, quality, and suitability for machine learning analysis.

1. **Column Selection:**  
   Removed unnecessary columns such as `Accession` and `Collection_Year`, retaining only the sequence and environmental variables needed for modeling.

2. **Missing Value Check:**  
   Checked for missing values; none were present in the cleaned dataset.

3. **Sequence Length Normalization:**  
   Analyzed sequence lengths and found variability (most sequences between 1800 and 2015 nucleotides).
   To ensure compatibility with machine learning models and minimize information loss:
     - Removed sequences shorter than 1800 nt.
     - Truncated sequences longer than 2015 nt to 2015 nt.
     - Padded shorter sequences with a special value (`N`) to reach a fixed length of 2015 nt.
   This approach preserves the majority of the data, reduces outlier impact, and aligns with biological and ML standards.

4. **Sequence Expansion:**  
   Split each sequence into individual position columns (`p_1`, `p_2`, ..., `p_2015`), resulting in a DataFrame shape of (359, 2017).

5. **Nucleotide Encoding:**  
   Encoded nucleotides as integers:  
   `A` → 1, `T` → 2, `G` → 3, `C` → 4, `N` (unknown/padded) → 0  
   This format is suitable for multi-class classification models.

> **See data_preprocessing.ipynb in the notebooks/ directory for the complete preprocessing pipeline and implementation details.**

## Exploratory Data Analysis (EDA)

A comprehensive EDA was performed to understand the dataset’s structure, assess feature distributions, and identify key patterns for downstream modeling.

- **Dataset Overview:**  
  359 samples × 2017 features:  
    - 2 environmental variables (`Avg_Temp_C`, `CO2_Emission_Mt`)  
    - 2015 nucleotide sequence positions (encoded as 0–4: N/A, A, T, G, C)
  
  No missing values; all data types are numeric for efficient analysis.

- **Environmental Variables:**  
  Both average temperature and CO₂ emissions are right-skewed, reflecting recent data concentration from Delhi, India.
  Strong positive correlation (r = 0.92) between temperature and CO₂, indicating these features trend together over time and may act as proxies for each other.

- **Sequence Features:**  
  Substantial base variability across all positions; no highly conserved sites (>95% consensus).
  A and G are the most prevalent bases, with a significant proportion of padding/unknowns (`N`) in later sequence positions due to length normalization.
  High Shannon entropy across most positions confirms considerable genetic diversity in the targeted regions.

- **Genotype–Environment Association:**  
  Quartile binning of environmental variables enabled categorical association analysis.
  Multiple sequence positions show significant associations with both temperature and CO₂ (even after multiple-testing correction).
  Cramér’s V analysis indicates CO₂ has a slightly stronger association with nucleotide variation than temperature, suggesting environmental variables may broadly influence DENV genomic evolution.

These findings confirm the dataset is well-suited for machine learning modeling and genotype–environment association studies, while highlighting the importance of accounting for padding, variable sequence coverage, and feature collinearity in downstream analysis.

> **See data_eda.ipynb in the notebooks/ directory for detailed visualizations and a full breakdown of EDA findings.**

## Modeling

This project formulates the prediction of DENV genomic evolution as a multi-class, multi-output classification task. For each sample, the goal is to predict the nucleotide (`A`, `T`, `G`, `C`, or `N`) at every position (1 to 2015) of the concatenated sequence, based on the corresponding environmental variables.

### Modeling Workflow

The following steps were **common to all models**:

1. **Load Processed Data**
   - Used the fully cleaned and encoded dataset from `processed_genomic_dataset.csv`.

2. **Define Features and Targets**
   - Features (`X`): Environmental variables (average annual temperature and CO₂ emissions).
   - Targets (`y`): Encoded nucleotide values for positions 1–2015 (columns `p_1` to `p_2015`).

3. **Train-Test Split**
   - Stratified 70/30 split based on collection year to avoid temporal data leakage and to ensure fair evaluation.

### Model Implementation

All models were benchmarked for the same prediction task. Key details are below:

#### Random Forest (RF)

- **Library:** `sklearn.ensemble.RandomForestClassifier`
- **Multi-Output Setup:**  
  Used `sklearn.multioutput.MultiOutputClassifier` to wrap the base Random Forest model, enabling simultaneous prediction of all 2015 nucleotide positions for each sample.
- **Fitting:**  
  Model was fit on the training data and used to predict nucleotide classes for all sequence positions in the test set.

#### Artificial Neural Network (ANN)

- **Library:** TensorFlow/Keras
- **Architecture:**  
  Multi-layer perceptron with two dense hidden layers with 128 neurons each, ReLU activations, and 30% dropout for regularization.
- **Output:**  
  Final layer configured for multi-class, multi-output prediction (2015 softmax units).
- **Optimization:**  
  Adam optimizer and sparse categorical cross-entropy loss.
- **Training:**  
  Epochs is set to 50 with early stopping to prevent overfitting and mini-batch size of 32 is used to strike a balance.

#### Long Short-Term Memory Network (LSTM)

- **Library:** TensorFlow/Keras
- **Architecture:**  
  LSTM layers with 64 memory units to capture potential sequential dependencies within the genomic regions, followed by dense output for multi-class prediction.
- **Optimization:**  
  Adam optimizer and sparse categorical cross-entropy loss.
- **Training:**  
  Epochs is set to 50 with early stopping to prevent overfitting and mini-batch size of 32 is used to strike a balance.

#### XGBoost

- **Library:** `xgboost.XGBClassifier`
- **Multi-Output Setup:**  
  Wrapped with `MultiOutputClassifier` for simultaneous multi-position prediction.
- **Fitting:**  
  Model was fit on the training data and used to predict nucleotide classes for all sequence positions in the test set.

#### LightGBM

- **Library:** `lightgbm.LGBMClassifier`
- **Multi-Output Setup:**  
  Wrapped with `MultiOutputClassifier` for simultaneous multi-position prediction.
- **Fitting:**  
  Model was fit on the training data and used to predict nucleotide classes for all sequence positions in the test set.

> **See individual Jupyter notebooks in the `notebooks/` directory for full implementation details, code, and intermediate outputs.**

## Evaluation

All models were evaluated using the same metrics to ensure fair comparison and robust performance analysis:

1. **Overall Accuracy:**  
   The proportion of correctly predicted nucleotides across all sequence positions and test samples.

2. **Per-Position Accuracy:**  
   Accuracy calculated separately for each nucleotide position, highlighting performance consistency (or variation) along the sequence.

3. **Per-Position Classification Report (Macro F1 and Per-Class F1):**  
   For each position, the classification report summarizes precision, recall, and F1-score for each nucleotide class, with special focus on the macro-average F1 (averaged across all classes).

4. **Per-Position Confusion Matrix:**  
   Confusion matrices for individual positions provide a detailed view of the distribution of true vs. predicted classes, making it easy to identify systematic errors or class imbalances.

> **Performance comparison with graphs and detailed analysis can be found in the `model_comparison.ipynb` notebook.**
> **All result tables and summary files are saved in the `results/` directory as CSV and JSON files.**

## Usage Instructions

To reproduce the results or use the code for further analysis, follow these steps:

1. **Clone this repository**

   ```bash
   git clone https://github.com/Achshah-RM/denv-genome-forecasting.git
   cd denv-genome-forecasting

2. **Install dependencies**

   Ensure you have Python 3.7+ installed. Then run:

   ```bash
   pip install -r requirements.txt

3. **Download the dataset**  
   - Download the combined dataset from [Kaggle](https://kaggle.com/datasets/a65efae08c47f91a4219ded7ab83a92a27eb75cb136b1c04495344a00a123a10)  
   - Place the `combined_dataset.csv` file into the `data/` folder in the project directory.

4. **Run the notebooks**

   Open the `notebooks/` folder and run the Jupyter notebooks in the following recommended order:
    
    1. `download_genbank.ipynb` *(optional: to replicate the raw data download)*
    2. `extract_regions.ipynb`
    3. `merge_environment_sequences.ipynb`
    4. `data_preprocessing.ipynb`
    5. `data_eda.ipynb`
    6. `random_forest_model.ipynb`
    7. `ann_model.ipynb`
    8. `lstm_model.ipynb`
    9. `xgboost_model.ipynb`
    10. `lightgbm_model.ipynb`
    11. `model_comparison.ipynb`
    
5. **Using Pre-trained Models for Prediction**

   - You can directly use any of the trained and saved models in the `models/` directory to make predictions on new genomic or environmental data.
   - Load the model file (using `joblib`), prepare your feature input in the expected format, and use the `.predict()` method to obtain predictions.

   *Example (Random Forest):*

    ```python
    import joblib
    model = joblib.load('models/rf_multioutput_model.joblib')
    # X_new: your new feature matrix (environmental variables)
    y_pred = model.predict(X_new)
  
## Conclusion

Forecasting the evolution of the dengue virus genome in response to environmental changes has important implications for public health, vaccine development, and epidemic preparedness. By integrating machine learning with real-world climate and genomic data, this project aims to:

- Support early identification of high-risk viral genotypes.
- Inform the design of more effective vaccines and therapeutic strategies.
- Enable policymakers to anticipate and respond proactively to climate-driven shifts in infectious disease patterns.

**Future Scope:**  
The project can be expanded by incorporating additional genomic regions, environmental variables, or data from other regions and time periods. Further work could include advanced model interpretability, validation on external datasets, or adaptation for other vector-borne diseases.

## Acknowledgements

This research project was conducted as part of the bachelor thesis at **IU International University of Applied Sciences, Germany**, under the supervision of **Prof. (Hon.) Dr. rer. pol. Fadi Mohsen**.

