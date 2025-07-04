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

This repository contains code and resources for a research project that uses machine learning to forecast likely genomic changes in the Dengue virus (DENV) based on environmental trends. Focusing on key genomic regions (Envelope (E) protein, 5′UTR, and 3′UTR), the project combines historical DENV sequences from Delhi, India, with environmental data—average temperature and CO₂ emissions—to explore how climate factors may influence viral evolution. The pipeline covers data collection, preprocessing, exploratory analysis, and predictive modeling using Random Forest, Artificial Neural Networks, and LSTM models.  The goal is to support vaccine design and epidemic preparedness by enabling more accurate predictions of dominant viral genotypes in response to environmental change.

## Project Structure
```plaintext
denv-genome-forecasting/
│
├── data/
│ ├── metadata.csv                  # NCBI metadata file
│ ├── indian_denv_genomes.gb        # Full GenBank records (downloaded)
│ ├── extracted_sequences.csv       # DENV regions (E, 5′UTR, 3′UTR) with accession/year
│ ├── delhi_env_1960_2024.csv       # Yearly Delhi temperature and CO₂ data
│ ├── combined_dataset.csv          # Merged genomic and environmental data
│ └── processed_genomic_dataset.csv # Final, processed input for ML modeling
│
├── notebooks/
│ ├── data_eda.ipynb                    # Exploratory data analysis
│ ├── data_preprocessing.ipynb          # Data cleaning and preparation
│ ├── download_genbank.ipynb            # Scripts to download GenBank records
│ ├── extract_regions.ipynb             # Extract E, 5′UTR, 3′UTR regions from genomes
│ └── merge_environment_sequences.ipynb # Merge genomic and environmental data
│
├── models/
│ └── (empty)            # (Will contain trained model checkpoints)
│
├── results/
│ └── (empty)            # (Will contain evaluation metrics and predictions)
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

The resulting dataset is fully processed and ready for machine learning analysis.

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

## Modeling

The following steps are planned for the predictive modeling phase (to be completed):

- **Train/test split** of the dataset for fair model evaluation
- **Model selection:** Random Forest (RF), Artificial Neural Network (ANN), and Long Short-Term Memory (LSTM)
- **Hyperparameter tuning** to optimize performance for each algorithm

## Evaluation

The evaluation stage will include:

- Calculation of performance metrics such as **accuracy**, **F1-score**, and **ROC-AUC** (where applicable)
- **Model comparison** through visualizations and summary tables
- **Error analysis** and interpretation of key results

*These steps will be completed and updated as the project progresses.*

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

   Open the `notebooks/` folder and run the Jupyter notebooks in the following order:

   - `download_genbank.ipynb` (if you want to replicate raw data download)
   - `extract_regions.ipynb`
   - `merge_environment_sequences.ipynb`
   - `data_preprocessing.ipynb`
   - `data_eda.ipynb`
   - Modeling and evaluation notebooks (to be added)

   Notebooks are designed to be run sequentially, but you can jump to any step using pre-processed files provided in `data/`.

5. **Model Training and Evaluation**

   - After completing EDA and preprocessing, proceed to model training (notebooks to be added).
   - Trained models (when available) will be saved in the `models/` folder and can be used as benchmarks or for further analysis.
  
## Conclusion

Forecasting the evolution of the dengue virus genome in response to environmental changes has important implications for public health, vaccine development, and epidemic preparedness. By integrating machine learning with real-world climate and genomic data, this project aims to:

- Support early identification of high-risk viral genotypes.
- Inform the design of more effective vaccines and therapeutic strategies.
- Enable policymakers to anticipate and respond proactively to climate-driven shifts in infectious disease patterns.

**Future Scope:**  
The project can be expanded by incorporating additional genomic regions, environmental variables, or data from other regions and time periods. Further work could include advanced model interpretability, validation on external datasets, or adaptation for other vector-borne diseases.

## Acknowledgments

This research project was conducted as part of the bachelor thesis at **IU International University of Applied Sciences, Germany**, under the supervision of **Prof. (Hon.) Dr. rer. pol. Fadi Mohsen**.

---

> **Project Status:**  
> The work is ongoing. Data collection, preprocessing, and exploratory analysis have been completed. The modeling and evaluation phase is currently in progress and will be updated in this repository upon completion.

---
