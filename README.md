# EY-DS-Challenge-2025

![Landing](images/covering.jpg)

This repository contains the code and workflow I used to build my dataset and models for the **2025 EY Open Data & AI Challenge – Cooling Urban Heat Islands**  
🔗 [Challenge Website](https://challenge.ey.com)

The main explanation of my approach, along with the code to reproduce my results, can be found in the notebook:

📘 **`Model Assessment John Bergmann.ipynb`**

To run the notebook successfully:
- Install the required libraries (tested with Python 3.12.9):

  ```bash
  pip install -r requirements.txt
  ```
- Please download the required data sources as specified within the notebook.
- Place them in a `raw` subdirectory inside the `DATA` folder.
- Note: You may need to adjust some file paths to get everything working on your machine – sorry! 😄

---

## 🏆 Results

The final model achieves a **validation R² score of 98.16%** using **58 features**.  
This solution was awarded **2nd place (First Runner-Up)** in the 2025 EY Open Data & AI Challenge (EY Participants).

---

## 🛠️ Additional Scripts

These scripts were primarily used during development and can be executed in the following order to replicate the workflow:

| Step | File | Description |
|------|------|-------------|
| 1 | `feature_engineering.py` | Collection of functions to generate features from the base dataset and additional sources |
| 2 | `build_feature_pool.py` | Builds an initial, extensive feature set prior to feature selection |
| 3 | `feature_elimination.py` | Performs SHAP-based recursive feature elimination with hyperparameter tuning using Optuna and LightGBM |
| 4 | `et_ffs_sk.py` | Forward feature selection with sampling to enhance the feature pool post-RFE |

---

Enjoy!