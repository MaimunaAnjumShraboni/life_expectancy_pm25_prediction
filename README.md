# Predicting Global Life Expectancy From Air Pollution (PM2.5)

Group project for **CSE427 (Machine Learning)**, BRAC University.

This project predicts country-level life expectancy from PM2.5 air pollution together with health, demographic and socio-economic indicators. It compares 10 regression models, from linear baselines to tree ensembles, XGBoost and a neural network (MLP).

## Results (held-out test set)

| Model | R² | RMSE | MAE |
|-------|----|------|-----|
| XGBoost | 0.9629 | 1.790 | 1.072 |
| Extra Trees | 0.9621 | 1.809 | 1.131 |
| Random Forest | 0.9578 | 1.909 | 1.196 |
| Gradient Boosting | 0.9573 | 1.921 | 1.257 |
| Decision Tree | 0.9138 | 2.728 | 1.951 |
| kNN | 0.8927 | 3.044 | 2.047 |
| MLP Regressor | 0.8863 | 3.133 | 2.004 |
| AdaBoost | 0.8764 | 3.267 | 2.541 |
| Ridge Regression | 0.8155 | 3.991 | 2.987 |
| Linear Regression | 0.8155 | 3.991 | 2.987 |

**Best model:** XGBoost. Tree-based ensembles clearly outperform linear and distance-based models, which suggests the relationships in this data are strongly non-linear.

## Dataset

Two public sources were merged on country and year:

- **Life Expectancy (WHO)** from Kaggle: <https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who>
- **PM2.5 air pollution, mean annual exposure** from the World Bank: <https://data.worldbank.org/indicator/EN.ATM.PM25.MC.M3>

The final processed dataset has **2,500 country-year records and 17 columns** (target: `life_expectancy`). It is included as `data/final_life_pm25.csv`, and it is the only file the modeling cells need.

## Pipeline

1. **Merge:** inner join of the two sources on country and year.
2. **Cleaning:** column-name cleanup, removal of invalid targets, median/mode imputation, outlier filtering, label-encoding of country, and removal of highly correlated or low-value columns.
3. **EDA:** life expectancy distribution, PM2.5 vs life expectancy scatter plot, correlation heatmap.
4. **Modeling:** 10 regressors, 80/20 train/test split (seed 42). Scaling is applied for Linear/Ridge, kNN and MLP only.
5. **Evaluation:** R², RMSE, MAE, comparison plots, and actual vs predicted for the best model.

## How to Run

The notebook was written for Google Colab and reads files from `/content/drive/MyDrive/`.

1. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. In the notebook, replace the path prefix `/content/drive/MyDrive/` with `data/` and remove the Google Drive mount cells.
3. In the XGBoost cell, change `device='cuda'` to `device='cpu'` if you have no GPU.
4. Run the notebook from the repository root:
   ```bash
   jupyter notebook life_expectancy_pm25_regression.ipynb
   ```

To skip preprocessing, run only the cells from the "EDA" section onward; they load `data/final_life_pm25.csv` directly.

## Limitations

- Data is aggregated at country level and merges two independent sources.
- Hyperparameters were chosen manually, not through exhaustive search.

## Repository Structure

```
.
├── life_expectancy_pm25_regression.ipynb   # Full code: merge, preprocessing, EDA, models, evaluation
├── report.pdf                              # Written project report
├── requirements.txt
├── README.md
└── data/
    └── final_life_pm25.csv                 # Processed dataset used for modeling
```
