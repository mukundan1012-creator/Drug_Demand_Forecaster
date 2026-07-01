# 💊 Drug Demand Forecaster

A time-series machine learning project that forecasts daily pharmacy drug demand across 8 ATC drug categories using ARIMA and Prophet models — with automatic per-category model selection based on RMSE performance.

---

## 🧠 Problem Statement

Pharmacies struggle to maintain optimal drug inventory — overstocking increases costs while understocking causes critical shortages. This project builds a forecasting system that predicts daily drug demand per ATC category, automatically selecting the best-performing model (ARIMA or Prophet) for each drug independently.

---

## 📁 Project Structure


Drug_Demand_Forecaster/
├── data/
│   └── salesdaily.csv           
├── src/
│   ├── data_loader.py           
│   ├── preprocessing.py         
│   ├── forecast_model.py        
│   └── evaluate.py              
├── Model/                       
├── app.py                       
├── final_notebook.ipynb         
├── model_comparison.csv         
├── forecast_comparison.png      
└── requirements.txt
---

## 🔬 Approach

### 1. Data
- Dataset: [Pharma Sales Data — Kaggle](https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data)
- 2,106 daily rows across 8 ATC drug categories (2014–2019)
- All 8 categories confirmed stationary via ADF test (p < 0.05)

### 2. Models
- **ARIMA(2,0,2):** Short-term lookback model — strong on low-variance, spike-dominated categories
- **Prophet:** Trend + seasonality decomposition — strong on categories with structured seasonal patterns

### 3. Per-Category Auto-Selection
Both models are trained on every drug category. The model with lower RMSE on the held-out last 30 days is automatically selected as the winner — no manual picking.

### 4. Evaluation
- **RMSE** — primary selection metric (same units as sales)
- **MAPE** — percentage error, computed excluding zero-actual days to avoid division by zero

---

## 📊 Results

| Drug | ARIMA RMSE | Prophet RMSE | Winner | Best RMSE |
|------|-----------|--------------|--------|-----------|
| M01AB | 2.680 | 2.695 | ARIMA | 2.680 |
| M01AE | 2.085 | 1.938 | Prophet | 1.938 |
| N02BA | 1.637 | 1.759 | ARIMA | 1.637 |
| N02BE | 19.328 | 12.255 | Prophet | 12.255 |
| N05B | 4.462 | 4.484 | ARIMA | 4.462 |
| N05C | 1.020 | 1.033 | **ARIMA** | **1.020** |
| R03 | 6.138 | 7.319 | ARIMA | 6.138 |
| R06 | 2.122 | 1.995 | Prophet | 1.995 |

**ARIMA wins 5 categories, Prophet wins 3** — neither model dominates universally, validating the per-category selection design.

**Best performer:** N05C (RMSE 1.020) — Hypnotics & Sedatives  
**Hardest category:** N02BE (RMSE 12.255) — Paracetamol demand highly volatile

---

## 📸 Screenshots

### App Overview — Sidebar + Drug Selector
![Sidebar Overview](sidebar_overview.png)

### Dropdown — All 8 ATC Categories
![Sidebar Dropdown](sidebar_overview2.png)

### Best Performer — N05C (ARIMA, RMSE 1.02)
![N05C Forecast](forecast_plot.png)

### Hardest Category — N02BE (Prophet, RMSE 12.255)
![N02BE Forecast](forecast_plot_hard.png)

### Full Model Comparison Summary Table
![Summary Table](summary_table.png)

---

## ⚙️ Key Engineering Decisions

- **Per-category model selection** over single-model-for-all — proven necessary by R03 (ARIMA wins) vs M01AE (Prophet wins)
- **Temporal train/test split** — last 30 days held out; no shuffling (shuffling destroys time ordering)
- **MAPE zero-division fix** — excluded zero-actual days via nonzero mask rather than epsilon padding
- **Models saved as `.pkl`** — dashboard loads pre-trained models instantly, no retraining on app startup

---

## ⚠️ Honest Limitations

- Fixed `ARIMA(2,0,2)` parameters used across all categories — auto-tuning (e.g., `auto_arima`) would improve results
- Random demand spikes driven by external events are inherently unpredictable from historical sales data alone
- N05C showed a convergence warning during ARIMA training — results should be interpreted cautiously
- Dataset covers 2014–2019 only — model may not generalize to post-2019 demand patterns

---

## 🚀 Run Locally

```bash
git clone https://github.com/mukundan1012-creator/Drug_Demand_Forecaster.git
cd Drug_Demand_Forecaster
pip install -r requirements.txt
streamlit run app.py
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![ARIMA](https://img.shields.io/badge/Model-ARIMA-green)
![Prophet](https://img.shields.io/badge/Model-Prophet-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data-lightgrey)

---

## 👤 Author

**Mukundan.D** | B.E. Electronics & Communication Engineering  
[GitHub](https://github.com/mukundan1012-creator)