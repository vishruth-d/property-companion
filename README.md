# 🏠 PropertyCompanion

**An ML-powered UK property valuation system with uncertainty quantification, ensemble predictions, and production-ready API.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## 📋 Overview

PropertyCompanion is an end-to-end machine learning system for valuing residential properties in England and Wales. Unlike traditional single-point predictions, this system provides:

- **Ensemble predictions** from 5 distinct modeling strategies
- **Uncertainty quantification** via quantile regression and conformal prediction
- **Confidence scoring** to indicate prediction reliability
- **Actionable verdicts** for property listings (undervalued/overvalued/fair value)
- **Comparable property analysis** using historical transactions

The system is designed for real-world use, featuring a FastAPI backend that can be integrated with any frontend application.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **5 Modeling Options** | From vanilla single-model to 5-segment price stratification |
| **Multi-Model Comparison** | LightGBM, CatBoost, and XGBoost with Optuna hyperparameter tuning |
| **Quantile Regression** | 80% prediction intervals (10th, 50th, 90th percentiles) |
| **Conformal Prediction** | Distribution-free uncertainty with guaranteed coverage |
| **MAPE-Weighted Ensemble** | Combines predictions weighted by segment accuracy |
| **Confidence Tiers** | HIGH/MEDIUM/LOW based on model agreement, spread, and segment MAPE |
| **Listing Verdicts** | Automatically classifies listings as STRONG_UNDERVALUED → STRONG_OVERVALUED |
| **Comparable Sales** | Finds similar properties with similarity scoring |
| **Production API** | FastAPI backend with Swagger documentation |

---

## 📊 Data

The model is trained on **~300,000 residential property transactions** from England and Wales, combining:

- **HM Land Registry Price Paid Data** — Transaction prices, property types, dates
- **Energy Performance Certificates (EPC)** — Floor area, bedrooms, bathrooms, energy ratings

### Features Used

| Category | Features |
|----------|----------|
| **Numeric** | `TOTAL_FLOOR_AREA`, `NUMBER_BEDROOMS`, `NUMBER_BATHROOMS` |
| **Categorical** | `POSTCODE_SECTOR`, `PROPERTY_TYPE`, `BUILT_FORM`, `TENURE_TYPE`, `LOCAL_AUTHORITY_LABEL`, `OLD/NEW`, `CURRENT_ENERGY_RATING` |
| **Engineered** | Log transforms, area ratios, polynomial features, interactions |

### Target Variable

Log-transformed price (`LOG_BASE_PRICE`) to handle the right-skewed distribution of property prices.

---

## 🔬 Methodology

### Modeling Options

The system implements 5 distinct modeling strategies to capture different aspects of the UK property market:

| Option | Strategy | Segmentation | Rationale |
|--------|----------|--------------|-----------|
| **1** | Vanilla | None | Baseline single model on all data |
| **2** | Stratified | None | Stratified sampling for balanced training |
| **3** | 3-Segment | 10/80/10 | Separate models for low/mid/high value properties |
| **4** | 3-Segment | 20/60/20 | Alternative segmentation boundaries |
| **5** | 5-Segment | 15/35/30/15/5 | Fine-grained price segments |

### Model Selection

Each option is trained with three gradient boosting frameworks:

- **LightGBM** — Fast training, native categorical support
- **CatBoost** — Robust categorical encoding, symmetric trees
- **XGBoost** — Industry standard with regularization

Hyperparameters are tuned using **Optuna** with Bayesian optimization (50-100 trials per segment).

### Feature Engineering Configurations

Multiple feature engineering strategies are tested:

```
base_3          → Raw numeric features only
log_transforms  → Log-transformed floor area/bedrooms/bathrooms
ratios          → Area per bedroom, area per bathroom, etc.
interactions    → Area × bedrooms, bedrooms × bathrooms
full_engineered → All of the above combined
```

---

## 📈 Uncertainty Quantification

### Quantile Regression

For each option, three quantile models predict the 10th, 50th, and 90th percentiles, providing an 80% prediction interval:

```
Lower Bound (q10) ─────── Median (q50) ─────── Upper Bound (q90)
     £450,000                £520,000               £610,000
```

### Conformal Prediction

Distribution-free uncertainty quantification that provides **guaranteed coverage** on held-out test data:

- Calibrated on test set residuals
- Per-segment thresholds for tighter intervals
- 80% coverage target

### Confidence Scoring

A multi-factor confidence score based on:

1. **Model Agreement** — Coefficient of variation across 5 options
2. **Interval Width** — Average quantile spread
3. **Segment MAPE** — Historical accuracy for the property's price segment

```
Score ≥ 4  → HIGH confidence   🟢
Score 1-3  → MEDIUM confidence 🟡
Score < 1  → LOW confidence    🔴
```

---

## 🎯 Results

### Model Performance (MAPE)

| Option | Segment | MAPE |
|--------|---------|------|
| Option 1 | Single | ~9-11% |
| Option 2 | Single | ~9-11% |
| Option 5 | Seg1 (0-15%) | ~8-10% |
| Option 5 | Seg2 (15-50%) | ~7-9% |
| Option 5 | Seg3 (50-80%) | ~8-10% |
| Option 5 | Seg4 (80-95%) | ~9-12% |
| Option 5 | Seg5 (95-100%) | ~12-15% |

*Note: Luxury properties (top 5%) are inherently harder to value due to unique characteristics.*

### Backtest Results

On held-out test data:
- **Within 10% of actual**: ~55-65%
- **Within 20% of actual**: ~85-90%
- **In 80% prediction interval**: ~78-82%

---

## 🚀 API

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check and loaded versions |
| `GET` | `/api/versions` | Available model versions |
| `POST` | `/api/predict` | Full property valuation |
| `POST` | `/api/preload/{version}` | Pre-load a model version |
| `POST` | `/api/upload-excel` | Batch valuations from file |

### Example Request

```bash
curl -X POST "http://localhost:8000/api/predict?version=v3" \
  -H "Content-Type: application/json" \
  -d '{
    "postcode_sector": "HA4 8",
    "property_type": "House",
    "built_form": "Detached",
    "total_floor_area": 317,
    "number_bedrooms": 6,
    "number_bathrooms": 4,
    "tenure_type": "F",
    "old_new": "O",
    "local_authority_label": "Hillingdon",
    "current_energy_rating": "C",
    "listing_price": 1650000
  }'
```

### Example Response

```json
{
  "version": "v3",
  "property": {
    "postcode_sector": "HA4 8",
    "property_type": "House",
    "total_floor_area": 317,
    "number_bedrooms": 6
  },
  "ml_valuation": {
    "mape_weighted_mean": 1542000,
    "ensemble_mean": 1528000,
    "coefficient_of_variation": 4.2,
    "point_predictions": {
      "Option 1 - Vanilla": 1510000,
      "Option 2 - Stratified": 1525000,
      "Option 3 - 3 Seg (10/80/10)": 1548000,
      "Option 4 - 3 Seg (20/60/20)": 1535000,
      "Option 5 - 5 Segments": 1562000
    }
  },
  "conformal": {
    "point_estimate": 1542000,
    "interval_lower": 1320000,
    "interval_upper": 1764000,
    "coverage": 0.80
  },
  "confidence": {
    "tier": "HIGH",
    "reasons": [
      "Strong model agreement (CV=4.2%)",
      "Narrow prediction interval (±14%)",
      "High accuracy segment (MAPE=8.5%)"
    ]
  },
  "verdict": {
    "classification": "LIKELY_OVERVALUED",
    "recommendation": "NEGOTIATE",
    "listing_price": 1650000,
    "predicted_price": 1542000,
    "difference_pct": 7.0
  },
  "negotiation": {
    "opening_offer": 1434000,
    "target_price": 1496000,
    "fair_value": 1542000,
    "walk_away": 1619000
  }
}
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- ~4GB RAM for model loading

### Setup

```bash
# Clone the repository
git clone https://github.com/vishruth-d/property-companion.git
cd property-companion

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
catboost>=1.2.0
xgboost>=2.0.0
optuna>=3.3.0
fastapi>=0.100.0
uvicorn>=0.23.0
joblib>=1.3.0
```

---

## 📁 Project Structure

```
propertycompanion/
├── 📓 housing.ipynb              # Main training notebook
├── 🚀 backend_api.py             # FastAPI backend
├── 📊 data/
│   ├── transactions_prepared.csv # Processed transaction data
│   └── raw/                      # Raw Land Registry + EPC data
├── 🤖 models/
│   ├── v1_baseline/              # Model version 1
│   ├── v2_extended/              # Model version 2
│   ├── v3_comprehensive/         # Model version 3 (recommended)
│   │   ├── configs_opt1_2.joblib
│   │   ├── models_opt5.joblib
│   │   ├── quantile_models.joblib
│   │   ├── quantile_configs.joblib
│   │   ├── conformal_thresholds.joblib
│   │   ├── segment_mapes.joblib
│   │   ├── percentiles.joblib
│   │   └── transactions_prepared.pkl
│   └── tuned_v3_option*_.joblib  # Optuna-tuned segment models
├── 📈 optuna_results_*.csv       # Hyperparameter tuning results
├── 📋 requirements.txt
└── 📖 README.md
```

---

## 🖥️ Usage

### Running the API

```bash
# Start the FastAPI server
python backend_api.py

# Or with uvicorn directly
uvicorn backend_api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **API Root**: http://localhost:8000/

### Training Models (Optional)

To retrain models with your own data:

1. Prepare transaction data with required columns
2. Open `housing.ipynb` in Jupyter
3. Run all cells sequentially
4. Models will be saved to `models/` directory

---

## 🔮 Future Improvements

- [ ] **Frontend Dashboard** — React/Next.js interface for property searches
- [ ] **Time-Series Features** — Price trend indicators per postcode
- [ ] **Image Integration** — Property photos for condition assessment
- [ ] **Rightmove/Zoopla Integration** — Auto-fetch listing details
- [ ] **Postcode API** — Validate and auto-complete postcodes
- [ ] **Model Monitoring** — Track prediction drift over time
- [ ] **A/B Testing** — Compare model versions in production

---

## 📄 License

This project's **source code** is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

> ⚠️ **Note**: Trained model files (`.joblib`) are not included in this repository. The code demonstrates the full training pipeline, but production models are kept private. Contact me if you'd like to discuss access for research or collaboration.

---

## 🙏 Acknowledgments

- **HM Land Registry** for Price Paid Data
- **Ministry of Housing, Communities & Local Government** for EPC data
- **Optuna** for hyperparameter optimization framework
- **LightGBM, CatBoost, XGBoost** teams for excellent gradient boosting libraries

---

## 📧 Contact

**Vishruth D** — [vishy.dhamo@gmail.com](mailto:vishy.dhamo@gmail.com)

Project Link: [https://github.com/vishruth-d/property-companion](https://github.com/vishruth-d/property-companion)

---

<p align="center">
  Made with ☕ and 🐍
</p>
