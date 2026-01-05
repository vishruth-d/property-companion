# PropertyCompanion 🏠

> ML-powered UK property valuation system with conformal prediction intervals

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**MSc Statistics (Data Science and Machine Learning) Dissertation Project**

---

## 🎯 Overview

PropertyCompanion is a full-stack property valuation application that uses ensemble machine learning models to predict UK residential property prices in North/West London. The system provides:

- **Point predictions** from 5 different model configurations
- **Calibrated confidence intervals** using conformal prediction (80% coverage)
- **Comparable transaction analysis** with similarity scoring
- **Investment recommendations** (Strong Buy / Buy / Fair / Negotiate / Avoid)
- **Mortgage calculations** and negotiation strategies
- **Batch processing** via Excel upload

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  React Frontend │────▶│  FastAPI Backend │────▶│   ML Models     │
│  (Vite + Tailwind)    │  (Python 3.10+)  │     │  (LightGBM/XGB) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │  Transaction DB  │
                        │  (Land Registry) │
                        └─────────────────┘
```

---

## 📊 Model Performance

| Version | Description | MAPE | Conformal Coverage |
|---------|-------------|------|-------------------|
| V1 | Baseline (single model) | 10.01% | - |
| V2 | Enhanced features | 10.31% | - |
| **V3** | **5-segment ensemble** | **9.95%** | **80%** |

### Segmentation Strategy (V3)

Properties are routed to segment-specific models based on estimated price percentile:

| Segment | Price Range | Properties | Segment MAPE |
|---------|-------------|------------|--------------|
| Seg1 (0-15%) | < £220k | Entry-level flats | ~8% |
| Seg2 (15-50%) | £220k - £450k | Average properties | ~8% |
| Seg3 (50-80%) | £450k - £750k | Mid-market | ~9% |
| Seg4 (80-95%) | £750k - £1.2M | Premium | ~10% |
| Seg5 (95-100%) | > £1.2M | Luxury | ~12% |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm or yarn

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/PropertyCompanion.git
cd PropertyCompanion

# Backend setup
pip install -r requirements.txt

# Frontend setup
cd frontend
npm install
cd ..
```

### Download Models & Data

Models and transaction data are not included due to size/sensitivity. 

**Option 1:** Download pre-trained models from [Google Drive Link]

**Option 2:** Train your own using `notebooks/PropertyCompanion.ipynb`

Place downloaded files:
```
PropertyCompanion/
├── models/
│   ├── option1/
│   ├── option2/
│   ├── option3/
│   ├── option4/
│   ├── option5/
│   ├── category_encodings/
│   ├── conformal_thresholds/
│   ├── quantile_models/
│   └── metadata/
└── transactions_prepared_v3.csv
```

### Run Application

**Terminal 1 - Backend:**
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

**Open:** http://localhost:5173

---

## 📖 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check & loaded models |
| `GET` | `/api/versions` | Available model versions |
| `POST` | `/api/predict` | Single property prediction |
| `POST` | `/api/upload-excel` | Batch Excel processing |
| `POST` | `/api/comparables` | Find comparable transactions |
| `GET` | `/api/stats/{version}` | Model statistics |
| `GET` | `/api/stats-summary` | Cross-version comparison |

### Example Request

```bash
curl -X POST "http://localhost:8000/api/predict?version=v3" \
  -H "Content-Type: application/json" \
  -d '{
    "postcode_sector": "HA3 8",
    "property_type": "House",
    "built_form": "Semi-Detached",
    "total_floor_area": 120,
    "number_bedrooms": 3,
    "number_bathrooms": 2,
    "listing_price": 550000,
    "local_authority_label": "Harrow",
    "current_energy_rating": "D",
    "old_new": "Y",
    "tenure_type": "F"
  }'
```

### Example Response

```json
{
  "version": "v3",
  "point_estimate": 580000,
  "predictions": {
    "Option 1 - Vanilla": 575000,
    "Option 2 - Stratified": 582000,
    "Option 3 - 3 Seg (10/80/10)": 578000,
    "Option 4 - 3 Seg (20/60/20)": 585000,
    "Option 5 - 5 Segments": 580000
  },
  "conformal_interval": {
    "lower": 510000,
    "upper": 650000,
    "coverage": 0.8
  },
  "confidence": "HIGH",
  "confidence_reasons": [
    "Strong model agreement (CV=4.0%)",
    "Narrow prediction interval (±12%)",
    "High accuracy segment (MAPE=8.5%)"
  ],
  "verdict": "FAIR_VALUE",
  "diff_pct": -5.2,
  "recommendation": "NEUTRAL",
  "recommendation_detail": "Fairly priced relative to market",
  "negotiation": {
    "opening": 506000,
    "target": 522500,
    "maximum": 539000,
    "walkaway": 591600
  },
  "mortgage": {
    "deposit": 137500,
    "loan_amount": 412500,
    "monthly_payment": 2534,
    "total_interest": 347700
  },
  "comparables": [...]
}
```

---

## 🖼️ Screenshots

| Home | Valuation Input |
|------|-----------------|
| ![Home](docs/screenshots/home.png) | ![Input](docs/screenshots/valuation.png) |

| Results | Comparables |
|---------|-------------|
| ![Results](docs/screenshots/results.png) | ![Comparables](docs/screenshots/comparables.png) |

---

## 📁 Project Structure

```
PropertyCompanion/
├── backend/                 # FastAPI application
│   ├── __init__.py
│   └── app.py              # API endpoints & ML inference
│
├── frontend/               # React application
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       └── App.jsx         # Main React component
│
├── notebooks/
│   └── PropertyCompanion.ipynb  # Complete analysis notebook
│                                # - Data preparation
│                                # - Feature engineering
│                                # - Model training (Optuna)
│                                # - Conformal calibration
│                                # - Evaluation & validation
│
├── models/                 # Trained models (not in repo)
│   ├── option1-5/         # Tuned models per segment
│   ├── category_encodings/
│   ├── conformal_thresholds/
│   └── quantile_models/
│
├── data/                   # Data files (not in repo)
│
└── docs/
    └── screenshots/
```

---

## 🔬 Methodology

### Data Sources

| Source | Description | Period |
|--------|-------------|--------|
| **UK Land Registry** | Price Paid Data | 2018-2025 |
| **EPC Register** | Energy Performance Certificates | 2018-2025 |
| **ONS** | House Price Index (temporal adjustment) | 2018-2025 |

### Coverage Area

15 Local Authorities in North/West London:
- Barnet, Brent, Ealing, Enfield, Harrow, Hillingdon, Hounslow
- Haringey, Richmond upon Thames
- Hertsmere, Three Rivers, Watford, Dacorum
- Buckinghamshire, Slough

### Feature Engineering

```python
# Base numerical features
TOTAL_FLOOR_AREA
NUMBER_BEDROOMS
NUMBER_BATHROOMS

# Derived ratio features
AREA_PER_BEDROOM = TOTAL_FLOOR_AREA / NUMBER_BEDROOMS
AREA_PER_BATHROOM = TOTAL_FLOOR_AREA / NUMBER_BATHROOMS
BEDROOMS_PER_BATHROOM = NUMBER_BEDROOMS / NUMBER_BATHROOMS

# Derived polynomial features
AREA_SQUARED, BEDROOMS_SQUARED, BATHROOMS_SQUARED
LOG_FLOOR_AREA

# Interaction features
AREA_x_BEDROOMS, AREA_x_BATHROOMS, BEDROOMS_x_BATHROOMS

# Categorical features
POSTCODE_SECTOR, PROPERTY_TYPE, BUILT_FORM, TENURE_TYPE
LOCAL_AUTHORITY_LABEL, CURRENT_ENERGY_RATING, OLD/NEW
```

### Model Training

1. **Hyperparameter Optimization**: Optuna with 100 trials per segment
2. **Models**: LightGBM, XGBoost, CatBoost (best selected per segment)
3. **Target**: LOG_BASE_PRICE (log-transformed for stability)
4. **Validation**: 5-fold cross-validation with temporal awareness

### Conformal Prediction

Split conformal prediction with segment-specific calibration:

```python
# Calibration (on held-out calibration set)
residuals = |y_true - y_pred| / y_pred  # Relative residuals
threshold = quantile(residuals, 0.80)    # 80% coverage

# Prediction interval
lower = prediction × (1 - threshold)
upper = prediction × (1 + threshold)
```

### Classification Logic

```python
segment_mape = get_segment_mape(predicted_price)
threshold = segment_mape × 1.3

if diff_pct < -threshold × 1.5:
    verdict = "STRONG_UNDERVALUED"
elif diff_pct < -threshold:
    verdict = "LIKELY_UNDERVALUED"
elif diff_pct > threshold × 1.5:
    verdict = "STRONG_OVERVALUED"
elif diff_pct > threshold:
    verdict = "LIKELY_OVERVALUED"
else:
    verdict = "FAIR_VALUE"
```

---

## 🧪 Validation

### Statistical Tests

- ✅ Residual normality (Shapiro-Wilk)
- ✅ Homoscedasticity check
- ✅ No systematic bias across price segments
- ✅ Conformal coverage validation (80% ± 2%)
- ✅ Cross-validation stability

### Backtesting

The system was backtested on 2024-2025 transactions not seen during training, achieving:
- MAPE: 9.95%
- Conformal coverage: 79.8%
- Classification accuracy: 73%

---

## 📈 Future Improvements

- [ ] Add location features (crime rates, school ratings, transport links)
- [ ] Implement automated Rightmove/Zoopla scraping
- [ ] Time-series forecasting for price trends
- [ ] Expand coverage to Greater London
- [ ] Deploy to cloud (AWS/GCP)
- [ ] Mobile application

---

## 🛠️ Tech Stack

**Backend:**
- Python 3.10+
- FastAPI
- LightGBM / XGBoost / CatBoost
- Pandas / NumPy / Scikit-learn
- Optuna (hyperparameter optimization)

**Frontend:**
- React 18
- Vite
- Tailwind CSS
- Lucide Icons

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- Imperial College London, Department of Mathematics
- UK Land Registry for Price Paid Data (Open Government Licence)
- EPC Register for Energy Performance data
- Office for National Statistics for House Price Index

---

## 📧 Contact

**Vishruth** - MSc Statistics, Imperial College London

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- Email: your.email@imperial.ac.uk

---

<p align="center">
  <i>Built with ❤️ for Imperial College London MSc Statistics Dissertation</i>
</p>
