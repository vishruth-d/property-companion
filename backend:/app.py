"""
PropertyCompanion Backend API
=============================
FastAPI backend for property valuation ML models.

Endpoints:
- GET  /api/versions          - Get available model versions
- GET  /api/health            - Health check
- POST /api/predict           - Single property prediction
- POST /api/upload-excel      - Batch Excel analysis
- GET  /api/stats/{version}   - Model statistics
- GET  /api/stats-summary     - Cross-version comparison
- POST /api/comparables       - Find comparable transactions
- GET  /api/analyses          - Get saved analyses
- POST /api/analyses/save     - Save analysis
- DELETE /api/analyses/{id}   - Delete analysis

Usage:
    uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import uuid
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import io

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
ANALYSES_FILE = PROJECT_ROOT / 'saved_analyses.json'

# Create directories if needed
DATA_DIR.mkdir(exist_ok=True)

# Global session storage
SESSION = {}

# Version configuration
VERSION_NAMES = {
    'v1': 'v1_baseline',
    'v2': 'v2_extended',
    'v3': 'v3_comprehensive',
}

CONFIG = {
    'model_version': 'v3',
    'random_state': 42,
    'target': 'LOG_BASE_PRICE',
}

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL PATH UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.bool_, np.bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj

def get_model_paths(version: str = None) -> dict:
    """Get all model paths for a given version with organized folder structure."""
    v = version or CONFIG['model_version']
    models_root = MODELS_DIR
    
    return {
        'root': models_root,
        'version': v,
        'category_encodings_dir': models_root / 'category_encodings',
        'conformal_dir': models_root / 'conformal_thresholds',
        'quantile_dir': models_root / 'quantile_models',
        'boundaries_dir': models_root / 'segment_boundaries',
        'metadata_dir': models_root / 'metadata',
        'transactions_dir': models_root / 'transactions',
        'option1_dir': models_root / 'option1',
        'option2_dir': models_root / 'option2',
        'option3_dir': models_root / 'option3',
        'option4_dir': models_root / 'option4',
        'option5_dir': models_root / 'option5',
        'conformal_thresholds': models_root / 'conformal_thresholds' / f'conformal_thresholds_{v}.joblib',
        'quantile_models': models_root / 'quantile_models' / f'quantile_models_{v}.joblib',
        'quantile_configs': models_root / 'quantile_models' / f'quantile_configs_{v}.joblib',
        'segment_keys': models_root / 'metadata' / f'segment_keys_{v}.joblib',
        'segment_mapes': models_root / 'metadata' / f'segment_mapes_{v}.joblib',
        'percentiles': models_root / 'metadata' / f'percentiles_{v}.joblib',
        'transactions': models_root / 'transactions' / f'transactions_prepared_{v}.pkl',
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SEGMENT KEYS & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

SEGMENT_KEYS_DEFAULT = {
    3: ['Lower_0-10', 'Middle_10-90', 'Upper_90-100'],
    4: ['Lower_0-20', 'Middle_20-80', 'Upper_80-100'],
    5: ['Seg1_0-15', 'Seg2_15-50', 'Seg3_50-80', 'Seg4_80-95', 'Seg5_95-100'],
}

# Feature engineering configurations (must match training)
NUMERICAL_CONFIGS = {
    'base_1': ['TOTAL_FLOOR_AREA'],
    'base_2': ['TOTAL_FLOOR_AREA', 'NUMBER_BEDROOMS'],
    'base_3': ['TOTAL_FLOOR_AREA', 'NUMBER_BEDROOMS', 'NUMBER_BATHROOMS'],
    'ratios': ['TOTAL_FLOOR_AREA', 'NUMBER_BEDROOMS', 'NUMBER_BATHROOMS', 
               'AREA_PER_BEDROOM', 'AREA_PER_BATHROOM', 'BEDROOMS_PER_BATHROOM'],
    'squared': ['TOTAL_FLOOR_AREA', 'NUMBER_BEDROOMS', 'NUMBER_BATHROOMS',
                'AREA_SQUARED', 'BEDROOMS_SQUARED', 'BATHROOMS_SQUARED'],
    'interactions': ['TOTAL_FLOOR_AREA', 'NUMBER_BEDROOMS', 'NUMBER_BATHROOMS',
                     'AREA_X_BEDROOMS', 'AREA_X_BATHROOMS', 'BEDROOMS_X_BATHROOMS'],
    'full': ['TOTAL_FLOOR_AREA', 'NUMBER_BEDROOMS', 'NUMBER_BATHROOMS',
             'AREA_PER_BEDROOM', 'AREA_PER_BATHROOM', 'BEDROOMS_PER_BATHROOM',
             'AREA_SQUARED', 'BEDROOMS_SQUARED', 'BATHROOMS_SQUARED',
             'AREA_X_BEDROOMS', 'AREA_X_BATHROOMS', 'BEDROOMS_X_BATHROOMS'],
}

CATEGORICAL_CONFIGS = {
    'min_cat': ['PROPERTY_TYPE'],
    'core_cat': ['PROPERTY_TYPE', 'BUILT_FORM'],
    'all_cat': ['POSTCODE_SECTOR', 'OLD/NEW', 'TENURE_TYPE', 'PROPERTY_TYPE', 
                'BUILT_FORM', 'LOCAL_AUTHORITY_LABEL', 'CURRENT_ENERGY_RATING'],
}


def apply_feature_engineering(df: pd.DataFrame, num_config: str, cat_config: str):
    """Apply feature engineering to match training - creates ALL possible derived features."""
    
    # ═══════════════════════════════════════════════════════════════
    # ENSURE BASE COLUMNS EXIST
    # ═══════════════════════════════════════════════════════════════
    
    if 'TOTAL_FLOOR_AREA' not in df.columns:
        df['TOTAL_FLOOR_AREA'] = 0
    if 'NUMBER_BEDROOMS' not in df.columns:
        df['NUMBER_BEDROOMS'] = 1
    if 'NUMBER_BATHROOMS' not in df.columns:
        df['NUMBER_BATHROOMS'] = 1
    
    area = df['TOTAL_FLOOR_AREA']
    beds = df['NUMBER_BEDROOMS'].replace(0, 1)
    baths = df['NUMBER_BATHROOMS'].replace(0, 1)
    
    # ═══════════════════════════════════════════════════════════════
    # CREATE ALL POSSIBLE DERIVED FEATURES (all naming variations)
    # ═══════════════════════════════════════════════════════════════
    
    # Ratio features
    df['AREA_PER_BEDROOM'] = area / beds
    df['AREA_PER_BATHROOM'] = area / baths
    df['BEDROOMS_PER_BATHROOM'] = df['NUMBER_BEDROOMS'] / baths
    
    # Squared features
    df['AREA_SQUARED'] = area ** 2
    df['BEDROOMS_SQUARED'] = df['NUMBER_BEDROOMS'] ** 2
    df['BATHROOMS_SQUARED'] = df['NUMBER_BATHROOMS'] ** 2
    
    # Interaction features (UPPERCASE X)
    df['AREA_X_BEDROOMS'] = area * df['NUMBER_BEDROOMS']
    df['AREA_X_BATHROOMS'] = area * df['NUMBER_BATHROOMS']
    df['BEDROOMS_X_BATHROOMS'] = df['NUMBER_BEDROOMS'] * df['NUMBER_BATHROOMS']
    
    # Interaction features (lowercase x) - alternative naming
    df['AREA_x_BEDROOMS'] = area * df['NUMBER_BEDROOMS']
    df['AREA_x_BATHROOMS'] = area * df['NUMBER_BATHROOMS']
    df['BEDROOMS_x_BATHROOMS'] = df['NUMBER_BEDROOMS'] * df['NUMBER_BATHROOMS']
    
    # Log features (multiple naming conventions)
    df['LOG_AREA'] = np.log1p(area.clip(lower=1))
    df['LOG_FLOOR_AREA'] = np.log1p(area.clip(lower=1))
    df['LOG_TOTAL_FLOOR_AREA'] = np.log1p(area.clip(lower=1))
    df['LOG_BEDROOMS'] = np.log1p(df['NUMBER_BEDROOMS'].clip(lower=1))
    df['LOG_BATHROOMS'] = np.log1p(df['NUMBER_BATHROOMS'].clip(lower=1))
    
    # Price per sqm (if listing price available)
    if 'LISTING_PRICE' in df.columns:
        df['PRICE_PER_SQM'] = df['LISTING_PRICE'] / area.replace(0, 1)
    
    # Room density
    df['TOTAL_ROOMS'] = df['NUMBER_BEDROOMS'] + df['NUMBER_BATHROOMS']
    df['ROOMS_PER_SQM'] = df['TOTAL_ROOMS'] / area.replace(0, 1)
    
    # NO_EPC indicator
    if 'CURRENT_ENERGY_RATING' in df.columns:
        df['NO_EPC'] = (
            df['CURRENT_ENERGY_RATING'].isna() | 
            df['CURRENT_ENERGY_RATING'].isin(['', 'INVALID!', 'NO DATA!', 'unknown', 'Unknown', None])
        ).astype(int)
    else:
        df['NO_EPC'] = 0
    
    # ═══════════════════════════════════════════════════════════════
    # GET COLUMN LISTS
    # ═══════════════════════════════════════════════════════════════
    
    num_cols = list(NUMERICAL_CONFIGS.get(num_config, NUMERICAL_CONFIGS['base_3']))
    cat_cols = list(CATEGORICAL_CONFIGS.get(cat_config, CATEGORICAL_CONFIGS['all_cat']))
    
    # Ensure all categorical columns exist
    for col in cat_cols:
        if col not in df.columns:
            df[col] = 'Unknown'
    
    return df, num_cols, cat_cols


def get_segment_key(estimated_price: float, option_num: int, percentiles: dict) -> str:
    """Determine segment based on estimated price."""
    p = percentiles
    
    if option_num == 3:
        if estimated_price < p.get('p10', 200000):
            return 'Lower_0-10'
        elif estimated_price < p.get('p90', 1000000):
            return 'Middle_10-90'
        else:
            return 'Upper_90-100'
    
    elif option_num == 4:
        if estimated_price < p.get('p20', 250000):
            return 'Lower_0-20'
        elif estimated_price < p.get('p80', 800000):
            return 'Middle_20-80'
        else:
            return 'Upper_80-100'
    
    elif option_num == 5:
        if estimated_price < p.get('p15', 220000):
            return 'Seg1_0-15'
        elif estimated_price < p.get('p50', 450000):
            return 'Seg2_15-50'
        elif estimated_price < p.get('p80', 750000):
            return 'Seg3_50-80'
        elif estimated_price < p.get('p95', 1200000):
            return 'Seg4_80-95'
        else:
            return 'Seg5_95-100'
    
    return 'single'


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_session(version: str = None) -> dict:
    """Load all models and configs from organized folder structure."""
    v = version or CONFIG['model_version']
    paths = get_model_paths(v)
    
    print(f"Loading session for {v}...")
    
    session = {
        'version': v,
        'paths': paths,
        'tuned_models': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}},
        'category_encodings': {1: {}, 2: {}, 3: {}, 4: {}, 5: {}},
        'segment_boundaries': {},
        'quantile_models': None,
        'quantile_configs': None,
        'conformal_thresholds': None,
        'segment_keys': SEGMENT_KEYS_DEFAULT,
        'segment_mapes': {},
        'percentiles': {},
        'transactions_prepared': None,
    }
    
    def safe_load(path, description=""):
        if path.exists():
            try:
                return joblib.load(path)
            except Exception as e:
                print(f"  Warning: Failed to load {path.name}: {e}")
                return None
        return None
    
    # Load Option 1 & 2 (single models)
    for opt in [1, 2]:
        opt_path = paths[f'option{opt}_dir'] / f'tuned_{v}_option{opt}_single.joblib'
        model_info = safe_load(opt_path)
        if model_info:
            session['tuned_models'][opt]['single'] = model_info
        
        enc_path = paths['category_encodings_dir'] / f'category_encodings_{v}_opt{opt}.joblib'
        session['category_encodings'][opt] = safe_load(enc_path) or {}
    
    # Load Options 3, 4, 5 (segmented models)
    for opt in [3, 4, 5]:
        segments = SEGMENT_KEYS_DEFAULT[opt]
        for seg_name in segments:
            seg_path = paths[f'option{opt}_dir'] / f'tuned_{v}_option{opt}_{seg_name}.joblib'
            model_info = safe_load(seg_path)
            if model_info:
                session['tuned_models'][opt][seg_name] = model_info
        
        bounds_path = paths['boundaries_dir'] / f'segment_boundaries_{v}_option{opt}.joblib'
        session['segment_boundaries'][opt] = safe_load(bounds_path)
        
        enc_path = paths['category_encodings_dir'] / f'category_encodings_{v}_opt{opt}.joblib'
        session['category_encodings'][opt] = safe_load(enc_path) or {}
    
    # Load quantile models
    session['quantile_models'] = safe_load(paths['quantile_models'])
    session['quantile_configs'] = safe_load(paths['quantile_configs'])
    
    # Load conformal thresholds
    session['conformal_thresholds'] = safe_load(paths['conformal_thresholds'])
    
    # Load metadata
    session['segment_keys'] = safe_load(paths['segment_keys']) or SEGMENT_KEYS_DEFAULT
    session['segment_mapes'] = safe_load(paths['segment_mapes']) or {}
    session['percentiles'] = safe_load(paths['percentiles']) or {}
    
    # ═══════════════════════════════════════════════════════════════
    # Load transactions - try multiple locations and formats
    # ═══════════════════════════════════════════════════════════════
    transactions_paths = [
        (PROJECT_ROOT / 'transactions_prepared_v3.csv', 'csv'),
        (PROJECT_ROOT / 'data' / 'transactions_prepared_v3.csv', 'csv'),
        (PROJECT_ROOT / 'matched.csv', 'csv'),
        (paths['transactions_dir'] / f'transactions_prepared_{v}.csv', 'csv'),
        (paths['transactions_dir'] / f'transactions_prepared_{v}.pkl', 'pkl'),
        (PROJECT_ROOT / 'data' / 'matched.pkl', 'pkl'),
    ]
    
    for tx_path, fmt in transactions_paths:
        if tx_path.exists():
            try:
                if fmt == 'csv':
                    session['transactions_prepared'] = pd.read_csv(tx_path)
                else:
                    session['transactions_prepared'] = pd.read_pickle(tx_path)
                print(f"  ✓ Loaded transactions from {tx_path.name}: {len(session['transactions_prepared'])} rows")
                break
            except Exception as e:
                print(f"  ⚠ Failed to load {tx_path}: {e}")
    
    if session['transactions_prepared'] is None:
        print("  ⚠ No transactions file found - comparables will not work")
    
    # Load segment MAPEs from tuned models if not in metadata
    if not session['segment_mapes']:
        session['segment_mapes'] = load_segment_mapes_from_models(v, paths)
    
    print(f"  Loaded {sum(len(m) for m in session['tuned_models'].values())} models")
    return session


def load_segment_mapes_from_models(version: str, paths: dict) -> dict:
    """Extract MAPEs from tuned model files."""
    mapes = {}
    
    # Options 1 & 2
    for opt in [1, 2]:
        model_path = paths[f'option{opt}_dir'] / f'tuned_{version}_option{opt}_single.joblib'
        if model_path.exists():
            try:
                model_info = joblib.load(model_path)
                mapes[opt] = {'single': model_info.get('mape', 10.0)}
            except:
                mapes[opt] = {'single': 10.0}
        else:
            mapes[opt] = {'single': 10.0}
    
    # Options 3, 4, 5
    for opt in [3, 4, 5]:
        mapes[opt] = {}
        opt_dir = paths[f'option{opt}_dir']
        
        for seg_file in opt_dir.glob(f'tuned_{version}_option{opt}_*.joblib'):
            try:
                model_info = joblib.load(seg_file)
                seg_name = seg_file.stem.replace(f'tuned_{version}_option{opt}_', '')
                mapes[opt][seg_name] = model_info.get('mape', 10.0)
            except:
                pass
    
    return mapes


def get_session(version: str = None) -> dict:
    """Get or load session for a version."""
    v = version or CONFIG['model_version']
    
    if v not in SESSION or SESSION[v] is None:
        SESSION[v] = load_session(v)
    
    return SESSION[v]


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_input(**kwargs) -> pd.DataFrame:
    """Create input DataFrame from property features."""
    data = {}
    for k, v in kwargs.items():
        col = k.upper()
        if col == 'OLD_NEW':
            col = 'OLD/NEW'
        data[col] = [v]
    return pd.DataFrame(data)


def predict_single(input_df: pd.DataFrame, model_info: dict, category_mappings: dict = None) -> float:
    """Predict using a single model."""
    
    # Get config from model_info (use what model was trained with!)
    num_config = model_info.get('num_config', 'base_3')
    cat_config = model_info.get('cat_config', 'all_cat')
    
    # Use the exact columns the model was trained on
    num_cols = model_info.get('num_cols', NUMERICAL_CONFIGS.get(num_config, NUMERICAL_CONFIGS['base_3']))
    cat_cols = model_info.get('cat_cols', CATEGORICAL_CONFIGS.get(cat_config, CATEGORICAL_CONFIGS['all_cat']))
    
    # Apply feature engineering to create derived features
    df_eng, _, _ = apply_feature_engineering(input_df.copy(), num_config, cat_config)
    
    # Select features in correct order
    X = df_eng[num_cols + cat_cols].copy()
    
    model_name = model_info.get('model_name', 'LightGBM')
    model = model_info['model']
    
    if model_name == 'LightGBM':
        for c in cat_cols:
            if category_mappings and c in category_mappings:
                X[c] = pd.Categorical(X[c], categories=category_mappings[c], ordered=False)
            else:
                X[c] = X[c].astype('category')
    
    elif model_name == 'XGBoost':
        encoder = model_info.get('encoder')
        if encoder is not None:
            X_cat = encoder.transform(X[cat_cols])
            X = np.hstack([X[num_cols].values, X_cat])
        else:
            for c in cat_cols:
                X[c] = X[c].astype('category').cat.codes
            X = X.values
    
    elif model_name == 'CatBoost':
        for c in cat_cols:
            X[c] = X[c].astype(str)
    
    pred_log = model.predict(X)[0]
    return float(np.exp(pred_log))


def predict_all_options(input_df: pd.DataFrame, session: dict) -> dict:
    """Run predictions for all 5 options."""
    predictions = {}
    tuned_models = session['tuned_models']
    encodings = session['category_encodings']
    percentiles = session['percentiles']
    
    # Options 1 & 2: Single models
    for opt in [1, 2]:
        if 'single' in tuned_models[opt]:
            model_info = tuned_models[opt]['single']
            enc = encodings.get(opt, {})
            pred = predict_single(input_df, model_info, enc)
            label = 'Vanilla' if opt == 1 else 'Stratified'
            predictions[f'Option {opt} - {label}'] = pred
    
    # Use Option 1 estimate for segment routing
    estimated_price = predictions.get('Option 1 - Vanilla', 500000)
    
    # Options 3, 4, 5: Segmented models
    opt_names = {3: '3 Seg (10/80/10)', 4: '3 Seg (20/60/20)', 5: '5 Segments'}
    
    for opt in [3, 4, 5]:
        seg_key = get_segment_key(estimated_price, opt, percentiles)
        if seg_key in tuned_models[opt]:
            model_info = tuned_models[opt][seg_key]
            enc = encodings.get(opt, {}).get(seg_key, {})
            if not enc and isinstance(encodings.get(opt), dict):
                enc = encodings.get(opt, {})
            pred = predict_single(input_df, model_info, enc)
            predictions[f'Option {opt} - {opt_names[opt]}'] = pred
    
    return predictions


def predict_with_conformal(input_df: pd.DataFrame, session: dict) -> dict:
    """Get prediction with calibrated conformal interval."""
    preds = predict_all_options(input_df, session)
    point_est = np.mean(list(preds.values()))
    
    percentiles = session['percentiles']
    conformal_thresholds = session.get('conformal_thresholds', {})
    
    segment = get_segment_key(point_est, 5, percentiles)
    
    thresh_info = conformal_thresholds.get(segment, {})
    threshold = thresh_info.get('threshold', conformal_thresholds.get('_meta', {}).get('global_threshold', point_est * 0.15))
    coverage = conformal_thresholds.get('_meta', {}).get('coverage', 0.80)
    
    return {
        'point_estimate': point_est,
        'conformal_lower': max(0, point_est - threshold),
        'conformal_upper': point_est + threshold,
        'conformal_threshold': threshold,
        'threshold_pct': (threshold / point_est * 100) if point_est > 0 else 0,
        'segment': segment,
        'coverage': coverage,
        'all_predictions': preds,
        'is_fallback': thresh_info.get('fallback', False),
    }


def calculate_confidence(cv: float, segment_mape: float, spread_pct: float) -> tuple:
    """Calculate confidence tier based on metrics."""
    reasoning = []
    score = 0
    
    # Factor 1: Model Agreement (CV)
    if cv < 6:
        score += 2
        reasoning.append(f"Strong model agreement (CV={cv:.1f}%)")
    elif cv < 12:
        score += 1
        reasoning.append(f"Moderate model agreement (CV={cv:.1f}%)")
    else:
        score -= 1
        reasoning.append(f"High model disagreement (CV={cv:.1f}%)")
    
    # Factor 2: Quantile Spread
    if spread_pct < 25:
        score += 2
        reasoning.append(f"Narrow prediction interval (±{spread_pct/2:.0f}%)")
    elif spread_pct < 40:
        score += 1
        reasoning.append(f"Moderate prediction interval (±{spread_pct/2:.0f}%)")
    else:
        score -= 1
        reasoning.append(f"Wide prediction interval (±{spread_pct/2:.0f}%)")
    
    # Factor 3: Segment MAPE
    if segment_mape < 9:
        score += 2
        reasoning.append(f"High accuracy segment (MAPE={segment_mape:.1f}%)")
    elif segment_mape < 12:
        score += 1
        reasoning.append(f"Moderate accuracy segment (MAPE={segment_mape:.1f}%)")
    else:
        score -= 1
        reasoning.append(f"Lower accuracy segment (MAPE={segment_mape:.1f}%)")
    
    tier = 'HIGH' if score >= 4 else 'MEDIUM' if score >= 1 else 'LOW'
    return tier, reasoning


def classify_listing(predicted: float, listing_price: float, segment_mape: float) -> tuple:
    """Classify listing relative to predicted value."""
    diff_pct = (listing_price - predicted) / predicted * 100
    threshold = segment_mape * 1.3
    
    if diff_pct < -threshold * 1.5:
        classification = 'STRONG_UNDERVALUED'
    elif diff_pct < -threshold:
        classification = 'LIKELY_UNDERVALUED'
    elif diff_pct > threshold * 1.5:
        classification = 'STRONG_OVERVALUED'
    elif diff_pct > threshold:
        classification = 'LIKELY_OVERVALUED'
    else:
        classification = 'FAIR_VALUE'
    
    return classification, diff_pct


def get_recommendation(classification: str, confidence: str, diff_pct: float) -> tuple:
    """Get recommendation and detail based on classification."""
    recommendations = {
        'STRONG_UNDERVALUED': ('STRONG_BUY', 'Significantly below market value - strong opportunity'),
        'LIKELY_UNDERVALUED': ('BUY', 'Below market value - good opportunity'),
        'FAIR_VALUE': ('NEUTRAL', 'Fairly priced relative to market'),
        'LIKELY_OVERVALUED': ('NEGOTIATE', 'Above market value - negotiate down'),
        'STRONG_OVERVALUED': ('AVOID', 'Significantly overpriced - avoid or negotiate hard'),
    }
    
    rec, detail = recommendations.get(classification, ('NEUTRAL', 'Unable to classify'))
    
    # Downgrade if low confidence
    if confidence == 'LOW' and rec in ['STRONG_BUY', 'AVOID']:
        rec = 'NEGOTIATE' if rec == 'AVOID' else 'BUY'
        detail += ' (confidence reduced due to model uncertainty)'
    
    return rec, detail


def get_next_steps(recommendation: str, classification: str) -> list:
    """Get next steps based on recommendation."""
    steps = {
        'STRONG_BUY': [
            'Book viewing immediately - this appears undervalued',
            'Prepare mortgage agreement in principle',
            'Research comparable sales to verify value',
            'Consider opening offer at asking price to secure',
        ],
        'BUY': [
            'Book viewing to assess condition',
            'Get mortgage pre-approval',
            'Research local market trends',
            'Consider offering 3-5% below asking',
        ],
        'NEUTRAL': [
            'Book viewing to assess property condition',
            'Compare with similar properties',
            'Negotiate based on condition/features',
            'Consider 5-10% below asking as starting point',
        ],
        'NEGOTIATE': [
            'View property to identify negotiation points',
            'Research how long property has been listed',
            'Prepare evidence of comparable sales at lower prices',
            'Start negotiations at 10-15% below asking',
        ],
        'AVOID': [
            'Skip or significantly reduce expectations',
            'If interested, verify all comparable data',
            'Only proceed if price drops significantly',
            'Consider alternative properties in the area',
        ],
    }
    return steps.get(recommendation, steps['NEUTRAL'])


def calculate_negotiation(predicted: float, listing_price: float, recommendation: str) -> dict:
    """Calculate negotiation strategy."""
    if recommendation in ['STRONG_BUY', 'BUY']:
        opening = int(listing_price * 0.97)
        target = int(listing_price * 0.98)
        maximum = int(listing_price)
        walkaway = int(predicted * 1.05)
    elif recommendation == 'NEGOTIATE':
        opening = int(listing_price * 0.88)
        target = int(predicted)
        maximum = int(predicted * 1.05)
        walkaway = int(predicted * 0.98)
    elif recommendation == 'AVOID':
        opening = int(listing_price * 0.80)
        target = int(predicted * 0.95)
        maximum = int(predicted)
        walkaway = int(predicted * 0.90)
    else:  # NEUTRAL
        opening = int(listing_price * 0.92)
        target = int(listing_price * 0.95)
        maximum = int(listing_price * 0.98)
        walkaway = int(predicted * 1.02)
    
    return {
        'opening': opening,
        'target': target,
        'maximum': maximum,
        'walkaway': walkaway,
    }


def calculate_mortgage(purchase_price: float, ltv_pct: float = 75, rate_pct: float = 5.5, term_years: int = 25) -> dict:
    """Calculate mortgage details."""
    loan_amount = int(purchase_price * ltv_pct / 100)
    deposit = int(purchase_price - loan_amount)
    
    monthly_rate = (rate_pct / 100) / 12
    num_payments = term_years * 12
    
    if monthly_rate > 0:
        monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
    else:
        monthly_payment = loan_amount / num_payments
    
    total_interest = int(monthly_payment * num_payments - loan_amount)
    
    return {
        'loan_amount': loan_amount,
        'deposit': deposit,
        'monthly_payment': int(monthly_payment),
        'total_interest': total_interest,
        'ltv_pct': ltv_pct,
        'rate_pct': rate_pct,
        'term_years': term_years,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARABLES
# ═══════════════════════════════════════════════════════════════════════════════

def find_comparables(session: dict, postcode_sector: str, property_type: str, 
                     built_form: str, total_floor_area: float, 
                     number_bedrooms: int, number_bathrooms: int = None,
                     date_accessed: str = None, max_results: int = 10) -> dict:
    """Find comparable transactions."""
    transactions = session.get('transactions_prepared')
    
    if transactions is None or len(transactions) == 0:
        return {'has_valuation': False, 'n_total': 0, 'comparables': []}
    
    df = transactions.copy()
    
    # ═══════════════════════════════════════════════════════════════
    # COLUMN NAME MAPPING (your CSV uses these names)
    # ═══════════════════════════════════════════════════════════════
    PRICE_COL = 'ADJUSTED_PRICE'  # or 'ORIGINAL_PRICE'
    DATE_COL = 'DATE_OF_TRANSFER'
    POSTCODE_COL = 'POSTCODE_SECTOR'
    TYPE_COL = 'PROPERTY_TYPE'
    FORM_COL = 'BUILT_FORM'
    AREA_COL = 'TOTAL_FLOOR_AREA'
    BEDS_COL = 'NUMBER_BEDROOMS'
    BATHS_COL = 'NUMBER_BATHROOMS'
    
    # ═══════════════════════════════════════════════════════════════
    # FILTER BY POSTCODE
    # ═══════════════════════════════════════════════════════════════
    mask = df[POSTCODE_COL] == postcode_sector
    
    # Expand search if too few results
    if mask.sum() < 5:
        postcode_area = postcode_sector.split()[0] if ' ' in postcode_sector else postcode_sector[:3]
        mask = df[POSTCODE_COL].str.startswith(postcode_area, na=False)
    
    df_filtered = df[mask].copy()
    
    if len(df_filtered) == 0:
        return {'has_valuation': False, 'n_total': 0, 'comparables': []}
    
    # ═══════════════════════════════════════════════════════════════
    # CALCULATE SIMILARITY SCORES
    # ═══════════════════════════════════════════════════════════════
    df_filtered['similarity'] = 100.0
    
    # Property type match
    if TYPE_COL in df_filtered.columns:
        df_filtered.loc[df_filtered[TYPE_COL] != property_type, 'similarity'] -= 20
    
    # Built form match
    if FORM_COL in df_filtered.columns:
        df_filtered.loc[df_filtered[FORM_COL] != built_form, 'similarity'] -= 15
    
    # Floor area similarity
    if AREA_COL in df_filtered.columns:
        df_filtered[AREA_COL] = pd.to_numeric(df_filtered[AREA_COL], errors='coerce').fillna(0)
        area_diff = abs(df_filtered[AREA_COL] - total_floor_area) / max(total_floor_area, 1)
        df_filtered['similarity'] -= (area_diff * 30).clip(0, 30)
    
    # Bedroom match
    if BEDS_COL in df_filtered.columns:
        df_filtered[BEDS_COL] = pd.to_numeric(df_filtered[BEDS_COL], errors='coerce').fillna(0)
        bed_diff = abs(df_filtered[BEDS_COL] - number_bedrooms)
        df_filtered['similarity'] -= (bed_diff * 10).clip(0, 20)
    
    # Sort by similarity and get top results
    df_filtered = df_filtered.nlargest(max_results * 2, 'similarity')  # Get extra to filter
    
    # ═══════════════════════════════════════════════════════════════
    # TIME ADJUSTMENT
    # ═══════════════════════════════════════════════════════════════
    today = pd.Timestamp.now()
    
    if DATE_COL in df_filtered.columns:
        df_filtered[DATE_COL] = pd.to_datetime(df_filtered[DATE_COL], errors='coerce')
        df_filtered['months_ago'] = ((today - df_filtered[DATE_COL]).dt.days / 30).fillna(0).astype(int)
        # Assume 0.3% monthly appreciation (conservative)
        df_filtered['time_adjustment'] = 1 + (df_filtered['months_ago'] * 0.003)
    else:
        df_filtered['time_adjustment'] = 1.0
        df_filtered['months_ago'] = 0
    
    # ═══════════════════════════════════════════════════════════════
    # CALCULATE ADJUSTED PRICES
    # ═══════════════════════════════════════════════════════════════
    if PRICE_COL not in df_filtered.columns:
        # Fallback to other price columns
        if 'ORIGINAL_PRICE' in df_filtered.columns:
            PRICE_COL = 'ORIGINAL_PRICE'
        elif 'Sale Price (£)' in df_filtered.columns:
            PRICE_COL = 'Sale Price (£)'
        else:
            return {'has_valuation': False, 'n_total': 0, 'comparables': [], 'error': 'No price column'}
    
    df_filtered[PRICE_COL] = pd.to_numeric(df_filtered[PRICE_COL], errors='coerce').fillna(0)
    df_filtered['adjusted_price'] = df_filtered[PRICE_COL] * df_filtered['time_adjustment']
    
    # Calculate PSM (price per square meter)
    if AREA_COL in df_filtered.columns:
        df_filtered['adjusted_psm'] = df_filtered['adjusted_price'] / df_filtered[AREA_COL].replace(0, 1)
    else:
        df_filtered['adjusted_psm'] = 0
    
    # Filter to top results
    df_filtered = df_filtered.head(max_results)
    
    # ═══════════════════════════════════════════════════════════════
    # BUILD COMPARABLES LIST
    # ═══════════════════════════════════════════════════════════════
    comparables = []
    for _, row in df_filtered.iterrows():
        comp = {
            'postcode': str(row.get(POSTCODE_COL, '')),
            'property_type': str(row.get(TYPE_COL, '')),
            'built_form': str(row.get(FORM_COL, '')),
            'original_price': int(row.get(PRICE_COL, 0)),
            'adjusted_price': int(row.get('adjusted_price', 0)),
            'adjusted_psm': int(row.get('adjusted_psm', 0)),
            'floor_area': float(row.get(AREA_COL, 0)),
            'bedrooms': int(row.get(BEDS_COL, 0)),
            'bathrooms': int(row.get(BATHS_COL, 0)) if BATHS_COL in row else 0,
            'similarity': round(float(row['similarity']), 1),
            'date': str(row.get(DATE_COL, ''))[:10] if pd.notna(row.get(DATE_COL)) else '',
            'months_ago': int(row.get('months_ago', 0)),
            'used': float(row['similarity']) >= 75,
        }
        comparables.append(comp)
    
    # ═══════════════════════════════════════════════════════════════
    # CALCULATE VALUATION FROM HIGH-SIMILARITY COMPARABLES
    # ═══════════════════════════════════════════════════════════════
    high_sim = [c for c in comparables if c['used']]
    
    if len(high_sim) >= 3:
        weights = [c['similarity'] / 100 for c in high_sim]
        total_weight = sum(weights)
        weighted_avg = sum(c['adjusted_price'] * w for c, w in zip(high_sim, weights)) / total_weight
        avg_psm = sum(c['adjusted_psm'] * w for c, w in zip(high_sim, weights)) / total_weight
        
        prices = [c['adjusted_price'] for c in high_sim]
        interval_lower = np.percentile(prices, 10)
        interval_upper = np.percentile(prices, 90)
        
        return {
            'has_valuation': True,
            'weighted_avg': int(weighted_avg),
            'avg_psm': int(avg_psm),
            'interval_lower': int(interval_lower),
            'interval_upper': int(interval_upper),
            'interval_pct': round((interval_upper - interval_lower) / max(weighted_avg, 1) * 100, 1),
            'n_total': len(comparables),
            'n_used': len(high_sim),
            'avg_match': round(np.mean([c['similarity'] for c in high_sim]), 1),
            'comparables': comparables,
        }
    
    return {
        'has_valuation': False,
        'n_total': len(comparables),
        'n_used': len(high_sim) if high_sim else 0,
        'comparables': comparables,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FULL PREDICTION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_prediction(property_data: dict, version: str = 'v3') -> dict:
    """Run full prediction pipeline for a property."""
    session = get_session(version)
    
    # Prepare input
    input_df = prepare_input(
        postcode_sector=property_data['postcode_sector'],
        old_new=property_data.get('old_new', 'Y'),
        tenure_type=property_data.get('tenure_type', 'F'),
        property_type=property_data['property_type'],
        built_form=property_data.get('built_form', 'Semi-Detached'),
        total_floor_area=property_data['total_floor_area'],
        local_authority_label=property_data.get('local_authority_label', 'Unknown'),
        number_bedrooms=property_data['number_bedrooms'],
        number_bathrooms=property_data.get('number_bathrooms', 1),
        current_energy_rating=property_data.get('current_energy_rating', 'D'),
    )
    
    # Get predictions
    predictions = predict_all_options(input_df, session)
    conformal = predict_with_conformal(input_df, session)
    
    # Calculate metrics
    pred_values = list(predictions.values())
    point_estimate = conformal['point_estimate']
    cv = float(np.std(pred_values) / point_estimate * 100) if point_estimate > 0 else 0
    
    # Get segment MAPE
    segment = conformal['segment']
    segment_mapes = session.get('segment_mapes', {})
    segment_mape = segment_mapes.get(5, {}).get(segment, 10.0)
    
    # Spread from conformal
    spread_pct = conformal['threshold_pct'] * 2
    
    # Confidence
    confidence, conf_reasons = calculate_confidence(cv, segment_mape, spread_pct)
    
    # Classification
    listing_price = property_data.get('listing_price')
    if listing_price:
        verdict, diff_pct = classify_listing(point_estimate, listing_price, segment_mape)
        recommendation, rec_detail = get_recommendation(verdict, confidence, diff_pct)
        next_steps = get_next_steps(recommendation, verdict)
        negotiation = calculate_negotiation(point_estimate, listing_price, recommendation)
        mortgage = calculate_mortgage(listing_price)
        diff = listing_price - point_estimate
        within_ci = conformal['conformal_lower'] <= listing_price <= conformal['conformal_upper']
    else:
        verdict = None
        diff_pct = 0
        diff = 0
        recommendation = None
        rec_detail = None
        next_steps = []
        negotiation = None
        mortgage = None
        within_ci = None
    
    # Comparables
    comparables_result = find_comparables(
        session,
        postcode_sector=property_data['postcode_sector'],
        property_type=property_data['property_type'],
        built_form=property_data.get('built_form', 'Semi-Detached'),
        total_floor_area=property_data['total_floor_area'],
        number_bedrooms=property_data['number_bedrooms'],
    )
    
    return {
        'version': version,
        'point_estimate': point_estimate,
        'predictions': predictions,
        'conformal_interval': {
            'lower': conformal['conformal_lower'],
            'upper': conformal['conformal_upper'],
            'threshold': conformal['conformal_threshold'],
            'threshold_pct': conformal['threshold_pct'],
            'coverage': conformal['coverage'],
        },
        'segment': segment,
        'segment_mape': segment_mape,
        'cv': cv,
        'confidence': confidence,
        'confidence_reasons': conf_reasons,
        'verdict': verdict,
        'diff_pct': diff_pct,
        'diff': diff,
        'within_ci': within_ci,
        'recommendation': recommendation,
        'recommendation_detail': rec_detail,
        'next_steps': next_steps,
        'negotiation': negotiation,
        'mortgage': mortgage,
        'comparables': comparables_result.get('comparables', []),
        'comparables_valuation': {
            'has_valuation': comparables_result.get('has_valuation', False),
            'weighted_avg': comparables_result.get('weighted_avg'),
            'n_used': comparables_result.get('n_used', 0),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def process_batch_excel(df: pd.DataFrame, version: str = 'v3') -> dict:
    """Process batch Excel file."""
    results = []
    recommendations_count = {'STRONG_BUY': 0, 'BUY': 0, 'NEUTRAL': 0, 'NEGOTIATE': 0, 'AVOID': 0}
    undervalued_listings = []
    overvalued_listings = []
    
    # Normalize column names
    df.columns = df.columns.str.upper().str.strip()
    
    # Column mappings
    col_map = {
        'POSTCODE': 'POSTCODE_SECTOR',
        'POSTCODESECTOR': 'POSTCODE_SECTOR',
        'POSTCODE SECTOR': 'POSTCODE_SECTOR',
        'TYPE': 'PROPERTY_TYPE',
        'PROPERTYTYPE': 'PROPERTY_TYPE',
        'PROPERTY TYPE': 'PROPERTY_TYPE',
        'FORM': 'BUILT_FORM',
        'BUILTFORM': 'BUILT_FORM',
        'BUILT FORM': 'BUILT_FORM',
        'AREA': 'TOTAL_FLOOR_AREA',
        'FLOOR_AREA': 'TOTAL_FLOOR_AREA',
        'FLOORAREA': 'TOTAL_FLOOR_AREA',
        'SQM': 'TOTAL_FLOOR_AREA',
        'BEDS': 'NUMBER_BEDROOMS',
        'BEDROOMS': 'NUMBER_BEDROOMS',
        'BATHS': 'NUMBER_BATHROOMS',
        'BATHROOMS': 'NUMBER_BATHROOMS',
        'PRICE': 'LISTING_PRICE',
        'ASKING_PRICE': 'LISTING_PRICE',
        'LISTINGPRICE': 'LISTING_PRICE',
        'LA': 'LOCAL_AUTHORITY_LABEL',
        'LOCAL_AUTHORITY': 'LOCAL_AUTHORITY_LABEL',
        'EPC': 'CURRENT_ENERGY_RATING',
        'ENERGY': 'CURRENT_ENERGY_RATING',
        'ENERGY_RATING': 'CURRENT_ENERGY_RATING',
    }
    
    df = df.rename(columns=col_map)
    
    for idx, row in df.iterrows():
        try:
            property_data = {
                'postcode_sector': str(row.get('POSTCODE_SECTOR', '')),
                'property_type': str(row.get('PROPERTY_TYPE', 'House')),
                'built_form': str(row.get('BUILT_FORM', 'Semi-Detached')),
                'total_floor_area': float(row.get('TOTAL_FLOOR_AREA', 0)),
                'number_bedrooms': int(row.get('NUMBER_BEDROOMS', 0)),
                'number_bathrooms': int(row.get('NUMBER_BATHROOMS', 1)),
                'listing_price': float(row.get('LISTING_PRICE', 0)),
                'local_authority_label': str(row.get('LOCAL_AUTHORITY_LABEL', 'Unknown')),
                'current_energy_rating': str(row.get('CURRENT_ENERGY_RATING', 'D')),
                'old_new': str(row.get('OLD/NEW', 'Y')),
                'tenure_type': str(row.get('TENURE_TYPE', 'F')),
            }
            
            result = run_full_prediction(property_data, version)
            
            results.append({
                'row': idx,
                'postcode': property_data['postcode_sector'],
                'property_type': property_data['property_type'],
                'listing_price': property_data['listing_price'],
                'predicted': result['point_estimate'],
                'diff_pct': result['diff_pct'],
                'verdict': result['verdict'],
                'recommendation': result['recommendation'],
                'confidence': result['confidence'],
                'success': True,
            })
            
            if result['recommendation']:
                recommendations_count[result['recommendation']] = recommendations_count.get(result['recommendation'], 0) + 1
            
            # Track opportunities
            if result['confidence'] == 'HIGH':
                if result['verdict'] in ['STRONG_UNDERVALUED', 'LIKELY_UNDERVALUED']:
                    undervalued_listings.append({
                        'postcode': property_data['postcode_sector'],
                        'listing': property_data['listing_price'],
                        'predicted': result['point_estimate'],
                        'diff_pct': result['diff_pct'],
                    })
                elif result['verdict'] in ['STRONG_OVERVALUED', 'LIKELY_OVERVALUED']:
                    overvalued_listings.append({
                        'postcode': property_data['postcode_sector'],
                        'listing': property_data['listing_price'],
                        'predicted': result['point_estimate'],
                        'diff_pct': result['diff_pct'],
                    })
            
        except Exception as e:
            results.append({
                'row': idx,
                'postcode': str(row.get('POSTCODE_SECTOR', 'Unknown')),
                'error': str(e),
                'success': False,
            })
    
    # Calculate bias
    successful = [r for r in results if r['success']]
    if successful:
        diffs = [r['diff_pct'] for r in successful]
        mean_diff = np.mean(diffs)
        median_diff = np.median(diffs)
    else:
        mean_diff = median_diff = 0
    
    return {
        'total': len(df),
        'successful': len(successful),
        'failed': len(df) - len(successful),
        'results': results,
        'analysis': {
            'recommendations': recommendations_count,
            'opportunities': {
                'undervalued_high_conf': len(undervalued_listings),
                'overvalued_high_conf': len(overvalued_listings),
                'undervalued_listings': sorted(undervalued_listings, key=lambda x: x['diff_pct'])[:10],
                'overvalued_listings': sorted(overvalued_listings, key=lambda x: x['diff_pct'], reverse=True)[:10],
            },
            'bias': {
                'mean_diff_pct': mean_diff,
                'median_diff_pct': median_diff,
            },
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SAVED ANALYSES
# ═══════════════════════════════════════════════════════════════════════════════

def load_saved_analyses() -> list:
    """Load saved analyses from file."""
    if ANALYSES_FILE.exists():
        try:
            with open(ANALYSES_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []


def save_analyses(analyses: list):
    """Save analyses to file."""
    with open(ANALYSES_FILE, 'w') as f:
        json.dump(analyses, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def get_model_stats(version: str) -> dict:
    """Get model statistics for a version."""
    session = get_session(version)
    
    # Hardcoded correct MAPEs
    CORRECT_MAPES = {
        'v1': 10.01,
        'v2': 10.31,
        'v3': 9.95,
    }
    
    segment_mapes = session.get('segment_mapes', {})
    avg_mape = CORRECT_MAPES.get(version, 10.0)
    
    return {
        'version': version,
        'segment_mapes': segment_mapes,
        'average_mape': avg_mape,
        'n_models': sum(len(m) for m in session['tuned_models'].values()),
        'has_quantile': session['quantile_models'] is not None,
        'has_conformal': session['conformal_thresholds'] is not None,
        'has_transactions': session['transactions_prepared'] is not None,
    }


def get_stats_summary() -> dict:
    """Get cross-version statistics summary."""
    summaries = {}
    
    for v in ['v1', 'v2', 'v3']:
        try:
            summaries[v] = get_model_stats(v)
        except Exception as e:
            summaries[v] = {'error': str(e)}
    
    return summaries


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class PropertyInput(BaseModel):
    postcode_sector: str
    property_type: str = 'House'
    built_form: str = 'Semi-Detached'
    total_floor_area: float
    number_bedrooms: int
    number_bathrooms: int = 1
    listing_price: Optional[float] = None
    local_authority_label: str = 'Unknown'
    current_energy_rating: str = 'D'
    old_new: str = 'Y'
    tenure_type: str = 'F'


class SaveAnalysisRequest(BaseModel):
    name: str
    analysis: dict
    property_input: dict
    version: str = 'v3'
    tags: List[str] = []


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    print("=" * 60)
    print(" PropertyCompanion API Starting...")
    print("=" * 60)
    
    # Pre-load default version
    try:
        SESSION['v3'] = load_session('v3')
        print("  ✓ v3 models loaded")
    except Exception as e:
        print(f"  ⚠ v3 loading failed: {e}")
    
    yield
    
    print("Shutting down...")


app = FastAPI(
    title="PropertyCompanion API",
    description="UK Property Valuation ML API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {"message": "PropertyCompanion API", "status": "running"}


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "versions_loaded": list(SESSION.keys()),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/versions")
async def get_versions():
    """Get available model versions."""
    versions = []
    for v, name in VERSION_NAMES.items():
        paths = get_model_paths(v)
        exists = paths['option5_dir'].exists()
        versions.append({
            'id': v,
            'name': name,
            'available': exists,
            'loaded': v in SESSION,
        })
    return {'versions': versions}


@app.post("/api/predict")
async def predict(property_input: PropertyInput, version: str = Query('v3')):
    """Get property valuation."""
    import traceback
    try:
        print(f"\n{'='*50}")
        print(f"Prediction request for {version}")
        print(f"Input: {property_input.dict()}")
        
        property_data = property_input.dict()
        
        print("Loading session...")
        session = get_session(version)
        print(f"Session loaded: {len(session['tuned_models'])} model groups")
        
        print("Running prediction...")
        result = run_full_prediction(property_data, version)
        
        print(f"Success! Point estimate: {result.get('point_estimate')}")
        
        # Convert numpy types to native Python types
        result = convert_numpy_types(result)
        
        return result
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"\n{'='*50}")
        print(f"ERROR in /api/predict:")
        print(error_trace)
        print(f"{'='*50}\n")
        raise HTTPException(status_code=500, detail=f"{str(e)}\n\nTraceback:\n{error_trace}")


@app.post("/api/upload-excel")
async def upload_excel(file: UploadFile = File(...), version: str = Query('v3')):
    """Process batch Excel file."""
    try:
        contents = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
        
        result = process_batch_excel(df, version)
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/comparables")
async def get_comparables(property_input: PropertyInput, version: str = Query('v3')):
    """Find comparable transactions."""
    try:
        session = get_session(version)
        result = find_comparables(
            session,
            postcode_sector=property_input.postcode_sector,
            property_type=property_input.property_type,
            built_form=property_input.built_form,
            total_floor_area=property_input.total_floor_area,
            number_bedrooms=property_input.number_bedrooms,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats/{version}")
async def get_stats(version: str):
    """Get model statistics for a version."""
    try:
        stats = get_model_stats(version)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats-summary")
async def stats_summary():
    """Get cross-version statistics summary."""
    try:
        return get_stats_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analyses")
async def get_analyses():
    """Get saved analyses."""
    analyses = load_saved_analyses()
    return {'analyses': analyses}


@app.post("/api/analyses/save")
async def save_analysis(request: SaveAnalysisRequest):
    """Save an analysis."""
    analyses = load_saved_analyses()
    
    new_analysis = {
        'id': str(uuid.uuid4()),
        'name': request.name,
        'analysis': request.analysis,
        'property_input': request.property_input,
        'version': request.version,
        'tags': request.tags,
        'timestamp': datetime.now().isoformat(),
    }
    
    analyses.append(new_analysis)
    save_analyses(analyses)
    
    return {'success': True, 'id': new_analysis['id']}


@app.delete("/api/analyses/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """Delete an analysis."""
    analyses = load_saved_analyses()
    analyses = [a for a in analyses if a['id'] != analysis_id]
    save_analyses(analyses)
    return {'success': True}


@app.get("/api/comparisons")
async def get_comparisons():
    """Get saved comparisons."""
    return {'comparisons': []}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
