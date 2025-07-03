from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import pandas as pd
import numpy as np
from datetime import timedelta
import os

# --- Machine Learning Specific Imports ---
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, auc, f1_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE 
import xgboost as xgb # NEW IMPORT FOR XGBOOST

# --- Configuration for Data Saving/Loading ---
CATALOG_FILE = "california_earthquake_catalog_M2+.csv"
FORCE_FETCH = False

# --- 1. Data Acquisition ---
print("--- Step 1: Data Acquisition ---")

df_catalog = pd.DataFrame()

if os.path.exists(CATALOG_FILE) and not FORCE_FETCH:
    print(f"Loading earthquake catalog from {CATALOG_FILE}...")
    try:
        df_catalog = pd.read_csv(CATALOG_FILE, parse_dates=['origin_time'])
        df_catalog['magnitude'] = pd.to_numeric(df_catalog['magnitude'], errors='coerce')
        df_catalog['depth_km'] = pd.to_numeric(df_catalog['depth_km'], errors='coerce')
        print(f"Loaded {len(df_catalog)} events from CSV.")
    except Exception as e:
        print(f"Error loading CSV: {e}. Attempting to fetch data from USGS instead.")
        FORCE_FETCH = True
else:
    print(f"'{CATALOG_FILE}' not found or FORCE_FETCH is True. Proceeding to fetch from USGS.")


if FORCE_FETCH or not os.path.exists(CATALOG_FILE) or df_catalog.empty:
    client = Client("USGS")

    start_time_full_catalog = UTCDateTime("2005-01-01")
    end_time_full_catalog = UTCDateTime.now()

    minlatitude = 32.0
    maxlatitude = 42.0
    minlongitude = -125.0
    maxlongitude = -114.0
    minmagnitude_fetch = 2.0

    print(f"Fetching full catalog for California from {start_time_full_catalog} to {end_time_full_catalog} (M{minmagnitude_fetch}+)...")

    all_records = []
    current_chunk_start = start_time_full_catalog
    chunk_duration_years = 1

    while current_chunk_start < end_time_full_catalog:
        current_chunk_end = min(current_chunk_start + timedelta(days=chunk_duration_years * 365.25), end_time_full_catalog)
        
        if current_chunk_end <= current_chunk_start:
             current_chunk_start = current_chunk_end + timedelta(days=1)
             continue

        print(f"  Fetching chunk from {current_chunk_start} to {current_chunk_end}...")
        try:
            events_chunk = client.get_events(starttime=current_chunk_start,
                                            endtime=current_chunk_end,
                                            minlatitude=minlatitude,
                                            maxlatitude=maxlatitude,
                                            minlongitude=minlongitude,
                                            maxlongitude=maxlongitude,
                                            minmagnitude=minmagnitude_fetch,
                                            orderby="time-asc")

            for event in events_chunk:
                origin = event.preferred_origin() or event.origins[0]
                magnitude = event.preferred_magnitude() or event.magnitudes[0]
                
                time = origin.time.datetime if origin.time else None
                lat = origin.latitude if origin.latitude is not None else None
                lon = origin.longitude if origin.longitude is not None else None
                depth = origin.depth / 1000 if origin.depth is not None else None
                mag = magnitude.mag if magnitude.mag is not None else None
                mag_type = magnitude.magnitude_type if magnitude.magnitude_type else None

                all_records.append({
                    "origin_time": time,
                    "latitude": lat,
                    "longitude": lon,
                    "depth_km": depth,
                    "magnitude": mag,
                    "magnitude_type": mag_type
                })
            print(f"  Fetched {len(events_chunk)} events in this chunk.")
            
        except Exception as e:
            print(f"  Error fetching chunk {current_chunk_start} to {current_chunk_end}: {e}")
            print("  This might be due to exceeding the 20k limit within this chunk. Consider reducing `chunk_duration_years`.")
            break

        current_chunk_start = current_chunk_end

    df_catalog = pd.DataFrame(all_records)

    df_catalog = df_catalog.sort_values("origin_time").reset_index(drop=True)

    print(f"\nTotal events retrieved for full catalog: {len(df_catalog)}")
    print("Head of raw catalog data (first 5 events from full period):")
    print(df_catalog.head())
    print("\n")

    print(f"Saving fetched catalog to {CATALOG_FILE}...")
    df_catalog.to_csv(CATALOG_FILE, index=False)
    print("Catalog saved.\n")
else:
    print("Skipping data fetching. Using loaded catalog.\n")


# --- 2. Feature Engineering Function ---
print("--- Step 2: Feature Engineering (b-value calculation) ---")

def calculate_b_value(magnitudes, m_min):
    magnitudes = magnitudes[magnitudes >= m_min]
    if len(magnitudes) < 5:
        return np.nan 
    
    mean_mag = np.mean(magnitudes)
    if (mean_mag - m_min) <= 0:
        return np.nan
        
    b = np.log10(np.e) / (mean_mag - m_min)
    return b

print("b-value calculation function defined.\n")

# --- 3. Sliding Window Feature Extraction and Labelling (REVERTED temporal features) ---
print("--- Step 3: Sliding Window Feature Extraction and Labelling ---")

window_size_days = 90
step_days = 7
m_min_for_features = 2.0

target_magnitude = 6.0
prediction_horizon_days = 30

if len(df_catalog) == 0:
    print("No events in catalog. Cannot perform feature extraction and labeling.")
else:
    start_date_catalog = df_catalog["origin_time"].min()
    end_date_catalog = df_catalog["origin_time"].max() 

    current_window_start = start_date_catalog
    feature_rows = []

    df_large_events = df_catalog[df_catalog["magnitude"] >= target_magnitude].copy()

    while current_window_start + timedelta(days=window_size_days + prediction_horizon_days) <= end_date_catalog:
        current_window_end = current_window_start + timedelta(days=window_size_days)
        
        df_window = df_catalog[
            (df_catalog["origin_time"] >= current_window_start) & 
            (df_catalog["origin_time"] < current_window_end)
        ].copy()

        df_window['magnitude'] = pd.to_numeric(df_window['magnitude'], errors='coerce')
        df_window['depth_km'] = pd.to_numeric(df_window['depth_km'], errors='coerce')
        
        seismicity_rate = len(df_window) 
        b_value = calculate_b_value(df_window["magnitude"], m_min_for_features)

        mean_magnitude = df_window["magnitude"].mean() if seismicity_rate > 0 else np.nan
        std_magnitude = df_window["magnitude"].std() if seismicity_rate > 1 else np.nan
        max_magnitude = df_window["magnitude"].max() if seismicity_rate > 0 else np.nan

        inter_event_times = np.diff(df_window["origin_time"].astype(np.int64)) / 10**9
        mean_inter_event_time = np.mean(inter_event_times) if len(inter_event_times) > 0 else np.nan
        std_inter_event_time = np.std(inter_event_times) if len(inter_event_times) > 1 else np.nan
        cv_inter_event_time = (std_inter_event_time / mean_inter_event_time) if mean_inter_event_time > 0 else np.nan

        mean_depth = df_window["depth_km"].mean() if seismicity_rate > 0 else np.nan
        std_depth = df_window["depth_km"].std() if seismicity_rate > 1 else np.nan
        
        # Skip window if core features are NaN or seismicity is 0
        if pd.isna(b_value) or seismicity_rate == 0 or pd.isna(mean_magnitude):
            current_window_start += timedelta(days=step_days)
            continue
        
        future_start_time = current_window_end
        future_end_time = current_window_end + timedelta(days=prediction_horizon_days)
        
        large_quake_in_future = not df_large_events[
            (df_large_events["origin_time"] >= future_start_time) &
            (df_large_events["origin_time"] < future_end_time)
        ].empty
        
        feature_rows.append({
            "window_end": current_window_end,
            "seismicity_rate": seismicity_rate,
            "b_value": b_value,
            "mean_magnitude": mean_magnitude,
            "std_magnitude": std_magnitude,
            "max_magnitude": max_magnitude,
            "mean_inter_event_time": mean_inter_event_time,
            "cv_inter_event_time": cv_inter_event_time,
            "mean_depth": mean_depth,
            "std_depth": std_depth,
            "target_label": 1 if large_quake_in_future else 0
        })
        
        current_window_start += timedelta(days=step_days)

    df_features = pd.DataFrame(feature_rows)

    print("Feature DataFrame created.")
    print("Head of engineered features:")
    print(df_features.head())
    print("\n")

    # --- 4. Label Imbalance Analysis ---
    print("--- Step 4: Label Imbalance Analysis ---")
    num_positive_labels = df_features['target_label'].sum()
    num_total_labels = len(df_features)
    num_negative_labels = num_total_labels - num_positive_labels

    print(f"Total feature windows: {num_total_labels}")
    print(f"Number of 'large earthquake' labels (1s): {num_positive_labels}")
    print(f"Number of 'no large earthquake' labels (0s): {num_negative_labels}")
    if num_total_labels > 0:
        print(f"Percentage of 'large earthquake' labels: {num_positive_labels / num_total_labels * 100:.2f}%")
    else:
        print("No feature windows generated.")

# --- 5. Machine Learning Model Training and Evaluation (XGBoost with Tuning) ---
print("\n--- Step 5: Machine Learning Model Training and Evaluation ---")

if 'df_features' in locals() and not df_features.empty:
    # 5.1. Prepare Data for ML - REVERTED X to previous features
    X = df_features[['seismicity_rate', 'b_value', 'mean_magnitude', 'std_magnitude', 
                     'max_magnitude', 'mean_inter_event_time', 'cv_inter_event_time', 
                     'mean_depth', 'std_depth']] 
    y = df_features['target_label']

    print(f"Features (X) shape: {X.shape}")
    print(f"Labels (y) shape: {y.shape}")

    if X.isnull().sum().sum() > 0:
        print("\nWARNING: NaN values found in features. These rows will be dropped for training.")
        original_rows = X.shape[0]
        X = X.dropna()
        y = y.loc[X.index] 
        print(f"Dropped {original_rows - X.shape[0]} rows due to NaN values.")
        print(f"New Features (X) shape: {X.shape}")
        print(f"New Labels (y) shape: {y.shape}")
    
    if len(X) == 0:
        print("No valid feature rows after dropping NaNs. Cannot train ML model.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        print(f"Train label distribution (before SMOTE):\n{y_train.value_counts(normalize=True)}")
        print(f"Test label distribution:\n{y_test.value_counts(normalize=True)}")

        print("\nApplying SMOTE to balance the training data...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        print(f"Train set shape after SMOTE: {X_train_resampled.shape}")
        print(f"Train label distribution after SMOTE:\n{y_train_resampled.value_counts(normalize=True)}")

        # --- Step 6: Hyperparameter Tuning with GridSearchCV for XGBoost ---
        print("\n--- Step 6: Hyperparameter Tuning with GridSearchCV for XGBoost ---")
        
        # Define the parameter grid for XGBoost
        param_grid = {
            'n_estimators': [100, 200, 300], # Number of boosting rounds
            'learning_rate': [0.05, 0.1, 0.2], # Step size shrinkage to prevent overfitting
            'max_depth': [3, 5, 7], # Maximum depth of a tree
            'subsample': [0.7, 0.9], # Subsample ratio of the training instance
            'colsample_bytree': [0.7, 0.9], # Subsample ratio of columns when constructing each tree
            'gamma': [0, 0.1, 0.2] # Minimum loss reduction required to make a further partition on a leaf node
        }

        # Create a scorer for F1-score for the positive class (1)
        scorer = make_scorer(f1_score, pos_label=1)

        # Initialize GridSearchCV with XGBClassifier
        grid_search = GridSearchCV(
            estimator=xgb.XGBClassifier(
                objective='binary:logistic', # For binary classification
                use_label_encoder=False,     # Suppress warning for label encoder
                eval_metric='logloss',       # Evaluation metric for the objective function
                random_state=42
            ),
            param_grid=param_grid,
            scoring=scorer,
            cv=5,
            verbose=2,
            n_jobs=-1
        )

        print("Starting GridSearchCV for XGBoost...")
        grid_search.fit(X_train_resampled, y_train_resampled)
        print("GridSearchCV complete.")

        best_model = grid_search.best_estimator_
        print(f"\nBest parameters found for XGBoost: {grid_search.best_params_}")
        print(f"Best cross-validation F1-score: {grid_search.best_score_:.4f}")

        # --- 5.4. Evaluation (now using the best_model found by GridSearchCV) ---
        print("\n--- Model Evaluation on Original Test Set (Best XGBoost Model) ---")
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Large Quake', 'Large Quake']))

        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.show()

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(7, 6))
        plt.plot(recall, precision, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall (True Positive Rate)')
        plt.ylabel('Precision (Positive Predictive Value)')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.show()

        print("\n--- Feature Importance (XGBoost) ---")
        feature_importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print(feature_importances)
else:
    print("Skipping ML model training as no features were generated or loaded.")