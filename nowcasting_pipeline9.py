import joblib # Added for model serialization
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import pandas as pd
import numpy as np
from datetime import timedelta
import os
import time # Import time for sleep

# --- Machine Learning Specific Imports ---
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, auc, f1_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# --- Configuration for Data Saving/Loading ---
CATALOG_FILE = "california_earthquake_catalog_M2+_1990_Present.csv"
FORCE_FETCH = False # Set to True for the first run or if you need to re-fetch

# --- Model Deployment Paths ---
MODEL_SAVE_PATH = "earthquake_prediction_model.joblib"
FEATURE_COLUMNS_SAVE_PATH = "model_feature_columns.joblib"


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

    start_time_full_catalog = UTCDateTime("1990-01-01")
    end_time_full_catalog = UTCDateTime.now()

    minlatitude_region = 32.0
    maxlatitude_region = 42.0
    minlongitude_region = -125.0
    maxlongitude_region = -114.0
    minmagnitude_fetch = 2.0

    print(f"Fetching full catalog for California from {start_time_full_catalog} to {end_time_full_catalog} (M{minmagnitude_fetch}+)...")

    all_records = []
    
    # Use a stack to manage time windows to process, starting with the full range
    # Each item in the stack is (start_time, end_time, level) where level indicates chunking depth
    # Level 0: Years, Level 1: Months, Level 2: Weeks, Level 3: Days
    time_windows_to_process = [(start_time_full_catalog, end_time_full_catalog, 0)]

    retries = 0
    max_retries_per_chunk = 5 # Prevent infinite loops for persistent issues

    while time_windows_to_process:
        current_chunk_start, current_chunk_end, level = time_windows_to_process.pop(0) # Process in FIFO order
        
        if current_chunk_end <= current_chunk_start:
            continue

        print(f"  Fetching chunk (Level {level}) from {current_chunk_start} to {current_chunk_end}...")
        try:
            events_chunk = client.get_events(starttime=current_chunk_start,
                                            endtime=current_chunk_end,
                                            minlatitude=minlatitude_region,
                                            maxlatitude=maxlatitude_region,
                                            minlongitude=minlongitude_region,
                                            maxlongitude=maxlongitude_region,
                                            minmagnitude=minmagnitude_fetch,
                                            orderby="time-asc")

            for event in events_chunk:
                origin = event.preferred_origin() or event.origins[0]
                magnitude = event.preferred_magnitude() or event.magnitudes[0]
                
                time_val = origin.time.datetime if origin.time else None
                lat = origin.latitude if origin.latitude is not None else None
                lon = origin.longitude if origin.longitude is not None else None
                depth = origin.depth / 1000 if origin.depth is not None else None
                mag = magnitude.mag if magnitude.mag is not None else None
                mag_type = magnitude.magnitude_type if magnitude.magnitude_type else None

                all_records.append({
                    "origin_time": time_val,
                    "latitude": lat,
                    "longitude": lon,
                    "depth_km": depth,
                    "magnitude": mag,
                    "magnitude_type": mag_type
                })
            print(f"  Fetched {len(events_chunk)} events in this chunk.")
            retries = 0 # Reset retries on success

        except Exception as e:
            error_message = str(e)
            print(f"  Error fetching chunk {current_chunk_start} to {current_chunk_end}: {error_message}")
            
            if "Error 400: Bad Request" in error_message and "exceeds search limit" in error_message:
                print(f"  Chunk too large. Splitting Level {level} further...")
                if retries < max_retries_per_chunk:
                    retries += 1
                    # Divide the current problematic chunk into smaller sub-chunks
                    mid_time = current_chunk_start + (current_chunk_end - current_chunk_start) / 2
                    
                    if level == 0: # Divide year into months
                        step_duration = timedelta(days=30) # Approximate month
                    elif level == 1: # Divide month into weeks
                        step_duration = timedelta(days=7) # Week
                    elif level == 2: # Divide week into days
                        step_duration = timedelta(days=1) # Day
                    else: # If even days are too much, we have a problem (very high seismicity)
                        print("  Cannot split further (reached daily chunks or already too many retries). Skipping this sub-chunk.")
                        continue # Skip this problematic sub-chunk

                    sub_chunk_start = current_chunk_start
                    while sub_chunk_start < current_chunk_end:
                        sub_chunk_end = min(sub_chunk_start + step_duration, current_chunk_end)
                        time_windows_to_process.insert(0, (sub_chunk_start, sub_chunk_end, level + 1)) # Add to front to process smaller chunks next
                        sub_chunk_start = sub_chunk_end
                    time.sleep(1) # Be polite to the server
                else:
                    print(f"  Max retries reached for chunk {current_chunk_start} to {current_chunk_end}. Skipping to next window.")
                    retries = 0 # Reset for the next top-level chunk
            else:
                print(f"  Non-limit related error or max retries hit. Skipping this chunk: {e}")
                retries = 0

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


# --- 2. Feature Engineering Functions and Grid Definition ---
print("--- Step 2: Feature Engineering Functions and Grid Definition ---")

def calculate_b_value(magnitudes, m_min):
    magnitudes = magnitudes[magnitudes >= m_min]
    if len(magnitudes) < 5: 
        return np.nan 
    
    mean_mag = np.mean(magnitudes)
    if (mean_mag - m_min) <= 0: 
        return np.nan
        
    b = np.log10(np.e) / (mean_mag - m_min)
    return b

print("b-value calculation function defined.")

LAT_MIN = 32.0
LAT_MAX = 42.0
LON_MIN = -125.0
LON_MAX = -114.0
GRID_SIZE_DEG = 0.5 

lat_bins = np.arange(LAT_MIN, LAT_MAX + GRID_SIZE_DEG, GRID_SIZE_DEG)
lon_bins = np.arange(LON_MIN, LON_MAX + GRID_SIZE_DEG, GRID_SIZE_DEG)

grid_cells = []
for i in range(len(lat_bins) - 1):
    for j in range(len(lon_bins) - 1):
        cell_lat_min = lat_bins[i]
        cell_lat_max = lat_bins[i+1]
        cell_lon_min = lon_bins[j]
        cell_lon_max = lon_bins[j+1]
        grid_cells.append({
            'lat_min': cell_lat_min, 'lat_max': cell_lat_max,
            'lon_min': cell_lon_min, 'lon_max': cell_lon_max,
            'id': f"lat{cell_lat_min:.1f}_lon{cell_lon_min:.1f}"
        })
print(f"Defined {len(grid_cells)} grid cells for spatial features.\n")


# --- 3. Sliding Window Feature Extraction and Labelling (INCLUDING SPATIAL FEATURES) ---
print("--- Step 3: Sliding Window Feature Extraction and Labelling (INCLUDING SPATIAL FEATURES) ---")

window_size_days = 90
step_days = 7
m_min_for_features = 2.0
min_events_per_cell = 3 

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
        
        # --- Overall Region Features (Still useful as a general context) ---
        row_features = {}
        row_features["window_end"] = current_window_end
        row_features["seismicity_rate_region"] = len(df_window)
        row_features["b_value_region"] = calculate_b_value(df_window["magnitude"], m_min_for_features)
        row_features["mean_magnitude_region"] = df_window["magnitude"].mean() if len(df_window) > 0 else np.nan
        row_features["std_magnitude_region"] = df_window["magnitude"].std() if len(df_window) > 1 else np.nan
        row_features["max_magnitude_region"] = df_window["magnitude"].max() if len(df_window) > 0 else np.nan
        
        inter_event_times = np.diff(df_window["origin_time"].astype(np.int64)) / 10**9
        row_features["mean_inter_event_time_region"] = np.mean(inter_event_times) if len(inter_event_times) > 0 else np.nan
        row_features["cv_inter_event_time_region"] = (np.std(inter_event_times) / np.mean(inter_event_times)) if np.mean(inter_event_times) > 0 else np.nan
        
        row_features["mean_depth_region"] = df_window["depth_km"].mean() if len(df_window) > 0 else np.nan
        row_features["std_depth_region"] = df_window["depth_km"].std() if len(df_window) > 1 else np.nan

        # --- Spatial Features per Grid Cell ---
        for cell in grid_cells:
            df_cell_window = df_window[
                (df_window["latitude"] >= cell['lat_min']) & (df_window["latitude"] < cell['lat_max']) &
                (df_window["longitude"] >= cell['lon_min']) & (df_window["longitude"] < cell['lon_max'])
            ].copy()
            
            num_events_cell = len(df_cell_window)
            
            if num_events_cell >= min_events_per_cell: 
                cell_seismicity_rate = num_events_cell
                cell_b_value = calculate_b_value(df_cell_window["magnitude"], m_min_for_features)
                cell_mean_magnitude = df_cell_window["magnitude"].mean() if num_events_cell > 0 else np.nan 
                
                row_features[f"seismicity_rate_{cell['id']}"] = cell_seismicity_rate
                row_features[f"b_value_{cell['id']}"] = cell_b_value
                row_features[f"mean_magnitude_{cell['id']}"] = cell_mean_magnitude 

        # --- Target Labeling ---
        future_start_time = current_window_end
        future_end_time = current_window_end + timedelta(days=prediction_horizon_days)
        
        large_quake_in_future = not df_large_events[
            (df_large_events["origin_time"] >= future_start_time) &
            (df_large_events["origin_time"] < future_end_time)
        ].empty
        
        row_features["target_label"] = 1 if large_quake_in_future else 0
        
        if not pd.isna(row_features["b_value_region"]) and row_features["seismicity_rate_region"] > 0:
            feature_rows.append(row_features)
        
        current_window_start += timedelta(days=step_days)

    df_features = pd.DataFrame(feature_rows)
    
    spatial_feature_cols = [col for col in df_features.columns if "lat" in col and "lon" in col]
    for col in spatial_feature_cols:
        df_features[col] = df_features[col].fillna(0) 

    print("Feature DataFrame created.")
    print("Head of engineered features (first 5 rows and some columns):")
    print(df_features.iloc[:, :15].head()) 
    print(f"\nTotal number of features (columns): {df_features.shape[1] - 2}")
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

# --- 5. Machine Learning Model Training and Evaluation (XGBoost with Spatial Features) ---
print("\n--- Step 5: Machine Learning Model Training and Evaluation ---")

if 'df_features' in locals() and not df_features.empty:
    X = df_features.drop(columns=['window_end', 'target_label']) 
    y = df_features['target_label']

    print(f"Features (X) shape: {X.shape}")
    print(f"Labels (y) shape: {y.shape}")

    if X.isnull().sum().sum() > 0:
        print("\nWARNING: NaN values still found in features after spatial processing. These rows will be dropped for training.")
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
        # Dynamically set k_neighbors to avoid error if minority class is small
        # The number of samples for k_neighbors must be less than or equal to the number of minority samples - 1
        num_minority_train_samples = y_train.value_counts()[1] if 1 in y_train.value_counts() else 0
        smote_k_neighbors = min(5, max(1, num_minority_train_samples - 1)) 
        
        if num_minority_train_samples <= 1: # SMOTE needs at least 2 samples to create synthetic ones
            print(f"  Warning: Not enough minority samples ({num_minority_train_samples}) in training set for SMOTE. Skipping SMOTE.")
            X_train_resampled, y_train_resampled = X_train, y_train # Skip SMOTE
        else:
            smote = SMOTE(random_state=42, k_neighbors=smote_k_neighbors)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            print(f"  SMOTE k_neighbors used: {smote_k_neighbors}")


        print(f"Train set shape after SMOTE: {X_train_resampled.shape}")
        print(f"Train label distribution after SMOTE:\n{y_train_resampled.value_counts(normalize=True)}")

        # --- Step 6: Hyperparameter Tuning with GridSearchCV for XGBoost ---
        print("\n--- Step 6: Hyperparameter Tuning with GridSearchCV for XGBoost ---")
        
        # We will use the best parameters found in the previous run as a single combination
        # since we are focusing on threshold optimization, not hyperparameter tuning.
        param_grid = {
            'n_estimators': [250], 
            'learning_rate': [0.1], 
            'max_depth': [7], 
            'subsample': [0.8],
            'colsample_bytree': [0.7],
            'gamma': [0] 
        }

        scorer = make_scorer(f1_score, pos_label=1)

        grid_search = GridSearchCV(
            estimator=xgb.XGBClassifier(
                objective='binary:logistic',
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            ),
            param_grid=param_grid,
            scoring=scorer,
            cv=5,
            verbose=2,
            n_jobs=-1
        )

        print("Starting GridSearchCV for XGBoost with extended catalog and best known parameters...")
        grid_search.fit(X_train_resampled, y_train_resampled)
        print("GridSearchCV complete.")

        best_model = grid_search.best_estimator_
        print(f"\nBest parameters found for XGBoost: {grid_search.best_params_}")
        print(f"Best cross-validation F1-score: {grid_search.best_score_:.4f}")

        # --- Deployment Step 1: Save the trained model and feature column names ---
        print(f"\nSaving trained model to {MODEL_SAVE_PATH}...")
        joblib.dump(best_model, MODEL_SAVE_PATH)
        print("Model saved.")

        print(f"Saving feature column names to {FEATURE_COLUMNS_SAVE_PATH}...")
        joblib.dump(X.columns.tolist(), FEATURE_COLUMNS_SAVE_PATH) # X holds the original feature column names
        print("Feature column names saved.")
        # --- End Deployment Step 1 ---


        # --- 5.4. Evaluation (using the best_model) ---
        print("\n--- Model Evaluation on Original Test Set (Default Threshold) ---")
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        print("\nClassification Report (Default Threshold 0.5):")
        print(classification_report(y_test, y_pred, target_names=['No Large Quake', 'Large Quake']))

        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix (Default Threshold 0.5):")
        print(cm)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Predicted 0', 'Predicted 1'],
                            yticklabels=['Actual 0', 'Actual 1'])
        plt.title('Confusion Matrix (Default Threshold 0.5)')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.show()

        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(7, 6))
        plt.plot(recall, precision, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall (True Positive Rate)')
        plt.ylabel('Precision (Positive Predictive Value)')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.show()

        print("\n--- Feature Importance (XGBoost with Extended Catalog) ---")
        feature_importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print(feature_importances)

        # --- Step 7: Threshold Optimization ---
        print("\n--- Step 7: Threshold Optimization ---")

        # Calculate F1-scores for various thresholds
        f1_scores = []
        for thresh in thresholds:
            y_pred_thresh = (y_proba >= thresh).astype(int)
            f1_scores.append(f1_score(y_test, y_pred_thresh, pos_label=1))

        # Find the threshold that maximizes F1-score
        optimal_f1_threshold_idx = np.argmax(f1_scores)
        optimal_f1_threshold = thresholds[optimal_f1_threshold_idx]
        print(f"Threshold for optimal F1-score: {optimal_f1_threshold:.4f}")

        # Plot Precision, Recall, and F1-score vs. Threshold
        plt.figure(figsize=(10, 7))
        plt.plot(thresholds, precision[:-1], label='Precision')
        plt.plot(thresholds, recall[:-1], label='Recall')
        plt.plot(thresholds, f1_scores, label='F1-score')
        plt.axvline(x=optimal_f1_threshold, color='r', linestyle='--', label=f'Optimal F1 Threshold ({optimal_f1_threshold:.2f})')
        plt.xlabel('Probability Threshold')
        plt.ylabel('Score')
        plt.title('Precision, Recall, and F1-score vs. Probability Threshold')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Let's evaluate at a couple of interesting thresholds based on common trade-offs:
        # 1. The F1-optimal threshold (already found)
        # 2. A threshold that prioritizes higher Recall (e.g., trying to catch 80% of quakes)
        # 3. A threshold that prioritizes higher Precision (e.g., reducing false alarms)

        # Example: Threshold for higher recall (e.g., recall >= 0.8 if possible)
        # Find threshold where recall is at least 0.8, taking the lowest such threshold to maximize precision
        # Check if any threshold gives recall >= 0.8
        high_recall_threshold_idx = np.where(recall[:-1] >= 0.8)[0]
        if len(high_recall_threshold_idx) > 0:
            # We want the highest precision for the desired recall, which corresponds to the highest threshold
            # among those meeting the recall criteria.
            high_recall_threshold = thresholds[high_recall_threshold_idx[-1]] 
            print(f"\nEvaluating at a High Recall Threshold (e.g., recall >= 0.8): {high_recall_threshold:.4f}")
            y_pred_high_recall = (y_proba >= high_recall_threshold).astype(int)
            print(classification_report(y_test, y_pred_high_recall, target_names=['No Large Quake', 'Large Quake']))
            cm_high_recall = confusion_matrix(y_test, y_pred_high_recall)
            print("Confusion Matrix (High Recall Threshold):")
            print(cm_high_recall)
        else:
            print("\nNo threshold achieves recall >= 0.8. Consider a different target recall or note the maximum achievable recall.")
            max_recall = recall[0] # Recall at lowest threshold (usually 1.0)
            print(f"Maximum achievable recall: {max_recall:.2f}")


        # Example: Threshold for higher precision (e.g., precision >= 0.9 if possible)
        # Find threshold where precision is at least 0.9, taking the highest such threshold to maximize recall
        high_precision_threshold_idx = np.where(precision[:-1] >= 0.9)[0]
        if len(high_precision_threshold_idx) > 0:
            # We want the highest threshold to get the highest precision, while also checking the corresponding recall.
            # To get higher recall for a given precision, we might need a lower threshold.
            # Let's aim for the lowest threshold that still gives us precision >= 0.9 (this usually maximizes recall).
            high_precision_threshold = thresholds[high_precision_threshold_idx[0]] 
            print(f"\nEvaluating at a High Precision Threshold (e.g., precision >= 0.9): {high_precision_threshold:.4f}")
            y_pred_high_precision = (y_proba >= high_precision_threshold).astype(int)
            print(classification_report(y_test, y_pred_high_precision, target_names=['No Large Quake', 'Large Quake']))
            cm_high_precision = confusion_matrix(y_test, y_pred_high_precision)
            print("Confusion Matrix (High Precision Threshold):")
            print(cm_high_precision)
        else:
            print("\nNo threshold achieves precision >= 0.9. Consider a different target precision or note the maximum achievable precision.")
            max_precision = precision[np.argmax(thresholds)] # Precision at highest threshold (usually 1.0 if any true positives are found)
            print(f"Maximum achievable precision: {max_precision:.2f}")

else:
    print("Skipping ML model training and evaluation as no features were generated or loaded.")