from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import pandas as pd
import numpy as np
from datetime import timedelta
import os # Import the os module for file path operations

# --- Configuration for Data Saving/Loading ---
CATALOG_FILE = "california_earthquake_catalog_M2+.csv" # Name of the file to save/load
FORCE_FETCH = False # Set to True to force re-fetching from USGS, even if file exists

# --- 1. Data Acquisition ---
print("--- Step 1: Data Acquisition ---")

df_catalog = pd.DataFrame() # Initialize an empty DataFrame

if os.path.exists(CATALOG_FILE) and not FORCE_FETCH:
    print(f"Loading earthquake catalog from {CATALOG_FILE}...")
    try:
        df_catalog = pd.read_csv(CATALOG_FILE, parse_dates=['origin_time'])
        print(f"Loaded {len(df_catalog)} events from CSV.")
    except Exception as e:
        print(f"Error loading CSV: {e}. Attempting to fetch data from USGS instead.")
        FORCE_FETCH = True # Fallback to fetching if loading fails
else:
    print(f"'{CATALOG_FILE}' not found or FORCE_FETCH is True. Proceeding to fetch from USGS.")


if FORCE_FETCH or not os.path.exists(CATALOG_FILE) or df_catalog.empty:
    # Initialize USGS client
    client = Client("USGS")

    # Define full time range for the entire catalog
    start_time_full_catalog = UTCDateTime("2005-01-01")
    end_time_full_catalog = UTCDateTime.now() # Get current UTC time for up-to-date data

    minlatitude = 32.0
    maxlatitude = 42.0
    minlongitude = -125.0
    maxlongitude = -114.0
    minmagnitude_fetch = 2.0 # Fetching M2.0+ events for microseismicity analysis

    print(f"Fetching full catalog for California from {start_time_full_catalog} to {end_time_full_catalog} (M{minmagnitude_fetch}+)...")

    all_records = []
    current_chunk_start = start_time_full_catalog
    # Adjust chunk_duration_years if you still hit the 20k limit (e.g., to 0.5 for 6 months)
    # A 1-year chunk for M2+ in California might still hit limits sometimes.
    # If you get "exceeds search limit" again, reduce this (e.g., to 0.5 for 6 months)
    chunk_duration_years = 1 

    while current_chunk_start < end_time_full_catalog:
        current_chunk_end = min(current_chunk_start + timedelta(days=chunk_duration_years * 365.25), end_time_full_catalog)
        
        # Ensure we don't request a chunk ending before it starts (can happen with min/max rounding)
        if current_chunk_end <= current_chunk_start:
             current_chunk_start = current_chunk_end + timedelta(days=1) # Advance by a day to avoid infinite loop
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
                depth = origin.depth / 1000 if origin.depth is not None else None # Convert to km
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
            # If an error occurs, it's safer to break the loop to avoid repeated errors
            break
            
        current_chunk_start = current_chunk_end # Move to the next chunk

    df_catalog = pd.DataFrame(all_records)

    # Sort by origin_time is crucial for sliding window
    df_catalog = df_catalog.sort_values("origin_time").reset_index(drop=True)

    print(f"\nTotal events retrieved for full catalog: {len(df_catalog)}")
    print("Head of raw catalog data (first 5 events from full period):")
    print(df_catalog.head())
    print("\n")

    # --- SAVE THE FETCHED DATA ---
    print(f"Saving fetched catalog to {CATALOG_FILE}...")
    df_catalog.to_csv(CATALOG_FILE, index=False)
    print("Catalog saved.\n")
else:
    print("Skipping data fetching. Using loaded catalog.\n")


# --- 2. Feature Engineering Function ---
print("--- Step 2: Feature Engineering (b-value calculation) ---")

def calculate_b_value(magnitudes, m_min):
    """
    Calculates the b-value using the maximum likelihood estimate.
    Requires at least 5 events for a reliable estimate.
    """
    magnitudes = magnitudes[magnitudes >= m_min]
    if len(magnitudes) < 5:
        return np.nan 
    
    mean_mag = np.mean(magnitudes)
    if (mean_mag - m_min) <= 0:
        return np.nan
        
    b = np.log10(np.e) / (mean_mag - m_min)
    return b

print("b-value calculation function defined.\n")

# --- 3. Sliding Window Feature Extraction and Labelling ---
print("--- Step 3: Sliding Window Feature Extraction and Labelling ---")

# Parameters for feature extraction
window_size_days = 90 # 90-day nowcasting window
step_days = 7         # Slide window forward by 7 days
m_min_for_features = 2.0 # Minimum magnitude for events included in feature calculation (e.g., Mc)

# Parameters for LABELLING
target_magnitude = 6.0 # M >= 6.0 is our 'large earthquake'
prediction_horizon_days = 30 # Look 30 days into the future for a target event

# Ensure the catalog has enough data for at least one full window + prediction horizon
if len(df_catalog) == 0:
    print("No events in catalog. Cannot perform feature extraction and labeling.")
else:
    # Use the min and max dates from the actual fetched catalog for loop bounds
    start_date_catalog = df_catalog["origin_time"].min()
    end_date_catalog = df_catalog["origin_time"].max() 

    current_window_start = start_date_catalog
    feature_rows = []

    # Filter the original df_catalog to include only events relevant for labelling (M >= target_magnitude)
    df_large_events = df_catalog[df_catalog["magnitude"] >= target_magnitude].copy()

    # Loop through the catalog with a sliding window
    # The loop needs to continue until the end of the catalog minus the prediction horizon
    # because we need a future window to check for labels.
    while current_window_start + timedelta(days=window_size_days + prediction_horizon_days) <= end_date_catalog:
        current_window_end = current_window_start + timedelta(days=window_size_days)
        
        # Extract events within the current feature calculation window
        df_window = df_catalog[
            (df_catalog["origin_time"] >= current_window_start) & 
            (df_catalog["origin_time"] < current_window_end)
        ]
        
        # Skip window if it doesn't have enough events for reliable feature calculation
        seismicity_rate = len(df_window)
        b_value = calculate_b_value(df_window["magnitude"], m_min_for_features)

        if pd.isna(b_value) or seismicity_rate == 0: 
            current_window_start += timedelta(days=step_days)
            continue
        
        # --- Labelling Logic ---
        future_start_time = current_window_end
        future_end_time = current_window_end + timedelta(days=prediction_horizon_days)
        
        # Check if a large earthquake (M >= target_magnitude) occurs in this future window
        large_quake_in_future = not df_large_events[
            (df_large_events["origin_time"] >= future_start_time) &
            (df_large_events["origin_time"] < future_end_time)
        ].empty
        
        # Append features and the generated label
        feature_rows.append({
            "window_end": current_window_end,
            "seismicity_rate": seismicity_rate,
            "b_value": b_value,
            "target_label": 1 if large_quake_in_future else 0
        })
        
        # Move the window forward by the defined step
        current_window_start += timedelta(days=step_days)

    # Final feature DataFrame
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

# --- 5. Machine Learning Model Training and Evaluation ---
print("\n--- Step 5: Machine Learning Model Training and Evaluation ---")

if 'df_features' in locals() and not df_features.empty:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, auc
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 5.1. Prepare Data for ML
    X = df_features[['seismicity_rate', 'b_value']] # Features
    y = df_features['target_label'] # Labels

    print(f"Features (X) shape: {X.shape}")
    print(f"Labels (y) shape: {y.shape}")

    # Check for NaN values in features (e.g., from b-value calculation with insufficient data)
    # This is important! Models cannot handle NaNs directly.
    if X.isnull().sum().sum() > 0:
        print("\nWARNING: NaN values found in features. These rows will be dropped for training.")
        original_rows = X.shape[0]
        X = X.dropna()
        y = y.loc[X.index] # Ensure labels correspond to cleaned features
        print(f"Dropped {original_rows - X.shape[0]} rows due to NaN values.")
        print(f"New Features (X) shape: {X.shape}")
        print(f"New Labels (y) shape: {y.shape}")
    
    if len(X) == 0:
        print("No valid feature rows after dropping NaNs. Cannot train ML model.")
    else:
        # 5.2. Data Splitting (Stratified)
        # Using a fixed random_state for reproducibility
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        print(f"Train label distribution:\n{y_train.value_counts(normalize=True)}")
        print(f"Test label distribution:\n{y_test.value_counts(normalize=True)}")

        # 5.3. Model Selection and Training (with Class Weights)
        print("\nTraining RandomForestClassifier with 'balanced' class weights...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        print("Model training complete.")

        # 5.4. Evaluation
        print("\n--- Model Evaluation on Test Set ---")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] # Probability of the positive class

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Large Quake', 'Large Quake']))

        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

        # Confusion Matrix for detailed view
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        # Visualize Confusion Matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.show()

        # Precision-Recall Curve (Very important for imbalanced data)
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

        # Feature Importance (for Random Forest)
        print("\n--- Feature Importance ---")
        feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print(feature_importances)
else:
    print("Skipping ML model training as no features were generated or loaded.")