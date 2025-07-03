import joblib
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import pandas as pd
import numpy as np
from datetime import timedelta
import os

# --- Configuration for Prediction (Must match training where applicable) ---
MODEL_PATH = "earthquake_prediction_model.joblib"
FEATURE_COLUMNS_PATH = "model_feature_columns.joblib" # Path to saved feature column names
OPTIMAL_THRESHOLD = 0.3593 # Use the exact optimal F1-score threshold found from your previous run
TARGET_MAGNITUDE = 6.0
PREDICTION_HORIZON_DAYS = 30
WINDOW_SIZE_DAYS = 90
M_MIN_FOR_FEATURES = 2.0
MIN_EVENTS_PER_CELL = 3 # Needs to be consistent with training

# --- Region and Grid Definitions (Must match training exactly) ---
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

# --- Feature Engineering Functions (Must match training exactly) ---
def calculate_b_value(magnitudes, m_min):
    magnitudes = magnitudes[magnitudes >= m_min]
    if len(magnitudes) < 5:
        return np.nan
    mean_mag = np.mean(magnitudes)
    if (mean_mag - m_min) <= 0:
        return np.nan
    b = np.log10(np.e) / (mean_mag - m_min)
    return b

# --- Main Prediction Logic ---
def make_prediction():
    print("--- Starting Prediction Cycle ---")

    # 1. Load the trained model and feature column names
    try:
        model = joblib.load(MODEL_PATH)
        trained_feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        print(f"Model loaded from {MODEL_PATH}")
        print(f"Feature column order loaded from {FEATURE_COLUMNS_PATH}")
    except FileNotFoundError:
        print(f"Error: Model or feature columns file not found. Ensure '{MODEL_PATH}' and '{FEATURE_COLUMNS_PATH}' exist.")
        return
    except Exception as e:
        print(f"Error loading model/feature columns: {e}")
        return

    client = Client("USGS")

    # 2. Define the current prediction window (e.g., last 90 days ending now)
    current_time = UTCDateTime.now()
    window_end_for_features = current_time
    window_start_for_features = window_end_for_features - timedelta(days=WINDOW_SIZE_DAYS)

    print(f"Fetching data for feature window: {window_start_for_features} to {window_end_for_features}")

    try:
        events_in_window = client.get_events(starttime=window_start_for_features,
                                            endtime=window_end_for_features,
                                            minlatitude=LAT_MIN,
                                            maxlatitude=LAT_MAX,
                                            minlongitude=LON_MIN,
                                            maxlongitude=LON_MAX,
                                            minmagnitude=M_MIN_FOR_FEATURES,
                                            orderby="time-asc")

        recent_records = []
        for event in events_in_window:
            origin = event.preferred_origin() or event.origins[0]
            magnitude = event.preferred_magnitude() or event.magnitudes[0]
            recent_records.append({
                "origin_time": origin.time.datetime,
                "latitude": origin.latitude,
                "longitude": origin.longitude,
                "depth_km": origin.depth / 1000,
                "magnitude": magnitude.mag
            })
        df_recent_events = pd.DataFrame(recent_records)
        print(f"Fetched {len(df_recent_events)} recent events.")

    except Exception as e:
        print(f"Error fetching recent data: {e}. Cannot make prediction.")
        return

    if df_recent_events.empty or len(df_recent_events) < MIN_EVENTS_PER_CELL:
        print("Not enough recent events to form a valid feature window. Skipping prediction.")
        return

    # 3. Engineer features for the current window (must match training logic)
    current_features = {}
    # 'window_end' is for info, not a model feature
    current_features_info = {"window_end": window_end_for_features.datetime}

    current_features["seismicity_rate_region"] = len(df_recent_events)
    current_features["b_value_region"] = calculate_b_value(df_recent_events["magnitude"], M_MIN_FOR_FEATURES)
    current_features["mean_magnitude_region"] = df_recent_events["magnitude"].mean() if len(df_recent_events) > 0 else np.nan
    current_features["std_magnitude_region"] = df_recent_events["magnitude"].std() if len(df_recent_events) > 1 else np.nan
    current_features["max_magnitude_region"] = df_recent_events["magnitude"].max() if len(df_recent_events) > 0 else np.nan

    inter_event_times = np.diff(df_recent_events["origin_time"].astype(np.int64)) / 10**9
    current_features["mean_inter_event_time_region"] = np.mean(inter_event_times) if len(inter_event_times) > 0 else np.nan
    current_features["cv_inter_event_time_region"] = (np.std(inter_event_times) / np.mean(inter_event_times)) if np.mean(inter_event_times) > 0 else np.nan

    current_features["mean_depth_region"] = df_recent_events["depth_km"].mean() if len(df_recent_events) > 0 else np.nan
    current_features["std_depth_region"] = df_recent_events["depth_km"].std() if len(df_recent_events) > 1 else np.nan

    # Spatial Features (ensure this matches the training logic exactly)
    for cell in grid_cells:
        df_cell_window = df_recent_events[
            (df_recent_events["latitude"] >= cell['lat_min']) & (df_recent_events["latitude"] < cell['lat_max']) &
            (df_recent_events["longitude"] >= cell['lon_min']) & (df_recent_events["longitude"] < cell['lon_max'])
        ]
        num_events_cell = len(df_cell_window)

        # Initialize to 0 or NaN if no events, then calculate if enough events
        current_features[f"seismicity_rate_{cell['id']}"] = 0 # Default to 0
        current_features[f"b_value_{cell['id']}"] = np.nan
        current_features[f"mean_magnitude_{cell['id']}"] = np.nan

        if num_events_cell >= MIN_EVENTS_PER_CELL:
            current_features[f"seismicity_rate_{cell['id']}"] = num_events_cell
            current_features[f"b_value_{cell['id']}"] = calculate_b_value(df_cell_window["magnitude"], M_MIN_FOR_FEATURES)
            current_features[f"mean_magnitude_{cell['id']}"] = df_cell_window["magnitude"].mean()

    df_current_features = pd.DataFrame([current_features])

    # IMPORTANT: Reindex the DataFrame to match the order of columns the model was trained on
    # Fill any missing columns (e.g., a specific grid cell had no events in current window
    # but had events during training) with 0, consistent with how you handle NaNs for spatial features.
    df_current_features_reindexed = df_current_features.reindex(columns=trained_feature_columns, fill_value=0)

    # Handle any remaining NaNs from regional features (e.g., b_value_region if very few events)
    # Ensure consistency with how NaNs were handled in X before training (e.g., X.dropna() or fillna)
    # Your training script used X.dropna() so here we must ensure no NaNs reach the model.
    if df_current_features_reindexed.isnull().values.any():
        print("Warning: NaN values found in final feature vector after reindexing. Attempting to fill with 0.")
        # This fillna(0) is a pragmatic choice to allow prediction.
        # Ideally, ensure your feature engineering always produces non-NaNs or handle them based on training strategy.
        df_current_features_reindexed = df_current_features_reindexed.fillna(0) # Fill for prediction, assuming 0 is a safe default for NaNs not handled by reindex fill_value.

    if df_current_features_reindexed.empty:
        print("No valid features after processing for prediction. Cannot make prediction.")
        return

    # 4. Make prediction
    prediction_proba = model.predict_proba(df_current_features_reindexed)[0, 1]
    prediction_label = 1 if prediction_proba >= OPTIMAL_THRESHOLD else 0

    # 5. Report prediction
    print(f"\n--- Earthquake Prediction for {current_time.strftime('%Y-%m-%d %H:%M:%S')} ---")
    print(f"Prediction Window: {window_start_for_features.strftime('%Y-%m-%d')} to {window_end_for_features.strftime('%Y-%m-%d')}")
    print(f"Probability of Large Earthquake (M{TARGET_MAGNITUDE}+ in next {PREDICTION_HORIZON_DAYS} days): {prediction_proba:.4f}")

    if prediction_label == 1:
        print(f"Prediction: ALERT! A large earthquake (M{TARGET_MAGNITUDE}+) is predicted in California within the next {PREDICTION_HORIZON_DAYS} days.")
        # --- Placeholder for actual alerting mechanism ---
        # You would integrate email, SMS, or other notification services here.
        # Example: send_email_alert("Earthquake Alert!", f"High probability of M6+ quake: {prediction_proba:.2f}")
    else:
        print(f"Prediction: No large earthquake (M{TARGET_MAGNITUDE}+) predicted within the next {PREDICTION_HORIZON_DAYS} days.")

    print("--- Prediction Cycle Complete ---")

if __name__ == "__main__":
    make_prediction()