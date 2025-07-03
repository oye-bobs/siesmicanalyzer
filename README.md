
#  California Earthquake Nowcasting Project

## Predicting Tomorrow's Tremors: A Machine Learning Approach to Earthquake Nowcasting in California

This project develops and deploys a robust machine learning model to nowcast the likelihood of significant earthquakes (Magnitude 6.0+) in California within a 30-day horizon. Leveraging historical seismic data and advanced feature engineering, the system provides automated, data-driven insights into current seismic risk.

---

##  Features

- **Comprehensive Data Acquisition**: Fetches M2+ earthquake data from the USGS FDSN client for California (1990-Present) with robust chunking to handle API limits.
- **Advanced Feature Engineering**: Calculates both regional and fine-grained spatial seismic features (seismicity rate, b-value, mean magnitude, etc.) from sliding time windows.
- **XGBoost Classification**: Utilizes a highly optimized XGBoost model for predicting large earthquake probabilities.
- **Class Imbalance Handling**: Employs SMOTE to balance the highly imbalanced earthquake dataset during training.
- **Optimal Thresholding**: Identifies and applies an optimal probability threshold for binary predictions, balancing precision and recall.
- **Automated Prediction Pipeline**: A dedicated script for real-time (or near-real-time) predictions, loading the trained model and processing the latest seismic data.
- **Automated Scheduling**: Configured for automated execution via system schedulers (cron/Task Scheduler).
- **Robust Alerting**: Includes logging to file and email notifications for predicted large earthquake events.

---

##  Technologies Used

- Python 3.x
- ObsPy: For seismic data acquisition.
- Pandas & NumPy: For data manipulation and numerical operations.
- Scikit-learn: For machine learning utilities (train/test split, GridSearchCV, metrics).
- Imbalanced-learn (imblearn): For handling class imbalance (SMOTE).
- XGBoost: The core machine learning model.
- joblib: For model serialization.
- Matplotlib & Seaborn: For data visualization.
- smtplib & email.mime.text: For email notifications.

---

##  Getting Started

### Prerequisites

- Python 3.x installed.
- Git installed.
- A GitHub account (for pushing your code).
- VS Code (recommended IDE).

### Installation

Clone the repository:

```bash
git clone https://github.com/oye-bobs/siesmicanalyzer
cd siesmicanalyzer
```

Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On macOS/Linux:
source ./.venv/bin/activate
```

Install dependencies:

```bash
pip install obspy pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn
```

---

## Configuration

Before running, you might need to adjust some configurations:

### nowcasting_pipeline.py:
- `CATALOG_FILE`: Name of the CSV file to store the earthquake catalog.
- `FORCE_FETCH`: Set to True for the first run to download the full catalog.
- `start_time_full_catalog`: Defines the start date for data fetching (currently "1990-01-01").
- `minlatitude_region`, `maxlatitude_region`, `minlongitude_region`, `maxlongitude_region`: Define the geographical area of interest.
- `target_magnitude`: The magnitude threshold for "large earthquakes" (currently 6.0).
- `prediction_horizon_days`: The future window to predict large earthquakes (currently 30 days).

### predict_earthquake.py:
- `OPTIMAL_THRESHOLD`: Must be set to the exact optimal F1-score threshold found during training.
- Email Configuration: Update `EMAIL_SENDER`, `EMAIL_PASSWORD`, and `EMAIL_RECEIVER` for alerts.

---

##  How to Run

### 1. Train the Model
```bash
python nowcasting_pipeline9.py
```

### 2. Make Predictions
```bash
python predict_earthquake.py
```

### 3. Automate Predictions

#### On Linux/macOS (using cron):
```bash
crontab -e
```

Add a line:

```bash
0 3 * * * /path/to/your/venv/bin/python /path/to/your/project/predict_earthquake.py >> /path/to/your/project/prediction_log.txt 2>&1
```

#### On Windows (using Task Scheduler):
- Search for "Task Scheduler"
- Create a "Basic Task"
- Set the schedule and point to your Python interpreter and script.

---

##  Model Performance Highlights

- True Positives: 18
- False Positives: 0
- False Negatives: 4
- **Precision (Large Quake)**: 1.00
- **Recall (Large Quake)**: 0.82
- **F1-score (Large Quake)**: 0.90
- **ROC-AUC Score**: 0.9982
- **Precision-Recall AUC**: 0.94

---

##  Future Work

- Advanced Feature Engineering (e.g., stress accumulation metrics).
- Time Series Cross-Validation (e.g., walk-forward validation).
- Ensemble Methods (LightGBM, CatBoost).
- Web Dashboard for visualization and interaction.

---

##  Contributing

Feel free to fork this repository, open issues, or submit pull requests.

---

##  License

MIT License.

---

##  Contact

Your Name - adeoyeayan@gmail.com
