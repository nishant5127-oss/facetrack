# AI-Based Missing Person Identification and Tracking System

Streamlit + MediaPipe Face Landmarker + scikit-learn KNN + SQLModel/SQLite + Folium.

## First-time setup
    python -m venv venv
    venv\Scripts\activate
    python -m pip install -r requirements.txt

## Run (two terminals, venv activated in each)
Officer portal (http://localhost:8501):

    $env:OFFICER_PASSWORD="your-password"     # optional, default admin123
    python -m streamlit run Home.py

Public portal (http://localhost:8502):

    $env:DEV_MODE="1"                         # optional: threshold slider + debug distances
    $env:SMTP_HOST="smtp.gmail.com"           # optional: email alerts
    $env:SMTP_USER="sender@gmail.com"
    $env:SMTP_PASS="gmail-app-password"
    python -m streamlit run mobile_app.py --server.port 8502

Other optional setting: MATCH_THRESHOLD (default 0.08).
The face model (face_landmarker.task, ~4 MB) downloads automatically on first use.

## Adding Kaggle Face Recognition Datasets

You can populate the database with face recognition datasets (such as celebrity/person face datasets from Kaggle) using `seed_dataset.py`:

### 1. Import already downloaded Kaggle dataset
If you have a Kaggle dataset folder (with folders for each person):
```powershell
# Import default local dataset (or specify with --data-dir)
python seed_dataset.py

# Import all individuals without limit:
python seed_dataset.py --limit 0

# Import from a custom folder:
python seed_dataset.py --data-dir "C:\path\to\Kaggle\Dataset\Faces" --limit 20
```

### 2. Download directly via Kaggle API
1. Place your `kaggle.json` API token in `~/.kaggle/kaggle.json`.
2. Run:
```powershell
python seed_dataset.py --kaggle-dataset "vasukannan/face-recognition-dataset" --limit 15
```

