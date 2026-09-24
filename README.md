# Customer Segmentation App

A web application that predicts which customer segment a person belongs to based on their demographics, spending behavior, and purchase history. Built with Streamlit and trained using KMeans clustering following the CRISP-DM methodology.

## Features

- Input customer attributes via an interactive form
- Instant cluster prediction using a pre-trained KMeans model
- Displays the matching cluster profile for business insights

## Tech Stack

| Layer | Technology |
|---|---|
| Web UI | Streamlit |
| Machine Learning | scikit-learn (KMeans, PCA, StandardScaler) |
| Data Processing | pandas, numpy |
| Model Serialization | joblib |

## ML Pipeline

The model was developed in `customer_segmentation_crisp_dm.ipynb` following the CRISP-DM framework:

1. **Business Understanding** — Identify segments for targeted marketing
2. **Data Understanding** — EDA on the Customer Personality Analysis dataset (2,240 rows)
3. **Data Preparation** — Imputation, outlier removal, feature engineering, scaling, PCA
4. **Modeling** — KMeans (k=4 selected via Elbow + Silhouette analysis)
5. **Evaluation** — Silhouette Score, Davies-Bouldin Score, Calinski-Harabasz Score
6. **Deployment** — Artifacts exported as `.pkl` files, served via Streamlit

## Getting Started

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd <repo-folder>
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows CMD
.\venv\Scripts\activate.bat
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

## Project Structure

```
.
├── app.py                                # Streamlit web application
├── preprocess.py                         # Feature engineering logic
├── customer_segmentation_crisp_dm.ipynb  # Full ML pipeline notebook
├── requirements.txt                      # Python dependencies
├── cluster_profile.csv                   # Cluster mean profiles
├── kmeans_model.pkl                      # Trained KMeans model
├── scaler.pkl                            # Fitted StandardScaler
├── pca.pkl                               # Fitted PCA transformer
└── cluster_labels.pkl                    # Cluster ID to label mapping
```

## Dataset

[Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) — Kaggle
