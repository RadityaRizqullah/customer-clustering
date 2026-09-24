# Customer Segmentation App

A web application that predicts which customer segment a person belongs to based on their demographics, spending behavior, and purchase history. Built with Streamlit and trained using KMeans clustering following the CRISP-DM methodology.

## Features

- Input customer attributes via an interactive form
- Instant cluster prediction using a pre-trained KMeans model (k=3)
- Displays the matching cluster profile for business insights

## Tech Stack

| Layer | Technology |
|---|---|
| Web UI | Streamlit |
| Machine Learning | scikit-learn (KMeans, PCA, StandardScaler) |
| Data Processing | pandas, numpy |
| Additional | kneed (automatic elbow detection), scipy |
| Model Serialization | joblib |

## ML Pipeline

The model was developed in `customer_segmentation_crisp_dm.ipynb` following the CRISP-DM framework:

1. **Business Understanding** — Identify segments to enable targeted marketing strategies
2. **Data Understanding** — EDA on the Customer Personality Analysis dataset (2,240 rows × 29 columns): distribution analysis, correlation heatmap, boxplots, scatter plots
3. **Data Preparation**
   - Missing value imputation (median for `Income`)
   - Noise removal in `Marital_Status` (`Alone`, `Absurd`, `YOLO` → `Single`)
   - Outlier removal (Age > 90 years, Income > 99th percentile)
   - Feature engineering (10 derived features — see table below)
   - StandardScaler normalization
   - PCA (2 components, for visualization only)
4. **Modeling** — KMeans (k=2–10 evaluated via Elbow Method + Silhouette Score; **k=3** selected) and Agglomerative Hierarchical Clustering (Ward linkage, for comparison)
5. **Evaluation** — Silhouette Score, Davies-Bouldin Score, Calinski-Harabasz Score, ARI, Chi-Square validation, stability analysis
6. **Deployment** — Artifacts exported as `.pkl` files, served via Streamlit

### Feature Engineering

| Feature | Derivation |
|---|---|
| `Age` | `2014 - Year_Birth` (static reference date: 2014-12-31) |
| `TotalSpending` | Sum of all 6 `Mnt*` product columns |
| `TotalPurchases` | Sum of web + catalog + store + deals purchases |
| `Dependents` | `Kidhome + Teenhome` |
| `TotalCampaignsAccepted` | Sum of `AcceptedCmp1–5` |
| `SpendingRatio` | `TotalSpending / (Income + 1)` |
| `DaysSinceEnrollment` | Days from `Dt_Customer` to 2014-12-31 |

### Cluster Results (k=3)

| Cluster | Label | Description | Size |
|---|---|---|---|
| 0 | At-Risk / Dormant | Low spending, long since last purchase | ~56.7% |
| 1 | Premium Loyal | High income & spending, few dependents | ~43.2% |
| 2 | Mid-Tier Family | Moderate spending, has dependents | minority |

**Key metrics:** Silhouette Score = 0.2731 · Davies-Bouldin = 1.01 · Calinski-Harabasz = 602.29

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

**4. (Notebook only) Download the dataset**

Place `marketing_campaign.csv` (tab-separated) in the project root before running the notebook.
Source: [Customer Personality Analysis — Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)

**5. Run the app**
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

## Project Structure

```
.
├── app.py                                # Streamlit web application
├── preprocess.py                         # Feature engineering logic
├── customer_segmentation_crisp_dm.ipynb  # Full ML pipeline notebook (CRISP-DM)
├── requirements.txt                      # Python dependencies
├── cluster_profile.csv                   # Cluster mean profiles
├── kmeans_model.pkl                      # Trained KMeans model (k=3)
├── scaler.pkl                            # Fitted StandardScaler
├── pca.pkl                               # Fitted PCA transformer (2 components)
└── cluster_labels.pkl                    # Cluster ID to label mapping
```
