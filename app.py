import streamlit as st
import pandas as pd
import joblib

# Load artefak
@st.cache_resource
def load_models():
    km = joblib.load('kmeans_model.pkl')
    sc = joblib.load('scaler.pkl')
    pc = joblib.load('pca.pkl')
    labels = joblib.load('cluster_labels.pkl')
    profile = pd.read_csv('cluster_profile.csv', index_col=0)
    return km, sc, pc, labels, profile

km_model, scaler, pca, cluster_label_map, cluster_profile = load_models()

# ── Judul ─────────────────────────────────────────────────────
st.title("🛒 Customer Segmentation App")
st.markdown("Masukkan data pelanggan untuk mengetahui segmennya.")

# ── Input Form ────────────────────────────────────────────────
with st.form("input_form"):
    st.subheader("Data Pelanggan")
    col1, col2 = st.columns(2)
    
    with col1:
        year_birth = st.number_input("Tahun Lahir", 1940, 2000, 1975)
        income = st.number_input("Income (USD/tahun)", 0, 200000, 50000)
        recency = st.slider("Recency (hari sejak pembelian terakhir)", 0, 100, 30)
        kidhome = st.selectbox("Jumlah Anak Kecil di Rumah", [0, 1, 2])
        teenhome = st.selectbox("Jumlah Remaja di Rumah", [0, 1, 2])
        num_web_visits = st.slider("Kunjungan Web per Bulan", 0, 20, 5)
    
    with col2:
        mnt_wines = st.number_input("Belanja Wine (USD)", 0, 1500, 200)
        mnt_meat = st.number_input("Belanja Daging (USD)", 0, 1800, 100)
        mnt_fruits = st.number_input("Belanja Buah (USD)", 0, 200, 20)
        mnt_fish = st.number_input("Belanja Ikan (USD)", 0, 300, 30)
        mnt_sweet = st.number_input("Belanja Makanan Manis (USD)", 0, 300, 20)
        mnt_gold = st.number_input("Belanja Produk Premium (USD)", 0, 400, 50)
    
    st.subheader("Riwayat Kampanye")
    cmp_cols = st.columns(5)
    cmp_vals = []
    for i, col in enumerate(cmp_cols):
        cmp_vals.append(col.checkbox(f"Kampanye {i+1}"))
    
    num_web = st.slider("Pembelian via Web", 0, 30, 4)
    num_catalog = st.slider("Pembelian via Katalog", 0, 30, 2)
    num_store = st.slider("Pembelian di Toko", 0, 20, 5)
    num_deals = st.slider("Pembelian Pakai Diskon", 0, 20, 2)
    dt_customer = st.date_input("Tanggal Bergabung")
    
    submitted = st.form_submit_button("🔍 Prediksi Segmen")

# ── Prediksi ─────────────────────────────────────────────────
if submitted:
    from preprocess import preprocess_input  # fungsi dari notebook
    
    input_data = {
        'Year_Birth': year_birth, 'Income': income, 'Recency': recency,
        'Kidhome': kidhome, 'Teenhome': teenhome,
        'MntWines': mnt_wines, 'MntFruits': mnt_fruits,
        'MntMeatProducts': mnt_meat, 'MntFishProducts': mnt_fish,
        'MntSweetProducts': mnt_sweet, 'MntGoldProds': mnt_gold,
        'NumDealsPurchases': num_deals, 'NumWebPurchases': num_web,
        'NumCatalogPurchases': num_catalog, 'NumStorePurchases': num_store,
        'NumWebVisitsMonth': num_web_visits,
        'AcceptedCmp1': int(cmp_vals[0]), 'AcceptedCmp2': int(cmp_vals[1]),
        'AcceptedCmp3': int(cmp_vals[2]), 'AcceptedCmp4': int(cmp_vals[3]),
        'AcceptedCmp5': int(cmp_vals[4]),
        'Dt_Customer': dt_customer.strftime('%d-%m-%Y'),
    }
    
    X_new = preprocess_input(input_data)
    X_scaled = scaler.transform(X_new)
    cluster = km_model.predict(X_scaled)[0]
    label = cluster_label_map[cluster]
    
    st.success(f"### Segmen: {label} (Kluster {cluster})")
    st.dataframe(cluster_profile.loc[[cluster]])