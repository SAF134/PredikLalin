import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, r2_score
import datetime

# --- Configuration ---
st.set_page_config(page_title="PredikLalin.com", layout="wide")

# --- 1. Data Loading & Caching ---
@st.cache_data
def load_data():
    """Loads and cleans the initial dataset."""
    df = pd.read_csv('essen.csv')
    
    # Filter error rows
    if 'error' in df.columns:
        df = df[df['error'] == 0.0]
        df.drop(columns=['error'], inplace=True)
        
    # Convert 'day' to datetime
    df['day'] = pd.to_datetime(df['day'])
    
    return df

# --- 2. Preprocessing & Feature Engineering ---
@st.cache_data
def preprocess_data(df):
    """Aggregates data and creates features."""
    # Aggregation per interval
    df_agg = df.groupby(['day', 'interval'])[['flow', 'occ', 'speed']].mean().reset_index()
    
    # Time Features
    df_agg['year'] = df_agg['day'].dt.year
    df_agg['month'] = df_agg['day'].dt.month
    df_agg['day_of_month'] = df_agg['day'].dt.day
    df_agg['day_of_week'] = df_agg['day'].dt.dayofweek
    df_agg['hour'] = (df_agg['interval'] // 3600).astype(int)
    df_agg['minute'] = ((df_agg['interval'] % 3600) // 60).astype(int)
    
    # Target Variable: Status (Legacy Crisp Logic for initial labeling/training)
    def get_status(speed):
        if speed <= 30:
            return 'Macet'
        elif speed <= 55:
            return 'Padat'
        else:
            return 'Lancar'
    
    df_agg['status'] = df_agg['speed'].apply(get_status)
    
    return df_agg

# --- Fuzzy Logic System ---
def determine_status_fuzzy(speed, flow, occ):
    """
    Menentukan status lalu lintas menggunakan logika fuzzy dengan batas user-defined.
    
    Fuzzy Sets:
    - Speed: Pelan (<30), Normal (25-55), Kencang (>50)
    - Flow: Low (<300), Medium (200-700), High (>600)
    - Occ: Low (<0.02), Medium (0.01-0.04), High (>0.03)
    """
    
    # --- Fuzzification (Membership Functions) ---
    
    # 1. Speed Membership
    # Pelan (Macet): 1 at 0, 0 at 30
    speed_pelan = np.clip((30 - speed) / (30 - 0), 0, 1) if speed < 30 else 0
    if speed <= 0: speed_pelan = 1
    
    # Normal (Padat): 0 at 25, 1 at 40, 0 at 55 (Triangle)
    if 25 < speed <= 40:
        speed_normal = (speed - 25) / (40 - 25)
    elif 40 < speed < 55:
        speed_normal = (55 - speed) / (55 - 40)
    else:
        speed_normal = 0
        
    # Kencang (Lancar): 0 at 50, 1 at 100+
    speed_kencang = np.clip((speed - 50) / (100 - 50), 0, 1)
    if speed >= 100: speed_kencang = 1

    # 2. Occupancy Membership
    # Low: 1 at 0, 0 at 0.02
    occ_low = np.clip((0.02 - occ) / 0.02, 0, 1)
    
    # Medium: 0 at 0.01, 1 at 0.025, 0 at 0.04 (Triangle)
    if 0.01 < occ <= 0.025:
        occ_med = (occ - 0.01) / (0.025 - 0.01)
    elif 0.025 < occ < 0.04:
        occ_med = (0.04 - occ) / (0.04 - 0.025)
    else:
        occ_med = 0
        
    # High: 0 at 0.03, 1 at 0.05 (Ramp up)
    occ_high = np.clip((occ - 0.03) / (0.05 - 0.03), 0, 1)

    # 3. Flow Membership (Used for boosting confidence)
    # Low: 1 at 0, 0 at 300
    flow_low = np.clip((300 - flow) / 300, 0, 1)
    
    # Medium: 0 at 200, 1 at 450, 0 at 700
    if 200 < flow <= 450:
        flow_med = (flow - 200) / (450 - 200)
    elif 450 < flow < 700:
        flow_med = (700 - flow) / (700 - 450)
    else:
        flow_med = 0
        
    # High: 0 at 600, 1 at 1000
    flow_high = np.clip((flow - 600) / (1000 - 600), 0, 1)
    if flow >= 1000: flow_high = 1
    
    # --- Rule Evaluation (Inference) ---
    # Logika Dasar: Speed adalah penentu utama.
    score_pelan = speed_pelan
    score_normal = speed_normal
    score_kencang = speed_kencang
    
    # Rule Adjustment:
    # Jika Speed Normal TAPI Occ High ATAU Flow High -> Condong ke Pelan
    if speed_normal > 0:
        if occ_high > 0.5 or flow_high > 0.5:
             score_pelan = max(score_pelan, 0.6) # Boost Pelan
             score_normal = score_normal * 0.5 # Suppress Normal
        elif occ_low > 0.5 and flow_low > 0.5:
             # Biarkan tetap Normal
             pass
             
    # Jika Speed Kencang TAPI Occ High (Aneh) -> Turunkan confidence
    if speed_kencang > 0 and occ_high > 0.8:
         score_kencang = score_kencang * 0.7
         score_normal = max(score_normal, 0.4)

    # --- Defuzzification (Max Aggregation) ---
    # Mapping Speed Parameters to Traffic Status:
    # Pelan -> Macet
    # Normal -> Padat
    # Kencang -> Lancar
    scores = {'Macet': score_pelan, 'Padat': score_normal, 'Lancar': score_kencang}
    final_status = max(scores, key=scores.get)
    
    return final_status, scores


# --- 3. Model Training ---
@st.cache_resource
def train_models(df):
    """Trains RF and XGBoost for Status and Regressors for Speed/Flow/Occ."""
    
    # Features
    features = ['month', 'day_of_week', 'hour', 'minute']
    X = df[features]
    
    # Targets
    y_status = df['status']
    y_speed = df['speed']
    y_flow = df['flow']
    y_occ = df['occ']
    
    # Label Encoder for XGBoost
    le = LabelEncoder()
    y_status_encoded = le.fit_transform(y_status)
    
    # --- Random Forest ---
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X, y_status)
    
    rf_speed = RandomForestRegressor(n_estimators=50, random_state=42)
    rf_speed.fit(X, y_speed)
    
    rf_flow = RandomForestRegressor(n_estimators=50, random_state=42)
    rf_flow.fit(X, y_flow)
    
    rf_occ = RandomForestRegressor(n_estimators=50, random_state=42)
    rf_occ.fit(X, y_occ)

    # --- XGBoost ---
    # XGBoost classifier generally needs encoded labels 0,1,2..., not strings
    xgb_clf = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
    xgb_clf.fit(X, y_status_encoded)
    
    xgb_speed = XGBRegressor(n_estimators=100, random_state=42)
    xgb_speed.fit(X, y_speed)
    
    xgb_flow = XGBRegressor(n_estimators=100, random_state=42)
    xgb_flow.fit(X, y_flow)
    
    xgb_occ = XGBRegressor(n_estimators=100, random_state=42)
    xgb_occ.fit(X, y_occ)
    
    return {
        'rf': (rf_clf, rf_speed, rf_flow, rf_occ),
        'xgb': (xgb_clf, xgb_speed, xgb_flow, xgb_occ),
        'le': le,
        'features': features
    }

# --- Pages Functions ---

def show_dataset_overview(raw_df):
    st.header("📋 Dataset Kota Essen")
    st.write("### Menampilkan Semua Data (essen.csv)")
    st.dataframe(raw_df)
    st.write(f"Total Baris: {raw_df.shape[0]}, Total Kolom: {raw_df.shape[1]}")

def show_data_loading(raw_df):
    st.header("📂 Pemuatan & Pembersihan Data")
    st.write("### Review Data Asli (Head)")
    st.write(raw_df.head())
    st.write(f"Dimensi Data: {raw_df.shape}")
    st.info("Data telah dimuat dan baris dengan 'error' = 1.0 telah dihapus.")

def show_eda(df):
    st.header("📊 Analisis Data Eksplorasi (EDA)")
    
    st.write("### Distribusi Status Lalu Lintas")
    # Custom order and colors
    order = ['Macet', 'Padat', 'Lancar']
    colors = ['red', 'orange', 'green']
    
    # Get counts ensuring all categories are present (even if 0)
    status_counts = df['status'].value_counts().reindex(order, fill_value=0)
    
    fig_status, ax_status = plt.subplots(figsize=(8, 5))
    ax_status.bar(status_counts.index, status_counts.values, color=colors)
    ax_status.set_title('Distribusi Status Lalu Lintas (Macet, Padat, Lancar)')
    ax_status.set_xlabel('Status')
    ax_status.set_ylabel('Jumlah Data')
    ax_status.grid(axis='y', alpha=0.3)
    
    ax_status.grid(axis='y', alpha=0.3)
    
    # Add text labels on bars
    for i, v in enumerate(status_counts.values):
        ax_status.text(i, v + (v*0.01), str(v), ha='center', fontweight='bold')
        
    st.pyplot(fig_status)
    st.info("""
    **Kriteria Penentuan Status (Logic):**
    - 🔴 **Macet**: Kecepatan ≤ 30 km/h
    - 🟠 **Padat**: 30 km/h < Kecepatan ≤ 55 km/h
    - 🟢 **Lancar**: Kecepatan > 55 km/h
    """)
    
    st.write("### Tren Harian (Weekday vs Weekend)")
    
    # Reuse the visualization logic for Weekday vs Weekend
    df_viz = df.copy()
    df_viz['day_type'] = df_viz['day_of_week'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
    
    # Aggregate for plotting
    plot_data = df_viz.groupby(['hour', 'day_type'])[['speed', 'flow', 'occ']].mean().reset_index()
    
    # Add 'All Days'
    all_days = df_viz.groupby('hour')[['speed', 'flow', 'occ']].mean().reset_index()
    all_days['day_type'] = 'All Days'
    plot_data = pd.concat([plot_data, all_days])

    # Speed Plot
    fig_speed, ax_speed = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='hour', y='speed', hue='day_type', data=plot_data, style='day_type', markers=True, ax=ax_speed)
    ax_speed.set_title('Rata-rata Kecepatan per Jam')
    ax_speed.set_xticks(range(24))
    ax_speed.grid(True)
    st.pyplot(fig_speed)
    
    # Flow Plot
    fig_flow, ax_flow = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='hour', y='flow', hue='day_type', data=plot_data, style='day_type', markers=True, ax=ax_flow)
    ax_flow.set_title('Rata-rata Flow per Jam')
    ax_flow.set_xticks(range(24))
    ax_flow.grid(True)
    st.pyplot(fig_flow)

    # Occupancy Plot
    fig_occ, ax_occ = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='hour', y='occ', hue='day_type', data=plot_data, style='day_type', markers=True, ax=ax_occ)
    ax_occ.set_title('Rata-rata Occupancy per Jam')
    ax_occ.set_xticks(range(24))
    ax_occ.grid(True)
    st.pyplot(fig_occ)

    st.write("### Korelasi Antar Fitur (Heatmap)")
    # Select numeric columns for correlation
    corr_cols = ['speed', 'flow', 'occ', 'hour', 'day_of_week', 'month']
    corr_matrix = df[corr_cols].corr()
    
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
    ax_corr.set_title('Correlation Heatmap')
    st.pyplot(fig_corr)

def show_feature_engineering(df):
    st.header("🛠️ Rekayasa Fitur")
    st.write("Fitur berikut telah diekstrak dari kolom 'day' dan 'interval':")
    st.write("- **year, month, day_of_month, day_of_week**: Dari kolom Tanggal.")
    st.write("- **hour, minute**: Dari kolom Interval.")
    st.write("- **status**: Target klasifikasi berdasarkan kecepatan.")
    
    st.dataframe(df[['day', 'interval', 'speed', 'year', 'month', 'day_of_week', 'hour', 'minute', 'status']])

def show_Pemodelan(df, models_dict):
    st.header("🤖 Pemodelan")
    features = models_dict['features']
    le = models_dict['le']
    
    # Split Data
    X = df[features]
    y_status = df['status']
    y_speed = df['speed']
    y_flow = df['flow']
    y_occ = df['occ']
    
    # Split
    X_train, X_test, y_status_train, y_status_test = train_test_split(X, y_status, test_size=0.2, random_state=42)
    _, _, y_speed_train, y_speed_test = train_test_split(X, y_speed, test_size=0.2, random_state=42)
    _, _, y_flow_train, y_flow_test = train_test_split(X, y_flow, test_size=0.2, random_state=42)
    _, _, y_occ_train, y_occ_test = train_test_split(X, y_occ, test_size=0.2, random_state=42)
    
    st.write("### Perbandingan Performa Model (Classification)")
    
    model_choice = st.radio("Pilih Model untuk Evaluasi Detail:", ["Random Forest", "XGBoost"])
    
    if model_choice == "Random Forest":
        clf, r_speed, r_flow, r_occ = models_dict['rf']
        y_pred = clf.predict(X_test)
    else: # XGBoost
        clf, r_speed, r_flow, r_occ = models_dict['xgb']
        y_pred_encoded = clf.predict(X_test)
        y_pred = le.inverse_transform(y_pred_encoded)
    
    st.write(f"#### Classification Report: {model_choice}")
    report = classification_report(y_status_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.write(f"### Evaluasi Regresi (Speed, Flow, Occupancy) - {model_choice}")
    
    # Predict Regression
    pred_speed = r_speed.predict(X_test)
    pred_flow = r_flow.predict(X_test)
    pred_occ = r_occ.predict(X_test)
    
    # Calculate Metrics
    metrics_data = {
        'Metric': ['MAE (Mean Absolute Error)', 'MSE (Mean Squared Error)', 'R2 Score'],
        'Speed': [
            mean_absolute_error(y_speed_test, pred_speed),
            mean_squared_error(y_speed_test, pred_speed),
            r2_score(y_speed_test, pred_speed)
        ],
        'Flow': [
            mean_absolute_error(y_flow_test, pred_flow),
            mean_squared_error(y_flow_test, pred_flow),
            r2_score(y_flow_test, pred_flow)
        ],
        'Occupancy': [
            mean_absolute_error(y_occ_test, pred_occ),
            mean_squared_error(y_occ_test, pred_occ),
            r2_score(y_occ_test, pred_occ)
        ]
    }
    
    st.dataframe(pd.DataFrame(metrics_data))

def show_prediction(models_dict):
    st.header("🎯 Prediksi Lalu Lintas")
    st.write("Sistem ini menggunakan **Hybrid Prediction**: Regresi (untuk nilai) + Fuzzy Logic (untuk status).")
    
    features = models_dict['features']
    le = models_dict['le']

    # Model Selection
    model_type = st.selectbox("Pilih Model AI (Regresi):", ["Random Forest", "XGBoost"])
    
    if model_type == "Random Forest":
        clf, reg_speed, reg_flow, reg_occ = models_dict['rf']
    else: # XGBoost
        clf, reg_speed, reg_flow, reg_occ = models_dict['xgb']

    col1, col2, col3 = st.columns(3)
    with col1:
        input_date = st.date_input("Pilih Tanggal", value=datetime.date(2017, 3, 27))
    with col2:
        input_hour = st.number_input("Jam (0-23)", min_value=0, max_value=23, value=8, step=1)
    with col3:
        input_minute = st.number_input("Menit (0-59)", min_value=0, max_value=59, value=0, step=1)
    
    input_dt = datetime.datetime.combine(input_date, datetime.time(input_hour, input_minute))
    
    if st.button("Mulai Prediksi"):
        # Create input features
        input_features = pd.DataFrame([{
            'month': input_dt.month,
            'day_of_week': input_dt.weekday(),
            'hour': input_dt.hour,
            'minute': (input_dt.minute // 5) * 5
        }])
        
        # 1. Predict Numeric Values
        speed_pred = reg_speed.predict(input_features)[0]
        flow_pred = reg_flow.predict(input_features)[0]
        occ_pred = reg_occ.predict(input_features)[0]
        
        # 2. Determine Status using Fuzzy Logic
        status_fuzzy, fuzzy_scores = determine_status_fuzzy(speed_pred, flow_pred, occ_pred)

        # 3. Predict Status using Crisp Classifier (For Comparison)
        status_crisp_val = clf.predict(input_features)
        if model_type == "XGBoost":
             status_crisp = le.inverse_transform(status_crisp_val)[0]
        else:
             status_crisp = status_crisp_val[0]
        
        
        st.write(f"### Hasil Prediksi AI ({model_type})")
        
        # Metrics Display
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Status (Prediksi)", status_fuzzy)
        c2.metric("Speed (Prediksi)", f"{speed_pred:.2f} km/h")
        c3.metric("Flow (Prediksi)", f"{flow_pred:.2f}")
        c4.metric("Occupancy (Prediksi)", f"{occ_pred:.2f}")
        
        # Status Alert based on Fuzzy
        if status_fuzzy == 'Pelan' or status_fuzzy == 'Macet':
            st.error("Lalu lintas Macet")
        elif status_fuzzy == 'Normal' or status_fuzzy == 'Padat':
            st.warning("Lalu lintas Padat.")
        else:
            st.success("Lalu lintas Lancar.")
            
        # Analysis Expander
        with st.expander("Lihat Detail 'Fuzzy Logic' vs 'Crisp Classifier'"):
            st.markdown("""
            ### 🧐 Apa Bedanya?
            
            **1. Fuzzy Logic (Logika Samar)**
            *   **Cara Kerja:** Meniru cara berpikir manusia. Tidak ada batas tegas "Ya" atau "Tidak".
            *   **Contoh:** Kecepatan 31 km/h mungkin dianggap "80% Macet" dan "20% Padat". 
            *   **Kelebihan:** Lebih halus (smooth) dan bisa menangani ketidakpastian. Sangat baik untuk kasus perbatasan/ambigu.
            
            **2. Crisp Classifier (Klasifikasi Tegas - Random Forest/XGBoost)**
            *   **Cara Kerja:** Belajar pola dari data masa lalu secara matematis kaku.
            *   **Contoh:** Jika data latih bilang ≤ 30 km/h itu Macet, maka 30.0001 km/h langsung dianggap Padat (beda kelas total).
            *   **Kelebihan:** Sangat akurat jika pola datanya konsisten dan jelas.
            """)
            st.divider()
            
            st.write(f"**Keputusan Fuzzy Logic:** {status_fuzzy}")
            st.write("Skor Probabilitas Fuzzy:")
            st.json(fuzzy_scores)
            
            st.divider()
            st.write(f"**Keputusan Classifier Biasa (Crisp):** {status_crisp}")
            if status_fuzzy != status_crisp:
                st.info("💡 Menarik! Logika Fuzzy memberikan hasil berbeda karena mempertimbangkan nuansa data (seperti kombinasi Speed dan Kepadatan), bukan hanya batas angka kaku.")

        # --- Visualisasi Fuzzy Membership ---
        st.write("### 📈 Visualisasi Membership Function Fuzzy Logic")
        st.write("Grafic ini menunjukkan bagaimana kecepatan dan occupancy dipetakan ke dalam derajat keanggotaan (0-1).")
        
        # Helper func for plotting triangles
        def plot_mf_triangle(ax, x_range, peak, low, high, label, color):
            y_vals = []
            for x in x_range:
                if x <= low: val = 0
                elif low < x <= peak: val = (x - low) / (peak - low)
                elif peak < x < high: val = (high - x) / (high - peak)
                else: val = 0
                y_vals.append(val)
            ax.plot(x_range, y_vals, label=label, color=color, linewidth=2)
            
        # Helper func for Trapezoid Left (Decreasing)
        def plot_mf_trap_left(ax, x_range, zero_at, label, color):
            y_vals = np.clip((zero_at - x_range) / zero_at, 0, 1) # Simple linear ramp down
            # Correct trapezoid logic based on request "0-30 Macet" (1 at 0, 0 at 30)
            # Logic: 0 is 1, 30 is 0.
            y_vals = np.array([np.clip((zero_at - x) / zero_at if x < zero_at else 0, 0, 1) for x in x_range])
            ax.plot(x_range, y_vals, label=label, color=color, linewidth=2)

        # Helper func for Trapezoid Right (Increasing)
        def plot_mf_trap_right(ax, x_range, start_at, max_val, label, color):
             # Logic: start_at is 0, max_val is 1
            y_vals = np.array([np.clip((x - start_at) / (max_val - start_at) if x > start_at else 0, 0, 1) for x in x_range])
            ax.plot(x_range, y_vals, label=label, color=color, linewidth=2)

        # 1. Visualization for Speed
        x_speed = np.linspace(0, 100, 200)
        
        fig_speed, ax_speed = plt.subplots(figsize=(10, 4))
        # Pelan: 0-30
        plot_mf_trap_left(ax_speed, x_speed, 30, 'Pelan', 'red')
        # Normal: 25-55, peak 40
        plot_mf_triangle(ax_speed, x_speed, 40, 25, 55, 'Normal', 'orange')
        # Kencang: 50-100
        plot_mf_trap_right(ax_speed, x_speed, 50, 100, 'Kencang', 'green')
        
        ax_speed.axvline(x=speed_pred, color='blue', linestyle='--', label=f'Prediksi: {speed_pred:.1f}')
        ax_speed.set_title('Membership Function: Speed (Kecepatan)')
        ax_speed.set_xlabel('Speed (km/h)')
        ax_speed.legend(loc='center right')
        ax_speed.grid(True, alpha=0.3)
        st.pyplot(fig_speed)

        # 2. Visualization for Occupancy
        x_occ = np.linspace(0, 0.05, 200) # Zoom in to 0-0.05 range
        if occ_pred > 0.05: x_occ = np.linspace(0, max(occ_pred * 1.2, 0.1), 200)
        
        fig_occ, ax_occ = plt.subplots(figsize=(10, 4))
        # Low: 0-0.02
        plot_mf_trap_left(ax_occ, x_occ, 0.02, 'Low', 'green')
        # Med: 0.01-0.04, peak 0.025
        plot_mf_triangle(ax_occ, x_occ, 0.025, 0.01, 0.04, 'Medium', 'orange')
        # High: 0.03-0.05 (Revised)
        plot_mf_trap_right(ax_occ, x_occ, 0.03, 0.05, 'High', 'red')
        
        ax_occ.axvline(x=occ_pred, color='blue', linestyle='--', label=f'Prediksi: {occ_pred:.4f}')
        ax_occ.set_title('Membership Function: Occupancy')
        ax_occ.set_xlabel('Occupancy (0-1)')
        ax_occ.legend(loc='center right')
        ax_occ.grid(True, alpha=0.3)
        st.pyplot(fig_occ)

        # 3. Visualization for Flow
        x_flow = np.linspace(0, 1200, 200) # 0-1200 cover range
        
        fig_flow, ax_flow = plt.subplots(figsize=(10, 4))
        # Low: 0-300
        plot_mf_trap_left(ax_flow, x_flow, 300, 'Rendah', 'green')
        # Med: 200-700, peak 450
        plot_mf_triangle(ax_flow, x_flow, 450, 200, 700, 'Sedang', 'orange')
        # High: 600-1000
        plot_mf_trap_right(ax_flow, x_flow, 600, 1000, 'Tinggi', 'red')
        
        ax_flow.axvline(x=flow_pred, color='grey', linestyle='--', label=f'Prediksi: {flow_pred:.1f}')
        ax_flow.set_title('Membership Function: Flow (Jumlah Kendaraan)')
        ax_flow.set_xlabel('Flow (kendaraan/jam)')
        ax_flow.legend(loc='center right')
        ax_flow.grid(True, alpha=0.3)
        st.pyplot(fig_flow)

        # 4. Visualization for Decision (Defuzzification)
        st.write("### 🧠 Visualisasi Keputusan (Defuzzifikasi)")
        st.write("Grafic ini menunjukkan bagaimana input digabungkan untuk menghasilkan keputusan akhir.")
        
        # Output Universe: 0-100 (Score Kemacetan/Kelancaran)
        x_out = np.linspace(0, 100, 200)
        
        # Output Sets Definitions (Standardized 0-100)
        # Macet (Pelan): 0-40 (Trapezoid Left)
        y_out_macet = np.clip((40 - x_out) / 40, 0, 1)
        if hasattr(y_out_macet, "__len__"): # Handle numpy array check logic safely if needed
             y_out_macet = np.array([np.clip((40 - x)/40 if x < 40 else 0, 0, 1) for x in x_out])
        
        # Padat (Normal): 30-70 (Triangle, Peak 50)
        y_out_padat = np.zeros_like(x_out)
        mask_padat = (x_out > 30) & (x_out < 70)
        # Safe triangle creation
        for i, x in enumerate(x_out):
            if 30 < x <= 50: y_out_padat[i] = (x - 30) / 20
            elif 50 < x < 70: y_out_padat[i] = (70 - x) / 20
        
        # Lancar (Kencang): 60-100 (Trapezoid Right)
        y_out_lancar = np.array([np.clip((x - 60)/40 if x > 60 else 0, 0, 1) for x in x_out])
        
        # Alpha Cuts (Clipping based on inference scores)
        # Map Input Names to Output Sets: Pelan->Macet, Normal->Padat, Kencang->Lancar
        alpha_macet = fuzzy_scores['Macet']
        alpha_padat = fuzzy_scores['Padat']
        alpha_lancar = fuzzy_scores['Lancar']
        
        y_cut_macet = np.minimum(y_out_macet, alpha_macet)
        y_cut_padat = np.minimum(y_out_padat, alpha_padat)
        y_cut_lancar = np.minimum(y_out_lancar, alpha_lancar)
        
        # Aggregation (Union / Max)
        y_aggregated = np.maximum(y_cut_macet, np.maximum(y_cut_padat, y_cut_lancar))
        
        # Calculate Centroid (Defuzzification)
        if np.sum(y_aggregated) == 0:
            centroid = 50 # Default if no activation
        else:
            centroid = np.sum(x_out * y_aggregated) / np.sum(y_aggregated)
            
        fig_defuz, ax_defuz = plt.subplots(figsize=(10, 4))
        
        # Plot Base Sets (Dashed/Transparent)
        ax_defuz.plot(x_out, y_out_macet, '--', color='red', alpha=0.3, label='Macet (Base)')
        ax_defuz.plot(x_out, y_out_padat, '--', color='orange', alpha=0.3, label='Padat (Base)')
        ax_defuz.plot(x_out, y_out_lancar, '--', color='green', alpha=0.3, label='Lancar (Base)')
        
        # Plot Aggregated Area (Filled)
        ax_defuz.fill_between(x_out, y_aggregated, color='blue', alpha=0.4, label='Area Keputusan')
        ax_defuz.plot(x_out, y_aggregated, color='blue', linewidth=2)
        
        # Plot Centroid Line
        ax_defuz.axvline(x=centroid, color='black', linewidth=3, linestyle='-', label=f'Centroid (Score: {centroid:.1f})')
        
        ax_defuz.set_title(f'Output Fuzzy & Defuzzification (Status: {status_fuzzy})')
        ax_defuz.set_xlabel('Score Kelancaran (0=Macet Total, 100=Sangat Lancar)')
        ax_defuz.legend(loc='center right')
        ax_defuz.grid(True, alpha=0.3)
        st.pyplot(fig_defuz)



# --- Main App Logic ---
def main():
    st.sidebar.title("DashboardPredikLalin")
    menu_options = [
        "Dataset Kota Essen",
        "Pemuatan & Pembersihan Data",
        "Analisis Data Eksplorasi (EDA)",
        "Rekayasa Fitur",
        "Pemodelan",
        "Prediksi Lalu Lintas"
    ]
    selection = st.sidebar.radio("Pilih Halaman :", menu_options)
    
    # Load Data Once
    with st.spinner('Memuat data...'):
        raw_df = load_data()
        if raw_df is None:
            return
        df = preprocess_data(raw_df)
    
    # Train Models Once (Cached)
    # Returns dictionary of models
    models_dict = train_models(df)

    # Route to Page
    if selection == "Dataset Kota Essen":
        show_dataset_overview(raw_df)
    elif selection == "Pemuatan & Pembersihan Data":
        show_data_loading(raw_df)
    elif selection == "Analisis Data Eksplorasi (EDA)":
        show_eda(df)
    elif selection == "Rekayasa Fitur":
        show_feature_engineering(df)
    elif selection == "Pemodelan":
        show_Pemodelan(df, models_dict)
    elif selection == "Prediksi Lalu Lintas":
        show_prediction(models_dict)

if __name__ == "__main__":
    main()
