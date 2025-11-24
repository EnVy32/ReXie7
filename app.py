import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io
import time

# Backend imports
from src.data_loader import load_raw_data, save_processed_data, generate_synthetic_data
from src.preprocessing import clean_price_data, filter_target_car, encode_categorical_features, remove_outliers, simplify_grades
from src.model import split_data, train_model, evaluate_model, save_model, get_feature_importance, calculate_advanced_metrics
from src.scraper import scrape_listings

# --- PAGE CONFIG (The Polish) ---
st.set_page_config(
    page_title="ReXie7 | JDM Forecaster", 
    page_icon="🏎️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (The Vibe) ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    /* Hide Streamlit Footer */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
col_head1, col_head2 = st.columns([1, 5])
with col_head1:
    st.markdown("# 🏎️")
with col_head2:
    st.title("ReXie7 AI")
    st.markdown("#### *Architect Your Dream Car Analysis*")

st.markdown("---")

# --- SIDEBAR ---
st.sidebar.header("⚙️ System Control")
data_source = st.sidebar.radio("Input Stream:", ("📁 Local CSV", "🧪 Synthetic Data", "🔴 Live Market (TC-V)"))

# --- DATA LOGIC ---
def get_live_data():
    progress_text = "Connecting to Neural Network..."
    my_bar = st.progress(0, text=progress_text)
    
    target_url = "https://www.tc-v.com/used_car/honda/fit/"
    PAGES_TO_SCRAPE = 100 
    
    def progress_callback(completed, total):
        percent = int((completed / total) * 100)
        # Smooth progress update
        my_bar.progress(percent, text=f"Ingesting Data Node {completed}/{total}...")

    data = scrape_listings(target_url, max_pages=PAGES_TO_SCRAPE, progress_callback=progress_callback)
    
    if not data:
        my_bar.empty()
        return None
        
    my_bar.progress(100, text=f"Processing {len(data)} Units...")
    df = pd.DataFrame(data)
    time.sleep(0.5)
    my_bar.empty()
    return df

def load_data(source):
    df = None
    if source == "📁 Local CSV":
        raw_data_path = Path.cwd() / 'data' / 'raw' / 'final_cars_datasets.csv'
        if raw_data_path.exists(): df = load_raw_data(raw_data_path)
    elif source == "🧪 Synthetic Data":
        df = generate_synthetic_data(n_samples=1000)
    elif source == "🔴 Live Market (TC-V)":
        if st.sidebar.button("⚡ Ignite Scraper (100 Pages)"):
            df = get_live_data()
            if df is not None and not df.empty:
                st.session_state['live_df'] = df
            else:
                st.error("Scraper Disconnected.")
        if 'live_df' in st.session_state:
            df = st.session_state['live_df']
        else:
            st.info("Awaiting Command: Ignite Scraper.")
            st.stop()
    return df

# Load & Clean
df = load_data(data_source)
if df is not None: df = clean_price_data(df)
else: st.stop()

if df.empty:
    st.error("Data Stream Empty.")
    st.stop()

# Filters
available_marks = df['mark'].value_counts().index.tolist()
selected_mark = st.sidebar.selectbox("Target Make:", available_marks)
mark_mask = df['mark'] == selected_mark
available_models = df[mark_mask]['model'].value_counts().index.tolist()
selected_model = st.sidebar.selectbox("Target Model:", available_models)

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Market Overview", "🧠 Train ReXie7", "🔮 The Oracle"])

# --- TAB 1: OVERVIEW ---
with tab1:
    st.subheader(f"Market Intel: {selected_mark.upper()} {selected_model.upper()}")
    df_target = filter_target_car(df, selected_mark, selected_model)
    
    if df_target.empty:
        st.warning("No Assets Found.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Market Volume", len(df_target))
        c2.metric("Avg Valuation", f"{int(df_target['price'].mean()):,} '000 JPY")
        c3.metric("Avg Mileage", f"{int(df_target['mileage'].mean()):,} km")
        
        st.markdown("### Asset List")
        
        column_config = {
            "price": st.column_config.NumberColumn("Price ('000 JPY)", format="%d"),
            "year": st.column_config.NumberColumn("Year", format="%d"),
            "mileage": st.column_config.NumberColumn("Mileage", format="%d km")
        }
        if 'link' in df_target.columns:
            column_config["link"] = st.column_config.LinkColumn("Source", display_text="View Listing 🔗")
            
        st.dataframe(
            df_target.head(50), 
            hide_index=True, 
            column_config=column_config,
            use_container_width=True
        )
        
        st.download_button("📥 Export Intel (CSV)", df_target.to_csv(index=False).encode('utf-8'), "jdm_intel.csv", "text/csv")

# --- TAB 2: TRAINING ---
with tab2:
    st.subheader("🧠 Neural Training Ground")
    
    if df_target.empty:
        st.warning("Insufficient Data.")
    else:
        if st.button("🚀 Train ReXie7 Model", type="primary"):
            
            my_bar = st.progress(0, text="Pipeline Initialization...")
            
            my_bar.progress(10, text="Sanitizing Data & Removing Anomalies (IQR)...")
            df_cleaned_ai = remove_outliers(df_target)
            
            my_bar.progress(25, text="Engineering Grade Features...")
            df_engineered = simplify_grades(df_cleaned_ai)
            
            my_bar.progress(40, text="Encoding Categorical Signals...")
            df_encoded = encode_categorical_features(df_engineered)
            
            my_bar.progress(50, text="Splitting Train/Test Vectors...")
            X_train, X_test, y_train, y_test = split_data(df_encoded)
            
            my_bar.progress(70, text="Gradient Boosting (XGBoost Grid Search)...")
            model = train_model(X_train, y_train)
            
            my_bar.progress(90, text="Running Diagnostics...")
            metrics, predictions = evaluate_model(model, X_test, y_test)
            adv_metrics = calculate_advanced_metrics(model, X_train, y_train, X_test, y_test, predictions)
            
            # Save State
            st.session_state['ai_model'] = model
            st.session_state['ai_metrics'] = metrics
            st.session_state['ai_adv_metrics'] = adv_metrics
            st.session_state['ai_predictions'] = predictions
            st.session_state['ai_y_test'] = y_test
            st.session_state['ai_cols'] = X_train.columns
            
            my_bar.progress(100, text="ReXie7 Online.")
            time.sleep(0.5)
            my_bar.empty()
            st.success("✅ Model Successfully Calibrated.")

        # DASHBOARD
        if 'ai_model' in st.session_state:
            metrics = st.session_state['ai_metrics']
            adv = st.session_state['ai_adv_metrics']
            y_test = st.session_state['ai_y_test']
            preds = st.session_state['ai_predictions']
            cols = st.session_state['ai_cols']
            
            # KPI ROW
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Precision (MAE)", f"± {int(metrics['mae']):,} k JPY", help="Mean Absolute Error")
            kpi2.metric("Confidence (R²)", f"{metrics['r2']:.2%}")
            kpi3.metric("Deviation (RMSE)", f"{int(metrics['rmse']):,} k JPY")
            
            delta = adv['train_r2'] - metrics['r2']
            color = "normal" if delta < 0.05 else "inverse"
            kpi4.metric("Learn Rate (Train)", f"{adv['train_r2']:.2%}", delta=f"{delta:.2%} Gap", delta_color=color)
            
            st.markdown("---")
            
            # VIZ ROW
            v1, v2 = st.columns(2)
            with v1:
                st.markdown("#### 🎯 Accuracy Matrix")
                res_df = pd.DataFrame({"Actual": y_test, "Predicted": preds}).sort_values(by="Actual").reset_index(drop=True)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(res_df.index, res_df["Actual"], color='#ff4b4b', alpha=0.5, s=15, label='Market Value')
                ax.scatter(res_df.index, res_df["Predicted"], color='#00cc96', alpha=0.5, marker='x', s=15, label='ReXie7 Value')
                ax.legend()
                ax.set_ylabel("Price ('000 JPY)")
                st.pyplot(fig)
                
            with v2:
                st.markdown("#### 🧠 Feature Weight")
                imp_df = get_feature_importance(st.session_state['ai_model'], cols)
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                sns.barplot(data=imp_df.head(8), x='Importance', y='Feature', palette='mako', ax=ax2)
                st.pyplot(fig2)
            
            # SAVE
            buffer = io.BytesIO()
            joblib.dump(st.session_state['ai_model'], buffer)
            buffer.seek(0)
            st.download_button("💾 Save ReXie7 Core (.pkl)", buffer, "rexie7_core.pkl")

# --- TAB 3: THE ORACLE (PREDICTOR) ---
with tab3:
    st.subheader("🔮 The Oracle")
    st.markdown("Input vehicle parameters to receive an instant valuation from ReXie7.")
    
    if 'ai_model' not in st.session_state:
        st.warning("⚠️ ReXie7 is offline. Please go to the 'Train ReXie7' tab and train the model first.")
    else:
        # INPUT FORM
        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                in_year = st.number_input("Year", min_value=1990, max_value=2025, value=2015)
                in_mileage = st.number_input("Mileage (km)", min_value=0, max_value=300000, value=50000, step=1000)
                in_engine = st.number_input("Engine (cc)", min_value=660, max_value=4000, value=1500)
            
            with c2:
                in_trans = st.selectbox("Transmission", ["at", "mt"])
                in_drive = st.selectbox("Drive", ["2wd", "4wd"])
                in_grade = st.selectbox("Grade Category", ["13G (Base)", "Hybrid", "RS (Sport)", "15X (Mid)", "Other"])
            
            submitted = st.form_submit_button("🔮 Consult ReXie7")
            
            if submitted:
                # 1. Reconstruct DataFrame from inputs
                input_data = pd.DataFrame({
                    'year': [in_year],
                    'mileage': [in_mileage],
                    'engine_capacity': [in_engine],
                    'transmission': [in_trans],
                    'drive': [in_drive],
                    'grade_category': [in_grade],
                    'fuel': ['gasoline'],      # Default assumption
                    'hand_drive': ['rhd']      # Default assumption
                })
                
                # 2. Alignment (One-Hot Encoding Matching)
                # We need to ensure the input dataframe has the exact same columns as the training data
                model_cols = st.session_state['ai_cols']
                
                # Pre-encode
                input_encoded = pd.get_dummies(input_data)
                
                # Reindex to match model columns (fill missing with 0)
                final_input = input_encoded.reindex(columns=model_cols, fill_value=0)
                
                # 3. Predict
                model = st.session_state['ai_model']
                prediction = model.predict(final_input)[0]
                
                st.markdown("---")
                st.success(f"### Estimated Valuation: {int(prediction):,} '000 JPY")
                st.caption("*Estimate based on FOB Price (Japan Port)*")