import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io
import time # Needed for progress bar simulation

# Backend imports
from src.data_loader import load_raw_data, save_processed_data, generate_synthetic_data
from src.preprocessing import clean_price_data, filter_target_car, encode_categorical_features, remove_outliers
from src.model import split_data, train_model, evaluate_model, save_model
from src.scraper import fetch_page, parse_search_results # <--- NEW IMPORTS

# --- PAGE CONFIG ---
st.set_page_config(page_title="JDM Price Forecaster", page_icon="🚗", layout="wide")

# --- HEADER ---
st.title("🇯🇵 JDM Price Forecaster")
st.markdown("### Architect Your Dream Car Analysis")

# --- 1. SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Configuration")

# DATA SOURCE SELECTOR (Radio Button is cleaner than Toggles for 3 options)
data_source = st.sidebar.radio(
    "Data Source:",
    ("📁 Local CSV", "🧪 Synthetic (Demo)", "🔴 Live Data (TC-V)"),
    help="Choose where the data comes from."
)

# --- 2. DATA LOADING LOGIC ---
@st.cache_data(ttl=600) # Cache live data for 10 minutes to avoid banning
def get_live_data():
    # Progress Bar UI Container
    progress_text = "Connecting to Japan (TC-V)..."
    my_bar = st.progress(0, text=progress_text)
    
    # STEP 1: Fetch
    target_url = "https://www.tc-v.com/used_car/honda/fit/"
    html = fetch_page(target_url)
    
    # Update Bar
    my_bar.progress(50, text="Downloaded HTML. Parsing Data...")
    time.sleep(0.5) # Small UX pause so user sees the change
    
    if not html:
        my_bar.empty()
        return None
        
    # STEP 2: Parse
    data = parse_search_results(html)
    my_bar.progress(90, text=f"Found {len(data)} cars. Finalizing...")
    
    # STEP 3: Convert to DataFrame
    df = pd.DataFrame(data)
    
    my_bar.progress(100, text="Complete!")
    time.sleep(0.5)
    my_bar.empty() # Hide bar after success
    
    return df

def load_data(source):
    df = None
    if source == "📁 Local CSV":
        project_root = Path.cwd()
        raw_data_path = project_root / 'data' / 'raw' / 'final_cars_datasets.csv'
        if raw_data_path.exists():
            df = load_raw_data(raw_data_path)
    
    elif source == "🧪 Synthetic (Demo)":
        df = generate_synthetic_data(n_samples=1000)
        
    elif source == "🔴 Live Data (TC-V)":
        # We use a button to trigger scraping so we don't spam the server on reload
        if st.sidebar.button("⚡ Scrape New Data"):
            df = get_live_data()
            if df is not None and not df.empty:
                st.session_state['live_df'] = df # Save to session state
            else:
                st.error("Failed to scrape data or no cars found.")
        
        # Retrieve from session state if available (persistent across tabs)
        if 'live_df' in st.session_state:
            df = st.session_state['live_df']
        else:
            st.info("Click '⚡ Scrape New Data' to fetch live listings.")
            st.stop() # Stop execution until data is present

    return df

# LOAD AND INITIAL CLEAN
df = load_data(data_source)

if df is not None:
    # Universal Cleaning (applies to all sources)
    # Note: Scraped data might not need 'clean_price_data' if we parsed it well,
    # but it's safe to run to remove empty rows.
    df = clean_price_data(df)
else:
    st.stop() # Wait for data

# --- 3. FILTERING UI ---
# Determine available marks (Dynamic based on source)
available_marks = df['mark'].value_counts().index.tolist()
if not available_marks:
    st.error("Dataset is empty. Check your source.")
    st.stop()

selected_mark = st.sidebar.selectbox("Select Make:", available_marks)

mark_mask = df['mark'] == selected_mark
available_models = df[mark_mask]['model'].value_counts().index.tolist()

selected_model = st.sidebar.selectbox("Select Model:", available_models)

# --- 4. MAIN TABS ---
tab1, tab2 = st.tabs(["📊 Data Overview", "🤖 AI Training Studio"])

# --- TAB 1: DATA OVERVIEW ---
with tab1:
    st.subheader(f"Analysis: {selected_mark.upper()} {selected_model.upper()}")
    
    # Filter
    df_target = filter_target_car(df, selected_mark, selected_model)
    
    if df_target.empty:
        st.warning("No cars match this filter.")
    else:
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Offers", len(df_target))
        
        # Safe metric calculation (handle empty or zero)
        avg_price = int(df_target['price'].mean() * 1000) if not df_target.empty else 0
        avg_mile = int(df_target['mileage'].mean()) if not df_target.empty else 0
        
        col2.metric("Avg Price (FOB)", f"{avg_price:,} JPY")
        col3.metric("Avg Mileage", f"{avg_mile:,} km")
        
        # Show DataFrame
        st.dataframe(df_target.head(), hide_index=True)
        
        # Download Button
        csv_data = df_target.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download This Data (CSV)",
            data=csv_data,
            file_name=f"data_{data_source}_{selected_mark}.csv",
            mime="text/csv"
        )
        
        # Diagnostics
        st.divider()
        st.markdown("#### 🔍 Data Diagnostics")
        numeric_df = df_target.select_dtypes(include=['float64', 'int64'])
        if not numeric_df.empty:
            corr = numeric_df.corr()
            st.dataframe(corr[['price']].sort_values(by='price', ascending=False).style.background_gradient(cmap='coolwarm'))

# --- TAB 2: AI TRAINING STUDIO ---
with tab2:
    st.subheader("🤖 AI Training Studio")
    
    if df_target.empty:
        st.warning("No data available to train.")
    else:
        if st.button("🚀 Start Advanced Training", type="primary"):
            
            # 1. PRE-PROCESSING
            progress_text = "Initializing Training Pipeline..."
            my_bar = st.progress(0, text=progress_text)
            
            # A. Outlier Removal (Skip for Synthetic/Live if you trust it, but safer to keep)
            my_bar.progress(10, text="Cleaning Data...")
            df_cleaned_ai = remove_outliers(df_target)
            
            # B. Encoding
            my_bar.progress(30, text="Encoding Features...")
            df_encoded = encode_categorical_features(df_cleaned_ai)
            
            # C. Splitting
            my_bar.progress(50, text="Splitting Train/Test...")
            X_train, X_test, y_train, y_test = split_data(df_encoded)
            
            # 2. TRAINING
            my_bar.progress(70, text="Training Random Forest...")
            model = train_model(X_train, y_train)
            
            # 3. EVALUATION
            my_bar.progress(90, text="Evaluating...")
            mae, r2, predictions = evaluate_model(model, X_test, y_test)
            
            my_bar.progress(100, text="Done!")
            time.sleep(0.5)
            my_bar.empty()
            
            st.success("✅ Model Trained Successfully!")
            
            # METRICS
            m_col1, m_col2 = st.columns(2)
            mae_real = int(mae * 1000)
            m_col1.metric("Mean Absolute Error (MAE)", f"± {mae_real:,} JPY")
            m_col2.metric("Accuracy (R² Score)", f"{r2:.2%}")
            
            # CHARTS
            st.markdown("#### 📉 Actual vs. Predicted")
            results_df = pd.DataFrame({"Actual": y_test, "Predicted": predictions}).sort_values(by="Actual").reset_index(drop=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(results_df.index, results_df["Actual"], color='red', alpha=0.6, label='Real Price')
            ax.scatter(results_df.index, results_df["Predicted"], color='blue', alpha=0.6, label='AI Prediction', marker='x')
            ax.vlines(results_df.index, results_df["Actual"], results_df["Predicted"], color='gray', alpha=0.2)
            ax.legend()
            st.pyplot(fig)
            
            # DOWNLOAD MODEL
            buffer = io.BytesIO()
            joblib.dump(model, buffer)
            buffer.seek(0)
            st.download_button("📥 Download Model (.pkl)", buffer, "model.pkl", "application/octet-stream")