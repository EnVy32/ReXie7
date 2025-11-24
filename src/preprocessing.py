import pandas as pd
import re

def clean_price_data(df):
    """
    Basic technical cleaning: Drops duplicates and missing values.
    """
    initial_count = len(df)
    
    if 'link' in df.columns:
        df.drop_duplicates(subset=['link'], keep='first', inplace=True)
    else:
        df.drop_duplicates(inplace=True)
        
    df.dropna(subset=['price'], inplace=True)
    
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    print(f"Dropped: {initial_count - len(df)} rows (duplicates/empty)")
    return df

def filter_target_car(df, target_mark, target_model):
    mask = (df['mark'] == target_mark.lower()) & (df['model'] == target_model.lower())
    df_filtered = df[mask].copy()
    df_filtered.drop(columns=['mark', 'model'], axis=1, inplace=True)
    return df_filtered

def simplify_grades(df):
    """
    Consolidates messy grade names into 'Elite' categories.
    Logic: Priority matching (RS > Hybrid > Base).
    """
    print("--- FEATURE ENGINEERING: Smart Grade Grouping ---")
    
    if 'grade' not in df.columns:
        return df

    def map_grade(grade_text):
        text = str(grade_text).lower()
        if 'rs' in text: return 'RS (Sport)'
        if 'hybrid' in text: return 'Hybrid'
        if '13g' in text: return '13G (Base)'
        if '15x' in text or '15xl' in text: return '15X (Mid)'
        if 'luxe' in text: return 'Luxe'
        if 'modulo' in text: return 'Modulo'
        return 'Other'

    df['grade_category'] = df['grade'].apply(map_grade)
    
    # Drop the original messy 'grade' column to prevent noise
    df.drop(columns=['grade'], inplace=True)
    
    print(f"--> Grades grouped into: {df['grade_category'].unique()}")
    return df

def remove_outliers(df):
    """
    UPGRADE: Uses Interquartile Range (IQR) for statistical cleaning.
    """
    print("--- REMOVING OUTLIERS (Statistical IQR) ---")
    initial_count = len(df)
    
    # 1. Engine Logic (Hard Limits)
    df = df[df['engine_capacity'].between(600, 4000)]
    
    # 2. Price Logic (IQR Method)
    Q1 = df['price'].quantile(0.05) # 5th Percentile
    Q3 = df['price'].quantile(0.95) # 95th Percentile (We keep expensive cars, just cut extremes)
    
    # Filter
    df = df[(df['price'] >= Q1) & (df['price'] <= Q3)]
    
    print(f"Rows retained: {len(df)}")
    print(f"Outliers dropped: {initial_count - len(df)}")
    
    return df

def encode_categorical_features(df):
    print("--- ENCODING (One-Hot) ---")
    
    if 'link' in df.columns:
        df = df.drop(columns=['link'])
    
    # Note: We use 'grade_category' now instead of 'grade'
    candidates = ['transmission', 'drive', 'fuel', 'hand_drive', 'grade_category']
    cols_to_encode = [col for col in candidates if col in df.columns]
    
    if not cols_to_encode:
        return df

    df_encoded = pd.get_dummies(df, columns=cols_to_encode, dtype=int)
    print(f"Encoded columns: {cols_to_encode}")
    return df_encoded