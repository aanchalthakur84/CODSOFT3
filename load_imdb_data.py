import pandas as pd
import numpy as np
import re

def load_imdb_india_data(file_path):
    """Load and clean the IMDb India movies dataset"""
    
    # Read the data with tab separation
    try:
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, sep='\t', encoding='latin-1')
        except:
            df = pd.read_csv(file_path, sep='\t', encoding='cp1252')
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Handle the year column (remove parentheses and convert to numeric)
    if 'Year' in df.columns:
        df['Year'] = df['Year'].astype(str).str.replace('(', '').str.replace(')', '').str.strip()
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # Clean Duration column (extract minutes)
    if 'Duration' in df.columns:
        df['Duration'] = df['Duration'].astype(str).str.extract('(\d+)').astype(float)
    
    # Clean Rating column
    if 'Rating' in df.columns:
        df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    
    # Clean Votes column (remove commas and convert to numeric)
    if 'Votes' in df.columns:
        df['Votes'] = df['Votes'].astype(str).str.replace(',', '').str.replace('"', '').str.strip()
        df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
    
    # Clean Genre column (remove quotes and split)
    if 'Genre' in df.columns:
        df['Genre'] = df['Genre'].astype(str).str.replace('"', '').str.strip()
    
    # Clean text columns
    text_columns = ['Name', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            # Replace empty strings with NaN
            df[col] = df[col].replace('', np.nan)
    
    # Remove rows with missing target (Rating)
    df = df.dropna(subset=['Rating'])
    
    print(f"Data shape after cleaning: {df.shape}")
    print(f"Data types:")
    print(df.dtypes)
    
    return df

def preprocess_for_modeling(df):
    """Preprocess data for the movie rating prediction model"""
    
    # Create a copy
    df_processed = df.copy()
    
    # Handle missing values
    # For numeric columns, fill with median
    numeric_columns = ['Year', 'Duration', 'Votes']
    for col in numeric_columns:
        if col in df_processed.columns:
            median_val = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_val)
    
    # For categorical columns, fill with 'Unknown'
    categorical_columns = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
    for col in categorical_columns:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna('Unknown')
    
    # Create additional features
    # Number of actors (some movies might have missing actors)
    actor_cols = ['Actor 1', 'Actor 2', 'Actor 3']
    df_processed['num_actors'] = df_processed[actor_cols].notna().sum(axis=1)
    
    # Genre count (some movies have multiple genres)
    if 'Genre' in df_processed.columns:
        df_processed['genre_count'] = df_processed['Genre'].astype(str).str.split(',').apply(len)
    
    # Decade from year
    if 'Year' in df_processed.columns:
        df_processed['decade'] = (df_processed['Year'] // 10) * 10
    
    print(f"Final processed data shape: {df_processed.shape}")
    return df_processed

if __name__ == "__main__":
    # Test the data loading
    file_path = r"C:\Users\VICTUS\ARPAN DOC\IMDb Movies India.txt"
    
    try:
        # Load and clean data
        df = load_imdb_india_data(file_path)
        
        # Preprocess for modeling
        df_processed = preprocess_for_modeling(df)
        
        print("\nSample of processed data:")
        print(df_processed.head())
        
        print("\nBasic statistics:")
        print(df_processed.describe())
        
        # Save processed data for easier use
        output_path = r"C:\Users\VICTUS\CascadeProjects\movie-rating-prediction\imdb_india_processed.csv"
        df_processed.to_csv(output_path, index=False)
        print(f"\nProcessed data saved to: {output_path}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please check the file path and format.")
