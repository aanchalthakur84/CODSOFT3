import pandas as pd
import numpy as np
from movie_rating_prediction import MovieRatingPredictor

def load_imdb_data(file_path):
    """Load the IMDb India dataset with UTF-16 encoding"""
    
    # The file appears to be UTF-16 encoded with null bytes
    try:
        df = pd.read_csv(file_path, sep='\t', encoding='utf-16')
    except:
        try:
            df = pd.read_csv(file_path, sep='\t', encoding='utf-16-le')
        except:
            df = pd.read_csv(file_path, sep='\t', encoding='utf-16-be')
    
    print(f"Loaded data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df

def clean_data(df):
    """Clean and preprocess the data"""
    
    # Remove empty rows
    df = df.dropna(how='all')
    
    # Clean Year column
    df['Year'] = df['Year'].astype(str).str.extract('(\d{4})')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # Clean Duration - extract minutes
    df['Duration'] = df['Duration'].astype(str).str.extract('(\d+)')
    df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
    
    # Clean Rating
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    
    # Clean Votes - remove commas and quotes
    df['Votes'] = df['Votes'].astype(str).str.replace(',', '').str.replace('"', '').str.strip()
    df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
    
    # Remove rows with missing ratings (our target)
    df_clean = df.dropna(subset=['Rating']).copy()
    
    print(f"After removing missing ratings: {len(df_clean)} records")
    
    if len(df_clean) == 0:
        print("No valid records found!")
        return None
    
    # Fill missing numeric values
    df_clean['Year'] = df_clean['Year'].fillna(df_clean['Year'].median())
    df_clean['Duration'] = df_clean['Duration'].fillna(df_clean['Duration'].median())
    df_clean['Votes'] = df_clean['Votes'].fillna(0)
    
    # Fill missing categorical values
    categorical_cols = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
    for col in categorical_cols:
        df_clean[col] = df_clean[col].fillna('Unknown')
    
    print(f"Final cleaned dataset shape: {df_clean.shape}")
    print(f"Rating range: {df_clean['Rating'].min():.1f} - {df_clean['Rating'].max():.1f}")
    
    return df_clean

def main():
    # File path
    file_path = r"C:\Users\VICTUS\ARPAN DOC\IMDb Movies India.txt"
    
    print("Loading IMDb India dataset...")
    df = load_imdb_data(file_path)
    
    print("\nCleaning data...")
    df_clean = clean_data(df)
    
    if df_clean is None:
        print("Failed to clean data. Exiting.")
        return
    
    # Save cleaned data
    df_clean.to_csv(r"C:\Users\VICTUS\CascadeProjects\movie-rating-prediction\imdb_cleaned.csv", index=False)
    
    # Show sample
    print("\nSample of cleaned data:")
    print(df_clean.head())
    
    # Initialize and train model
    print("\nTraining movie rating prediction model...")
    predictor = MovieRatingPredictor()
    
    try:
        train_metrics, test_metrics, X_test, y_test, y_pred = predictor.train_model(df_clean, target_column='Rating')
        
        # Show feature importance
        importance_df = predictor.feature_importance()
        print("\nTop 10 Important Features:")
        print(importance_df.head(10))
        
        # Example prediction
        sample_movie = {
            'Name': 'Test Bollywood Movie',
            'Year': 2023,
            'Duration': 135,
            'Genre': 'Drama',
            'Rating': 0,  # Will be predicted
            'Votes': 5000,
            'Director': 'Director_1',
            'Actor 1': 'Actor_1',
            'Actor 2': 'Actor_2',
            'Actor 3': 'Actor_3'
        }
        
        predicted_rating = predictor.predict(sample_movie)
        print(f"\nPredicted rating for sample movie: {predicted_rating:.2f}")
        
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    main()
