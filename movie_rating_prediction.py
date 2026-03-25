import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

class MovieRatingPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_columns = None
        
    def load_data(self, file_path):
        """Load movie data from CSV or TXT file"""
        try:
            # Try different encodings for the IMDb dataset
            encodings = ['utf-8', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    if file_path.endswith('.txt'):
                        # For tab-separated IMDb data
                        df = pd.read_csv(file_path, sep='\t', encoding=encoding)
                    else:
                        # For regular CSV files
                        df = pd.read_csv(file_path, encoding=encoding)
                    
                    print(f"Data loaded successfully with {encoding} encoding. Shape: {df.shape}")
                    print(f"Columns: {list(df.columns)}")
                    return df
                except (UnicodeDecodeError, pd.errors.EmptyDataError):
                    continue
            
            print("Error: Could not read file with any encoding.")
            return None
            
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self, df):
        """Preprocess the movie data"""
        # Make a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Handle missing values
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_columns] = df_processed[numeric_columns].fillna(df_processed[numeric_columns].median())
        
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df_processed[col] = df_processed[col].fillna('Unknown')
        
        return df_processed
    
    def build_model(self, df, target_column='rating'):
        """Build the regression model with preprocessing pipeline"""
        # Identify feature columns (excluding target)
        self.feature_columns = [col for col in df.columns if col != target_column]
        
        # Separate numeric and categorical features
        numeric_features = df[self.feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df[self.feature_columns].select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numeric features: {numeric_features}")
        print(f"Categorical features: {categorical_features}")
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Create full pipeline with model
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        return self.model
    
    def train_model(self, df, target_column='rating'):
        """Train the model"""
        # Preprocess data
        df_processed = self.preprocess_data(df)
        
        # Split features and target
        X = df_processed[self.feature_columns] if self.feature_columns else df_processed.drop(columns=[target_column])
        y = df_processed[target_column]
        
        # Split train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build model if not already built
        if self.model is None:
            self.build_model(df_processed, target_column)
        
        # Train model
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Evaluate model
        train_metrics = self.calculate_metrics(y_train, y_train_pred, "Training")
        test_metrics = self.calculate_metrics(y_test, y_test_pred, "Testing")
        
        return train_metrics, test_metrics, X_test, y_test, y_test_pred
    
    def calculate_metrics(self, y_true, y_pred, dataset_name):
        """Calculate evaluation metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"\n{dataset_name} Metrics:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def predict(self, movie_data):
        """Make prediction for new movie data"""
        if self.model is None:
            print("Model not trained yet. Please train the model first.")
            return None
        
        # Ensure input is a DataFrame
        if isinstance(movie_data, dict):
            movie_data = pd.DataFrame([movie_data])
        
        # Preprocess the input data
        movie_data_processed = self.preprocess_data(movie_data)
        
        # Make prediction
        prediction = self.model.predict(movie_data_processed)
        return prediction[0]
    
    def plot_predictions(self, y_true, y_pred):
        """Plot actual vs predicted ratings"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Ratings')
        plt.ylabel('Predicted Ratings')
        plt.title('Actual vs Predicted Movie Ratings')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def feature_importance(self, feature_names=None):
        """Get feature importance from the trained model"""
        if self.model is None:
            print("Model not trained yet.")
            return None
        
        # Get feature names after preprocessing
        if hasattr(self.preprocessor, 'get_feature_names_out'):
            feature_names = self.preprocessor.get_feature_names_out()
        else:
            feature_names = [f"feature_{i}" for i in range(len(self.model.named_steps['regressor'].feature_importances_))]
        
        # Get importance scores
        importances = self.model.named_steps['regressor'].feature_importances_
        
        # Create DataFrame for better visualization
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df

def create_sample_data():
    """Create sample movie data for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample data
    data = {
        'title': [f'Movie_{i}' for i in range(n_samples)],
        'year': np.random.randint(1990, 2024, n_samples),
        'duration': np.random.randint(80, 180, n_samples),
        'budget': np.random.randint(1000000, 200000000, n_samples),
        'genre': np.random.choice(['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi'], n_samples),
        'director': np.random.choice([f'Director_{i}' for i in range(50)], n_samples),
        'main_actor': np.random.choice([f'Actor_{i}' for i in range(100)], n_samples),
        'rating': np.random.uniform(1.0, 10.0, n_samples)
    }
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Create and run the movie rating predictor
    predictor = MovieRatingPredictor()
    
    # Create sample data (replace with actual data loading)
    print("Creating sample data...")
    df = create_sample_data()
    print(f"Sample data shape: {df.shape}")
    print("\nSample data preview:")
    print(df.head())
    
    # Train the model
    print("\nTraining model...")
    train_metrics, test_metrics, X_test, y_test, y_pred = predictor.train_model(df)
    
    # Plot predictions
    predictor.plot_predictions(y_test, y_pred)
    
    # Show feature importance
    importance_df = predictor.feature_importance()
    print("\nTop 10 Important Features:")
    print(importance_df.head(10))
    
    # Example prediction
    sample_movie = {
        'title': 'New Movie',
        'year': 2023,
        'duration': 120,
        'budget': 50000000,
        'genre': 'Action',
        'director': 'Director_1',
        'main_actor': 'Actor_1'
    }
    
    predicted_rating = predictor.predict(sample_movie)
    print(f"\nPredicted rating for sample movie: {predicted_rating:.2f}")
