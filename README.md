# Movie Rating Prediction

A machine learning project that predicts movie ratings using regression techniques. This baseline implementation uses one-hot encoding for categorical features like actors and directors.

## Features

- **Regression Model**: Uses RandomForestRegressor for predicting movie ratings
- **Data Preprocessing**: Handles missing values and categorical encoding
- **Evaluation Metrics**: MSE, RMSE, MAE, and R² scores
- **Feature Importance**: Shows which features contribute most to predictions
- **Visualization**: Plots actual vs predicted ratings

## Installation

1. Clone or download this project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from movie_rating_prediction import MovieRatingPredictor

# Initialize predictor
predictor = MovieRatingPredictor()

# Load your data (CSV format)
df = predictor.load_data('your_movie_data.csv')

# Train the model
train_metrics, test_metrics, X_test, y_test, y_pred = predictor.train_model(df)

# Make predictions
sample_movie = {
    'title': 'New Movie',
    'year': 2023,
    'duration': 120,
    'budget': 50000000,
    'genre': 'Action',
    'director': 'Director_Name',
    'main_actor': 'Actor_Name'
}
predicted_rating = predictor.predict(sample_movie)
print(f"Predicted rating: {predicted_rating:.2f}")
```

### Running the Sample

```bash
python movie_rating_prediction.py
```

This will create sample data and train the model to demonstrate functionality.

## Data Format

Your CSV file should include the following columns (or similar):

- `title`: Movie title (string)
- `year`: Release year (numeric)
- `duration`: Movie duration in minutes (numeric)
- `budget`: Movie budget (numeric)
- `genre`: Movie genre (categorical)
- `director`: Director name (categorical)
- `main_actor`: Main actor/actress (categorical)
- `rating`: Target rating (numeric, 1-10 scale)

## Model Details

### Current Implementation
- **Algorithm**: Random Forest Regressor
- **Categorical Encoding**: One-Hot Encoding
- **Numeric Scaling**: StandardScaler
- **Cross-validation**: Train/test split (80/20)

### Future Improvements
- **Target Encoding**: For better handling of high-cardinality categorical features
- **Hyperparameter Tuning**: GridSearchCV or RandomizedSearchCV
- **Advanced Models**: XGBoost, LightGBM, or Neural Networks
- **Feature Engineering**: Text features from descriptions, sentiment analysis

## Evaluation Metrics

The model is evaluated using:
- **Mean Squared Error (MSE)**: Average squared difference between predicted and actual values
- **Root Mean Squared Error (RMSE)**: Square root of MSE, in the same units as the target
- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual values
- **R² Score**: Coefficient of determination, proportion of variance explained

## File Structure

```
movie-rating-prediction/
├── movie_rating_prediction.py  # Main implementation
├── requirements.txt            # Python dependencies
├── README.md                  # This file
└── data/                      # Your data files (create as needed)
```

## Contributing

Feel free to improve the model by:
1. Adding better encoding techniques for categorical features
2. Implementing more sophisticated models
3. Adding cross-validation
4. Improving data preprocessing
5. Adding more evaluation metrics

## License

This project is open source and available under the MIT License.
=======
# CODSOFT3
>>>>>>> 0f2275cec094864b51fd8e6b24cb677a411a74e1
