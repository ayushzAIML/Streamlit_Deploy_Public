# Student Exam Score Prediction

A machine learning web application that predicts student exam scores based on study habits and academic performance indicators.

## Overview

This project uses a Linear Regression model to predict student exam scores based on four key factors:
- Hours studied per week
- Sleep hours per night
- Attendance percentage
- Previous exam averages

The model achieves 85.37% accuracy on test data and is deployed as an interactive Streamlit web application.

## Features

- **Interactive Web Interface**: User-friendly Streamlit dashboard with sliders for input parameters
- **Real-time Predictions**: Instant score predictions with confidence metrics
- **Data Visualizations**: 
  - Student performance profile radar chart
  - Feature importance analysis
  - Performance categorization
- **Model Insights**: Display of key performance indicators and improvement recommendations

## Model Performance

- **Algorithm**: Linear Regression
- **Training R²**: 83.64%
- **Testing R²**: 85.37%
- **RMSE**: 0.4114

### Feature Importance
1. Hours Studied: 73.65%
2. Previous Scores: 42.23%
3. Attendance Percentage: 22.87%
4. Sleep Hours: 19.50%

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/student-score-prediction.git
cd student-score-prediction
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run app.py
```

## Dependencies

```
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
plotly
seaborn
```

## File Structure

```
student-score-prediction/
├── app.py                                    # Main Streamlit application
├── main.ipynb                               # Jupyter notebook with model development
├── linear_regression_student_scores.pkl    # Trained model file
├── feature_scaler.pkl                       # Feature scaler (if used)
├── scaler.pkl                              # Additional scaler
├── target_scaler.pkl                       # Target variable scaler
├── student_exam_scores.csv                 # Dataset
└── requirements.txt                        # Python dependencies
```

## Usage

1. Launch the web application using `streamlit run app.py`
2. Use the sliders to input student information:
   - Hours studied per week (0-20 hours)
   - Sleep hours per night (4-12 hours)
   - Attendance percentage (0-100%)
   - Previous exam average (0-100%)
3. Click "Predict Exam Score" to get the prediction
4. View the results including:
   - Predicted score percentage
   - Performance category
   - Change from previous scores
   - Student profile visualization
   - Feature importance analysis

## Model Development

The model was developed using the following process:

1. **Data Preprocessing**: 
   - Feature scaling and normalization
   - Handling of outliers and missing values

2. **Model Training**:
   - Linear Regression algorithm implementation
   - Cross-validation for performance evaluation
   - Feature importance analysis

3. **Model Evaluation**:
   - Train/test split validation
   - R-squared score calculation
   - Root Mean Square Error (RMSE) assessment

## Key Insights

- Study hours have the strongest impact on exam performance (73.7% importance)
- Previous academic performance is a strong predictor (42.2% importance)
- Regular attendance significantly affects outcomes (22.9% importance)
- Adequate sleep quality contributes to better performance (19.5% importance)

## Improvement Recommendations

Based on the model analysis, students can improve their exam scores by:
- Targeting 12-15 study hours per week
- Maintaining 90%+ class attendance
- Getting 7-9 hours of sleep nightly
- Building on previous academic performance
- Using active learning techniques

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or suggestions, please open an issue on GitHub or contact the maintainer.

## Acknowledgments

- Dataset source: Student performance data
- Built with Streamlit for the web interface
- Scikit-learn for machine learning implementation
- Plotly for interactive visualizations