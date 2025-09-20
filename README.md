# Flipkart Customer Support Analysis

This project analyzes customer support data to predict Customer Satisfaction (CSAT) scores using machine learning techniques. The analysis includes data preprocessing, exploratory data analysis, sentiment analysis, and predictive modeling using XGBoost.

## Dataset Description

The dataset contains customer support interactions with the following key features:
- **Customer interaction details**: Channel, category, sub-category
- **Temporal data**: Issue reported time, response time, survey date
- **Agent information**: Agent name, supervisor, manager, tenure, shift
- **Customer data**: Location, product category, item price
- **Response metrics**: Connected handling time, response time analysis
- **Target variable**: CSAT Score (1-5 scale)

## Project Structure

```
├── FlipkartAnalysis.ipynb    # Main analysis notebook
├── README.md                 # Project documentation
└── data/                     # Data directory (add your CSV file here)
    └── Customer_support_data.csv
```

## Analysis Overview

### 1. Data Preprocessing
- **Missing value handling**: Filled missing values appropriately
- **Date/time processing**: Converted timestamps and calculated response times
- **Feature engineering**: 
  - Created response speed categories (Very Fast, Fast, Slow, Very Slow)
  - Mapped tenure buckets to numerical values
  - Generated sentiment scores from customer remarks using TextBlob
  - Created binary CSAT target variable (satisfied vs unsatisfied)

### 2. Exploratory Data Analysis
- Customer satisfaction score distribution
- Response time analysis across different CSAT levels
- Agent shift performance comparison
- Visualization of key patterns and trends

### 3. Machine Learning Model
- **Algorithm**: XGBoost Classifier
- **Target**: Binary classification (CSAT ≥ 4 = satisfied, < 4 = unsatisfied)
- **Features**: Categorical encoding of all relevant variables
- **Optimization**: Grid search with cross-validation for hyperparameter tuning

### 4. Model Performance
- **Accuracy**: ~84.7%
- **Hyperparameter optimization** using GridSearchCV
- **Evaluation metrics**: Precision, recall, F1-score, confusion matrix

## Key Findings

1. **Response Time Impact**: Faster response times correlate with higher customer satisfaction
2. **Shift Analysis**: Different agent shifts show varying levels of customer satisfaction
3. **Predictive Accuracy**: The XGBoost model achieves good performance in predicting customer satisfaction

## Required Libraries

Make sure you have the following Python packages installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost textblob
```

## Usage Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Nagaraj335/Labmentix-Internship.git
   cd Labmentix-Internship
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt  # or install packages individually
   ```

3. **Prepare your data**:
   - Place your `Customer_support_data.csv` file in the project directory
   - Update the file path in the notebook if necessary

4. **Run the analysis**:
   - Open `FlipkartAnalysis.ipynb` in Jupyter Notebook or JupyterLab
   - Execute all cells to reproduce the analysis

## Model Features

The model uses the following types of features:
- **Categorical variables**: Channel, category, sub-category, agent details, shift information
- **Numerical variables**: Response time, item price, tenure bucket
- **Engineered features**: Response speed categories, sentiment scores

## Results Interpretation

- **High Precision for Satisfied Customers**: The model performs well at identifying satisfied customers
- **Response Time Insights**: Faster response times are strong predictors of satisfaction
- **Agent Performance**: Different shifts and agent characteristics impact satisfaction levels

## Future Improvements

1. **Feature Engineering**: 
   - Add more temporal features (day of week, time of day patterns)
   - Include interaction features between variables

2. **Model Enhancement**:
   - Try ensemble methods combining multiple algorithms
   - Implement deep learning approaches for text analysis

3. **Business Applications**:
   - Real-time satisfaction prediction
   - Agent performance optimization
   - Response time targets based on predictions

## Technical Notes

- **Data Preprocessing**: Handles missing values and converts categorical variables using one-hot encoding
- **Cross-validation**: Uses 3-fold CV for reliable model evaluation
- **Hyperparameter Tuning**: Grid search across multiple XGBoost parameters
- **Evaluation**: Comprehensive metrics including confusion matrix and classification report

## Contributing

Feel free to fork this repository and submit pull requests for improvements. Please ensure your code follows the existing style and includes appropriate documentation.

## License

This project is part of the Labmentix Internship program. Please refer to the organization's guidelines for usage and distribution.
