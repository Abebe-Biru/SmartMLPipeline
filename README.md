# SmartMLPipeline

SmartMLPipeline simplifies the process of building machine learning models for classification tasks. It automates data preprocessing, feature engineering, model training, evaluation, and visualization.

## **Features**

- **Data Cleaning**: Handles missing values and scales numerical features.
- **Encoding**: Encodes categorical variables.
- **Feature Selection**: Selects the most relevant features.
- **Model Training**: Trains multiple models with hyperparameter tuning using Optuna.
- **Evaluation**: Provides detailed evaluation metrics.
- **Visualization**: Offers insights through plots and graphs.

## **Installation**

```bash
pip install SmartMLPipeline
```
## **Quickstart**
```python
from smartmlpipeline import SmartMLPipeline
import pandas as pd

# Load your data
df = pd.read_csv('data/dataset.csv')

# Define target and feature columns
target_column = 'target'
categorical_columns = ['cat_feature1', 'cat_feature2']
numerical_columns = ['num_feature1', 'num_feature2']

# Initialize and run the pipeline
pipeline = SmartMLPipeline()
results = pipeline.run_pipeline(
    df,
    target_column=target_column,
    categorical_columns=categorical_columns,
    numerical_columns=numerical_columns
)

# Access results
print("Best Model:", results['best_model'])
print("Evaluation Metrics:", results['evaluation_results'])
```