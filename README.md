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

<!-- ---

## **5. Including the `LICENSE` File**

Use the MIT License for simplicity.


---

## **6. Developing the Package Modules**

### **6.1. `smartmlpipeline/__init__.py`**

Initialize the package and define the `SmartMLPipeline` class.

```python
# smartmlpipeline/__init__.py

from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .model_training import ModelTrainer
from .evaluation import ModelEvaluator
from .visualization import Visualizer

class SmartMLPipeline:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.engineer = FeatureEngineer()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.visualizer = Visualizer()

    def run_pipeline(
        self,
        df,
        target_column,
        categorical_columns=[],
        numerical_columns=[],
        impute_strategy='mean',
        test_size=0.2,
        random_state=42,
        feature_selection_k='all',
        n_trials=50,
        optimize_direction='minimize',
    ):
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Data Preprocessing
        X = self.preprocessor.handle_missing_values(X, strategy=impute_strategy)
        y = y.fillna(y.mode()[0])

        X = self.preprocessor.encode_categorical(X, categorical_columns)
        X = self.preprocessor.scale_features(X, numerical_columns)

        # Feature Engineering
        X_new = self.engineer.select_best_features(X, y, k=feature_selection_k)
        selected_features = X.columns[self.engineer.get_support_indices()]

        # Train-Test Split
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(
            X_new, y, test_size=test_size, random_state=random_state
        )

        # Model Training
        self.best_model = self.trainer.train_and_select_model(
            X_train, y_train, n_trials=n_trials, direction=optimize_direction
        )

        # Evaluation
        evaluation_results = self.evaluator.evaluate_model(self.best_model, X_test, y_test)

        # Visualization
        self.visualizer.plot_confusion_matrix(
            evaluation_results['confusion_matrix'], labels=y.unique()
        )
        if hasattr(self.best_model, 'feature_importances_'):
            self.visualizer.plot_feature_importances(self.best_model, selected_features)

        return {
            'best_model': self.best_model,
            'best_params': self.trainer.best_params,
            'evaluation_results': evaluation_results,
            'selected_features': selected_features,
            'study': self.trainer.study,
        }
``` -->