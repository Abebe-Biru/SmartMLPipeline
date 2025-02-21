from smartmlpipeline import SmartMLPipeline
from smartmlpipeline.utils import load_data
from smartmlpipeline.visualization import plot_optimization_history, plot_param_importances

# Load dataset
df = load_data('data/dataset.csv')

# Specify columns
target_column = 'target'
categorical_columns = ['cat_feature1', 'cat_feature2']
numerical_columns = df.columns.drop([target_column] + categorical_columns).tolist()

# Initialize pipeline
pipeline = SmartMLPipeline()

# Run pipeline
results = pipeline.run_pipeline(
    df,
    target_column=target_column,
    categorical_columns=categorical_columns,
    numerical_columns=numerical_columns,
    n_trials=30  # Adjust the number of trials as needed
)

# Access results
print("Best Model:", type(results['best_model']).__name__)
print("Best Hyperparameters:", results['best_params'])
print("Evaluation Results:", results['evaluation_results'])
print("Selected Features:", results['selected_features'])

# Visualize Optuna study (Optional)
plot_optimization_history(results['study']).show()
plot_param_importances(results['study']).show()
