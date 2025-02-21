# smartmlpipeline/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Visualizer:
    def __init__(self):
        pass

    def plot_confusion_matrix(self, cm, labels):
        sns.set(style='whitegrid')
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    def plot_feature_importances(self, model, feature_names):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            names = [feature_names[i] for i in indices]
            plt.figure(figsize=(10, 6))
            plt.title('Feature Importances')
            plt.bar(range(len(names)), importances[indices])
            plt.xticks(range(len(names)), names, rotation=90)
            plt.tight_layout()
            plt.show()
        else:
            print("Model does not support feature importances.")
