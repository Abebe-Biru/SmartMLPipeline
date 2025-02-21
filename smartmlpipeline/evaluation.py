from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

class ModelEvaluator:
    def __init__(self):
        pass

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_auc_score': roc_auc,
        }
