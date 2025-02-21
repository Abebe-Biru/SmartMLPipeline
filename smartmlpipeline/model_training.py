# smartmlpipeline/model_training.py

import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

class ModelTrainer:
    def __init__(self):
        self.best_model = None
        self.best_params = None
        self.study = None
        self.best_model_name = None

    def objective(self, trial, model_name, X, y):
        if model_name == 'RandomForest':
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
            )
        elif model_name == 'SVM':
            C = trial.suggest_loguniform('C', 1e-3, 1e3)
            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
            model = SVC(C=C, kernel=kernel, probability=True)
        elif model_name == 'LogisticRegression':
            C = trial.suggest_loguniform('C', 1e-3, 1e3)
            penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
            solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
            model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000)
        else:
            raise ValueError("Unsupported model type")

        score = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        accuracy = score.mean()
        return 1.0 - accuracy  # Optuna minimizes the objective

    def train_and_select_model(self, X_train, y_train, n_trials=50, direction='minimize'):
        search_space = ['RandomForest', 'SVM', 'LogisticRegression']
        best_score = float('inf')

        for model_name in search_space:
            print(f"Optimizing {model_name}...")
            study = optuna.create_study(direction=direction)
            study.optimize(
                lambda trial: self.objective(trial, model_name, X_train, y_train),
                n_trials=n_trials,
            )

            if study.best_value < best_score:
                best_score = study.best_value
                self.best_params = study.best_params
                self.study = study
                self.best_model_name = model_name

        # Retrain the best model on the entire training set
        self.best_model = self.create_model(self.best_model_name, self.best_params)
        self.best_model.fit(X_train, y_train)
        return self.best_model

    @staticmethod
    def create_model(model_name, params):
        if model_name == 'RandomForest':
            return RandomForestClassifier(**params)
        elif model_name == 'SVM':
            return SVC(**params, probability=True)
        elif model_name == 'LogisticRegression':
            solver = 'liblinear' if params.get('penalty') == 'l1' else 'lbfgs'
            params['solver'] = solver
            params['max_iter'] = 1000
            return LogisticRegression(**params)
        else:
            raise ValueError("Unsupported model type")
