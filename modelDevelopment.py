import shutil
import pickle

from pathlib import Path

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from dataPreProcessingAndFeatureEngg import DataPreProcessingAndFeatureEngg


class ModelDevelopmentPipleLine:
    def __init__(self):
        self.preProcessingPipeLineObj = DataPreProcessingAndFeatureEngg()
        # This returns your Pipeline(steps=[('cleaner',...), ('preprocessor',...)])
        self.preProcessPipeLine = self.preProcessingPipeLineObj.build_reproducible_preprocessing_pipeline()
        self.artifacts_dir = Path("artifacts")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def export_reusable_artifacts(self, fitted_pipeline, input_example, model_name):
        """Saves the best model and handles MLflow directory conflicts."""
        safe_name = model_name.lower().replace(" ", "_")

        # Paths
        pickle_path = self.artifacts_dir / f"heart_disease_{safe_name}_pipeline.pkl"
        mlflow_path = self.artifacts_dir / "mlflow"

        # Overwrite protection for MLflow
        if mlflow_path.exists():
            shutil.rmtree(mlflow_path)

        # Save Pickle
        with open(pickle_path, "wb") as f:
            pickle.dump(fitted_pipeline, f)

        # Save MLflow Model
        mlflow.sklearn.save_model(
            sk_model=fitted_pipeline,
            path=str(mlflow_path),
            input_example=input_example
        )
        print(f"\n[SAVED] Best model '{model_name}' saved to {self.artifacts_dir}")

    def executeModelPipeLine(self):
        # 1. Load Data
        features, _ = self.preProcessingPipeLineObj.before_clean_data()
        target_df = self.preProcessingPipeLineObj.get_binary_target()
        target = target_df.values.ravel()

        # 2. Split
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42, stratify=target
        )

        # 3. Models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
            "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
        }

        best_model_name = None
        best_model_pipeline = None
        best_cv_roc_auc = -1.0

        print(f"\n{'Model':<20} | {'CV ROC-AUC':<12} | {'Test Accuracy':<12}")
        print("-" * 50)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, clf in models.items():
            with mlflow.start_run(run_name=name):
                # Build end-to-end pipeline
                # We use the existing preProcessPipeLine directly
                model_pipeline = Pipeline(steps=[
                    ('preprocessing', self.preProcessPipeLine),
                    ('classifier', clf)
                ])

                # Cross-validate
                cv_results = cross_validate(model_pipeline, X_train, y_train, cv=skf, scoring='roc_auc')
                mean_roc_auc = cv_results['test_score'].mean()

                # Fit on full training set
                model_pipeline.fit(X_train, y_train)

                # Test set evaluation
                y_pred = model_pipeline.predict(X_test)
                test_acc = accuracy_score(y_test, y_pred)

                print(f"{name:<20} | {mean_roc_auc:<12.4f} | {test_acc:<12.4f}")

                # Tracking
                mlflow.log_metric("cv_roc_auc", mean_roc_auc)
                mlflow.log_metric("test_accuracy", test_acc)

                # Update best model
                if mean_roc_auc > best_cv_roc_auc:
                    best_cv_roc_auc = mean_roc_auc
                    best_model_name = name
                    best_model_pipeline = model_pipeline

        # Export the champion
        if best_model_pipeline:
            self.export_reusable_artifacts(best_model_pipeline, X_test.iloc[:1], best_model_name)


def main():
    mlflow.set_experiment("Heart_Disease_Final")
    pipeline = ModelDevelopmentPipleLine()
    pipeline.executeModelPipeLine()


if __name__ == "__main__":
    main()
