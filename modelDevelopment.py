from dataPreProcessingAndFeatureEngg import DataPreProcessingAndFeatureEngg
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, recall_score, accuracy_score
from datetime import datetime
import pandas as pd
import mlflow
import matplotlib.pyplot as plt

import mlflow.sklearn


class ModelDevelopmentPipleLine:
    def __init__(self):
        self.preProcessingPipeLineObj = DataPreProcessingAndFeatureEngg()
        self.preProcessPipeLine = self.preProcessingPipeLineObj.build_preprocessing_pipeline()


    def executeModelPipeLine(self):
        # 1. Clean the raw data first
        features, target = self.preProcessingPipeLineObj.clean_data()

        # FIX: Convert target to 1D array
        # Use .values.ravel() if target is a pandas Series/DataFrame
        # Use .ravel() if target is a NumPy array
        target = target.values.ravel() if hasattr(target, 'values') else target.ravel()
        

        features_train, features_test, target_train, target_test = train_test_split(
            features, target, test_size=0.2, random_state=42, stratify=target
        )

        # Define 3 Candidate Models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
            "SVM": SVC(probability=True, kernel='rbf'),
            "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
        }

        results = {}

        print(f"{'Model':<20} | {'CV Accuracy':<12} | {'CV Recall':<10} | {'ROC-AUC':<10}")
        print("-" * 60)

        for name, clf in models.items():
            with mlflow.start_run(run_name=name):
                # Create Pipeline
                model_pipeline = Pipeline(steps=[
                    ('preprocessor', self.preProcessPipeLine), # Feature Engineering (Scaling/Encoding)
                    ('classifier', clf)
                ])

                # Cross-Validation
                cv_scores = cross_validate(
                    model_pipeline, features_train, target_train, 
                    cv=5, 
                    scoring=['accuracy', 'recall', 'roc_auc']
                )

                # Train final version for Test Metrics
                model_pipeline.fit(features_train, target_train)
                target_pred = model_pipeline.predict(features_test)
                targer_proba = model_pipeline.predict_proba(features_test)[:, 1]
                test_auc = roc_auc_score(target_test, targer_proba)

                metrics = {
                "accuracy": accuracy_score(target_test, target_pred),
                "recall": recall_score(target_test, target_pred),
                "roc_auc": roc_auc_score(target_test, targer_proba)
                }

                # 3. Log Parameters and Metrics
                mlflow.log_params(clf.get_params())
                mlflow.log_metrics(metrics)
                mlflow.set_tag("model_type", name)

                # 4. Generate and Log Plot (Artifact)
                plt.figure(figsize=(6,6))
                ConfusionMatrixDisplay.from_predictions(target_test, target_pred, cmap='Blues')
                plot_path = f"confusion_matrix_{name}.png"
                plt.title(f"Confusion Matrix - {name}")
                plt.savefig(plot_path)
                mlflow.log_artifact(plot_path)
                plt.close()

                # 5. Log the Model (Artifact)
                # This allows for one-click deployment later
                mlflow.sklearn.log_model(model_pipeline, name="heart-disease-pipeline", 
                                        input_example=features_test.iloc[:1])

modelPipeLine = ModelDevelopmentPipleLine()
modelPipeLine.executeModelPipeLine()




        
        
        

