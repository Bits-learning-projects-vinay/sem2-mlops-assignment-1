import pandas as pd
import pickle
import sys

from pathlib import Path


class TestModel:
    def __init__(self):
        self.artifacts_dir = Path(__file__).resolve().parent / "artifacts"

        # Auto-detect the first available pipeline pickle in artifacts/
        pkl_candidates = sorted(self.artifacts_dir.glob("heart_disease_*_pipeline.pkl"))
        if not pkl_candidates:
            sys.exit(
                f"[ERROR] No pipeline pickle found in {self.artifacts_dir}.\n"
                "Run modelDevelopment.py first to generate the model artifact."
            )
        self.model_path = pkl_candidates[0]
        print(f"[INFO] Loading model: {self.model_path.name}")
        self.sample_patient = pd.DataFrame([{
            "age": 63,
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 0,
            "ca": 0,
            "thal": '?'  # Testing the '?' handling in your HeartDiseaseCleaner
        }])

    def test_production_model(self):
        # 1. Path to your saved best model
        # Note: Ensure this matches the filename in your /artifacts folder

        with open(self.model_path, "rb") as f:
            model = pickle.load(f)
        print("\nInput Patient Data:")
        print(self.sample_patient)

        # 4. Perform Prediction
        # The pipeline automatically cleans, scales, and encodes this raw input
        prediction = model.predict(self.sample_patient)
        probability = model.predict_proba(self.sample_patient)[:, 1]

        # 5. Output Results
        result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease Detected"

        print("\n" + "="*30)
        print(f"PREDICTION: {result}")
        print(f"CONFIDENCE: {probability[0]:.2%}")
        print("="*30)


if __name__ == "__main__":
    loader = TestModel()
    loader.test_production_model()