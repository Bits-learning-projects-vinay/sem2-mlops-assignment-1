import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from dataSetLoad import DataSetLoader


class DataPreProcessingAndFeatureEngg:
    def __init__(self, dataset_id: int = 45):
        loader = DataSetLoader()
        self.features = loader.get_features()
        self.targets = loader.get_targets()
        self.features_before_clean = self.features.copy()
        self.targets_before_clean = self.targets.copy()

    def before_clean_data(self):
        return  self.features_before_clean, self.targets_before_clean    

    def clean_data(self):
        """
        Performs data cleaning: handles missing values, removes duplicates, 
        and ensures correct data types.
        """
        # 1. Handle Missing Values
        # The UCI dataset sometimes has '?' or NaN in 'ca' and 'thal' columns
        # We fill numeric columns with the median and categorical with the mode
        print(self.features)
        for col in self.features.columns:
            if self.features[col].isnull().sum() > 0:
                if self.features[col].dtype == 'Categorical':
                    # Categorical: use mode
                    self.features[col] = self.features[col].fillna(features[col].mode()[0])
                elif self.features[col].dtype == 'Integer':
                    # Numerical: use mean
                    self.features[col] = self.features[col].fillna(features[col].mean())

        # 2. Target Encoding
        # The UCI target is 0 for 'no disease', and 1, 2, 3, 4 for 'disease'
        # For binary classification, we collapse 1-4 into 1
        self.targets['num'] = self.targets['num'].apply(lambda x: 1 if x > 0 else 0)

        # 3. Ensure numeric types (UCI data can sometimes load as objects)
        self.features = self.features.apply(pd.to_numeric, errors='coerce')

        
        return self.features, self.targets

    # --- 3. EXPLORATORY DATA ANALYSIS (EDA) ---
    def run_visual_eda(self, X, y):
        """
        Generates professional visualizations for class balance and correlations.
        X: Features DataFrame
        y: Targets DataFrame
        """
        # Combine into one DataFrame for easier plotting
        df = pd.concat([X, y], axis=1)
        target_name = y.columns[0]

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Class Balance
        sns.countplot(data=df, x=target_name, ax=axes[0], palette="viridis")
        axes[0].set_title("Class Balance (Healthy vs Disease)")

        # 2. Age vs Disease
        if 'age' in df.columns:
            sns.histplot(data=df, x='age', hue=target_name, kde=True, ax=axes[1], element="step")
            axes[1].set_title("Age Distribution by Heart Disease")
        else:
            axes[1].set_text(0.5, 0.5, "'age' column not found")

        # 3. Correlation Heatmap
        corr = df.corr()
        sns.heatmap(corr, annot=False, cmap='coolwarm', ax=axes[2])
        axes[2].set_title("Feature Correlation Heatmap")

        plt.tight_layout()
        plt.show()

    def build_preprocessing_pipeline(self):
        """Creates an automated cleaning and scaling pipeline."""
        # Define feature groups
        numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

        # Cleaning logic for numeric: Impute gaps with mean, then Scale
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Cleaning logic for categorical: Impute with Mode, then Encode
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return self.preprocessor        

    def get_processed_data(self):
        """Returns cleaned features and targets."""
        return self.clean_data()
    
  

dataPreProcessing = DataPreProcessingAndFeatureEngg()
features, targets = dataPreProcessing.get_processed_data() 
features_before_clean, targets_before_clean =  dataPreProcessing.before_clean_data()
print("features ", features)
print("targets ", targets);   

dataPreProcessing.run_visual_eda(features_before_clean, targets)
dataPreProcessing.run_visual_eda(features, targets) 

pipeline = dataPreProcessing.build_preprocessing_pipeline()
processed_features = pipeline.fit_transform(features)

print("\n--- Final Results ---")
print(f"Original Shape: {features.shape}")
print(f"Processed Shape (after One-Hot Encoding): {processed_features.shape}")
print("Data is now cleaned, scaled, and ready for model training.")

