from ucimlrepo import fetch_ucirepo
import pandas as pd

class DataSetLoader:
    def __init__(self, dataset_id: int = 45):
        """
        Initializes the loader and fetches the dataset from UCI.
        Dataset ID 45 corresponds to the Heart Disease dataset.
        """
        # Fetching data during initialization ensures it's available for all methods
        self.dataset = fetch_ucirepo(id=dataset_id)
        
        # Storing data internally as DataFrames/Series
        self.X = self.dataset.data.features
        self.y = self.dataset.data.targets

    def get_features(self) -> pd.DataFrame:
        """Returns the 14 clinical features."""
        return self.X
        
    def get_targets(self) -> pd.DataFrame:
        """Returns the binary target (presence/absence of disease)."""
        return self.y

    def get_metadata(self):
        """Returns extra info like variable descriptions and units."""
        return self.dataset.variables


     
