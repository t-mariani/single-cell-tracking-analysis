from pathlib import Path

import pandas as pd

spot_csv_name = "_allspots.csv"


class DataLoader:
    def __init__(self, experiment_path):
        self.experiment_path = Path(experiment_path)

    def load_spot_data(self):
        spot_csv_path = self.experiment_path / spot_csv_name
        # Remove row 1,2,3 because unused header
        spot_df = pd.read_csv(spot_csv_path, header=0, skiprows=[1, 2, 3])
        return spot_df
