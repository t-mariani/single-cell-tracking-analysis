from pathlib import Path

import pandas as pd
from constants import SPOT_CSV_NAME


class DataLoader:
    """Class to load spot data from a given experiment path."""

    def __init__(self, experiment_folder):
        self.experiment_folder = Path(experiment_folder)

    def load_spot_data(self):
        spot_csv_path = self.experiment_folder / SPOT_CSV_NAME
        # Remove row 1,2,3 because unused header
        spot_df = pd.read_csv(spot_csv_path, header=0, skiprows=[1, 2, 3])
        return spot_df
