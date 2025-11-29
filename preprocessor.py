import pandas as pd
import numpy as np

from constants import (
    DISTANCE_TO_NEXT_POINT,
    SPEED,
    TIME_INTERVAL,
    RATIO_MICRON_PER_PIXEL,
)


class Preprocessor:
    """
    This class handles preprocessing of spot data DataFrame.
    The goal is to centralize all preprocessing steps here for easier maintenance.
    """

    def __init__(self, config):
        self.threshold_nspots = config.threshold_nspots

    def sort_by_track_and_frame(self, spot_df):
        """Sort spot_df by TRACK_ID and FRAME"""
        sorted_df = spot_df.sort_values(by=["TRACK_ID", "FRAME"]).reset_index(drop=True)
        return sorted_df

    def filter_tracks_nspots(self, spot_df):
        """Filter tracks with nspots < threshold_nspots"""

        print(f" Starting spot_df size: {len(spot_df)}")
        track_counts = spot_df.TRACK_ID.value_counts()
        valid_tracks = track_counts[track_counts >= self.threshold_nspots].index
        filtered_df = spot_df[spot_df.TRACK_ID.isin(valid_tracks)].reset_index(
            drop=True
        )
        print(f" Filtered spot_df size: {len(filtered_df)}")
        return filtered_df

    def interpolate_missing_frames(self, spot_df):
        """Interpolate missing frames in tracks (linear interpolation)
        Only valid if if the number of missing frames is small in time and we assume a constant speed between frames
        """

        interpolated_dfs = []
        counter = 0
        for track_id, group in spot_df.groupby("TRACK_ID"):
            new_group = group.set_index("FRAME").reindex(
                range(group["FRAME"].min(), group["FRAME"].max() + 1)
            )
            counter += new_group.shape[0] - group.shape[0]
            new_group["TRACK_ID"] = track_id
            new_group[["POSITION_X", "POSITION_Y"]] = new_group[
                ["POSITION_X", "POSITION_Y"]
            ].interpolate()
            interpolated_dfs.append(new_group.reset_index())
        interpolated_df = pd.concat(interpolated_dfs).reset_index(drop=True)
        print(f"Interpolated {counter} missing frames across all tracks.")

        return interpolated_df

    def add_distance_to_next_point(self, spot_df):
        """Add a column with distance to next point for each spot"""

        df = spot_df.copy()
        df[DISTANCE_TO_NEXT_POINT] = 0.0
        for track_id, group in df.groupby("TRACK_ID"):
            positions = group[["POSITION_X", "POSITION_Y"]].values
            distances = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
            df.loc[group.index[:-1], DISTANCE_TO_NEXT_POINT] = distances
            df.loc[group.index[-1], DISTANCE_TO_NEXT_POINT] = (
                0.0  # Last point has no next point
            )
        return df

    def compute_speed(self, spot_df):
        """
        Compute speed as distance to next point divided by time interval (assuming constant time interval)
        Only valid if distance to next point has been computed and interpolated missing frames
        """
        df = spot_df.copy()
        df[SPEED] = (
            df[DISTANCE_TO_NEXT_POINT] / TIME_INTERVAL * RATIO_MICRON_PER_PIXEL
        )  # Assuming time interval = 1 unit
        return df
