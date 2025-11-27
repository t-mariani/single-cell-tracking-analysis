import pandas as pd
import numpy as np

from constants import DISTANCE_TO_NEXT_POINT, SPEED

time_interval = 0.1  # video time interval in seconds
ratio_micron_per_pixel = 0.947  # microns per pixel


class Preprocessor:
    def __init__(self, spot_df, config):
        self.spot_df = spot_df
        self.threshold_nspots = config.threshold_nspots

    def sort_by_track_and_frame(self, inplace=False):
        """Sort spot_df by TRACK_ID and FRAME"""
        sorted_df = self.spot_df.sort_values(by=["TRACK_ID", "FRAME"]).reset_index(
            drop=True
        )
        if inplace:
            self.spot_df = sorted_df
        return sorted_df

    def filter_tracks_nspots(self, inplace=False):
        """Filter tracks with nspots < threshold_nspots"""
        print(f" Starting spot_df size: {len(self.spot_df)}")
        track_counts = self.spot_df.TRACK_ID.value_counts()
        valid_tracks = track_counts[track_counts >= self.threshold_nspots].index
        filtered_df = self.spot_df[
            self.spot_df.TRACK_ID.isin(valid_tracks)
        ].reset_index(drop=True)
        print(f" Filtered spot_df size: {len(filtered_df)}")
        if inplace:
            self.spot_df = filtered_df
        return filtered_df

    def interpolate_missing_frames(self, inplace=False):
        """Interpolate missing frames in tracks (linear interpolation)"""
        interpolated_dfs = []
        counter = 0
        for track_id, group in self.spot_df.groupby("TRACK_ID"):
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
        if inplace:
            self.spot_df = interpolated_df
        return interpolated_df

    def add_distance_to_next_point(self, inplace=False):
        """Add a column with distance to next point for each spot"""
        df = self.spot_df.copy()
        df[DISTANCE_TO_NEXT_POINT] = 0.0
        for track_id, group in df.groupby("TRACK_ID"):
            positions = group[["POSITION_X", "POSITION_Y"]].values
            distances = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
            df.loc[group.index[:-1], DISTANCE_TO_NEXT_POINT] = distances
            df.loc[group.index[-1], DISTANCE_TO_NEXT_POINT] = (
                0.0  # Last point has no next point
            )
        if inplace:
            self.spot_df = df
        return df

    def compute_speed(self, inplace=False):
        """Compute speed as distance to next point divided by time interval (assuming constant time interval)"""
        df = self.spot_df.copy()
        df[SPEED] = (
            df[DISTANCE_TO_NEXT_POINT] / time_interval * ratio_micron_per_pixel
        )  # Assuming time interval = 1 unit
        if inplace:
            self.spot_df = df
        return df
