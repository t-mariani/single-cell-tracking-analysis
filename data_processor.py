import pandas as pd
import matplotlib.pyplot as plt

from preprocessor import Preprocessor
from run_tumble_labeler import RunTumbleLabeler
from plotter import Plotter
from constants import RUN_TUMBLE_LABEL, RUN_TUMBLE_INDEX, RUN_TUMBLE, SPEED


class DataProcessor:
    def __init__(self, spot_df, config, name=None):
        self.name = name
        self.spot_df = spot_df
        self.config = config
        self.plotter = Plotter(spot_df)
        self.preprocessor = Preprocessor(spot_df, config)
        self.run_tumble_labeler = RunTumbleLabeler(config=config)
        self.list_longest = spot_df.TRACK_ID.value_counts().index.tolist()[:10]
        if self.config.preprocess_to_apply_default:
            self.preprocess(self.config.preprocess_to_apply_default)
        self.print_stats()

    def _propagate_spot_df_changes(self):
        self.preprocessor.spot_df = self.spot_df
        self.plotter.spot_df = self.spot_df

    def preprocess(self, specific_steps=None):
        if specific_steps is None or "filter_nspots" in specific_steps:
            print("Filtering tracks with nspots <", self.config.threshold_nspots)
            self.spot_df = self.preprocessor.filter_tracks_nspots(inplace=True)
        if specific_steps is None or "interpolate_missing_frames" in specific_steps:
            print("Interpolating missing frames")
            self.spot_df = self.preprocessor.interpolate_missing_frames(inplace=True)
        if specific_steps is None or "add_distance_to_next_point" in specific_steps:
            print("Adding distance to next point")
            self.spot_df = self.preprocessor.add_distance_to_next_point(inplace=True)
        if specific_steps is None or "compute_speed" in specific_steps:
            print("Computing speed")
            self.spot_df = self.preprocessor.compute_speed(inplace=True)
        if specific_steps is None or "sort_by_track_and_frame" in specific_steps:
            print("Sorting by TRACK_ID and FRAME")
            self.spot_df = self.preprocessor.sort_by_track_and_frame(inplace=True)
        self._propagate_spot_df_changes()

    def label_tracks(self):
        labels = []
        for track_id, group in self.spot_df.groupby("TRACK_ID"):
            track_pos = group[["POSITION_X", "POSITION_Y"]].values
            track_labels = self.run_tumble_labeler.label(track_pos)
            labels.extend(track_labels)
        self.spot_df[RUN_TUMBLE_LABEL] = labels
        # Create run-tumble index: increment on each transition
        run_tumble_indices = []
        for track_id, group in self.spot_df.groupby("TRACK_ID"):
            track_labels = group[RUN_TUMBLE_LABEL].values
            indices = [0]
            current_index = 0
            for i in range(1, len(track_labels)):
                if track_labels[i] != track_labels[i - 1]:
                    current_index += 1
                indices.append(current_index)
            run_tumble_indices.extend(indices)

        self.spot_df[RUN_TUMBLE_INDEX] = run_tumble_indices
        self.spot_df[RUN_TUMBLE] = (
            self.spot_df[RUN_TUMBLE_LABEL]
            + "_"
            + self.spot_df[RUN_TUMBLE_INDEX].astype(str)
        )
        self._propagate_spot_df_changes()

    def plots(self):
        self.plotter.plot_distribution_nspots(
            self.config.threshold_nspots, show_plot=False
        )
        self.plotter.plot_track_start_zero()

    def compute_metrics(self):
        n_tracks = self.spot_df.TRACK_ID.nunique()
        n_spots = len(self.spot_df)
        n_tracks_with_more_than_100_spots = (
            self.spot_df.TRACK_ID.value_counts() > 100
        ).sum()
        mean_speed = self.spot_df[SPEED].mean()

        self.metrics = pd.Series(
            {
                "n_tracks": n_tracks,
                "n_spots": n_spots,
                "n_tracks_with_more_than_100_spots": n_tracks_with_more_than_100_spots,
                "mean_speed": mean_speed,
            }
        )

    def print_stats(self):
        # Recompute metrics
        self.compute_metrics()
        print(f"Number of tracks: {self.metrics.n_tracks}")
        print(
            f"Number of tracks with more than 100 spots: {self.metrics.n_tracks_with_more_than_100_spots}"
        )
        print(f"Number of spots: {self.metrics.n_spots}")
        print(f"Mean speed: {self.metrics.mean_speed:.2f} µm/s")
