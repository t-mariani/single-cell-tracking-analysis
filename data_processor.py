import pandas as pd

from preprocessor import Preprocessor
from run_tumble_labeler import RunTumbleLabeler
from plotter import Plotter
from constants import RUN_TUMBLE_LABEL, RUN_TUMBLE_INDEX, RUN_TUMBLE, SPEED


class DataProcessor:
    """
    Class to handle data processing for a single experiment.

    Attributes
    ----------
    name : str
        Name of the experiment.
    spot_df : pd.DataFrame
        DataFrame containing spot data.
    config : Config
        Configuration object with processing parameters.
    plotter : Plotter
        Plotter object for generating plots.
    preprocessor : Preprocessor
        Preprocessor object for data preprocessing.
    run_tumble_labeler : RunTumbleLabeler
        RunTumbleLabeler object for labeling runs and tumbles.
    list_longest : list
        List of TRACK_IDs of the longest tracks.

    """

    def __init__(self, spot_df, config, name=None):
        self.name = name
        self.spot_df = spot_df
        self.config = config
        self.plotter = Plotter(spot_df)
        self.preprocessor = Preprocessor(config)
        self.run_tumble_labeler = RunTumbleLabeler(config=config)
        self.list_longest = spot_df.TRACK_ID.value_counts().index.tolist()[:10]
        self.is_track_labeled = False
        if self.config.preprocess_to_apply_default:
            self.preprocess(self.config.preprocess_to_apply_default)
        self.print_stats()

    def _propagate_spot_df_changes(self):
        """As spot_df is modified in DataProcessor, propagate changes to Plotter"""
        self.plotter.spot_df = self.spot_df

    def preprocess(self, specific_steps=None):
        """
        Preprocess spot_df using Preprocessor methods.
        If specific_steps is None, apply all default preprocessing steps in order.
        Otherwise, apply only the specified steps in the given order.
        """

        if specific_steps is None or "filter_nspots" in specific_steps:
            print("Filtering tracks with nspots <", self.config.threshold_nspots)
            self.spot_df = self.preprocessor.filter_tracks_nspots(self.spot_df)
        if specific_steps is None or "interpolate_missing_frames" in specific_steps:
            print("Interpolating missing frames")
            self.spot_df = self.preprocessor.interpolate_missing_frames(self.spot_df)
        if specific_steps is None or "add_distance_to_next_point" in specific_steps:
            print("Adding distance to next point")
            self.spot_df = self.preprocessor.add_distance_to_next_point(self.spot_df)
        if specific_steps is None or "compute_speed" in specific_steps:
            print("Computing speed")
            self.spot_df = self.preprocessor.compute_speed(self.spot_df)
        if specific_steps is None or "sort_by_track_and_frame" in specific_steps:
            print("Sorting by TRACK_ID and FRAME")
            self.spot_df = self.preprocessor.sort_by_track_and_frame(self.spot_df)

        self._propagate_spot_df_changes()

    def label_tracks(self):
        """
        Label tracks into runs and tumbles using RunTumbleLabeler
        """

        self.is_track_labeled = True
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
                indices.append(f"{current_index:04d}")
            run_tumble_indices.extend(indices)

        self.spot_df[RUN_TUMBLE_INDEX] = run_tumble_indices

        # Create combined RUN_TUMBLE column
        self.spot_df[RUN_TUMBLE] = (
            self.spot_df[RUN_TUMBLE_LABEL]
            + "_"
            + self.spot_df[RUN_TUMBLE_INDEX].astype(str)
        )

        self._propagate_spot_df_changes()

    def compute_metrics(self):
        """Compute basic metrics and store in self.metrics as a pd.Series"""

        n_tracks = self.spot_df.TRACK_ID.nunique()
        n_spots = len(self.spot_df)
        n_tracks_with_more_than_100_spots = (
            self.spot_df.TRACK_ID.value_counts() > 100
        ).sum()
        try:
            mean_speed = self.spot_df[SPEED].mean()
        except KeyError:  # If preprocessing step to compute speed not done
            mean_speed = float("nan")

        if self.is_track_labeled:
            n_runs = self.spot_df[RUN_TUMBLE].nunique()
            mean_speed_tumble = self.spot_df[
                self.spot_df[RUN_TUMBLE_LABEL] == "tumble"
            ][SPEED].mean()
            mean_speed_run = self.spot_df[self.spot_df[RUN_TUMBLE_LABEL] == "run"][
                SPEED
            ].mean()

            self.metrics = pd.Series(
                {
                    "n_tracks": n_tracks,
                    "n_spots": n_spots,
                    "n_tracks_with_more_than_100_spots": n_tracks_with_more_than_100_spots,
                    "n_runs": n_runs,
                    "mean_speed": mean_speed,
                    "mean_speed_tumble": mean_speed_tumble,
                    "mean_speed_run": mean_speed_run,
                }
            )
        else:
            self.metrics = pd.Series(
                {
                    "n_tracks": n_tracks,
                    "n_spots": n_spots,
                    "n_tracks_with_more_than_100_spots": n_tracks_with_more_than_100_spots,
                    "mean_speed": mean_speed,
                }
            )

    def print_stats(self):
        # (Re)compute metrics
        self.compute_metrics()
        print("=== Experiment statistics ===")
        print(f"Number of tracks: {self.metrics.n_tracks}")
        print(
            f"Number of tracks with more than 100 spots: {self.metrics.n_tracks_with_more_than_100_spots}"
        )
        print(f"Number of spots: {self.metrics.n_spots}")
        print(f"Mean speed: {self.metrics.mean_speed:.2f} µm/s")
        if self.is_track_labeled:
            print(f"Number of runs: {self.metrics.n_runs}")
            print(
                f"Mean speed during tumble: {self.metrics.mean_speed_tumble:.2f} µm/s"
            )
            print(f"Mean speed during run: {self.metrics.mean_speed_run:.2f} µm/s")
