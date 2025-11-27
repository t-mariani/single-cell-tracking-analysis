from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from constants import RUN_TUMBLE_LABEL, SPEED


class Plotter:
    def __init__(self, spot_df):
        self.spot_df = spot_df

    def plot_distribution_nspots(self, threshold_nspots, ax=None, show_plot=True):
        """Plot distribution of nspots with threshold line"""
        if ax is None:
            fig, ax = plt.subplots()
        nspots = self.spot_df.TRACK_ID.value_counts().values
        ax.hist(nspots, bins=50, alpha=0.7)
        ax.axvline(threshold_nspots, color="r", linestyle="dashed", linewidth=2)
        ax.set_xlabel("Number of spots")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Number of Spots per Track")
        if show_plot:
            plt.show()
        return ax

    def plot_track_start_zero(self):
        """Plot all tracks with starting point at (0,0)"""
        for track_id, group in self.spot_df.groupby("TRACK_ID"):
            pos = group[["POSITION_X", "POSITION_Y"]].to_numpy()
            pos = pos - pos[0]  # shift to start at (0,0)
            plt.plot(pos[:, 0], pos[:, 1], "x-", alpha=0.1)
        plt.xlabel("X Position (relative)")
        plt.ylabel("Y Position (relative)")
        plt.title("All Tracks with Start at (0,0)")
        plt.show()

    def plot_distribution_speed(
        self,
        split_tracks=False,
        log=False,
        select_run=False,
        ax=None,
        legend=None,
        show_plot=True,
    ):
        """Plot distribution of speeds"""
        assert (
            SPEED in self.spot_df.columns
        ), "Speed column not found. Please compute speed first."
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        if select_run:
            speeds = self.spot_df[self.spot_df[RUN_TUMBLE_LABEL] == "run"][SPEED].values
        else:
            speeds = self.spot_df[SPEED].values
        bins = 100
        if split_tracks:
            unique_tracks = self.spot_df.TRACK_ID.unique()
            for track_id in unique_tracks:
                track_speeds = self.spot_df[self.spot_df.TRACK_ID == track_id][
                    SPEED
                ].values
                ax.hist(
                    track_speeds,
                    bins=bins,
                    alpha=0.5,
                    histtype="step",
                    cumulative=-1,
                    label=f"Track {track_id}",
                )
            ax.legend()
        else:
            ax.hist(
                speeds,
                bins=bins,
                alpha=0.7,
                density=True,
                histtype="step",
                cumulative=-1,
                label=legend,
            )
        ax.set_xlabel("Speed (µm/s)")
        ax.set_ylabel(
            f'Cumulative Frequency {"(log scale)" if log else "(linear scale)"}'
        )
        if log:
            ax.set_yscale("log")
        ax.set_title(
            f'Distribution of Speeds per Track {("(Runs only)" if select_run else "")}'
        )
        if show_plot:
            plt.show()

        return ax

    def plot_distribution_duration(
        self, group: Literal["run", "tumble"], ax=None, show_plot=True
    ):
        pass

    def plot_run_tumble(self, track_id, colorize_tumbles=False):
        """Plot trajectory of a single track with run-tumble labels.

        track_id : ID of the track to plot
        labels : array of "run" "tumble" labels for the track
        colorize_tumbles : if True, color each tumble differently
        """
        track_data = self.spot_df[self.spot_df.TRACK_ID == track_id]
        x = track_data.POSITION_X.values
        y = track_data.POSITION_Y.values
        labels = track_data[RUN_TUMBLE_LABEL].values

        if colorize_tumbles:
            unique_tumbles = np.unique(labels[labels == "tumble"])
            colors = plt.cm.get_cmap("hsv", len(unique_tumbles) + 1)
            tumble_color_map = {
                tumble: colors(i) for i, tumble in enumerate(unique_tumbles)
            }

        for i in range(len(x) - 1):
            if labels[i] == "tumble":  # Tumble
                if colorize_tumbles:
                    plt.plot(
                        x[i : i + 2],
                        y[i : i + 2],
                        "x-",
                        color=tumble_color_map[labels[i]],
                        linewidth=2,
                    )
                else:
                    plt.plot(x[i : i + 2], y[i : i + 2], "x-", color="red", linewidth=2)
            else:  # Run
                plt.plot(x[i : i + 2], y[i : i + 2], "x-", color="blue", linewidth=1)

        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title(f"Trajectory of Track ID {track_id}")
        plt.show()
