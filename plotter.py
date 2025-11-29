from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from constants import RUN_TUMBLE, RUN_TUMBLE_LABEL, SPEED, RATIO_MICRON_PER_PIXEL
from utils import compute_angle_single


class Plotter:
    """
    Plotter class for visualizing tracking data and analysis results.
    Initialized with a spot DataFrame containing tracking data.

    This class handles : visualizations for a single experiment's spot DataFrame.
    2 types of plots are available:
    - Summary statistics plots (distributions of speeds, durations, etc.)
    - Trajectory plots (individual track trajectories with run-tumble labels, direction changes, etc
    """

    def __init__(self, spot_df):
        self.spot_df = spot_df

    def plot_distribution_nspots(self, threshold_nspots=None, ax=None, show_plot=True):
        """Plot distribution of nspots with threshold line
        x-axis : Number of spots inside a track = Duration of the track due to interpolate_missing_frames preprocess
        y-axis : Frequency

        Parameters
        ----------
        threshold_nspots : int, optional
            If provided, draw a vertical line at this threshold.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, create a new figure and axes.
        show_plot : bool
            If True, display the plot immediately.
        """
        if ax is None:
            fig, ax = plt.subplots()

        nspots = self.spot_df.TRACK_ID.value_counts().values
        ax.hist(nspots, bins=50, alpha=0.7)

        if threshold_nspots is not None:
            ax.axvline(threshold_nspots, color="r", linestyle="dashed", linewidth=2)

        ax.set_xlabel("Number of spots")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Number of Spots per Track")
        if show_plot:
            plt.show()
        return ax

    def plot_distribution_speed(
        self,
        log=False,
        select: Literal["run", "tumble"] = None,
        ax=None,
        legend=None,
        show_plot=True,
    ):
        """Plot cumulative distribution of speeds
        x-axis : Speed (µm/s)
        y-axis : Cumulative Frequency (log or linear scale)

        Parameters
        ----------
        log : bool
            If True, use logarithmic scale for y-axis.
        select : Literal["run", "tumble"]
            If "run", plot speeds for runs only. If "tumble", plot speeds for tumbles only.
            If False or None, plot speeds for all data.
        legend : str
            Legend label for the plot.
        ax : matplotlib.axes.Axes
            Axes to plot on. If None, create a new figure and axes.
        show_plot : bool
            If True, display the plot immediately.
        """

        assert (
            SPEED in self.spot_df.columns
        ), "Speed column not found. Please compute speed first."
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        if select:
            speeds = self.spot_df[self.spot_df[RUN_TUMBLE_LABEL] == select][
                SPEED
            ].values
        else:
            speeds = self.spot_df[SPEED].values

        ax.hist(
            speeds,
            bins=100,
            alpha=0.7,
            density=True,
            histtype="step",
            cumulative=-1,  # Cumulative from the right
            label=legend,
        )
        ax.set_xlabel("Speed (µm/s)")
        ax.set_ylabel(
            f'Cumulative Frequency {"(log scale)" if log else "(linear scale)"}'
        )
        if log:
            ax.set_yscale("log")
        ax.set_title(
            f'Distribution of Speeds per Track {("(Runs only)" if select == "run" else ("(Tumbles only)" if select == "tumble" else ""))}'
        )

        if show_plot:
            plt.show()
        return ax

    def plot_distribution_duration(
        self,
        group: Literal["run", "tumble"],
        legend=None,
        density=False,
        ax=None,
        show_plot=True,
    ):
        """Plot distribution of durations of runs or tumbles
        x-axis : Duration (number of frames)
        y-axis : Frequency

        Parameters
        ----------
        group : Literal["run", "tumble"]
            Specify whether to plot durations for "run" or "tumble" segments.
        legend : str
            Legend label for the plot.
        density : bool
            If True, plot density instead of frequency.
        ax : matplotlib.axes.Axes
            Axes to plot on. If None, create a new figure and axes.
        show_plot : bool
            If True, display the plot immediately.
        """

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        self.spot_df["TRACK_ID_RUN_TUMBLE"] = (
            self.spot_df.TRACK_ID.astype(str) + "_" + self.spot_df[RUN_TUMBLE]
        )
        if group == "run":
            durations = (
                self.spot_df[self.spot_df[RUN_TUMBLE_LABEL] == "run"]
                .groupby("TRACK_ID_RUN_TUMBLE")
                .size()
                .values
            )
        elif group == "tumble":
            durations = (
                self.spot_df[self.spot_df[RUN_TUMBLE_LABEL] == "tumble"]
                .groupby("TRACK_ID_RUN_TUMBLE")
                .size()
                .values
            )
        else:
            raise ValueError(f"Unknown group: {group}")

        ax.hist(
            durations,
            bins=200,
            alpha=0.7,
            histtype="step",
            density=density,
            cumulative=-1,
            label=legend,
        )
        ax.set_xlabel("Duration (number of frames)")
        ax.set_ylabel("Cumulative Frequency")
        ax.set_title(f"Distribution of {group.capitalize()} durations")

        if show_plot:
            plt.show()
        return ax

    def plot_change_direction_inside_run(self, ax=None, show_plot=True):
        """Plot change in direction inside runs (angle between segments) vs run length"""

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        self.spot_df["TRACK_ID_RUN_TUMBLE"] = (
            self.spot_df.TRACK_ID.astype(str) + "_" + self.spot_df[RUN_TUMBLE]
        )
        for track_run_id, group in self.spot_df.groupby("TRACK_ID_RUN_TUMBLE"):
            if group[RUN_TUMBLE_LABEL].iloc[0] == "run":
                positions = group[["POSITION_X", "POSITION_Y"]].values
                if len(positions) < 3:
                    continue
                start = positions[0]
                end = positions[-1]
                mid_index = len(positions) // 2
                mid = positions[mid_index]

                vec_start_mid = mid - start
                vec_mid_end = end - mid

                angle = compute_angle_single(vec_start_mid, vec_mid_end)
                ax.plot(angle, len(positions), "x", alpha=0.5)

        ax.set_xlabel("Change in Direction (degrees)")
        ax.set_ylabel("Run Length (number of frames)")
        ax.set_title("Change in Direction Inside Runs vs Run Length")
        if show_plot:
            plt.show()
        return ax

    def plot_change_direction_between_runs(
        self, filter_run_longer=None, legend=None, ax=None, show_plot=True
    ):
        """Plot change in direction between consecutive runs (angle between runs) vs combined run length

        Parameters
        ----------
        filter_run_longer : int, optional
            If set, only consider runs longer than this number of frames.
        legend : str
            Legend label for the plot.
        ax : matplotlib.axes.Axes
            Axes to plot on. If None, create a new figure and axes.
        show_plot : bool
            If True, display the plot immediately.
        """

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        angles = []
        for track_id, group in self.spot_df.groupby("TRACK_ID"):
            run_df = group[group[RUN_TUMBLE_LABEL] == "run"].copy()

            runs = run_df.groupby(RUN_TUMBLE)
            if filter_run_longer is not None:
                runs = runs.filter(lambda x: len(x) >= filter_run_longer).groupby(
                    RUN_TUMBLE
                )

            runs_names = list(runs.groups.keys())
            runs_names.sort()  # VERY IMPORTANT TO HAVE RUNS IN ORDER
            for i in range(len(runs) - 1):
                run1 = runs.get_group(runs_names[i])
                run2 = runs.get_group(runs_names[i + 1])
                pos1_start = run1[["POSITION_X", "POSITION_Y"]].values[0]
                pos1_end = run1[["POSITION_X", "POSITION_Y"]].values[-1]
                pos2_start = run2[["POSITION_X", "POSITION_Y"]].values[0]
                pos2_end = run2[["POSITION_X", "POSITION_Y"]].values[-1]

                vec_run1 = pos1_end - pos1_start
                vec_run2 = pos2_end - pos2_start

                angles.append(compute_angle_single(vec_run1, vec_run2))

        ax.hist(angles, bins=18, alpha=0.7, histtype="step", label=legend, density=True)
        ax.set_xlabel("Change in Direction Between Runs (degrees)")
        ax.set_ylabel("Frequency")
        ax.set_title("Change in Direction Between Consecutive Runs")
        if show_plot:
            plt.show()
        return ax

    ## -----
    ## Trajectories plotting

    def plot_track_start_zero(self):
        """Plot all tracks with starting point at (0,0)"""
        for track_id, group in self.spot_df.groupby("TRACK_ID"):
            pos = group[["POSITION_X", "POSITION_Y"]].to_numpy()
            pos = pos - pos[0]  # shift to start at (0,0)
            plt.plot(pos[:, 0], pos[:, 1], "x-", alpha=0.1)
        plt.axis("scaled")
        plt.xlabel("X Position (relative)")
        plt.ylabel("Y Position (relative)")
        plt.title("All Tracks with Start at (0,0)")
        plt.show()

    def plot_trajectory_change_direction(
        self, track_id, filter_run_longer=None, ax=None, show_plot=True
    ):
        """Plot trajectory of a single track with change in direction between runs indicated.

        track_id : ID of the track to plot
        filter_run_longer : if set, only consider runs longer than this number of frames
        """
        track_data = self.spot_df[self.spot_df.TRACK_ID == track_id]
        x = track_data.POSITION_X.values
        y = track_data.POSITION_Y.values

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(x, y, "x-", color="gray", alpha=0.5)

        run_df = track_data[track_data[RUN_TUMBLE_LABEL] == "run"].copy()
        runs = run_df.groupby(RUN_TUMBLE)
        if filter_run_longer is not None:
            runs = runs.filter(lambda x: len(x) >= filter_run_longer).groupby(
                RUN_TUMBLE
            )

        runs_names = list(runs.groups.keys())
        runs_names.sort()  # VERY IMPORTANT TO HAVE RUNS IN ORDER
        for i in range(len(runs) - 1):
            run1 = runs.get_group(runs_names[i])
            run2 = runs.get_group(runs_names[i + 1])
            pos1_start = run1[["POSITION_X", "POSITION_Y"]].values[0]
            pos1_end = run1[["POSITION_X", "POSITION_Y"]].values[-1]
            pos2_start = run2[["POSITION_X", "POSITION_Y"]].values[0]
            pos2_end = run2[["POSITION_X", "POSITION_Y"]].values[-1]

            vec_run1 = pos1_end - pos1_start
            vec_run2 = pos2_end - pos2_start

            angle = compute_angle_single(vec_run1, vec_run2)

            ax.plot(
                [pos1_end[0], pos2_start[0]],
                [pos1_end[1], pos2_start[1]],
                "b--",
                linewidth=1,
            )

            ax.plot(
                [pos1_start[0], pos1_end[0]],
                [pos1_start[1], pos1_end[1]],
                "g-",
                linewidth=2,
            )
            ax.plot(
                [pos2_start[0], pos2_end[0]],
                [pos2_start[1], pos2_end[1]],
                "g-",
                linewidth=2,
            )

            ax.text(
                pos1_end[0],
                pos1_end[1],
                f"{angle:.1f}°",
                color="red",
                fontsize=8,
                weight="bold",
            )

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title(f"Trajectory of Track ID {track_id} with Direction Changes")
        if show_plot:
            plt.show()
        return ax

    def plot_run_tumble(
        self,
        track_id,
        colorize_tumbles=False,
        real_distance=False,
        ax=None,
        show_plot=True,
    ):
        """Plot trajectory of a single track with run-tumble labels.

        track_id : ID of the track to plot
        labels : array of "run" "tumble" labels for the track
        colorize_tumbles : if True, color each tumble differently
        """
        if ax is None:
            fig, ax = plt.subplots()

        track_data = self.spot_df[self.spot_df.TRACK_ID == track_id]
        if real_distance:
            x = track_data.POSITION_X.values * RATIO_MICRON_PER_PIXEL
            y = track_data.POSITION_Y.values * RATIO_MICRON_PER_PIXEL
        else:
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
                    ax.plot(
                        x[i : i + 2],
                        y[i : i + 2],
                        "x-",
                        color=tumble_color_map[labels[i]],
                        linewidth=2,
                    )
                else:
                    ax.plot(x[i : i + 2], y[i : i + 2], "x-", color="red", linewidth=2)
            else:  # Run
                ax.plot(x[i : i + 2], y[i : i + 2], "x-", color="blue", linewidth=1)

        ax.axis("scaled")
        ax.set_xlabel(f"X Position {'(µm)' if real_distance else ''}")
        ax.set_ylabel(f"Y Position {'(µm)' if real_distance else ''}")
        ax.set_title(f"Trajectory of Track ID {track_id} with Run-Tumble Labels")
        if show_plot:
            plt.show()
