from functools import partial
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from config import Config
from data_processor import DataProcessor
from dataloader import DataLoader
from plotter import Plotter
from constants import (
    PLOT_SPEED_DISTRIBUTION,
    PLOT_CHANGE_DIRECTION_BETWEEN_RUN,
    PLOT_DISTRIBUTION_DURATION,
)


class DataComparison:
    """
    Class to compare multiple experiments' data.
    Methods to load, process, and plot comparisons between experiments.

    Parameters
    ----------
    list_experiment : list[str]
        List of experiment folder names without the run suffix (_1, _2, ...)
    config : Config
        Configuration object with processing parameters
    pre_path : str or Path, optional
        Prefix path where experiment folders are located, by default "data" because of git repository structure
    """

    def __init__(self, list_experiment: list[str], config: Config, pre_path="data"):
        self.config = config
        self.list_experiment = list_experiment

        if pre_path is None:
            self.pre_path = Path("")
        else:
            self.pre_path = Path(pre_path)

    def get_path_experiments(self) -> list[list[Path]]:
        """
        Get list of paths for each experiment, handling multiple runs with suffix _1, _2, ...

        Returns
        -------
        paths: list[list[Path]]
            List of list of Paths for each experiment
            Example: [[Path('data/exp1_1'), Path('data/exp1_2')], [Path('data/exp2')]]
        """

        paths = []
        for experiment in self.list_experiment:
            if not (self.pre_path / experiment).exists():
                # check for multiple runs with suffix _1, _2, ...
                i = 1
                runs = []
                path = self.pre_path / (experiment + f"_{i}")
                while path.exists():
                    runs.append(path)
                    i += 1
                    path = self.pre_path / (experiment + f"_{i}")

                if len(runs) == 0:
                    raise FileNotFoundError(
                        f"Experiment path {self.pre_path / experiment} does not exist, and no runs with suffix _1, _2, ... found."
                    )
                paths.append(runs)
            else:
                paths.append([Path(self.pre_path / experiment)])
        return paths

    def process_all(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process all experiments by applying DataProcessor to each run.
        Returns concatenated results DataFrame and metrics DataFrame.
        Side effect: store list of DataProcessor instances in self.data_processor_list
        and results in self.results_df and self.results_metrics.
        """

        results_df = pd.DataFrame()
        results_metrics = pd.DataFrame()
        self.data_processor_list: dict[str, DataProcessor] = {}

        for experiment_runs, experiment_name in zip(
            self.get_path_experiments(), self.list_experiment
        ):
            experiment_df = []
            experiment_metrics = []
            run_data_processors = []
            for experiment_path in experiment_runs:
                print(f"⚙️ Processing experiment at {experiment_path}")
                data_loader = DataLoader(experiment_path)
                try:
                    spot_df = data_loader.load_spot_data()
                except FileNotFoundError:
                    print(f" No '_allspots.csv' found in provided path")
                    continue
                data_processor = DataProcessor(
                    spot_df, self.config, str(experiment_path)
                )
                data_processor.preprocess()
                data_processor.label_tracks()
                run_data_processors.append(data_processor)

                # Get spot_df
                df = data_processor.spot_df.copy()
                df["experiment_name"] = experiment_name
                df["experiment_path"] = str(experiment_path)
                experiment_df.append(df)

                # Get metrics
                data_processor.compute_metrics()
                metrics = data_processor.metrics.copy()
                # Add experiment info to metrics
                metrics["experiment_name"] = experiment_name
                metrics["experiment_path"] = str(experiment_path)
                metrics["medium"] = experiment_name.split("_")[0]
                try:
                    metrics["OD"] = float(
                        experiment_name.split("_")[-1].replace("P", ".")
                    )
                except ValueError:
                    pass
                experiment_metrics.append(metrics)
            experiment_df = pd.concat(experiment_df).reset_index(drop=True)
            experiment_metrics = pd.DataFrame(experiment_metrics)
            results_df = pd.concat([results_df, experiment_df], ignore_index=True)
            results_metrics = pd.concat(
                [results_metrics, experiment_metrics], ignore_index=True
            )

            self.data_processor_list[experiment_name] = run_data_processors

        self.results_df = results_df
        self.results_metrics = results_metrics

        return results_df, results_metrics

    def plot(
        self,
        type_plot,
        aggregate_runs=False,
        aggregate_media=False,
        show_plot=True,
        **kwargs,
    ) -> None:
        """
        Plot comparison between experiments for a given plot type.

        Parameters
        ----------
        type_plot : String from constants.py starting with PLOT_
            Type of plot to generate, linked to Plotter methods.
        aggregate_runs : bool, optional
            Whether to aggregate all runs/trials of an experiment into a single line/ditribution, by default False
        aggregate_media : bool, optional
            Whether to aggregate all experiments of the same medium into a single line/distribution, by default False
        show_plot : bool, optional
            Whether to show the plot immediately, by default True
        **kwargs : dict
            Additional keyword arguments to pass to the plotting functions.
            See Plotter methods for available arguments depending on type_plot.
        """

        # Use partial to create a function with fixed arguments
        # plot_function will be a function that takes (object_plotter, ax, show_plot, legend, **kwargs)
        # and corresponds to the type_plot Plotter method
        def helper_plot(object_plotter, function_name, **kwargs):
            plot_function = getattr(object_plotter, function_name)
            return plot_function(**kwargs)

        if type_plot == PLOT_SPEED_DISTRIBUTION:
            plot_function = partial(
                helper_plot, function_name="plot_distribution_speed"
            )
        elif type_plot == PLOT_CHANGE_DIRECTION_BETWEEN_RUN:
            plot_function = partial(
                helper_plot,
                function_name="plot_change_direction_between_runs",
            )
        elif type_plot == PLOT_DISTRIBUTION_DURATION:
            plot_function = partial(
                helper_plot,
                function_name="plot_distribution_duration",
            )
        else:
            raise ValueError(f"Unknown plot type: {type_plot}")

        if aggregate_media:
            # Refactor data_processor_list to aggregate by medium
            env_dict: dict[str, list[DataProcessor]] = {}
            for name, dps in self.data_processor_list.items():
                env = name.split("_")[0]
                if env not in env_dict:
                    env_dict[env] = []
                env_dict[env].extend(dps)
            data_processor_list = env_dict
            # Equal to aggregate_runs but by medium
            aggregate_runs = True
        else:
            data_processor_list = self.data_processor_list

        for i, (name, dps) in enumerate(data_processor_list.items()):
            print(f"Plotting {type_plot} for experiment: {name} with {len(dps)} runs")
            if i == 0:  # Find ax to plot all on the same
                if aggregate_runs:
                    big_df = pd.concat(
                        [dp.spot_df for dp in dps], keys=[dp.name for dp in dps]
                    )
                    # TODO : for PLOT_SPEED_DISTRIBUTION TRACK_ID is not unique anymore, need to fix
                    # big_df["TRACK_ID"] = (
                    #     big_df.index.get_level_values(1) + "_" + big_df["TRACK_ID"]
                    # )
                    ax = plot_function(
                        Plotter(big_df), ax=None, show_plot=False, legend=name, **kwargs
                    )
                else:
                    ax = plot_function(
                        dps[0].plotter,
                        ax=None,
                        show_plot=False,
                        legend=dps[0].name,
                        **kwargs,
                    )
                    for data_processor in dps[1:]:
                        plot_function(
                            data_processor.plotter,
                            ax=ax,
                            show_plot=False,
                            legend=data_processor.name,
                            **kwargs,
                        )

            else:
                if aggregate_runs:
                    big_df = pd.concat([dp.spot_df for dp in dps])
                    plot_function(
                        Plotter(big_df), ax=ax, show_plot=False, legend=name, **kwargs
                    )
                else:
                    for data_processor in dps:
                        plot_function(
                            data_processor.plotter,
                            ax=ax,
                            show_plot=False,
                            legend=data_processor.name,
                            **kwargs,
                        )

        plt.legend()
        if show_plot:
            plt.show()

    ### Comparison-specific plots
    # Plot aggegated metrics that are not at the experiment level but at the comparison level

    def plot_all_speed_Od(self, show_plot=True, **kwargs):
        """plot self.results_metrics mean_speed, mean_speed_run, mean_speed_tumble vs OD"""
        ax = self.plot_speed_Od(select="average", show_plot=False, **kwargs)
        self.plot_speed_Od(select="run", ax=ax, show_plot=False, **kwargs)
        self.plot_speed_Od(select="tumble", ax=ax, show_plot=show_plot, **kwargs)
        plt.title("Mean Speed vs OD")

    def plot_speed_Od(self, select="average", ax=None, show_plot=None, **kwargs):
        """plot self.results_metrics mean_speed of select vs OD"""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        if select == "run":
            df_plot = self.results_metrics[["OD", "mean_speed_run"]].copy()
            y_col = "mean_speed_run"
        elif select == "tumble":
            df_plot = self.results_metrics[["OD", "mean_speed_tumble"]].copy()
            y_col = "mean_speed_tumble"
        else:
            df_plot = self.results_metrics[["OD", "mean_speed"]].copy()
            y_col = "mean_speed"

        # Group by OD and plot with error bars
        grouped = df_plot.groupby("OD")[y_col].agg(["mean", "std"])
        ax.errorbar(
            grouped.index,
            grouped["mean"],
            yerr=grouped["std"],
            fmt="o-",
            capsize=5,
            label=select,
            **kwargs,
        )
        ax.scatter(df_plot["OD"], df_plot[y_col], alpha=0.3)
        ax.set_xlabel("OD")
        ax.set_ylabel("Mean speed (µm/s)")
        ax.set_title(
            f"Mean speed {"(Run only)" if select == "run" else ("(Tumble only)" if select == "tumble" else "")} vs OD"
        )
        plt.legend()
        if show_plot:
            plt.show()
        return ax
