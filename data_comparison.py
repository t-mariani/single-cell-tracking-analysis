from typing import Literal
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from data_processor import DataProcessor
from dataloader import DataLoader
from plotter import Plotter

PLOT_SPEED_DISTRIBUTION = "PLOT_SPEED_DISTRIBUTION"


class DataComparison:
    def __init__(self, list_experiment, config, pre_path=None):
        self.config = config
        self.list_experiment = list_experiment

        if pre_path is None:
            self.pre_path = Path("")
        else:
            self.pre_path = Path(pre_path)

    def get_path_experiments(self):
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
                paths.append(runs)
            else:
                paths.append([Path(self.pre_path / experiment)])
        return paths

    def process_all(self):
        """Process all experiments and return combined spot_df and metrics DataFrame
        Side effect: store list of DataProcessor instances in self.data_processor_list
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
                print(f"Processing experiment at {experiment_path}")
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
                metrics["experiment_name"] = experiment_name
                metrics["experiment_path"] = str(experiment_path)
                experiment_metrics.append(metrics)
            experiment_df = pd.concat(experiment_df).reset_index(drop=True)
            experiment_metrics = pd.DataFrame(experiment_metrics)
            results_df = pd.concat([results_df, experiment_df], ignore_index=True)
            results_metrics = pd.concat(
                [results_metrics, experiment_metrics], ignore_index=True
            )

            self.data_processor_list[experiment_name] = run_data_processors

        return results_df, results_metrics

    def plot(
        self,
        type_plot: Literal["PLOT_SPEED_DISTRIBUTION"],
        aggregate_runs=False,
        **kwargs,
    ):
        if type_plot == PLOT_SPEED_DISTRIBUTION:

            for i, (name, dps) in enumerate(self.data_processor_list.items()):
                print(
                    f"Plotting speed distribution for experiment: {name} with {len(dps)} runs"
                )
                if i == 0:  # Find ax to plot all on the same
                    if aggregate_runs:
                        big_df = pd.concat([dp.spot_df for dp in dps])
                        ax = Plotter(big_df).plot_distribution_speed(
                            ax=None, show_plot=False, legend=name, **kwargs
                        )
                    else:
                        ax = dps[0].plotter.plot_distribution_speed(
                            ax=None, show_plot=False, legend=dps[0].name, **kwargs
                        )
                        for data_processor in dps[1:]:
                            data_processor.plotter.plot_distribution_speed(
                                ax=ax,
                                show_plot=False,
                                legend=data_processor.name,
                                **kwargs,
                            )

                else:
                    if aggregate_runs:
                        big_df = pd.concat([dp.spot_df for dp in dps])
                        Plotter(big_df).plot_distribution_speed(
                            ax=ax, show_plot=False, legend=name, **kwargs
                        )
                    else:
                        for data_processor in dps:
                            data_processor.plotter.plot_distribution_speed(
                                ax=ax,
                                show_plot=False,
                                legend=data_processor.name,
                                **kwargs,
                            )

            plt.legend()
            plt.show()

        else:
            raise ValueError(f"Unknown plot type: {type_plot}")

    def plot_all(self):
        list_ax = self.data_processor_list[0].plots()
        for data_processor in self.data_processor_list:
            data_processor.plots()
