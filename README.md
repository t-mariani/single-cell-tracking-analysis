# single-cell-tracking-analysis
Path analysis of bacteries after single cell tracking with microscope 

## Data aquisition :
* Microscopy imaging of bacteria in different media (agarose pad with different nutrient concentration, semi-liquid media)
* Single-cell tracking data obtained using TrackMate (Fiji plugin) from microscopy imaging videos.

## Repository purpose :
* Process single-cell tracking data to extract meaningful metrics about bacterial motility.
* Provide visualizations of bacterial trajectories and motility patterns.
* Facilitate comparison of bacterial behavior under different environmental conditions (media, optical density, etc.)

## Usage :
1. Clone the repository
2. Install dependencies (see Dependencies section)
3. Prepare your data following the assumptions below
4. Define a `config` (see config.py for available options)
5. Run the data processor on your experiment folder :
```python
from data_loader import DataLoader
from data_processor import DataProcessor

data_loader = DataLoader(experiment_folder="path/to/your/experiment_folder") # example : "data/LB_100_0P8_1"
spot_df = data_loader.load_spot_data()
dp = DataProcessor(spot_df, config=config)
dp.preprocess() # apply preprocessing steps
dp.label_tracks() # apply run-tumble labeling
dp.print_stats() # print basic stats
dp.plotter.plot_track_start_zero() # plot all trajectories on a single plot, see Plotter class for more plotting options
```

Several Jupyter notebooks are provided to guide you through specific analyses :
* [determine_run_tumble_label.ipynb](determine_run_tumble_label.ipynb) : help to choose run-tumble labeling parameters
* [media_comparison.ipynb](media_comparison.ipynb) : compare different experiments when medium changes with optical density fixed using the same analysis pipeline
* [od_comparison.ipynb](od_comparison.ipynb) : compare different experiments whith fixed media and varying optical density using the same analysis pipeline

All these notebbok arleady have some experiments loaded as example with insightful visualizations.


## Assumptions : 
* Naming convention of data folders and files : 
    * Folder : {medium}_*_{optical_density}_{run_number} with optical density written as string (0P2 for 0.2)
    * File : _allspots.csv 
* Constants defined in constants.py : time interval between frames and micron per pixel ratio
* "Run" can have 2 significations in this repo : Do not mix them up !
    * "run" when it's in the context of experiment : repeat of the same experiment (same medium, same optical density)
    * "run" in context of track labeling : kinda linear segment, not a tumble 
* The spot data csv file has the expected format (best ooptions : exported by TrackMate (Fiji plugin)). 
Or at least contains the expected columns : 
    * TRACK_ID : unique identifier of a track
    * POSITION_X : X coordinate of a spot (pixel unit)
    * POSITION_Y : Y coordinate of a spot (pixel unit)
    * FRAME : frame number of a spot


## Dependencies :
Easieast is to use pixi 
First [install pixi](https://pixi.sh/latest/installation/) if you don't have it yet. 
Then run :
```bash
pixi install
```

Otherwise find in [pixi.toml](pixi.toml) the list of dependencies and install them manually using pip.