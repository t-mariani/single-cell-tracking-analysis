from dataclasses import dataclass, field


@dataclass
class Config:
    # Preprocess
    threshold_nspots: int = 5  # Minimum number of spots per track to keep the track

    # Run-tumble labeling
    labeling_method: str = "angle"  # "angle", "speed", "combined"
    threshold_angle: float = (
        30.0  # degrees : angle change threshold, >= which a tumble is detected
    )
    threshold_speed: float = (
        2.0  # pixel units # TODO allow micron units ? speed threshold, <= which a tumble is detected
    )
    min_size_cluster: int = 2  # minimum size of cluster to be considered a tumble
    combination_rule: str = "and"  # "and", "or" between angle and speed criteria

    # Preprocessing steps to apply when intializing DataProcessor
    preprocess_to_apply_default: list = field(
        default_factory=lambda: [
            "interpolate_missing_frames",
            "add_distance_to_next_point",
            "sort_by_track_and_frame",
            "compute_speed",
        ]
    )  # e.g. ["filter_nspots", "interpolate_missing_frames", "add_distance_to_next_point", "sort_by_track_and_frame", "compute_speed"]
