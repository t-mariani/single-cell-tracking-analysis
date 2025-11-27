from dataclasses import dataclass, field


@dataclass
class Config:
    threshold_nspots: int = 5
    labeling_method: str = "angle"  # "angle", "speed", "combined"
    threshold_angle: float = 30.0  # degrees
    threshold_speed: float = 2.0  # distance units
    combination_rule: str = "and"  # "and", "or"
    min_size_cluster: int = 2  # minimum size of cluster to be considered a tumble
    preprocess_to_apply_default: list = field(
        default_factory=lambda: [
            "interpolate_missing_frames",
            "add_distance_to_next_point",
            "sort_by_track_and_frame",
            "compute_speed",
        ]
    )  # e.g. ["filter_nspots", "interpolate_missing_frames", "add_distance_to_next_point", "sort_by_track_and_frame", "compute_speed"]
