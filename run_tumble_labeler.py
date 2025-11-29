import numpy as np


def compute_angle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute angle between successive edges defined by points (x[i],y[i])"""
    edge_x = x[1:] - x[:-1]  # edge[0] is the displacement to go from x[0] to x[1]
    edge_y = y[1:] - y[:-1]
    edge = np.array([edge_x, edge_y]).T  # shape N-1,2
    norm_edge = edge / (
        np.linalg.norm(edge, axis=1, keepdims=True) + 1e-7
    )  # Normalize for angle computing and avoid division by zero
    # compute dot product norm_edge[i] with norm_edge[i+1]
    dot_product = np.diag(norm_edge[1:] @ norm_edge[:-1].T)  # shape N-2
    angle = np.arccos(dot_product) / (np.pi) * 180
    return angle  # shape N-2


def get_clusters_labels(pos, d_threshold, min_size: int = 2):
    """Cluster positions based on distance threshold.
    pos : numpy array of shape (n_points, 2)
    d_threshold : distance threshold to define a new cluster
    min_size : minimum size of cluster to be kept, smaller clusters are labeled -1
    Return : numpy array of shape (n_points,) with cluster labels"""

    labels = [0]
    current_label = 0
    for i in range(1, len(pos)):
        dist = np.linalg.norm(pos[i] - pos[i - 1])

        if dist >= d_threshold:
            # close the previous cluster
            current_label += 1
        labels.append(current_label)
    labels = np.array(labels)
    for lab in np.unique(labels):
        mask = labels == lab
        if sum(mask) < min_size:
            labels[mask] = -1
    return labels


def barycenter_cluster(pos: np.ndarray, labels: np.ndarray):
    """Replace each cluster of positions by its barycenter.
    pos : numpy array of shape (n_points, 2)
    labels : numpy array of shape (n_points,) with cluster labels
    Return : numpy array of shape (n_points - n_removed_points, 2) with barycenter positions
    """
    new_pos = pos
    accumulated = 0
    for lab in np.unique(labels):
        # Idea : replace position at index start by barycenter and remove the other between start+1 and end
        if lab == -1:
            continue
        mask: np.ndarray = labels == lab
        start = mask.argmax()  # start of the sequence
        end = len(labels) - mask[::-1].argmax()  # end of the sequence
        accumulated += end - start - 1
        barycenter = np.sum(pos[mask], axis=0) / len(pos[mask])
        new_pos = np.concat(
            (
                new_pos[: start - accumulated],
                barycenter[None, :],
                new_pos[end - accumulated :],
            ),
            axis=0,
        )
    return new_pos


def translate_bool_to_runtumble(array):
    np.full_like(array, "run", dtype=object)
    array_labels = np.where(array, "tumble", "run")
    return array_labels


def translate_run_tumble_to_bool(array):
    return array == "tumble"


class RunTumbleLabeler:
    """
    Class to label run and tumble phases based on different methods:
    - angle: based on angle changes
    - speed: based on speed clusters
    - combined: combination of angle and speed criteria
    """

    def __init__(self, config):
        method_funcs = {
            "angle": self._label_by_angle,
            "speed": self._label_by_speed,
            "combined": self._label_by_combined,
        }
        self.method = config.labeling_method
        assert self.method in method_funcs, f"Unknown method: {self.method}"
        self.combination_rule = config.combination_rule
        self.threshold_angle = config.threshold_angle
        self.threshold_speed = config.threshold_speed
        self.min_size_cluster = config.min_size_cluster
        self.labeling_func = method_funcs[self.method]

    def _adjust_length_labels(self, labels):
        """Adjust length of labels to match input positions length"""
        return np.concatenate(([labels[0]], labels, [labels[-1]]))

    def _label_by_angle(self, track_pos):
        angles = compute_angle(track_pos[:, 0], track_pos[:, 1])
        self.labels = translate_bool_to_runtumble(angles > self.threshold_angle)
        return self._adjust_length_labels(self.labels)

    def _label_by_speed(self, track_pos):
        labels = get_clusters_labels(
            track_pos, self.threshold_speed, self.min_size_cluster
        )
        self.labels = translate_bool_to_runtumble(
            labels != -1
        )  # labels -1 are too small clusters, i.e one point of a run
        return self.labels

    def _label_by_combined(self, track_pos):
        angle_labels = translate_run_tumble_to_bool(self._label_by_angle(track_pos))
        speed_labels = translate_run_tumble_to_bool(self._label_by_speed(track_pos))
        if self.combination_rule == "and":
            return translate_bool_to_runtumble(angle_labels & speed_labels)
        elif self.combination_rule == "or":
            return translate_bool_to_runtumble(angle_labels | speed_labels)
        else:
            raise ValueError(f"Unknown combination rule: {self.combination_rule}")

    def label(self, track_pos):
        """track_pos : numpy array of (x,y) positions, size (n_steps, 2)
        Return : numpy array of labels ("run" or "tumble") of size (n_steps)"""
        return self.labeling_func(track_pos)
