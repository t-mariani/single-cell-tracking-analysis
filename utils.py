import numpy as np


def compute_angle_single(vec1, vec2):
    """Compute angle in degrees between two vectors"""
    unit_vec1 = vec1 / (np.linalg.norm(vec1) + 1e-10)
    unit_vec2 = vec2 / (np.linalg.norm(vec2) + 1e-10)
    dot_product = np.dot(unit_vec1, unit_vec2)
    angle_rad = np.arccos(dot_product)
    angle_deg = angle_rad * (180 / np.pi)
    return angle_deg
