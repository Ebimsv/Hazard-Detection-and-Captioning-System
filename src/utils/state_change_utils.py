from sklearn.linear_model import LinearRegression
import numpy as np


def detect_driver_state_change(median_dists):
    """
    Detect if the driver state has changed based on the trend of median distances.

    Parameters:
    - median_dists (list): List of median distances calculated across frames.

    Returns:
    - bool: True if a driver state change is detected, otherwise False.
    """
    if len(median_dists) <= 1:
        # Not enough data to detect a trend
        return False

    # Perform linear regression to detect trend in median distances
    x = np.arange(len(median_dists)).reshape(-1, 1)  # Frame indices
    y = np.array(median_dists)  # Median distances
    speed_model = LinearRegression().fit(x, y)

    # Check if the trend shows slowing down (negative slope)
    return speed_model.coef_[0] < 0
