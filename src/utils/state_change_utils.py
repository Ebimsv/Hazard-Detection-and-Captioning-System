from sklearn.linear_model import LinearRegression
import numpy as np


def detect_driver_state_change(median_dists):
    if len(median_dists) < 2:
        return False

    x = np.array(range(len(median_dists))).reshape(-1, 1)
    y = np.array(median_dists)
    speed_model = LinearRegression().fit(x, y)

    if (
        speed_model.coef_[0] < 0
    ):  # If we are slowing down, driver state has probably changed
        return True
    return False
