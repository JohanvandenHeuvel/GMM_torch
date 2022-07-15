from data import make_pinwheel_data, make_two_cluster_data
from gmm import gmm
import numpy as np


if __name__ == '__main__':
    data = make_pinwheel_data(radial_std=0.3, tangential_std=0.05, num_classes=5, num_per_class=100, rate=0.25)
    # data = make_two_cluster_data(100)
    # data = np.array([[1, 1], [5, 1]])
    gmm(data, 15)

