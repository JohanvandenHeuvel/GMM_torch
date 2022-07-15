import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F


def make_plot(data, z_weights, loc_list, var_list):
    def generate_ellipse(params):
        """
        Generate ellipse from (mu, Sigma)
        """
        mu, var = params
        # Sigma = np.exp(0.5 * log_var)
        t = np.linspace(0, 2 * np.pi, 100) % 2 * np.pi
        circle = np.vstack((np.sin(t), np.cos(t)))
        ellipse = 2.0 * np.dot(np.linalg.cholesky(var), circle)
        return ellipse[0] + mu[0], ellipse[1] + mu[1]

    # plt.ion()

    data_x, data_y = zip(*data)

    assignments = torch.argmax(F.softmax(z_weights, dim=1), dim=-1)

    for i, params in enumerate(zip(loc_list, var_list)):

        mask = (assignments == i)
        plt.scatter(np.array(data_x)[mask.cpu()], np.array(data_y)[mask.cpu()])

        x, y = generate_ellipse(params)
        plt.plot(x, y, alpha=0.2, linewidth=2, linestyle="-")

    plt.show()
    # plt.pause(0.001)
    # plt.clf()
