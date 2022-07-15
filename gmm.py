from torch import distributions as dist
import torch
import torch.nn.functional as F

from sklearn.cluster import KMeans

from vae.models import resVAE

import matplotlib

from plot import make_plot

# matplotlib.use("TkAgg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_elementwise_multiplication(x, y):
    """Computes x * y where the fist dimension is the batch, x, y is a scalar.
    """
    return torch.einsum("bi, bi -> b", (x, y))


def compute_log_prob(data, loc_list, var_list):
    clusters = [
        dist.MultivariateNormal(loc=loc, covariance_matrix=var)
        for (loc, var) in zip(loc_list, var_list)
    ]

    # get the log probability of the data
    log_probs = []
    for cluster in clusters:
        log_prob = cluster.log_prob(data)
        assert not torch.any(torch.isnan(log_prob))
        log_probs.append(log_prob)
    log_probs = torch.stack(log_probs).T
    return log_probs


def fit_labels(log_probs, z_weights, n_epoch=100):
    optimizer = torch.optim.Adam([z_weights], lr=1e-3)
    for epoch in range(n_epoch):
        # compute the loss
        assignments = F.softmax(z_weights, dim=1)
        weighted_log_probs = batch_elementwise_multiplication(log_probs, assignments)
        loss = -torch.mean(weighted_log_probs)

        # do optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return z_weights


def fit_clusters(data, assignments, loc_list, var_list, n_epoch=100):
    optimizer = torch.optim.Adam([loc_list, var_list], lr=1e-3)
    for epoch in range(n_epoch):
        # compute loss
        log_probs = compute_log_prob(data, loc_list, var_list)
        weighted_log_probs = batch_elementwise_multiplication(log_probs, assignments)
        loss = -torch.mean(weighted_log_probs)

        # do optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loc_list, var_list


def fit(data, z_weights, loc_list, var_list, encoder, n_epoch=10000):
    data = torch.tensor(data, device=device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    for epoch in range(n_epoch):

        x = encoder.reparameterize(*encoder.encode(data.float()))
        log_probs = compute_log_prob(x, loc_list.detach(), var_list.detach())

        # find optimal cluster assignments
        z_weights = fit_labels(log_probs.detach(), z_weights)
        assignments = F.softmax(z_weights, dim=1)

        # find optimal cluster parameters
        loc_list, var_list = fit_clusters(x.detach(), assignments.detach(), loc_list, var_list)

        weighted_log_probs = batch_elementwise_multiplication(
            log_probs, assignments.detach()
        )
        loss = -torch.mean(weighted_log_probs)

        # do optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                make_plot(
                    x.cpu().detach().numpy(),
                    z_weights,
                    loc_list.cpu().detach().numpy(),
                    var_list.cpu().detach().numpy(),
                )

    return z_weights, loc_list, var_list


def gmm(data, K):
    kmeans = KMeans(n_clusters=K).fit(data)

    encoder = resVAE(input_size=2, latent_dim=2, hidden_size=50, name="resvae")

    n, d = data.shape

    dirichlet = dist.Dirichlet(torch.ones(K) / K)
    z_weights = dirichlet.rsample(torch.tensor([n]))
    z_weights = z_weights.to(device).detach().requires_grad_()

    loc_list = torch.from_numpy(kmeans.cluster_centers_).to(device)
    loc_list = loc_list.requires_grad_()

    var_list = (
        torch.diag(torch.ones(d, device=device))
        .unsqueeze(0)
        .repeat_interleave(K, dim=0)
    )
    var_list = var_list.requires_grad_()

    z_weights, loc_list, var_list = fit(data, z_weights, loc_list, var_list, encoder)
