import numpy as np
import matplotlib.pyplot as plt

from fastcluster import linkage
from scipy.spatial.distance import squareform


def seriation(Z, N, cur_index):
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return seriation(Z, N, left) + seriation(Z, N, right)


def compute_serial_matrix(dist_mat, method="ward"):
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method, preserve_input=True)
    res_order = seriation(res_linkage, N, N + N - 2)
    seriated_dist = np.zeros((N, N))
    a, b = np.triu_indices(N, k=1)
    seriated_dist[a, b] = dist_mat[[res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b, a] = seriated_dist[a, b]

    return seriated_dist, res_order, res_linkage


def plot_corr(correlations, model_names, reorder=True, res_order=None):

    if reorder:
        m = 1.0 - 0.5 * (correlations + correlations.T)
        m[np.diag_indices_from(m)] = 0.0
        ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(m, "complete")

        names_order = [model_names[res_order[i]] for i in range(len(model_names))]
        corr = correlations[res_order, :][:, res_order]
    else:
        corr = correlations
        names_order = model_names

    if res_order is not None:
        names_order = [model_names[res_order[i]] for i in range(len(model_names))]
        corr = correlations[res_order, :][:, res_order]

    plt.figure(figsize=(15, 15))
    plt.imshow(corr)
    plt.xticks([i for i in range(len(model_names))], names_order, rotation=-85)
    plt.yticks([i for i in range(len(model_names))], names_order)

    for i in range(len(model_names)):
        for j in range(len(model_names)):
            c = corr[j, i]
            col = "white" if c < 0.9 else "black"
            # plt.text(i, j, f"{c:.3f}", va="center", ha="center", c=col)
            if c:
                plt.text(i, j, f"{c:.0f}", va="center", ha="center", c=col)

    plt.show()

    return res_order
