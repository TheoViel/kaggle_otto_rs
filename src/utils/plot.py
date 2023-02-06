import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_importances(imps, run=None):
    """
    Plots feature importance. Plot can be saved to neptune.

    Args:
        imps (pandas DataFrame): Features importance.
        run (neptune run, optional): Neptune run. Defaults to None.
    """
    imps.index.name = "index"
    importances = imps.rename(columns={0: "importance", "0": "importance"})

    importances["importance"] = importances["importance"].apply(np.abs)
    importances = importances.sort_values("importance", ascending=False).reset_index()

    fig = plt.figure(figsize=(15, 1 + len(imps) // 5))
    sns.barplot(x="importance", y="index", data=importances)
    plt.yticks(fontsize=11)
    plt.ylabel(None)

    if run is not None:
        run["global/ft_imp"].upload(fig)
        plt.close()
    else:
        plt.show()
