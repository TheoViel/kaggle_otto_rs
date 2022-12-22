import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_importances(imps):
    importances = pd.DataFrame(imps).rename(columns={0: "importance"})
    importances["importance"] = importances["importance"].apply(np.abs)
    importances = importances.sort_values("importance", ascending=False).reset_index()
    importances = importances[importances["importance"] != 0]

    plt.figure(figsize=(15, 25))
    sns.barplot(x="importance", y="index", data=importances)
    plt.yticks(fontsize=11)
    plt.ylabel(None)
    plt.show()
