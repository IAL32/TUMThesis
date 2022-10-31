import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def viz_column(chapter: str, filename: str, column: str, xlabel="", ylabel="", ylim=()):

    data = pd.read_csv(os.path.abspath('') + f"/{filename}")
    data["run_name"] = data.apply(lambda row: f"TBS={row['target_batch_size']}", axis=1)
    data = data.sort_values(by=[
        "batch_size_per_step", "optimizer_params.lr", "target_batch_size"
    ], ascending=[True, True, False])

    fig = plt.figure()
    ax = sns.catplot(data, x="run_name", y=column, kind="violin", col="batch_size_per_step", row="optimizer_params.lr", hue="target_batch_size", dodge=False, sharex=False)
    if len(ylim) > 0:
        plt.ylim(*ylim)
    # plt.ylim(0, 60)

    ax.set(xlabel=xlabel, ylabel=ylabel)

    fig = plt.gcf()
    stripped_name = column.split("/")[-1].replace("_", "-")
    fig.savefig(f"../../figures/{chapter}_{stripped_name}_{filename.replace('.csv', '.pdf')}", bbox_inches='tight')
    plt.show()

