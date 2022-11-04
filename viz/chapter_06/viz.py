from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import wandb.apis.public

def get_run_list(filters={}):
    import wandb
    from dotenv import load_dotenv
    load_dotenv(override=True)
    api = wandb.Api(timeout=30)

    # Project is specified by <entity/project-name>
    return api.runs("kubework/hivemind-parameter-averaging-resnet18-imagenet_scale-up", filters=filters)

def load_and_save_baseline_runs(filename):
    load_and_save_runs(filename, filters={
        "$and": [
            {"config.run_name": {"$regex": "baseline-*"}},
            {"summary_metrics.cpu/logical_core_count": 16},
        ]
    })

def load_and_save_hivemind_runs_nop(filename):
    load_and_save_runs(filename, filters={
        "$and": [
            {"config.run_name": {"$regex": "hivemind-*"}},
            {"summary_metrics.cpu/logical_core_count": {"$in": [1, 2, 4, 8]}},
            {"config.batch_size_per_step": {"$in": [32, 64, 128]}},
            {"config.target_batch_size": 1250},
            {"config.number_of_nodes": {"$in": [16, 8, 4, 2]}},
        ]
    })


def load_and_save_hivemind_runs(filename):
    load_and_save_runs(filename, filters={
        "$and": [
            {"config.run_name": {"$regex": "hivemind-*"}},
            {"summary_metrics.cpu/logical_core_count": 8},
            {"config.batch_size_per_step": {"$in": [32, 64, 128]}},
            {"config.target_batch_size": {"$lte": 10000}},
            {"config.number_of_nodes": 2},
        ]
    })

def scan_history(run: wandb.apis.public.Run):
    print(f"scanning run {run.id}")

    # run.config is the input metrics.
    # We remove special values that start with _.
    config = {k:v for k,v in run.config.items()}
    config["optimizer_params.lr"] = config["optimizer_params"]["lr"]
    config["optimizer_params.momentum"] = config["optimizer_params"]["momentum"]

    MAX_RETRIES = 5
    retry = 0
    while retry < MAX_RETRIES:
        try:
            history = run.history(keys=[
                "_timestamp",
                "train/step",
                "bandwidth/net_recv_sys_bandwidth_mbs",
                "bandwidth/net_sent_sys_bandwidth_mbs",
                "train/samples_ps",
                "train/data_load_s",
                "train/model_forward_s",
                ("train/model_backward_only_s" if "train/model_backward_only_s" in run.summary._json_dict else "train/model_backward_s"),
                *(["train/model_opt_s"] if "train/model_backward_only_s" in run.summary._json_dict else []),
            ], samples=run.summary.get("_step"), pandas=False)
            break
        except Exception as e:
            print(f"Run {run.id} failed with: ", e.args[0])
            retry += 1

    if retry >= MAX_RETRIES:
        raise Exception(f"Could not fetch run {run.id}")

    row_history = []
    sum_total_time_s = 0
    sum_missing_time_s = 0
    for row in history:
        total_time_s = config["batch_size_per_step"] / row["train/samples_ps"]
        missing_time_s = total_time_s - row["train/data_load_s"] - row["train/model_forward_s"]
        # for old runs incorporating both...
        if "train/model_backward_only_s" in row:
            missing_time_s -= row["train/model_backward_only_s"] - row["train/model_opt_s"]
        else:
            missing_time_s -= row["train/model_backward_s"]
        sum_total_time_s += total_time_s
        sum_missing_time_s += missing_time_s
        row_history.append({**row, **config, "name": run.name, "train/total_time_s": total_time_s, "train/missing_time_s": missing_time_s})

    row_summary = {
        **run.summary._json_dict,
        "train/total_time_s": sum_total_time_s / len(row_history),
        "train/missing_time_s": sum_missing_time_s / len(row_history),
    }
    row_summary["_runtime"] = row_summary["_runtime"] / 60
    row_summary["train/loss"] = row_summary["train/loss"]["min"]

    return row_history, row_summary, config

def load_and_save_runs(filename, filters={}):
    runs = get_run_list(filters)
    summary_list = []
    config_list = []


    def task_run_scan_history(run):
        row_history_list, row_summary, row_config = scan_history(run)
        return row_history_list, row_summary, row_config

    # making stuff faster
    with ThreadPoolExecutor(10) as executor:
        history_header_created = False
        futures = [executor.submit(task_run_scan_history, run) for run in runs]
        for future in as_completed(futures):
            row_history_list, row_summary, row_config = future.result()

            row_history_df = pd.DataFrame.from_records(row_history_list)
            # write to disk as soon as we have the list
            if not history_header_created:
                row_history_df.to_csv(filename)
                history_header_created = True
            else:
                row_history_df.to_csv(filename, mode="a", header=False)

            summary_list.append(row_summary)
            config_list.append(row_config)

    config_df = pd.DataFrame.from_records(config_list) 
    summary_df = pd.DataFrame.from_records(summary_list)
    all_df = pd.concat([config_df, summary_df], axis=1)
    all_df.to_csv(f"summary-{filename}")

def blended_transform(ax, value):
    import matplotlib.transforms as transforms
    trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0, value, "{:.03f}".format(value), color="red", transform=trans, ha="right", va="center", fontsize=6)

def draw_line(ax, value):
    plt.axhline(y=value, color="red", linestyle="--", lw=0.5)
    blended_transform(ax, value)


def viz_column_all(chapter: str, filename: str, column: str, **kwargs):
    sort_by = ["number_of_nodes", "batch_size_per_step", "optimizer_params.lr", "target_batch_size", "gradient_accumulation_steps", "use_local_updates"]
    data = pd.read_csv(os.path.abspath('') + f"/{filename}", usecols=[*sort_by, column])
    for gas in [1, 2]:
        data_gas = data[data["gradient_accumulation_steps"] == gas]
        data_gas = data_gas.sort_values(by=sort_by, ascending=[True, False, True, False, True, True])
        viz_column(data_gas, chapter, filename, column, gas, **kwargs)


def viz_column(data, chapter: str, filename: str, column: str, gas, is_nop=False, ylabel="", ylim=()):
    cleaned_data = data
    if len(ylim) > 0:
        cleaned_data = cleaned_data[cleaned_data[column] < ylim[1]]
    row = "number_of_nodes" if is_nop else "target_batch_size"
    g = sns.FacetGrid(cleaned_data, row="batch_size_per_step", col="use_local_updates", margin_titles=True)
    g.figure.suptitle(f"GAS = {gas}")
    g.set_xticklabels(rotation=45)
    g.map_dataframe(
        sns.violinplot,
        x=row,
        y=column,
        dodge=False,
        order=cleaned_data[row].unique(),
        palette="tab10",
        linewidth=0.2,
    )
    g.set_ylabels("")
    g.figure.supylabel(ylabel)
    g.set_titles(row_template="BS = {row_name}", col_template="LU = {col_name}")
    g.set_xlabels("TBS")
    if len(ylim) > 0:
        plt.ylim(*ylim)

    stripped_name = column.split("/")[-1].replace("_", "-")
    g.tight_layout()
    g.figure.savefig(f"../../figures/{chapter}_{stripped_name}_gas-{gas}_{filename.replace('.csv', '.pdf')}", bbox_inches='tight')
    plt.show()


import os
import pandas as pd

def load_baseline_data():
    baseline_data = pd.read_csv(os.path.abspath('') + f"/summary-baseline-16vCPUs-GAS-1.csv")

    baseline_data = baseline_data.groupby(["optimizer_params.lr", "batch_size_per_step", "run_name", "gradient_accumulation_steps"]) \
        ["_runtime", "train/loss"].describe().reset_index()
    baseline_data = baseline_data.sort_values(
        by=["optimizer_params.lr", "gradient_accumulation_steps"],
    )
    return baseline_data


def find_baseline(baseline_data, batch_size, lr, gas):
    found_run = baseline_data[
        (baseline_data["batch_size_per_step"] == batch_size) &
        (baseline_data["optimizer_params.lr"] == lr) &
        (baseline_data["gradient_accumulation_steps"] == gas)
    ]

    if len(found_run) == 1:
        return found_run.iloc[0]
    return None    

def calc_increase_info(current_value, to_compare_value, sign=1):
    runtime_increase = (current_value - to_compare_value) / to_compare_value * 100
    runtime_increase_sign = "+" if runtime_increase * sign > 0 else "-"
    runtime_increase_color = "red" if runtime_increase * sign > 0 else "ForestGreen"
    runtime_increase = abs(round(runtime_increase, 2))
    return runtime_increase, runtime_increase_sign, runtime_increase_color

def print_increase_info(current_value, to_compare_value):
    runtime_increase, runtime_increase_sign, runtime_increase_color = calc_increase_info(current_value, to_compare_value)
    return f"{current_value} ({runtime_increase_sign}\\textcolor{{{runtime_increase_color}}}{{{runtime_increase}\%}})"
