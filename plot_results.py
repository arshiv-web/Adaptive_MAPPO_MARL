import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

sns.set_theme(
    style="whitegrid",
    context="talk",
    font_scale=1.0
)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.titleweight": "bold",
    "axes.labelweight": "regular",
    "legend.frameon": True
})

EVAL_DIR = "eval_results"
PLOT_DIR = os.path.join(EVAL_DIR, "plot")
NON_PAPER_DIR = os.path.join(PLOT_DIR, "non_paper")
PAPER_DIR = os.path.join(PLOT_DIR, "paper")

SUMMARY_FILE = os.path.join(EVAL_DIR, "summary.csv")
AGG_FILE = os.path.join(EVAL_DIR, "aggregate.csv")

TASKS = [
    "vmas/navigation",
    "vmas/balance",
    "pettingzoo/multiwalker",
    "pettingzoo/simple_world_comm"
]

def load_seed_curves():
    """
    Returns dataframe:
    [task, algo, seed, frame, reward]
    """
    rows = []

    pattern = os.path.join(EVAL_DIR, "*", "*", "*", "seed_*", "eval_reward_episode_reward_mean.csv")
    files = glob.glob(pattern)

    for f in files:
        parts = Path(f).parts
        task = parts[-4]
        algo = parts[-3]
        seed = int(parts[-2].split("_")[-1])

        df = pd.read_csv(f, header=None, names=["frame", "reward"])
        df["task"] = task
        df["algo"] = algo
        df["seed"] = seed

        rows.append(df)

    return pd.concat(rows, ignore_index=True)


def load_alpha_summary(task):

    dfs = []
    for i in range(0, 3):
        pattern = os.path.join(EVAL_DIR, "*", task, "adaptive_mappo", f"seed_{i}", "agent_alpha_mean.csv")
        files = glob.glob(pattern)
        seed = pd.read_csv(files[0], header=None, names=["frame", f"alphas{i}"])
        dfs.append(seed)
        
    merged = dfs[0][["frame", "alphas0"]]
    for i in range(1, 3):
        merged = merged.merge(dfs[i][["frame", f"alphas{i}"]], on="frame", how="outer")
    merged = merged.sort_values("frame").reset_index(drop=True)
    return merged


def load_agg():
    return pd.read_csv(AGG_FILE)

def plot_non_paper(task, df):
    algos = df["algo"].unique()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, algo in enumerate(sorted(algos)):
        ax = axes[i]
        sub = df[(df.task == task) & (df.algo == algo)]

        sns.lineplot(
            data=sub,
            x="frame",
            y="reward",
            hue="seed",
            ax=ax,
            palette="colorblind",
            linewidth=1.5
        )

        ax.set_title(algo)
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    handles, labels = axes[0].get_legend_handles_labels()

    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    fig.suptitle(f"Mean Reward Curves: {task}", fontsize=18)

    fig.subplots_adjust(bottom=0.18, top=0.90)

    fig.legend(
        handles,
        labels,
        title="Seed",
        loc="lower center",
        ncol=min(len(labels), 6),
        bbox_to_anchor=(0.5, -0.08)
    )

    out_dir = os.path.join(NON_PAPER_DIR)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(
        os.path.join(out_dir, f"{task}_6subplot.png"),
        bbox_inches="tight",
        pad_inches=0.3
    )
    plt.close()


def plot_main_performance(task, df):
    plt.figure(figsize=(10, 6))
    
    sub = df[df.task == task]

    ax = sns.lineplot(
        data=sub,
        x="frame",
        y="reward",
        hue="algo",
        errorbar="sd", # Plots Standard Deviation
        linewidth=2.5
    )

    plt.title(f"Performance: {task}", fontsize=14, pad=20, fontweight='bold')
    plt.xlabel("Frames", fontsize=12, labelpad=10)
    plt.ylabel("Reward", fontsize=12, labelpad=10)

    plt.legend(
        title="Algorithms",
        bbox_to_anchor=(1.05, 1), 
        borderaxespad=0.,
    )

    out_dir = os.path.join(PAPER_DIR, task)
    os.makedirs(out_dir, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "f1_performance.png"), dpi=300)
    plt.close()

def plot_curves(df, algos, file_name, title):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, task in enumerate(df.task.unique()):
        ax = axes[i]
        sub = df[(df.task == task) & (df.algo.isin(algos))]

        sns.lineplot(
            data=sub,
            x="frame",
            y="reward",
            hue="algo",
            errorbar="sd",
            ax=ax
        )

        ax.set_title(task)
        ax.set_xlabel("Frames")
        ax.set_ylabel("Reward")
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    handles, labels = axes[0].get_legend_handles_labels()

    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 2),
        frameon=True,
        title="Algorithm",
    )

    fig.suptitle(title, fontsize=16)

    plt.tight_layout(rect=[0, 0.18, 1, 1]) 

    out_dir = PAPER_DIR
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{file_name}.png"))
    plt.close()


def plot_adaptive(task, df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sub = df[(df.task == task) & ((df.algo == "adaptive_mappo"))]

    sns.lineplot(
        data=sub,
        x="frame",
        y="reward",
        hue="algo",
        errorbar="sd",
        ax=axes[0]
    )
    axes[0].set_title("Reward")

    alpha_summary = load_alpha_summary(task)

    alpha_long = alpha_summary.melt(
        id_vars="frame",
        value_vars=["alphas0", "alphas1", "alphas2"],
        var_name="seed",
        value_name="alpha"
    )

    sns.lineplot(
        data=alpha_long,
        x="frame",
        y="alpha",
        hue="seed",
        ax=axes[1]
    )

    axes[1].set_title("Alpha Behavior")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Alpha")
    axes[1].legend(title="Seed")

    fig.suptitle(f"Adaptive MAPPO: {task}")
    plt.tight_layout()

    out_dir = os.path.join(PAPER_DIR, task)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "f4_adaptive.png"))
    plt.close()

def plot_stability():
    task_names = {
        "vmas/navigation": "Navigation",
        "vmas/balance": "Balance",
        "pettingzoo/simple_world_comm": "Comm World",
        "pettingzoo/multiwalker": "Multiwalker",
    }

    algo_names = {
        "adaptive_mappo": r"$\alpha$-AMAPPO",
        "fixed_alpha_mappo_0.50": r"MAPPO ($\alpha=0.50$)",
        "fixed_alpha_mappo_0.25": r"MAPPO ($\alpha=0.25$)",
        "fixed_alpha_mappo_0.75": r"MAPPO ($\alpha=0.75$)",
        "ippo": "IPPO",
        "mappo": "MAPPO (fixed)",
    }

    df = load_agg()
    df["Task"] = df["Task"].map(task_names)
    df["Algo"] = df["Algo"].map(algo_names)
    pivot = df.pivot(index="Task", columns="Algo", values="Relative_Drop_%")
    pivot = pivot.sub(pivot.min(axis=1), axis=0)
    pivot = pivot.div(pivot.max(axis=1), axis=0)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, cmap="viridis", cbar=True)

    plt.title("Normalized Stability Gap (Lower = Better)")
    plt.tight_layout()

    out_dir = os.path.join(PAPER_DIR)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "f5_stability_heatmap.png"))
    plt.close()

def main():
    df = load_seed_curves()

    for task in df.task.unique():
        plot_non_paper(task, df)
        plot_main_performance(task, df)
        plot_adaptive(task, df)
        
    plot_curves(df, ["ippo", "mappo"] , "f2_motivation" , "Motivation: IPPO vs MAPPO")
    plot_curves(df, ["ippo", "mappo", "fixed_alpha_mappo_0.25", "fixed_alpha_mappo_0.50", "fixed_alpha_mappo_0.75"] , "f3_fixed_alpha" , "Fixed Alpha Mean Rewards")
    plot_stability()


if __name__ == "__main__":
    main()
