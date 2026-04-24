import os
import glob
import shutil
import csv
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from PIL import Image
import plot_results as pr

ALGOS = [
    "ippo", "mappo", 
    "fixed_alpha_mappo_0.25", "fixed_alpha_mappo_0.50", "fixed_alpha_mappo_0.75", 
    "adaptive_mappo"
]
TASKS = [
    "vmas/navigation",
    "vmas/balance",
    "pettingzoo/multiwalker",
    "pettingzoo/simple_world_comm"
]
SEEDS = [0, 1, 2]

BASE_RUNS_DIR = "/scratch/engin_root/engin1/arshiv/ml/BenchMARL/runs"
EVAL_DIR = "eval_results"
SUMMARY_FILE = os.path.join(EVAL_DIR, "summary.csv")
AGGREGATE_FILE = os.path.join(EVAL_DIR, "aggregate.csv")

def get_static_alpha(algo_name):
    if algo_name == "ippo":
        return 0.0
    elif algo_name == "mappo":
        return 1.0
    elif "fixed_alpha_mappo" in algo_name:
        return float(algo_name.split("_")[-1])
    return None

def get_closest_alpha(alpha_dict, target_frame):
    if not alpha_dict:
        return 0.0
    closest_frame = min(alpha_dict.keys(), key=lambda k: abs(k - target_frame))
    return alpha_dict[closest_frame]

def aggregate():
    df = pd.read_csv(SUMMARY_FILE)

    agg_df = df.groupby(['Task', 'Algo']).agg({
        'Best_Value': ['mean', 'std'],
        'Last_Value': ['mean', 'std'],
        'Best_Alpha': ['mean', 'std'],
        'Last_Alpha': ['mean', 'std']
    }).reset_index()

    agg_df.columns = [
        'Task', 'Algo', 
        'Best_Value_Mean', 'Best_Value_Std', 
        'Last_Value_Mean', 'Last_Value_Std',
        'Best_Alpha_Mean', 'Best_Alpha_Std',
        'Last_Alpha_Mean', 'Last_Alpha_Std'
    ]

    merged = df.merge(agg_df[['Task', 'Algo', 'Best_Value_Mean', 'Last_Value_Mean']], on=['Task', 'Algo'])
    
    merged['Dist_to_Best'] = (merged['Best_Value'] - merged['Best_Value_Mean']).abs()
    idx_best = merged.groupby(['Task', 'Algo'])['Dist_to_Best'].idxmin()
    closest_best = merged.loc[idx_best].copy()
    closest_best['Closest_Best_Video'] = "seed_" + closest_best['Seed'].astype(str) + "_frame_" + closest_best['Best_Frame'].astype(str)
    
    merged['Dist_to_Last'] = (merged['Last_Value'] - merged['Last_Value_Mean']).abs()
    idx_last = merged.groupby(['Task', 'Algo'])['Dist_to_Last'].idxmin()
    closest_last = merged.loc[idx_last].copy()
    closest_last['Closest_Last_Video'] = "seed_" + closest_last['Seed'].astype(str) + "_frame_" + closest_last['Last_Frame'].astype(str)
    
    agg_df = agg_df.merge(closest_best[['Task', 'Algo', 'Closest_Best_Video']], on=['Task', 'Algo'])
    agg_df = agg_df.merge(closest_last[['Task', 'Algo', 'Closest_Last_Video']], on=['Task', 'Algo'])

    agg_df['Stability_Gap'] = (agg_df['Best_Value_Mean'] - agg_df['Last_Value_Mean']).abs()
    agg_df['Relative_Drop_%'] = ((agg_df['Best_Value_Mean'] - agg_df['Last_Value_Mean']) / agg_df['Best_Value_Mean'].abs()) * 100
    agg_df['Rank'] = agg_df.groupby('Task')['Relative_Drop_%'].rank(method='min', ascending=True).astype(int)
    agg_df['Relative_Drop_%'] = agg_df['Relative_Drop_%'].round(1)
    
    num_cols = [
        'Best_Value_Mean', 'Best_Value_Std', 
        'Last_Value_Mean', 'Last_Value_Std', 
        'Stability_Gap',
        'Best_Alpha_Mean', 'Best_Alpha_Std',
        'Last_Alpha_Mean', 'Last_Alpha_Std'
    ]
    agg_df[num_cols] = agg_df[num_cols].round(4)
    agg_df = agg_df.fillna(0.0)

    final_cols = [
        'Task', 'Algo', 'Rank', 'Stability_Gap', 'Relative_Drop_%',
        'Best_Value_Mean', 'Best_Value_Std', 'Closest_Best_Video',
        'Last_Value_Mean', 'Last_Value_Std', 'Closest_Last_Video',
        'Best_Alpha_Mean', 'Best_Alpha_Std', 'Last_Alpha_Mean', 'Last_Alpha_Std'
    ]
    agg_df = agg_df[final_cols]
    agg_df = agg_df.sort_values(by=['Task', 'Rank'])
    agg_df.to_csv(AGGREGATE_FILE, index=False)
    
    print(f"Aggregation complete! Results saved to {AGGREGATE_FILE}")

def run_evaluation():
    os.makedirs(EVAL_DIR, exist_ok=True)

    with open(SUMMARY_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Task", "Algo", "Seed", "Best_Frame", "Best_Value", "Best_Alpha", "Last_Frame", "Last_Value", "Last_Alpha"])

    for task in TASKS:
        for algo in ALGOS:
            for seed in SEEDS:

                run_dir = os.path.join(BASE_RUNS_DIR, task, algo, f"seed_{seed}")
                search_pattern = os.path.join(run_dir, "*_mlp__*", "*_mlp__*", "scalars", "eval_reward_episode_reward_mean.csv")
                csv_matches = glob.glob(search_pattern)    
                
                if not csv_matches:
                    continue  # Skip if the run hasn't generated data yet
                    
                csv_path = csv_matches[0] 
                scalar_dir = os.path.dirname(csv_path)
                video_dir = os.path.abspath(os.path.join(scalar_dir, "..", "videos"))
                
                static_alpha = get_static_alpha(algo)
                alpha_dict = {}
                alpha_csv_path = None
                
                if static_alpha is None:
                    alpha_patterns = [
                        "train_agent_alpha_mean.csv",
                        "train_agents_alpha_mean.csv",
                        "train_walker_alpha_mean.csv",
                        "train_*_alpha_mean.csv"
                    ]
                    for pat in alpha_patterns:
                        alpha_matches = glob.glob(os.path.join(scalar_dir, pat))
                        if alpha_matches:
                            alpha_csv_path = alpha_matches[0]
                            break
                            
                    if alpha_csv_path:
                        with open(alpha_csv_path, mode='r') as af:
                            reader = csv.reader(af)
                            for row in reader:
                                if len(row) != 2: continue
                                try:
                                    alpha_dict[int(row[0])] = float(row[1])
                                except ValueError:
                                    continue

                best_frame, best_val = None, float('-inf')
                last_frame, last_val = None, None
                
                with open(csv_path, mode='r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) != 2:
                            continue
                        try:
                            frame = int(row[0])
                            val = float(row[1])
                        except ValueError:
                            continue
                            
                        if val > best_val:
                            best_val = val
                            best_frame = frame
                        
                        last_frame = frame
                        last_val = val

                best_alpha = static_alpha if static_alpha is not None else get_closest_alpha(alpha_dict, best_frame)
                last_alpha = static_alpha if static_alpha is not None else get_closest_alpha(alpha_dict, last_frame)

                target_dir = os.path.join(EVAL_DIR, task, algo, f"seed_{seed}")
                os.makedirs(target_dir, exist_ok=True)
                shutil.copy2(csv_path, os.path.join(target_dir, "eval_reward_episode_reward_mean.csv"))
                
                if alpha_csv_path:
                    shutil.copy2(alpha_csv_path, os.path.join(target_dir, "agent_alpha_mean.csv"))
                
                best_video_path = os.path.join(video_dir, f"eval_video_{best_frame}.mp4")
                last_video_path = os.path.join(video_dir, f"eval_video_{last_frame}.mp4")
                
                if os.path.exists(best_video_path):
                    shutil.copy2(best_video_path, os.path.join(target_dir, f"seed_{seed}_frame_{best_frame}.mp4"))

                if best_video_path != last_video_path and os.path.exists(last_video_path):
                    shutil.copy2(last_video_path, os.path.join(target_dir, f"seed_{seed}_frame_{last_frame}.mp4"))

                with open(SUMMARY_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        task, algo, seed, 
                        best_frame, f"{best_val:.4f}", f"{best_alpha:.4f}", 
                        last_frame, f"{last_val:.4f}", f"{last_alpha:.4f}"
                    ])

    print(f"\nEvaluation extraction complete. Check {SUMMARY_FILE}")
    aggregate()


if __name__ == "__main__":
    run_evaluation()
    pr.main()
    
