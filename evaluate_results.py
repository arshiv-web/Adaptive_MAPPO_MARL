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

ALGOS = ["ippo", "mappo"]
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
AGGREGATE_FILE = os.path.join(EVAL_DIR, "aggreagte.csv")

video_files = [
    "eval_results/vmas/navigation/mappo/seed_2/seed_2_frame_279.mp4",
    "eval_results/vmas/balance/mappo/seed_1/seed_1_frame_159.mp4",
    "eval_results/pettingzoo/simple_world_comm/ippo/seed_0/seed_0_frame_59.mp4",
    "eval_results/pettingzoo/multiwalker/mappo/seed_2/seed_2_frame_159.mp4"
]

output_path = [
    "eval_results/navigation_stages.png",
    "eval_results/balance_stages.png",
    "eval_results/swc_stages.png",
    "eval_results/multiwalker_stages.png",
]

svg_paths = {
    "Navigation": "eval_results/plots/Navigation_PNG.png",
    "Balance": "eval_results/plots/Balance_PNG.png",
    "Multiwalker": "eval_results/plots/Multiwalker_PNG.png",
    "Simple World Comm": "eval_results/plots/SWC_PNG.png"
}


def aggregate():
    df = pd.read_csv(SUMMARY_FILE)

    agg_df = df.groupby(['Task', 'Algo']).agg({
        'Best_Value': ['mean', 'std'],
        'Last_Value': ['mean', 'std']
    }).reset_index()

    
    agg_df.columns = [
        'Task', 'Algo', 
        'Best_Value_Mean', 'Best_Value_Std', 
        'Last_Value_Mean', 'Last_Value_Std'
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
    num_cols = ['Best_Value_Mean', 'Best_Value_Std', 'Last_Value_Mean', 'Last_Value_Std', 'Stability_Gap']
    agg_df[num_cols] = agg_df[num_cols].round(4)
    agg_df = agg_df.fillna(0.0)

    final_cols = [
        'Task', 'Algo', 'Rank', 'Stability_Gap', 'Relative_Drop_%',
        'Best_Value_Mean', 'Best_Value_Std', 'Closest_Best_Video',
        'Last_Value_Mean', 'Last_Value_Std', 'Closest_Last_Video'
    ]
    agg_df = agg_df[final_cols]
    agg_df = agg_df.sort_values(by=['Task', 'Rank'])
    agg_df.to_csv(AGGREGATE_FILE, index=False)
    
    print(f"Aggregation complete! Results saved to {AGGREGATE_FILE}")

def run_evaluation():
    os.makedirs(EVAL_DIR, exist_ok=True)

    with open(SUMMARY_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Task", "Algo", "Seed", "Best_Frame", "Best_Value", "Last_Frame", "Last_Value"])

    for task in TASKS:
        for algo in ALGOS:
            for seed in SEEDS:

                run_dir = os.path.join(BASE_RUNS_DIR, task, algo, f"seed_{seed}")
                search_pattern = os.path.join(run_dir, "*_mlp__*", "*_mlp__*", "scalars", "eval_reward_episode_reward_mean.csv")
                csv_matches = glob.glob(search_pattern)    
                csv_path = csv_matches[0] # Take the first match
                video_dir = os.path.abspath(os.path.join(os.path.dirname(csv_path), "..", "videos"))
                
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

                target_dir = os.path.join(EVAL_DIR, task, algo, f"seed_{seed}")
                os.makedirs(target_dir, exist_ok=True)
                shutil.copy2(csv_path, os.path.join(target_dir, "eval_reward_episode_reward_mean.csv"))
                
                best_video_path = os.path.join(video_dir, f"eval_video_{best_frame}.mp4")
                last_video_path = os.path.join(video_dir, f"eval_video_{last_frame}.mp4")
                
                shutil.copy2(best_video_path, os.path.join(target_dir, f"seed_{seed}_frame_{best_frame}.mp4"))

                if best_video_path != last_video_path:
                    shutil.copy2(last_video_path, os.path.join(target_dir, f"seed_{seed}_frame_{last_frame}.mp4"))

                with open(SUMMARY_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([task, algo, seed, best_frame, f"{best_val:.4f}", last_frame, f"{last_val:.4f}"])
                    
                # print(f"Processed: {task} | {algo} | seed_{seed}")

    print(f"\nEvaluation extraction complete. Check {SUMMARY_FILE}")
    aggregate()

def _compute_union_crop_box(
    frames,
    bg_threshold=220,
    pad=8,
    keep_top_components=6,
):
    all_boxes = []

    for frame in frames:
        mask = np.any(frame < bg_threshold, axis=2).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        components = []
        for i in range(1, num_labels):  # skip background
            x, y, w, h, area = stats[i]
            components.append((area, x, y, w, h))

        if not components:
            continue

        components.sort(reverse=True, key=lambda t: t[0])
        kept = components[:keep_top_components]

        x_min = min(x for _, x, _, _, _ in kept)
        y_min = min(y for _, _, y, _, _ in kept)
        x_max = max(x + w for _, x, _, w, _ in kept)
        y_max = max(y + h for _, _, y, _, h in kept)

        all_boxes.append((x_min, y_min, x_max, y_max))

    if not all_boxes:
        h, w = frames[0].shape[:2]
        return 0, 0, w, h

    x_min = min(b[0] for b in all_boxes)
    y_min = min(b[1] for b in all_boxes)
    x_max = max(b[2] for b in all_boxes)
    y_max = max(b[3] for b in all_boxes)

    h, w = frames[0].shape[:2]
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(w, x_max + pad)
    y_max = min(h, y_max + pad)

    return x_min, y_min, x_max, y_max


def _add_border(img, border_px=1, color=(0, 0, 0)):
    return cv2.copyMakeBorder(
        img,
        border_px, border_px, border_px, border_px,
        cv2.BORDER_CONSTANT,
        value=color
    )


def _draw_label_inside(img, label, font_scale=0.6, thickness=1, margin=8):
    out = img.copy()
    h, w = out.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    x = (w - tw) // 2
    y = margin + th
    
    rect_pad_x = 6
    rect_pad_y = 4
    x1 = max(0, x - rect_pad_x)
    y1 = max(0, y - th - rect_pad_y)
    x2 = min(w, x + tw + rect_pad_x)
    y2 = min(h, y + baseline + rect_pad_y)

    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
    alpha = 0.55
    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

    cv2.putText(
        out,
        label,
        (x, y),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )
    return out


def create_stage_strip(
    video_path,
    output_path="stage_strip.png",
    sample_fracs=(0.10, 0.50, 0.90),
    labels=("Early", "Mid", "Late"),
    bg_threshold=220,
    pad=8,
    keep_top_components=6,
    border_px=1,
    gap_px=8,
    start_time_sec=None,
    end_time_sec=None,
):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    clip_start_sec = 0.0 if start_time_sec is None else float(start_time_sec)
    clip_end_sec = duration_sec if end_time_sec is None else float(end_time_sec)
    clip_end_sec = min(clip_end_sec, duration_sec)

    start_frame = int(clip_start_sec * fps)
    end_frame = min(total_frames - 1, int(clip_end_sec * fps))
    clip_num_frames = end_frame - start_frame + 1

    frame_indices = [
        min(end_frame, max(start_frame, start_frame  + int(frac * (clip_num_frames - 1))))
        for frac in sample_fracs
    ]

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    x_min, y_min, x_max, y_max = _compute_union_crop_box(
        frames,
        bg_threshold=bg_threshold,
        pad=pad,
        keep_top_components=keep_top_components,
    )
    frames = [f[y_min:y_max, x_min:x_max] for f in frames]

    processed = []
    for frame, label in zip(frames, labels):
        img = _add_border(frame, border_px=border_px)
        img = _draw_label_inside(img, label, font_scale=0.6, thickness=1, margin=8)
        processed.append(img)

    heights = [img.shape[0] for img in processed]
    widths = [img.shape[1] for img in processed]
    max_h = max(heights)

    aligned = []
    for img in processed:
        h, w = img.shape[:2]
        if h < max_h:
            pad_bottom = max_h - h
            img = cv2.copyMakeBorder(
                img, 0, pad_bottom, 0, 0,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255)
            )
        aligned.append(img)

    total_w = sum(img.shape[1] for img in aligned) + gap_px * (len(aligned) - 1)
    canvas = np.full((max_h, total_w, 3), 255, dtype=np.uint8)

    x = 0
    for img in aligned:
        h, w = img.shape[:2]
        canvas[:h, x:x+w] = img
        x += w + gap_px

    # Save as PNG
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, canvas_bgr)
    print(f"Saved strip to: {output_path}")

def crop_top(img: Image.Image, top_frac: float = 0.10) -> Image.Image:
    w, h = img.size
    top = int(h * top_frac)
    return img.crop((0, top, w, h))


def plot():

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    pretty_titles = [
        "VMAS Navigation",
        "VMAS Balance",
        "Multiwalker",
        "Simple World Comm",
    ]

    items = list(svg_paths.items())

    for idx, (ax, (_, path)) in enumerate(zip(axes, items)):
        ax.set_title(pretty_titles[idx], fontsize=14, pad=10)
        img = Image.open(path).convert("RGBA")
        img = crop_top(img, top_frac=0.15)
        ax.imshow(img)
        ax.axis("off")

    legend_handles = [
        Line2D([0], [0], color="#d62728", lw=2.5, linestyle="-", label="MAPPO-0"),
        Line2D([0], [0], color="#ff7f0e", lw=2.5, linestyle="-", label="MAPPO-1"),
        Line2D([0], [0], color="#9467bd", lw=2.5, linestyle="-", label="MAPPO-2"),
        Line2D([0], [0], color="#2ca02c", lw=2.5, linestyle="--", label="IPPO-0"),
        Line2D([0], [0], color="#8c564b", lw=2.5, linestyle="--", label="IPPO-1"),
        Line2D([0], [0], color="#1f77b4", lw=2.5, linestyle="--", label="IPPO-2"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.02),
    )

    # -----------------------------
    # Layout + save
    # -----------------------------
    plt.subplots_adjust(
        left=0.03,
        right=0.97,
        top=0.94,
        bottom=0.13,
        wspace=0.06,
        hspace=0.20,
    )

    output_path = "eval_results/plots/plot_2x2.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved to: {output_path}")


def stack_images():
    img1_path = "eval_results/navigation_stages.png"
    img2_path = "eval_results/balance_stages.png"
    img3_path = "eval_results/multiwalker_stages.png"
    img4_path = "eval_results/swc_stages.png"
    output_path = "eval_results/rollout_stack.png"

    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    img3 = Image.open(img3_path)
    img4 = Image.open(img4_path)

    if img1.width != img2.width:
        aspect_ratio = img2.height / img2.width
        new_height = int(img1.width * aspect_ratio)
        img2 = img2.resize((img1.width, new_height), Image.Resampling.LANCZOS)

    if img1.width != img3.width:
        aspect_ratio = img3.height / img3.width
        new_height = int(img1.width * aspect_ratio)
        img3 = img3.resize((img1.width, new_height), Image.Resampling.LANCZOS)

    if img1.width != img4.width:
        aspect_ratio = img4.height / img4.width
        new_height = int(img1.width * aspect_ratio)
        img4 = img4.resize((img1.width, new_height), Image.Resampling.LANCZOS)

    # Create a new blank canvas with the combined height
    combined_width = img1.width
    combined_height = img1.height + img2.height + img3.height + img4.height
    new_img = Image.new('RGB', (combined_width, combined_height))

    # Paste the images (x, y)
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (0, img1.height))
    new_img.paste(img3, (0, img1.height + img2.height))
    new_img.paste(img4, (0, img1.height + img2.height + img3.height))

    # Save the result
    new_img.save(output_path)
    print(f"Successfully saved to {output_path}")

if __name__ == "__main__":
    run_evaluation()
    # for i, j in zip(video_files, output_path):
    #     create_stage_strip(
    #         video_path=i,
    #         output_path=j,
    #         sample_fracs=(0.01, 0.40, 0.99),
    #         labels=("EARLY", "MID", "LATE"),
    #         bg_threshold=220,
    #         pad=8,
    #         keep_top_components=6,
    #         border_px=2,
    #         gap_px=6,
    #     )
    
