"""Analyze experiment results and generate figures.
Usage:
    python scripts/analyze.py                    # Analyze all results
    python scripts/analyze.py --results-dir DIR  # Custom results directory
"""

import argparse
import json
import glob
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def get_alpha_label(dirname: str) -> str:
    """Determine alpha config from Hydra config saved by BenchMARL."""
    try:
        import yaml
        hydra_cfgs = glob.glob(f'{dirname}/**/.hydra/config.yaml', recursive=True)
        if hydra_cfgs:
            with open(hydra_cfgs[0]) as f:
                cfg = yaml.safe_load(f)
            algo = cfg.get('algorithm', {})
            return str(algo.get('alpha_start', ''))
    except Exception:
        pass
    return None


def load_results(results_dir: str):
    """Load all experiment results from a directory."""
    results = defaultdict(dict)

    for d in sorted(glob.glob(f'{results_dir}/*_multiwalker_*')):
        jsons = list(glob.glob(f'{d}/*.json'))
        if not jsons:
            continue
        try:
            with open(jsons[0]) as f:
                data = json.load(f)
            env = data.get('pettingzoo', {}).get('multiwalker', {})
            if not env:
                continue
            algo = list(env.keys())[0]
            sk = list(env[algo].keys())[0]
            seed = int(sk.split('_')[1])
            sd = env[algo][sk]

            steps = sorted([k for k in sd if k.startswith('step_')],
                           key=lambda s: int(s.split('_')[1]))
            if len(steps) < 3:
                continue

            frames = sd[steps[-1]].get('step_count', 0)
            if frames < 1_800_000:
                continue

            peak = sd.get('absolute_metrics', {}).get('return', [0])[0] or 0
            rets = sd[steps[-1]].get('return', [])
            final = sum(rets) / len(rets) if rets else 0

            curve = []
            for step_key in steps:
                st = sd[step_key]
                f_count = st.get('step_count', 0)
                r = st.get('return', [])
                if r:
                    curve.append((f_count, np.mean(r)))

            if algo == 'progressive':
                alpha = get_alpha_label(d)
                label = f'Blend α={alpha}' if alpha else 'Blend (unknown)'
            elif algo == 'ippo':
                label = 'IPPO'
            elif algo == 'mappo':
                label = 'MAPPO'
            else:
                continue

            key = (label, seed)
            if key not in results or frames > results[key]['frames']:
                results[key] = {
                    'label': label, 'seed': seed, 'frames': frames,
                    'peak': peak, 'final': final, 'curve': curve,
                }
        except Exception:
            continue

    return dict(results)


def print_summary(results: dict):
    """Print statistical summary table."""
    from scipy import stats

    by_label = defaultdict(list)
    for (label, seed), run in results.items():
        by_label[label].append(run)

    print(f"\n{'Method':17s} {'N':>3s}  {'Peak (mean±std)':>16s}  {'Final (mean±std)':>16s}")
    print("=" * 60)

    order = ['IPPO', 'MAPPO', 'Blend α=0.25', 'Blend α=0.5', 'Blend α=0.75']
    for label in order:
        runs = by_label.get(label, [])
        if not runs:
            continue
        peaks = [r['peak'] for r in runs]
        finals = [r['final'] for r in runs]
        print(f"{label:17s} {len(runs):>3d}  "
              f"{np.mean(peaks):6.1f} ± {np.std(peaks):5.1f}  "
              f"{np.mean(finals):6.1f} ± {np.std(finals):5.1f}")

    # Statistical tests
    print(f"\n{'Comparison':35s} {'Δ':>7s} {'p-value':>8s}")
    print("-" * 55)
    tests = [
        ('Blend α=0.5', 'IPPO', 'peak'),
        ('MAPPO', 'IPPO', 'peak'),
        ('Blend α=0.75', 'IPPO', 'final'),
    ]
    for a, b, metric in tests:
        va = [r[metric] for r in by_label.get(a, [])]
        vb = [r[metric] for r in by_label.get(b, [])]
        if len(va) < 2 or len(vb) < 2:
            continue
        t, p = stats.ttest_ind(va, vb)
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "ns"
        print(f"  {a:15s} vs {b:10s} ({metric:5s}): "
              f"{np.mean(va) - np.mean(vb):+6.1f}   p={p:.4f} {sig}")


def plot_all(results: dict, output_dir: str):
    """Generate all figures."""
    by_label = defaultdict(list)
    for (label, seed), run in results.items():
        by_label[label].append(run)

    order = ['IPPO', 'MAPPO', 'Blend α=0.25', 'Blend α=0.5', 'Blend α=0.75']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Training curves
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, color in zip(order, colors):
        runs = by_label.get(label, [])
        if not runs:
            continue
        min_len = min(len(r['curve']) for r in runs)
        frames = np.array([runs[0]['curve'][i][0] for i in range(min_len)])
        returns = np.array([[r['curve'][i][1] for i in range(min_len)] for r in runs])
        mean, std = returns.mean(0), returns.std(0)
        ax.plot(frames / 1e6, mean, label=f"{label} (n={len(runs)})", color=color, lw=2)
        ax.fill_between(frames / 1e6, mean - std, mean + std, alpha=0.15, color=color)
    ax.set_xlabel("Training Frames (millions)")
    ax.set_ylabel("Mean Episode Return")
    ax.set_title("Training Curves")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', ls='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(out / "reward_curves.pdf", bbox_inches='tight')
    fig.savefig(out / "reward_curves.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: reward_curves")

    # 2. Box + strip
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in zip(axes, ['peak', 'final'], ['Peak Return', 'Final Return']):
        data_list = []
        for label in order:
            runs = by_label.get(label, [])
            data_list.append([r[metric] for r in runs] if runs else [0])
        bp = ax.boxplot(data_list, positions=range(len(order)), widths=0.5,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.3)
        np.random.seed(42)
        for i, (vals, color) in enumerate(zip(data_list, colors)):
            jitter = np.random.normal(0, 0.08, size=len(vals))
            ax.scatter([i + j for j in jitter], vals, color=color,
                       s=40, zorder=5, edgecolors='black', linewidth=0.5, alpha=0.8)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel(title)
        ax.set_title(title, fontweight='bold')
        ax.axhline(y=0, color='gray', ls='--', alpha=0.5)
        ax.grid(True, alpha=0.2, axis='y')
    plt.suptitle('Performance Distribution (each dot = 1 seed)', fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(out / "box_strip.pdf", bbox_inches='tight')
    fig.savefig(out / "box_strip.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: box_strip")

    # 3. Stability analysis
    fig, ax = plt.subplots(figsize=(7, 4))
    stds = []
    for label in order:
        runs = by_label.get(label, [])
        stds.append(np.std([r['final'] for r in runs]) if runs else 0)
    bars = ax.bar(range(len(order)), stds, color=colors, edgecolor='black', lw=0.5, alpha=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=20, ha='right')
    ax.set_ylabel("Std of Final Return (lower = more stable)")
    ax.set_title("Training Stability")
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(out / "stability.pdf", bbox_inches='tight')
    fig.savefig(out / "stability.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: stability")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--output-dir", default="figures", help="Output figures directory")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    print(f"Loaded {len(results)} complete runs")

    print_summary(results)
    print("\nGenerating figures...")
    plot_all(results, args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
