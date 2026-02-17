#!/usr/bin/env python3
"""
Texture randomization ablation study.
Tests the effect of varying the number of unique textures on model performance.
"""

import os
os.environ['MUJOCO_GL'] = 'egl'  # Use EGL for headless rendering

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from domrand.sim_manager import SimManager
from domrand.trainer import Trainer
import matplotlib.pyplot as plt

# configs, hard coded constants for now
NUM_SAMPLES = 10000
NUM_RUNS = 5
TEXTURE_COUNTS = [10, 50, 100, 500, 1000, 10000]
RESULTS_FILE = "texture_ablation_results.csv"
GRAPH_FILE = "texture_ablation_graph.png"

def run_experiment(num_textures, run_idx):
    """Run training with specified number of unique textures."""
    trainer = Trainer(
        filepath="xmls/kuka/lbr4_reflex.xml",
        num_samples=NUM_SAMPLES,
        num_textures=num_textures,
        batch_size=50,
        learning_rate=1e-4
    )

    final_error = trainer.train()
    return final_error

def main():
    results = {}

    # Run experiments
    for num_textures in TEXTURE_COUNTS:
        print(f"[{num_textures} textures]")
        errors = []
        for run_idx in range(NUM_RUNS):
            error = run_experiment(num_textures, run_idx)
            errors.append(error)
        results[num_textures] = errors

    # Save results
    with open(RESULTS_FILE, 'w') as f:
        f.write("num_textures,run,test_error_cm\n")
        for num_tex, errors in results.items():
            for run_idx, error in enumerate(errors):
                f.write(f"{num_tex},{run_idx+1},{error:.2f}\n")

    # Calculate statistics
    texture_counts = sorted(results.keys())
    means = [np.mean(results[n]) for n in texture_counts]
    stds = [np.std(results[n]) for n in texture_counts]

    # Generate graph
    plt.figure(figsize=(8, 5))

    # plot mean line
    plt.semilogx(texture_counts, means, 'o-', linewidth=2, markersize=8,
            color='#2E86AB', label='Mean', zorder=3)

    # Plot te shaded region
    lower = np.array(means) - np.array(stds)
    upper = np.array(means) + np.array(stds)
    plt.fill_between(texture_counts, lower, upper, alpha=0.3, color='#2E86AB')

    plt.xlabel('Number of unique textures')
    plt.ylabel('Average error (cm)')
    plt.title('Sensitivity to Texture Randomization')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPH_FILE, dpi=150)

if __name__ == "__main__":
    main()
