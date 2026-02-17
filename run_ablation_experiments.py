#!/usr/bin/env python3
"""
Run ablation study experiments for domain randomization.

Trains each ablation variant for 1 epoch per run.
All results are logged to a single CSV file.

How to use:
    python run_ablation_experiments.py
"""

import os
import csv
import glob
from datetime import datetime
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from domrand.trainer import train_simple
from domrand.define_flags import FLAGS

# configs
DATA_DIR = "./ablation_data"
RESULTS_CSV = "./ablation_results.csv"
CHECKPOINTS_DIR = "./ablation_checkpoints"

ABLATION_VARIANTS = [
    "full_method",
    "no_noise",
    "no_camera_rand",
    "no_distractors"
]


def setup_results_csv():
    """Create CSV with header if it doesn't exist."""
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'variant',
                'epoch',
                'train_loss',
                'train_euc_cm',
                'eval_euc_cm',
                'checkpoint_path'
            ])
        print(f"Created: {RESULTS_CSV}\n")
    else:
        print(f"Appending to: {RESULTS_CSV}\n")


def train_variant(variant_name):
    """Train one ablation variant for 1 epoch."""
    print("\n" + "="*80)
    print(f"TRAINING: {variant_name.upper().replace('_', ' ')}")
    print("="*80)

    # Setup paths
    train_data_dir = os.path.join(DATA_DIR, "train", variant_name)
    checkpoint_dir = os.path.join(CHECKPOINTS_DIR, variant_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Get training files
    train_files = sorted(glob.glob(os.path.join(train_data_dir, "*.tfrecords")))
    if not train_files:
        print(f"ERROR: No training data found. Please run: python generate_data.py")
        return None

    print(f"Training data: {len(train_files)} files")
    print(f"Checkpoint: {checkpoint_dir}")

    # Setting the Hyperparmeters for training here
    FLAGS.data_path = train_data_dir
    FLAGS.filenames = train_files
    FLAGS.checkpoint = checkpoint_dir
    FLAGS.num_epochs = 1
    FLAGS.lr = 1e-4
    FLAGS.bs = 64
    FLAGS.plot_preds = False
    FLAGS.notify = False

    # Train
    print("Training for 1 epoch...\n")
    results = train_simple()

    train_loss = 0.0
    train_euc = results['train_euc']
    eval_euc = results['eval_euc']

    # Get current epoch count
    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    epoch = 1
    if ckpt:
        epoch = int(ckpt.split('-')[-1])

    print(f"\n Training complete!")
    print(f"  Epoch: {epoch}")
    print(f"  Train error: {train_euc:.2f} cm")
    print(f"  Eval error (real images): {eval_euc:.2f} cm")

    # Log to a csv file
    with open(RESULTS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            variant_name,
            epoch,
            f'{train_loss:.6f}',
            f'{train_euc:.2f}',
            f'{eval_euc:.2f}',
            checkpoint_dir
        ])

    return {
        'variant': variant_name,
        'epoch': epoch,
        'train_euc': train_euc,
        'eval_euc': eval_euc
    }


def print_summary():
    """Print summary of all results."""
    if not os.path.exists(RESULTS_CSV):
        print("No results yet. Run training first.")
        return

    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)

    # Read all results
    results_by_variant = {}
    with open(RESULTS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            variant = row['variant']
            if variant not in results_by_variant:
                results_by_variant[variant] = []
            results_by_variant[variant].append(row)

    # Print for each variant
    for variant in ABLATION_VARIANTS:
        if variant not in results_by_variant:
            continue

        runs = results_by_variant[variant]
        latest = runs[-1]

        print(f"\n{variant.upper().replace('_', ' ')}:")
        print(f"  Total runs: {len(runs)}")
        print(f"  Latest epoch: {latest['epoch']}")
        print(f"  Latest train error: {latest['train_euc_cm']} cm")
        print(f"  Latest eval error (real): {latest.get('eval_euc_cm', 'N/A')} cm")
        print(f"  Checkpoint: {latest['checkpoint_path']}")

    print("\n" + "="*80)
    print(f"Full log: {RESULTS_CSV}")
    print("="*80 + "\n")


def main():
    print("\n" + "="*80)
    print("DOMAIN RANDOMIZATION ABLATION STUDY")
    print("="*80)
    print(f"\nVariants: {', '.join(ABLATION_VARIANTS)}")
    print(f"Epochs per run: 1")
    print(f"Results CSV: {RESULTS_CSV}")
    print()

    # setup csv
    setup_results_csv()

    # Train each variant
    for i, variant in enumerate(ABLATION_VARIANTS, 1):
        print(f"\n[{i}/{len(ABLATION_VARIANTS)}] Training {variant}...")
        result = train_variant(variant)

        if result:
            print(f"✓ {variant} complete")
        else:
            print(f"✗ {variant} failed")

        # Reset graph for next variant
        tf.reset_default_graph()

    print_summary() # print out a summary


if __name__ == '__main__':
    main()
