#!/usr/bin/env python
"""
Ablation Study Runner for Domain Randomization Experiments

This script reproduces the ablation study from the domain randomization paper.
It runs 4 experiments:
  1. full_method - All randomization techniques enabled (baseline)
  2. no_noise - Disable image noise augmentation
  3. no_camera_rand - Disable camera position randomization
  4. no_distractors - Disable distractor objects in scene

Results are logged to CSV files for analysis and plotting.

Usage:
    python run_ablation_experiments.py [--num_examples 10000] [--num_epochs 20]
    
For GPU training on H100, use XLA flags:
    TF_XLA_FLAGS="--tf_xla_auto_jit=2" XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false" python run_ablation_experiments.py
"""

import os
import sys
import subprocess
import argparse
import csv
from datetime import datetime


def generate_data(output_path, num_files=10, examples_per_file=1000):
    """Generate TFRecord training data using domain randomization."""
    import glob
    import time
    import numpy as np
    import tensorflow as tf
    
    # Add project to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from domrand.sim_manager import SimManager
    
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Generating {num_files} files with {examples_per_file} examples each...")
    
    sim = SimManager(filepath='xmls/fetch/main.xml')
    
    for file_idx in range(num_files):
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
        filename = os.path.join(output_path, f'{timestamp}-%f.tfrecords')
        
        print(f"  Generating file {file_idx + 1}/{num_files}: {filename}")
        
        writer = tf.io.TFRecordWriter(filename)
        
        for i in range(examples_per_file):
            sim.randomize()
            image = sim.render()
            pos = sim.get_actuator_positions()
            
            # Create TFRecord example
            feature = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
                'label': tf.train.Feature(float_list=tf.train.FloatList(value=pos)),
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[0]])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[1]])),
                'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[2]])),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        
        writer.close()
        time.sleep(0.1)  # Small delay between files
    
    print(f"Data generation complete: {num_files * examples_per_file} total examples")
    return glob.glob(os.path.join(output_path, '*.tfrecords'))


def run_training(config, data_path, num_epochs, results_dir):
    """Run training for a single experiment configuration."""
    name = config['name']
    flags = config.get('flags', [])
    
    checkpoint_dir = os.path.join(results_dir, 'checkpoints', name)
    log_dir = os.path.join(results_dir, 'logs', name)
    epoch_log = os.path.join(log_dir, 'epoch_metrics.csv')
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable, 'run_training.py',
        '--data_path', data_path,
        '--checkpoint', checkpoint_dir,
        '--logpath', log_dir,
        '--num_epochs', str(num_epochs),
        '--epoch_log', epoch_log,
    ] + flags
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {name}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    # Run training
    start_time = datetime.now()
    result = subprocess.run(cmd, capture_output=False)
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    
    # Read final metrics
    final_metrics = {'train_loss': 'N/A', 'train_euc': 'N/A'}
    if os.path.exists(epoch_log):
        with open(epoch_log, 'r') as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1].strip().split(',')
                if len(last_line) >= 3:
                    final_metrics['train_loss'] = last_line[1]
                    final_metrics['train_euc'] = last_line[2]
    
    return {
        'name': name,
        'train_loss': final_metrics['train_loss'],
        'train_euc': final_metrics['train_euc'],
        'duration_sec': duration,
        'return_code': result.returncode
    }


def main():
    parser = argparse.ArgumentParser(description='Run domain randomization ablation experiments')
    parser.add_argument('--num_examples', type=int, default=10000,
                        help='Total number of training examples to generate (default: 10000)')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs per experiment (default: 20)')
    parser.add_argument('--num_files', type=int, default=10,
                        help='Number of TFRecord files to generate (default: 10)')
    parser.add_argument('--skip_data_gen', action='store_true',
                        help='Skip data generation if data already exists')
    parser.add_argument('--results_dir', type=str, default='./experiment_results',
                        help='Directory to save results (default: ./experiment_results)')
    args = parser.parse_args()
    
    # Define experiment configurations (ablation study from paper)
    experiments = [
        {'name': 'full_method', 'flags': [], 'description': 'All randomization enabled (baseline)'},
        {'name': 'no_noise', 'flags': ['--no_noise'], 'description': 'Disable image noise'},
        {'name': 'no_camera_rand', 'flags': ['--no_camera_rand'], 'description': 'Disable camera randomization'},
        {'name': 'no_distractors', 'flags': ['--no_distractors'], 'description': 'Disable distractor objects'},
    ]
    
    # Setup directories
    data_path = os.path.join(args.results_dir, 'data')
    os.makedirs(args.results_dir, exist_ok=True)
    
    print("="*60)
    print("Domain Randomization Ablation Study")
    print("="*60)
    print(f"Examples: {args.num_examples}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Results directory: {args.results_dir}")
    print()
    
    # Generate data if needed
    examples_per_file = args.num_examples // args.num_files
    if not args.skip_data_gen:
        print("Step 1: Generating training data...")
        generate_data(data_path, num_files=args.num_files, examples_per_file=examples_per_file)
    else:
        print("Step 1: Skipping data generation (--skip_data_gen)")
    
    # Run experiments
    print("\nStep 2: Running ablation experiments...")
    results = []
    
    for config in experiments:
        result = run_training(config, data_path, args.num_epochs, args.results_dir)
        results.append(result)
        print(f"  {result['name']}: loss={result['train_loss']}, euc={result['train_euc']}, time={result['duration_sec']:.1f}s")
    
    # Save results
    results_file = os.path.join(args.results_dir, 'ablation_results.csv')
    with open(results_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'train_loss', 'train_euc', 'duration_sec', 'return_code'])
        writer.writeheader()
        writer.writerows(results)
    
    # Print summary
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS")
    print("="*60)
    print(f"{'Experiment':<20} {'Loss':<12} {'Euclidean':<12} {'Time (s)':<10}")
    print("-"*60)
    for r in results:
        print(f"{r['name']:<20} {r['train_loss']:<12} {r['train_euc']:<12} {r['duration_sec']:<10.1f}")
    print("-"*60)
    print(f"\nResults saved to: {results_file}")
    
    # Create plotting script
    plot_script = os.path.join(args.results_dir, 'plot_results.py')
    with open(plot_script, 'w') as f:
        f.write('''#!/usr/bin/env python
"""Plot ablation study results."""
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load results
results_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(results_dir, 'ablation_results.csv'))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Loss comparison
axes[0].bar(df['name'], df['train_loss'].astype(float))
axes[0].set_xlabel('Experiment')
axes[0].set_ylabel('Training Loss')
axes[0].set_title('Ablation Study: Training Loss')
axes[0].tick_params(axis='x', rotation=45)

# Euclidean error comparison  
axes[1].bar(df['name'], df['train_euc'].astype(float))
axes[1].set_xlabel('Experiment')
axes[1].set_ylabel('Euclidean Error')
axes[1].set_title('Ablation Study: Euclidean Error')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'ablation_results.png'), dpi=150)
plt.show()
print(f"Plot saved to: {os.path.join(results_dir, 'ablation_results.png')}")
''')
    
    print(f"Plotting script saved to: {plot_script}")
    print("\nTo generate plots, run:")
    print(f"  python {plot_script}")


if __name__ == '__main__':
    main()
