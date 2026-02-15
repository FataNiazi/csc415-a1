#!/usr/bin/env python
"""
Reproduce the ablation study from:
  "Domain Randomization for Transferring Deep Neural Networks
   from Simulation to the Real World" (Tobin et al., 2017)

Table 1 ablation: systematically remove one randomization component
at a time to measure its contribution to sim-to-real transfer.

Usage:
  TF_XLA_FLAGS="--tf_xla_auto_jit=2" python run_ablation_experiments.py
"""

import os, sys, subprocess, csv

EPOCHS     = 20
NUM_FILES  = 10
DATA_DIR   = "./experiment_results/data"
RESULTS_DIR= "./experiment_results"

EXPERIMENTS = [
    ("full_method",     []),
    ("no_noise",        ["--no_noise"]),
    ("no_camera_rand",  ["--no_camera_rand"]),
    ("no_distractors",  ["--no_distractors"]),
]

def generate_data():
    """Generate TFRecord training data via domain randomization."""
    import time, glob
    import tensorflow as tf
    sys.path.insert(0, os.path.dirname(__file__))
    from domrand.sim_manager import SimManager

    os.makedirs(DATA_DIR, exist_ok=True)
    if glob.glob(os.path.join(DATA_DIR, "*.tfrecords")):
        print(f"Data already exists in {DATA_DIR}, skipping generation.")
        return

    sim = SimManager(filepath="xmls/fetch/main.xml")
    for i in range(NUM_FILES):
        fname = os.path.join(DATA_DIR, f"{time.strftime('%Y%m%d-%H%M%S')}-{i}.tfrecords")
        writer = tf.io.TFRecordWriter(fname)
        for _ in range(1000):
            sim.randomize()
            img = sim.render()
            pos = sim.get_actuator_positions()
            feat = {
                "image":  tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
                "label":  tf.train.Feature(float_list=tf.train.FloatList(value=pos)),
                "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[img.shape[0]])),
                "width":  tf.train.Feature(int64_list=tf.train.Int64List(value=[img.shape[1]])),
                "depth":  tf.train.Feature(int64_list=tf.train.Int64List(value=[img.shape[2]])),
            }
            writer.write(tf.train.Example(features=tf.train.Features(feature=feat)).SerializeToString())
        writer.close()
        time.sleep(0.1)
        print(f"  [{i+1}/{NUM_FILES}] {fname}")
    print("Data generation complete.\n")

def train(name, flags):
    """Run one training experiment, return final metrics."""
    ckpt = os.path.join(RESULTS_DIR, "checkpoints", name)
    logs = os.path.join(RESULTS_DIR, "logs", name)
    csv_path = os.path.join(logs, "epoch_metrics.csv")
    os.makedirs(ckpt, exist_ok=True)
    os.makedirs(logs, exist_ok=True)

    cmd = [sys.executable, "run_training.py",
           "--data_path", DATA_DIR,
           "--checkpoint", ckpt,
           "--logpath", logs,
           "--num_epochs", str(EPOCHS),
           "--epoch_log", csv_path] + flags

    print(f"\n{'='*50}\n  {name}\n{'='*50}")
    subprocess.run(cmd)

    # read final epoch metrics
    loss, euc = "N/A", "N/A"
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for line in f: pass
        parts = line.strip().split(",")
        if len(parts) >= 3:
            loss, euc = parts[1], parts[2]
    return loss, euc

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Generating data...")
    generate_data()

    print("Running ablation experiments...")
    rows = []
    for name, flags in EXPERIMENTS:
        loss, euc = train(name, flags)
        rows.append((name, loss, euc))

    # save results
    out = os.path.join(RESULTS_DIR, "ablation_results.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "train_loss", "train_euc"])
        w.writerows(rows)

    # print table
    print(f"\n{'='*50}")
    print("ABLATION RESULTS")
    print(f"{'='*50}")
    print(f"{'Experiment':<20}{'Loss':<12}{'Euc Error':<12}")
    print("-"*44)
    for name, loss, euc in rows:
        print(f"{name:<20}{loss:<12}{euc:<12}")
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
