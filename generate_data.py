#!/usr/bin/env python3
"""
Generate training and test data for domain randomization ablation study.

Usage:
    python generate_data.py
"""

import os
import sys

os.environ['MUJOCO_GL'] = 'egl'

import time
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.lib.io.tf_record import TFRecordOptions, TFRecordCompressionType, TFRecordWriter
from domrand.sim_manager import SimManager

# Configuration
DATA_DIR = "./ablation_data"
NUM_TRAIN_SAMPLES = 10000  # Per ablation variant
NUM_TEST_PER_GROUP = 20    # Per test group
SAMPLES_PER_FILE = 1000    # Split into multiple files


class AblationSimManager(SimManager):
    """SimManager with configurable randomization for ablation study."""

    def __init__(self, filepath, enable_noise=True, enable_camera_rand=True,
                 enable_distractors=True, **kwargs):
        self.enable_noise = enable_noise
        self.enable_camera_rand = enable_camera_rand
        self.enable_distractors = enable_distractors
        super().__init__(filepath, **kwargs)

    def _randomize(self):
        """Randomize with ablation controls."""
        self._rand_textures()

        if self.enable_camera_rand:
            self._rand_camera()

        self._rand_lights()
        self._rand_object()
        self._rand_walls()

        if self.enable_distractors:
            self._rand_distract()
        else:
            # Hide all distractors (there are 20: distract0 - distract19)
            self._set_visible('distract', 20, False)

    def _get_cam_frame(self, ground_truth=None):
        """Get camera frame with optional noise."""
        import skimage
        from domrand.utils.image import preproc_image

        # Render using dm_control renderer
        self.renderer.update_scene(self.data, camera='camera1')
        cam_img = self.renderer.render()  # Returns RGB image (480, 640, 3)

        if self.enable_noise:
            noise_var = np.random.uniform(0.0, 0.001)
            cam_img = (skimage.util.random_noise(cam_img, mode='gaussian', var=noise_var) * 255).astype(np.uint8)

        if self.display_data:
            from domrand.utils.image import display_image
            print(ground_truth)
            display_image(cam_img, mode='preproc')

        cam_img = preproc_image(cam_img)
        return cam_img


def write_tfrecord(sim, filename, num_samples):
    """Write samples to TFRecord file."""
    with TFRecordWriter(filename, options=TFRecordOptions(TFRecordCompressionType.GZIP)) as writer:
        for i in range(num_samples):
            image, label = sim.get_data()
            assert image.dtype == np.uint8

            example = tf.train.Example(features=tf.train.Features(feature={
                'label_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.astype(np.float32).tobytes()])),
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
            }))
            writer.write(example.SerializeToString())


def generate_test_data():
    """Generate test data: 20 samples × 3 groups = 60 total."""
    print("Generating test data (60 samples)...")

    test_dir = os.path.join(DATA_DIR, "test")
    os.makedirs(test_dir, exist_ok=True)

    # Test group 1: Object only (no distractors)
    sim = AblationSimManager(
        filepath="xmls/kuka/lbr4_reflex.xml",
        gpu_render=True,
        enable_noise=False,
        enable_camera_rand=True,
        enable_distractors=False
    )
    write_tfrecord(sim, os.path.join(test_dir, "test_object_only.tfrecords"), NUM_TEST_PER_GROUP)

    # Test group 2: With distractors
    sim = AblationSimManager(
        filepath="xmls/kuka/lbr4_reflex.xml",
        gpu_render=True,
        enable_noise=False,
        enable_camera_rand=True,
        enable_distractors=True
    )
    write_tfrecord(sim, os.path.join(test_dir, "test_distractors.tfrecords"), NUM_TEST_PER_GROUP)

    # Test group 3: With occlusions
    sim = AblationSimManager(
        filepath="xmls/kuka/lbr4_reflex.xml",
        gpu_render=True,
        enable_noise=False,
        enable_camera_rand=True,
        enable_distractors=True
    )
    write_tfrecord(sim, os.path.join(test_dir, "test_occlusions.tfrecords"), NUM_TEST_PER_GROUP)

    print("Test data complete")


def generate_training_variant(variant_name, **kwargs):
    """Generate training samples for one ablation variant."""
    variant_dir = os.path.join(DATA_DIR, "train", variant_name)
    os.makedirs(variant_dir, exist_ok=True)

    sim = AblationSimManager(filepath="xmls/kuka/lbr4_reflex.xml", gpu_render=True, **kwargs)

    num_files = NUM_TRAIN_SAMPLES // SAMPLES_PER_FILE
    for file_idx in range(num_files):
        fname = os.path.join(variant_dir, f"data_{file_idx:03d}.tfrecords")
        write_tfrecord(sim, fname, SAMPLES_PER_FILE)


def generate_training_data():
    """Generate training data for all 4 ablation variants."""
    print(f"Generating training data ({NUM_TRAIN_SAMPLES * 4} samples)...")

    variants = [
        ("full_method", {
            "enable_noise": True,
            "enable_camera_rand": True,
            "enable_distractors": True
        }),
        ("no_noise", {
            "enable_noise": False,
            "enable_camera_rand": True,
            "enable_distractors": True
        }),
        ("no_camera_rand", {
            "enable_noise": True,
            "enable_camera_rand": False,
            "enable_distractors": True
        }),
        ("no_distractors", {
            "enable_noise": True,
            "enable_camera_rand": True,
            "enable_distractors": False
        })
    ]

    for i, (name, kwargs) in enumerate(variants, 1):
        print(f"  [{i}/4] {name}")
        generate_training_variant(name, **kwargs)

    print("Training data complete")


def main():
    print(f"Domain Randomization Ablation - Data Generation")
    print(f"Data directory: {DATA_DIR}")

    generate_test_data()
    generate_training_data()

    print(f"\nComplete! Data saved to {DATA_DIR}/")


if __name__ == "__main__":
    main()
