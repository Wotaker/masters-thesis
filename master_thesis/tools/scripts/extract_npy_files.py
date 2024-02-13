"""
Extracts .npy files from effective connectivity pipeline results directory `--input_dir`,
and saves them in a new directory `--output_dir`.
"""

from argparse import ArgumentParser

import os
import numpy as np

if __name__ == "__main__":

    # Parse arguments
    parser = ArgumentParser(description="Extract .npy files from a directory")
    parser.add_argument("-i", "--input_dir", help="Input directory", required=True)
    parser.add_argument("-o", "--output_dir", help="Output directory", required=True)
    args = parser.parse_args()

    # Get input and output directories
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Get subject directories
    subject_dirs = os.listdir(input_dir)
    subject_dirs = [os.path.join(input_dir, subject_dir) for subject_dir in subject_dirs]
    subject_dirs = [subject_dir for subject_dir in subject_dirs if os.path.isdir(subject_dir)]

    # Extract .npy files
    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)
        subject_path = os.path.join(input_dir, subject_id)
        os.system(f"cp {subject_path}/*.npy {output_dir}")
