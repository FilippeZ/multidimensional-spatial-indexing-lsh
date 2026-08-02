"""
Data Ingestion and Sampling Pipeline module.
"""

import pandas as pd
import os
import numpy as np


def measure_file_size_mb(filepath: str) -> float:
    return os.path.getsize(filepath) / (1024 * 1024)


def write_and_measure(df: pd.DataFrame, output_file: str) -> float:
    df.to_csv(output_file, index=False)
    return measure_file_size_mb(output_file)


def combine_csv_approx_199mb(csv_files, output_file="data/data.csv", 
                             target_mb=199.0, tolerance_mb=1.0, max_iter=10):
    dfs = []
    for f in csv_files:
        if os.path.exists(f):
            print(f"Reading {f} ...")
            temp_df = pd.read_csv(f, low_memory=False)
            dfs.append(temp_df)
        else:
            print(f"Warning: File {f} does not exist.")

    if not dfs:
        raise FileNotFoundError("No input CSV files found to combine.")

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Initial combined shape: {combined_df.shape}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    size_mb = write_and_measure(combined_df, output_file)
    print(f"Initial file size: {size_mb:.2f} MB")

    if size_mb <= target_mb:
        print(f"Final file size {size_mb:.2f} MB is <= target ({target_mb} MB).")
        return combined_df

    iteration = 0
    while iteration < max_iter:
        if abs(size_mb - target_mb) <= tolerance_mb:
            print(f"File size {size_mb:.2f} MB is within ±{tolerance_mb} MB of {target_mb} MB.")
            break

        if size_mb < target_mb:
            print(f"File size dropped below target at {size_mb:.2f} MB. Stopping iteration.")
            break
        
        fraction = target_mb / size_mb
        sample_fraction = fraction * 0.99

        if sample_fraction <= 0.0:
            print("sample_fraction is 0 or negative. Not possible to continue.")
            break

        new_count = int(len(combined_df) * sample_fraction)
        if new_count < 1:
            print("Only one or zero rows left after sampling. Stopping.")
            break

        print(f"Iteration {iteration+1}: size = {size_mb:.2f} MB, sampling fraction ~ {sample_fraction:.4f} => {new_count} rows")
        combined_df = combined_df.sample(n=new_count, random_state=42).reset_index(drop=True)
        size_mb = write_and_measure(combined_df, output_file)
        print(f"New file size: {size_mb:.2f} MB")
        iteration += 1

    final_size = measure_file_size_mb(output_file)
    print(f"Final file size after iteration {iteration}: {final_size:.2f} MB")
    print(f"Final shape: {combined_df.shape}")
    return combined_df
