"""
Data Ingestion Entry Point for Multidimensional Spatial Indexing & LSH.
Combines annual CMS HCAHPS CSV files into data/data.csv (~199MB).
"""

import os
import sys

# Ensure src directory is in Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pipeline.combine_csv import combine_csv_approx_199mb

if __name__ == "__main__":
    csv_files = [
        "cms_hospital_patient_satisfaction_2016.csv",
        "cms_hospital_patient_satisfaction_2017.csv",
        "cms_hospital_patient_satisfaction_2018.csv",
        "cms_hospital_patient_satisfaction_2019.csv",
        "cms_hospital_patient_satisfaction_2020.csv",
    ]
    
    # Also check data/ subdirectory if files were placed there
    resolved_files = []
    for f in csv_files:
        if os.path.exists(f):
            resolved_files.append(f)
        elif os.path.exists(os.path.join("data", f)):
            resolved_files.append(os.path.join("data", f))
        else:
            resolved_files.append(f)

    output_file = os.path.join("data", "data.csv")

    print(f"Starting data combination pipeline target: {output_file}...")
    final_df = combine_csv_approx_199mb(
        resolved_files,
        output_file=output_file,
        target_mb=199.0,
        tolerance_mb=1.0,
        max_iter=10
    )
    print("Ingestion pipeline completed successfully!")
