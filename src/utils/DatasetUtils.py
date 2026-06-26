import os, wfdb
import pandas as pd
import numpy as np

def clean_physio_data(input_file, output_file, z_threshold=4.0):
    """
    Cleans physiological time-series data by handling empty rows, NaNs, and outliers.
    Uses absolute paths for now.
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    initial_length = len(df)
    
    # Drop if a row has no data across all columns
    df.dropna(how='all', inplace=True)
    print(f"Dropped {initial_length - len(df)} empty rows.")

    # Interpolate across incomplete data points
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        df.interpolate(method='linear', limit_direction='both', inplace=True)
        print(f"Interpolated {nan_count} missing values (NaNs).")
    else:
        print("No NaNs found.")

    # Clip unacceptably large values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if "y" in numeric_cols:
        numeric_cols = numeric_cols.drop("y")

    outliers_capped = 0
    for col in numeric_cols:
        col_mean = df[col].mean()
        col_std = df[col].std()
        
        lower_bound = col_mean - (z_threshold * col_std)
        upper_bound = col_mean + (z_threshold * col_std)
    
        outliers_capped += ((df[col] > upper_bound) | (df[col] < lower_bound)).sum()
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
    print(f"Capped {outliers_capped} outlier data points.")

    df.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved successfully to {output_file}!")

def download_physionet_record(record_name, db_dir, download_path):
    '''
    Download a record from a physionet database.
    Mainly used for .dat and .hea files (ignore .atr files).
    To download record named "100" from the URL https://physionet.org/content/mitdb/1.0.0/, pass record_name=100 and db_dir="mitdb"
    '''
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_download_path=os.path.join(script_dir, download_path)
    os.makedirs(local_download_path, exist_ok=True)

    dat_path = os.path.join(local_download_path, f"{record_name}.dat")
    hea_path = os.path.join(local_download_path, f"{record_name}.hea")
    if os.path.exists(dat_path) and os.path.exists(hea_path):
        print("Files already exist! Skipping download.")
    else:
        print(f"Starting download of {record_name} from {db_dir} on Physionet...")
        wfdb.dl_database(db_dir, local_download_path, records=[record_name])