import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
import math

# --- Configuration ---
output_filename_base = '_lowest_relative_minima_results.csv'
header_row = None  # Set to 0 if your files have headers

# --- Function to find the LOWEST relative minimum ---
def find_lowest_relative_minimum(data_series, trim_percent=0.1):
    """
    Finds the lowest relative minimum in a numeric series,
    excluding the last `trim_percent` of points to avoid tail-end artifacts.

    Returns:
        tuple: (lowest_value, index_in_original_series) or (None, None)
    """
    numeric_values = pd.to_numeric(data_series, errors='coerce').dropna()

    if len(numeric_values) < 3:
        return None, None

    trim_n = int(len(numeric_values) * trim_percent)
    if trim_n >= len(numeric_values) - 2:
        return None, None

    numeric_values_trimmed = numeric_values.iloc[:-trim_n]

    relative_minima = []
    for i in range(1, len(numeric_values_trimmed) - 1):
        if numeric_values_trimmed.iloc[i] < numeric_values_trimmed.iloc[i - 1] and \
           numeric_values_trimmed.iloc[i] < numeric_values_trimmed.iloc[i + 1]:
            relative_minima.append((numeric_values_trimmed.iloc[i], numeric_values_trimmed.index[i]))

    if relative_minima:
        lowest_val, idx = min(relative_minima, key=lambda x: x[0])
        if math.isnan(lowest_val):
            return None, None
        return lowest_val, idx

    return None, None

# --- File Selection ---
print("Please select the CSV files you want to process.")
root = tk.Tk()
root.withdraw()

selected_files = filedialog.askopenfilenames(
    title="Select Input CSV Files (Lowest relative minimum in 2nd column read will be found)",
    filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
)

if not selected_files:
    print("No files selected. Exiting.")
    exit()

csv_files = sorted(list(selected_files))

print(f"\nSelected {len(csv_files)} files:")
for f in csv_files:
    print(f"  - {os.path.basename(f)}")

output_directory = os.path.dirname(csv_files[0])
output_csv_filepath = os.path.join(output_directory, output_filename_base)
print(f"\nResults will be saved to: {output_csv_filepath}")

# Convert tuple to list and sort for potentially more consistent processing order
csv_files = sorted(list(selected_files))

# --- Ask for output filename prefix ---
from tkinter.simpledialog import askstring

print("Please enter an optional prefix for the output filename.")

prefix_root = tk.Tk()
prefix_root.withdraw()
filename_prefix = askstring("Filename Prefix", "Enter a prefix for the output CSV filename:")

if filename_prefix is None:
    filename_prefix = ""  # User cancelled input
else:
    filename_prefix = filename_prefix.strip()

output_filename = f"{filename_prefix}{output_filename_base}"

print(f"\nSelected {len(csv_files)} files:")
for f in csv_files:
    print(f"  - {os.path.basename(f)}")

# Determine output path using prefix
output_directory = os.path.dirname(csv_files[0])
output_csv_filepath = os.path.join(output_directory, output_filename)
print(f"\nResults will be saved to: {output_csv_filepath}")

# --- Main Logic ---
all_minima_results = []

for i, file_path in enumerate(csv_files):
    sample_number = i + 1
    filename = os.path.basename(file_path)
    print(f"\nProcessing file {sample_number}/{len(csv_files)}: {filename}")

    try:
        df = pd.read_csv(
            file_path,
            usecols=[1, 2],
            header=header_row,
            names=['Col_B', 'Col_C'] if header_row is None else None
        )

        if df.empty:
            print(f"  Warning: File '{filename}' is empty or has no data in specified columns. Skipping.")
            continue

        strain_col = df.columns[0]  # First column read
        data_col = df.columns[1]   # Second column read

        print(f"  Analyzing data from column: '{data_col}' (originally column index 2)")

        # Find lowest relative minimum and its index
        lowest_val, idx_in_df = find_lowest_relative_minimum(df[data_col], trim_percent=0.1)

        if lowest_val is not None:
            strain_val = df[strain_col].iloc[idx_in_df] if idx_in_df in df.index else None
            all_minima_results.append([sample_number, lowest_val, strain_val])
            print(f"  Found lowest relative minimum: {lowest_val} at strain: {strain_val}")
        else:
            print(f"  Warning: No relative minima found in column '{data_col}' for file '{filename}'.")

    except FileNotFoundError:
        print(f"  Error: File not found at {file_path}. Skipping.")
    except pd.errors.EmptyDataError:
        print(f"  Warning: File '{filename}' is empty. Skipping.")
    except IndexError:
        print(f"  Error: File '{filename}' does not have enough columns (required index 1 and 2). Skipping.")
    except ValueError as ve:
        print(f"  Error: Problem selecting columns from {filename}. Details: {ve}. Skipping.")
    except Exception as e:
        print(f"  An unexpected error occurred processing {filename}: {e}. Skipping.")

# --- Save Results ---
if not all_minima_results:
    print("\nNo valid relative minima found in any processed files. No output file created.")
elif any(item[1] is None for item in all_minima_results if len(item) > 1):
    print("\nNote: Some files did not contain any relative minima.")

if all_minima_results:
    processed_count = len([item for item in all_minima_results if len(item) > 1 and item[1] is not None])
    print(f"\nFound lowest relative minima in {processed_count} files.")
    results_df = pd.DataFrame(all_minima_results, columns=['Sample Number', 'Lowest Relative Minimum', 'Strain'])

    try:
        results_df.to_csv(output_csv_filepath, index=False)
        print(f"\nResults successfully saved to '{output_csv_filepath}'")
    except Exception as e:
        print(f"\nError saving results to CSV '{output_csv_filepath}': {e}")
