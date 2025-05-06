import pandas as pd
import tkinter as tk
from tkinter import filedialog, simpledialog
import os # Import os for path manipulation

def merge_csv_files(csv_files, output_file, base_name):
    """
    Merges multiple CSV files into a single Excel file, with each CSV in a separate sheet.
    Attempts to convert string columns to numeric types (float/int) before saving.

    Args:
        csv_files (list): A list of paths to the CSV files.
        output_file (str): The path for the output Excel file.
        base_name (str): The base name to use for sheet names.
    """
    print(f"Starting merge process. Output will be saved to: {output_file}")
    try:
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            for i, file in enumerate(csv_files):
                print(f"Processing file {i+1}/{len(csv_files)}: {os.path.basename(file)}...")
                try:
                    # Read the CSV file
                    df = pd.read_csv(file)

                    if not df.empty:
                        # --- CONVERSION LOGIC START ---
                        print(f"  Attempting numeric conversion for columns in {os.path.basename(file)}...")
                        for col in df.columns:
                            # Try converting column to numeric.
                            # 'coerce' turns non-numeric values into NaN.
                            # Pandas will attempt int first, then float if necessary (e.g., decimals or NaN present)
                            original_dtype = df[col].dtype
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            # Optional: Print if dtype changed (for debugging/info)
                            # if df[col].dtype != original_dtype:
                            #     print(f"    Column '{col}' converted from {original_dtype} to {df[col].dtype}")
                        # --- CONVERSION LOGIC END ---

                        # Create a unique sheet name
                        sheet_name = f"{base_name}-{i+1}"
                        # Ensure sheet name is valid for Excel (max 31 chars, no invalid chars)
                        sheet_name = sheet_name[:31].replace(':', '_').replace('\\', '_').replace('/', '_').replace('?', '_').replace('*', '_').replace('[', '_').replace(']', '_')

                        print(f"  Writing data to sheet: '{sheet_name}'")
                        # Write the potentially converted DataFrame to the Excel sheet
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                    else:
                        print(f"  Warning: File {os.path.basename(file)} is empty. Skipping.")

                except FileNotFoundError:
                    print(f"  Error: File not found - {file}. Skipping.")
                except pd.errors.EmptyDataError:
                    print(f"  Warning: File {os.path.basename(file)} is empty or contains no data. Skipping.")
                except Exception as e:
                    print(f"  Error processing file {os.path.basename(file)}: {e}. Skipping.")

        print(f"\nMerge complete. Output saved to: {output_file}")

    except Exception as e:
        print(f"\nError creating or writing to Excel file {output_file}: {e}")
        print("Please ensure the file is not open elsewhere and you have write permissions.")


def get_csv_files():
    """Opens a file dialog for the user to select multiple CSV files."""
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    print("Opening file dialog to select CSV files...")
    files_selected = filedialog.askopenfilenames(
        title="Select CSV files to merge",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy() # Close the hidden tkinter window
    if not files_selected:
        print("No files selected. Exiting.")
        return None
    print(f"Selected {len(files_selected)} files.")
    return files_selected

def get_base_name():
    """Opens a dialog for the user to enter a base name."""
    root = tk.Tk()
    root.withdraw() # Hide the main tkinter window
    print("Opening dialog to enter base name...")
    base_name = simpledialog.askstring("Base Name", "Enter the base name for sheet names and output file:")
    root.destroy() # Close the hidden tkinter window
    if not base_name:
        print("No base name entered. Using 'merged_output'.")
        return "merged_output" # Provide a default if none entered
    return base_name

if __name__ == "__main__":
    csv_files = get_csv_files()

    if csv_files: # Proceed only if files were selected
        base_name = get_base_name()

        if base_name: # Proceed only if a base name was provided or defaulted
            # Get the directory of the *first* selected file to save the output there
            output_folder = os.path.dirname(csv_files[0])
            output_file = os.path.join(output_folder, f"{base_name}-refined.xlsx")

            merge_csv_files(csv_files, output_file, base_name)
        else:
             # This case is handled by the default in get_base_name now, but kept for robustness
             print("Base name input cancelled. Exiting.")
    # else: # Message handled in get_csv_files
    #    print("File selection cancelled. Exiting.")