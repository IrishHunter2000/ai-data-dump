import pandas as pd
import print_rows

# Define the filename of your filled CSV
input_csv_filename = "yamaha_r1_pricing.csv"

# Load the CSV file into a DataFrame
try:
    df = pd.read_csv(input_csv_filename)
    print(f"Successfully loaded '{input_csv_filename}'.")
    print("\nFirst 5 rows of your loaded DataFrame:")
    print(df.head())
    print(f"\nTotal rows loaded: {len(df)}")
    print("Printing 5 rows now")
    print_rows.print5(df)
except FileNotFoundError:
    print(f"Error: The file '{input_csv_filename}' was not found.")
    print("Please make sure the CSV file is in the same directory as your Jupyter Notebook.")
    print("Or provide the full path to the file.")