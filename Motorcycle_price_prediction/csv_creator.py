import pandas as pd
import numpy as np

# --- Configuration for your CSV ---
num_rows = 100 # Number of data entries you want
make_name = "Yamaha"
model_name = "R1"
year_range = (2015, 2025) # Inclusive range for years
mileage_range = (1000, 12000) # Inclusive range for mileage
condition_score_range = (7, 10) # Inclusive range for condition score (e.g., 7=good, 10=excellent)
output_csv_filename = "yamaha_r1_data_to_fill.csv"

# --- Generate the data ---

# Generate random years
# np.random.randint(low, high, size) generates integers in [low, high)
years = np.random.randint(year_range[0], year_range[1] + 1, size=num_rows)

# Generate random mileages
mileages = np.random.randint(mileage_range[0], mileage_range[1] + 1, size=num_rows)

# Generate random condition scores
condition_scores = np.random.randint(condition_score_range[0], condition_score_range[1] + 1, size=num_rows)

# Create constant lists for Make and Model
makes = [make_name] * num_rows
models = [model_name] * num_rows

# Create an empty list/array for prices, which you will fill in
# Using np.nan (Not a Number) is a common way to represent missing numerical data
prices = [np.nan] * num_rows

# Create a dictionary from the generated lists
data = {
    'Make': makes,
    'Model': models,
    'Year': years,
    'Mileage': mileages,
    'Condition_Score': condition_scores,
    'Price': prices
}

# Create the DataFrame
df_r1_empty_price = pd.DataFrame(data)

# Sort the DataFrame by Year and then Mileage for better readability
df_r1_empty_price = df_r1_empty_price.sort_values(by=['Year', 'Mileage']).reset_index(drop=True)

# Save the DataFrame to a CSV file
df_r1_empty_price.to_csv(output_csv_filename, index=False)

print(f"CSV file '{output_csv_filename}' created successfully!")
print("It contains randomized 'Year', 'Mileage', and 'Condition_Score' for Yamaha R1.")
print("The 'Price' column is empty for you to fill in.")
print("\nFirst 5 rows of the generated DataFrame:")
print(df_r1_empty_price.head())