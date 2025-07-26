import pandas as pd

# Define the filename of your filled CSV
input_csv_filename = "yamaha_r1_pricing.csv"

# Load the CSV file into a DataFrame
try:
    df = pd.read_csv(input_csv_filename)
    print(f"Successfully loaded '{input_csv_filename}'.")
    print("\nFirst 5 rows of your loaded DataFrame:")
    print(df.head())
    print(f"\nTotal rows loaded: {len(df)}")
except FileNotFoundError:
    print(f"Error: The file '{input_csv_filename}' was not found.")
    print("Please make sure the CSV file is in the same directory as your Jupyter Notebook.")
    print("Or provide the full path to the file.")

X = df[['Make', 'Model', 'Year', 'Mileage', 'Condition_Score']]
y = df['Price']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression # We'll use this for the model

# Define which columns are categorical and which are numerical
# Even though Make and Model are constant, we treat them as categorical for pipeline consistency
categorical_features = ['Make', 'Model']
numerical_features = ['Year', 'Mileage', 'Condition_Score']

# Create a preprocessor using ColumnTransformer
# It applies OneHotEncoder to categorical features and 'passthrough' for numerical
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

# Split the data into training and testing sets
# test_size=0.2 means 20% of the data will be used for testing, 80% for training
# random_state ensures that your split is the same every time you run the code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a machine learning pipeline
# Step 1: Preprocess the data using our defined 'preprocessor'
# Step 2: Train a Linear Regression model
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', LinearRegression())])

# Train the model using the training data
print("\nTraining the Linear Regression model pipeline...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

from sklearn.metrics import mean_absolute_error, r2_score

# Make predictions on the test set
y_pred = model_pipeline.predict(X_test)

print("\n--- Model Evaluation ---")

# Mean Absolute Error (MAE): Average absolute difference between predicted and actual values
# A lower MAE means better accuracy.
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): ${mae:.2f}")

# R-squared (R2 Score): Indicates how well the model's predictions approximate the real data points.
# 1.0 is a perfect fit, 0.0 means the model performs no better than predicting the mean.
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R2 Score): {r2:.2f}")

print("\n--- Sample Predictions vs. Actuals (from Test Set) ---")
# Display some actual vs. predicted values from the test set
results_df = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred.round(2)})
# Reset index to align for display, as y_test might have original DataFrame indices
results_df = results_df.reset_index(drop=True)
print(results_df)

# --- Predict a new, unseen motorcycle price ---
print("\n--- Predicting a new Yamaha R1 price ---")
new_motorcycle_data = pd.DataFrame([{
    'Make': 'Yamaha',
    'Model': 'R1',
    'Year': 2024, # A new year within your range
    'Mileage': 1500, # Low mileage
    'Condition_Score': 10 # Excellent condition
}])

predicted_new_price = model_pipeline.predict(new_motorcycle_data)
print(f"Predicted price for a 2024 Yamaha R1, 1500 miles, Condition 10: ${predicted_new_price[0]:.2f}")

# Another example
new_motorcycle_data_2 = pd.DataFrame([{
    'Make': 'Yamaha',
    'Model': 'R1',
    'Year': 2016, # Older year
    'Mileage': 10000, # High mileage
    'Condition_Score': 7 # Good condition
}])

predicted_new_price_2 = model_pipeline.predict(new_motorcycle_data_2)
print(f"Predicted price for a 2016 Yamaha R1, 10000 miles, Condition 7: ${predicted_new_price_2[0]:.2f}")

import numpy as np

# Assuming model_pipeline is your trained pipeline from previous steps
regressor = model_pipeline.named_steps['regressor']
preprocessor = model_pipeline.named_steps['preprocessor']

# Get the feature names in the order they were processed by the preprocessor
feature_names_out = preprocessor.get_feature_names_out()
print("Order of features used by the model:", feature_names_out)

# Get the coefficients and intercept from the trained Linear Regression model
coefficients = regressor.coef_
intercept = regressor.intercept_

print("\nRaw Coefficients:", coefficients)
print("Raw Intercept:", intercept)

# --- Calculate the 'effective_intercept' and individual coefficients for JS ---
# In your R1-only dataset, 'Make_Yamaha' and 'Model_R1' are always 1 after one-hot encoding.
# Their coefficients effectively get added to the intercept.

effective_intercept_for_js = intercept

# Find indices for 'cat__Make_Yamaha' and 'cat__Model_R1'
# These might not exist if your OneHotEncoder only saw one category, but it's safer to check.
idx_make_yamaha = np.where(feature_names_out == 'cat__Make_Yamaha')[0][0] if 'cat__Make_Yamaha' in feature_names_out else -1
idx_model_r1 = np.where(feature_names_out == 'cat__Model_R1')[0][0] if 'cat__Model_R1' in feature_names_out else -1

if idx_make_yamaha != -1:
    effective_intercept_for_js += coefficients[idx_make_yamaha]
if idx_model_r1 != -1:
    effective_intercept_for_js += coefficients[idx_model_r1]

print("\nEffective Intercept for JS (includes constant Make/Model effects):", effective_intercept_for_js)

# Find indices for numerical features
idx_year = np.where(feature_names_out == 'num__Year')[0][0]
idx_mileage = np.where(feature_names_out == 'num__Mileage')[0][0]
idx_condition = np.where(feature_names_out == 'num__Condition_Score')[0][0]

print("Coefficient for Year (coef_year):", coefficients[idx_year])
print("Coefficient for Mileage (coef_mileage):", coefficients[idx_mileage])
print("Coefficient for Condition_Score (coef_condition_score):", coefficients[idx_condition])

output_csv_filename = "coefficients.csv"

data = {
    'effective_intercept': [effective_intercept_for_js],
    'coef_year': [coefficients[idx_year]],
    'coef_mileage': [coefficients[idx_mileage]],
    'coef_condition_score': [coefficients[idx_condition]]
}

# Create the DataFrame
df_r1_empty_price = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df_r1_empty_price.to_csv(output_csv_filename, index=False)
print(f"CSV file '{output_csv_filename}' created successfully!")