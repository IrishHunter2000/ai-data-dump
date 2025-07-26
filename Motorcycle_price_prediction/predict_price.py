import math as Math
import pandas as pd

input_csv_filename = "coefficients.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(input_csv_filename)

first_row = df.iloc[0]
effective_intercept = first_row['effective_intercept']
coef_year = first_row['coef_year']
coef_mileage = first_row['coef_mileage']
coef_condition_score = first_row['coef_condition_score']

#effective_intercept = -1183416.758331413 # Example: intercept + (coef_Make_Yamaha * 1) + (coef_Model_R1 * 1)
#coef_year = 593.5466154704053
#coef_mileage = -0.10354793106844397
#coef_condition_score = 62.34381838486461


yearInput = input("Enter Year: ")
mileageInput = input("Enter mileage: ")
conditionInput = input("Enter Contition Score: ")


# Get values and convert to numbers
year = float(yearInput)
mileage = float(mileageInput)
condition_score = float(conditionInput)

# Basic input validation
if Math.isnan(year) or Math.isnan(mileage) or Math.isnan(condition_score) or year < 2015 or year > 2025 or mileage < 1000 or mileage > 12000 or condition_score < 7 or condition_score > 10:
    result = 'Please enter valid numbers within the specified ranges. Invalid input'


# Calculate the predicted price using the Linear Regression formula
# Price = Intercept + (Coef_Year * Year) + (Coef_Mileage * Mileage) + (Coef_Condition * Condition_Score)
# Note: Make_Yamaha and Model_R1 are implicitly handled by the 'effective_intercept'
# because they are constant (always 1) for this specific R1 predictor.
predictedPrice = effective_intercept + (coef_year * year) + (coef_mileage * mileage) + (coef_condition_score * condition_score)

# Ensure price is not negative (though unlikely with realistic coefficients)
predictedPrice = round(max(0, predictedPrice), 2)

result = f"Predicted Price: ${predictedPrice}" 
print(result)