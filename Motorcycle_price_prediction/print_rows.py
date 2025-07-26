# Define features (X) and target (y)
# 'Make' and 'Model' are constant for this dataset, but we include them for consistency
# in case you expand later. 'Condition_Score' is now numerical.
def print5(df):


    X = df[['Make', 'Model', 'Year', 'Mileage', 'Condition_Score']]
    y = df['Price']

    print("\nFeatures (X) snapshot:")
    print(X.head())
    print("\nTarget (y) snapshot:")
    print(y.head())