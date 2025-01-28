import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the Excel file into a DataFrame
file_path = 'threshold_prediction.xlsx'  # Replace with your actual file path
df = pd.read_excel(file_path, header=0)

# Separate features (X) and target variables (y)
X = df.iloc[:, :7]  # First 7 columns as features
inventory_threshold = df.iloc[:, 7]  # 8th column as target variable
lifetime_threshold = df.iloc[:, 8]  # 9th column as target variable

# Add a constant to the features (intercept term)
X = sm.add_constant(X)

# Split the data into 20% train and 80% test
X_train, X_test, y_train_inventory, y_test_inventory = train_test_split(
    X, inventory_threshold, test_size=0.8, random_state=42)
_, _, y_train_lifetime, y_test_lifetime = train_test_split(
    X, lifetime_threshold, test_size=0.8, random_state=42)

# Fit a linear regression model for inventory threshold
model_inventory = sm.OLS(y_train_inventory, X_train)
results_inventory = model_inventory.fit()

# Fit a linear regression model for lifetime threshold
model_lifetime = sm.OLS(y_train_lifetime, X_train)
results_lifetime = model_lifetime.fit()

# Predict on the test set
inventory_predictions = results_inventory.predict(X_test)
lifetime_predictions = results_lifetime.predict(X_test)

# Calculate R^2 for inventory threshold
r2_inventory = r2_score(y_test_inventory, inventory_predictions)

# Calculate R^2 for lifetime threshold
r2_lifetime = r2_score(y_test_lifetime, lifetime_predictions)

# Predict on the complete data set and round predictions to nearest integer
complete_inventory_predictions = results_inventory.predict(X).round()
complete_lifetime_predictions = results_lifetime.predict(X).round()

# Create a DataFrame with the original data and the rounded predictions
predictions_df = df.copy()
predictions_df['Rounded_Inventory_Predictions'] = complete_inventory_predictions
predictions_df['Rounded_Lifetime_Predictions'] = complete_lifetime_predictions

print("Coefficients for Inventory Threshold Regression:")
print(results_inventory.params)
print("R^2 for Inventory Threshold Model:", r2_inventory)

print("\nCoefficients for Lifetime Threshold Regression:")
print(results_lifetime.params)
print("R^2 for Lifetime Threshold Model:", r2_lifetime)

# Save the predictions to an Excel file
# output_file_path = 'rounded_predictions.xlsx'  # Output file path
# predictions_df.to_excel(output_file_path, index=False)

# print(f"Rounded predictions have been saved to {output_file_path}")