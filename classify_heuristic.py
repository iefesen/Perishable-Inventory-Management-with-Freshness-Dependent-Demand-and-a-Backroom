import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

# Ignore warnings
warnings.simplefilter('ignore')

# Load the Excel file into a DataFrame
file_path = 'heuristic_classification.xlsx'  # Replace with your actual file path
df = pd.read_excel(file_path, header=0)

# Separate features (X) and target variable (y)
X = df.iloc[:, :7]  # First 7 columns as features
y = df.iloc[:, 7]   # 8th column as target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Print the coefficients
coefficients = model.coef_[0]
print(f"Coefficients: {coefficients}")

# Print the intercept
intercept = model.intercept_[0]
print(f"Intercept: {intercept}")

# Get the probabilities for each class
probabilities = model.predict_proba(X_test)[:, 1]  # Probability of class 1

# Apply the custom threshold of 0.3
threshold = 0.5
y_pred_custom = (probabilities >= threshold).astype(int)

# Calculate and print the accuracy with the custom threshold
accuracy_custom = accuracy_score(y_test, y_pred_custom)
print(f"Accuracy with custom threshold of {threshold}: {accuracy_custom:.4f}")

# Make predictions on the entire dataset using the custom threshold
probabilities_full = model.predict_proba(X)[:, 1]
predictions_custom = (probabilities_full >= threshold).astype(int)

# Create a DataFrame with the predictions
output_df = X.copy()
output_df['Predicted_Custom'] = predictions_custom

# Export the predictions to an Excel file
output_file_path = 'predictions_logistic_regression_with_custom_threshold.xlsx'  # Output file path
output_df.to_excel(output_file_path, index=False)

# print(f"Predictions exported successfully to {output_file_path}")