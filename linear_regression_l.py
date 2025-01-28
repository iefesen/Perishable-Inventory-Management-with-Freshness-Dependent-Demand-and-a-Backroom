# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 16:55:54 2024

@author: 20224695
"""



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Excel file
file_path = 'average_rewards_threshold_l_analysis.xlsx'
df = pd.read_excel(file_path)

# Select the first six columns (assuming columns are labeled 'A' to 'F') and column 'W'
selected_columns = ['K', 'I', 'L', 'p', 's', 'lambda_c', 'P_b']
target_column = 'max_threshold'
X = df[selected_columns]
y = df[target_column]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print("Mean Squared Error:", mse)
print("R-squared:", r2)


coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)
