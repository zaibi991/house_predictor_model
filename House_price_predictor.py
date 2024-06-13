# Importing necessary libraries
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Reading the housing data file
file_path = r'C:\Users\gulza\OneDrive\Desktop\Housing.csv'
house_data = pd.read_csv(file_path)

# Separating target variable 'price' and features
Y = house_data['price']
X_before_encoding = house_data[['guestroom', 'basement', 'furnishingstatus']]

# Encoding categorical features using LabelEncoder
label_encoder = LabelEncoder()
X_encoded = X_before_encoding.apply(lambda col: label_encoder.fit_transform(col))

# Selecting other numerical features
X_other = house_data[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]

# Concatenating numerical features and encoded categorical features
X = pd.concat([X_other, X_encoded], axis=1)

# Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# Printing the first 10 actual values of Y_test
print("Actual Y_test values:", Y_test[:10].tolist())

# Initializing and training the RandomForestRegressor model
house_model = RandomForestRegressor(random_state=1)
house_model.fit(X_train, Y_train)

# Predicting the target variable for the test set
Y_prediction = house_model.predict(X_test)

# Printing the first 10 predicted values
print("Predicted Y values:", Y_prediction[:10].tolist())

# Calculating evaluation metrics
mae = mean_absolute_error(Y_test, Y_prediction)
mse = mean_squared_error(Y_test, Y_prediction)
r2 = r2_score(Y_test, Y_prediction)
percentage_errors = (abs(Y_test - Y_prediction) / Y_test) * 100
percentage_errors_mean = percentage_errors.mean()

# Printing evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) Score:", r2)
print("Mean Percentage Error:", percentage_errors_mean)
