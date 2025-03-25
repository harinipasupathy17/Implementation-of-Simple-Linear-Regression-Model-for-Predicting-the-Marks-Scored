# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import Libraries – Load necessary Python libraries. 2.Load Dataset – Read the dataset containing study hours and marks. 3.Preprocess Data – Check for missing values and clean the data if needed. 4.Split Data – Divide the dataset into training and testing sets. 5.Train Model – Fit a Simple Linear Regression model to the training data. 6.Make Predictions – Use the trained model to predict marks on the test data. 7.Evaluate Model – Calculate Mean Absolute Error (MAE) and R² score. 8.Visualize Results – Plot the regression line along with actual data points.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Harini P
RegisterNumber: 212224230082

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv("/content/drive/MyDrive/student_scores.csv")

print("Dataset Preview:\n", data.head())

print("\nMissing Values:\n", data.isnull().sum())

X = data[['Hours']]  # Study hours
y = data['Scores']   # Marks scored

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nIntercept:", model.intercept_)
print("Slope:", model.coef_[0])

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error:", mae)
print("R² Score:", r2)

plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression - Marks Prediction")
plt.legend()
plt.show()

*/
```

## Output:
![Screenshot 2025-03-25 205253](https://github.com/user-attachments/assets/df0be18e-c7d6-4027-a087-e567da9ea222)

![Screenshot 2025-03-25 205306](https://github.com/user-attachments/assets/b39d9c54-95b5-4f2d-ac89-d05f616b763c)

![Screenshot 2025-03-25 205329](https://github.com/user-attachments/assets/0ef586e2-6a62-4656-9781-128c78f8e9d6)

![Screenshot 2025-03-25 205341](https://github.com/user-attachments/assets/1673b25b-d84d-4806-86f4-25a114380498)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
