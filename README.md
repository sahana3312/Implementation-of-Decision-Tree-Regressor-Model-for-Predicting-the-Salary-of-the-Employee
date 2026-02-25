# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Collect and prepare the employee dataset with features (experience, age, skills) and salary as the target variable.

2. Split the dataset into training and testing sets and train the Decision Tree Regressor model on the training data.

3. The model creates decision rules by splitting the data based on feature values to minimize error (MSE).

4. Use the trained model to predict employee salary and evaluate performance using metrics like MSE or R².

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SAHANA S
RegisterNumber:  212225040356
*/
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
dataset = pd.read_csv("C:/Users/acer/Downloads/Salary (1).csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)
salary_pred = regressor.predict([[6.5]])
print("Predicted Salary:", salary_pred)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Decision Tree Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

```
## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)

<img width="817" height="627" alt="image" src="https://github.com/user-attachments/assets/7a13af2a-1847-4d37-8684-6787566065d4" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
