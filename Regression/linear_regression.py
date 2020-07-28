# Linear Regression

# Importing the libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../Datasets/Admission_Predict.csv')
x = dataset.iloc[:,1].values  #independent variable 
y = dataset.iloc[:, -1].values  #dependent variable
x = x.reshape(len(x),1)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/4, random_state = 0)


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)  #here we train the linear regression model with training set


# Visualising the Training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Selection Chance vs GRE Marks (Training set)')
plt.xlabel('GRE Marks')
plt.ylabel('Percentage Chance of getting selected')
plt.show()

# Predicting the Test set results
y_pred = regressor.predict(x_test)

# Visualising the Test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue') #here we plot the line that we got from the training data,so we can compare test set results
plt.title('Selection Chance vs GRE Marks (Test Set)')
plt.xlabel('GRE Marks')
plt.ylabel('Percentage Chance of getting selected')
plt.show()