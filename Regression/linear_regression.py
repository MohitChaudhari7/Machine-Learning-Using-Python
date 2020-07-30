# Linear Regression

# Importing the libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../Datasets/Admission_Predict.csv')
ind = dataset.iloc[:,1].values  #independent variable 
dep = dataset.iloc[:, -1].values  #dependent variable
ind = ind.reshape(len(ind),1)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
ind_train, ind_test, dep_train, dep_test = train_test_split(ind, dep, test_size = 1/4, random_state = 0)


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(ind_train, dep_train)  #here we train the linear regression model with training set


# Visualising the Training set results
plt.scatter(ind_train, dep_train, color = 'red')
plt.plot(ind_train, regressor.predict(ind_train), color = 'blue')
plt.title('Selection Chance vs GRE Marks (Training set)')
plt.xlabel('GRE Marks')
plt.ylabel('Percentage Chance of getting selected')
plt.show()

# Predicting the Test set results
dep_pred = regressor.predict(ind_test)

# Visualising the Test set results
plt.scatter(ind_test, dep_test, color = 'red')
plt.plot(ind_train, regressor.predict(ind_train), color = 'blue') #here we plot the line that we got from the training data,so we can compare test set results
plt.title('Selection Chance vs GRE Marks (Test Set)')
plt.xlabel('GRE Marks')
plt.ylabel('Percentage Chance of getting selected')
plt.show()