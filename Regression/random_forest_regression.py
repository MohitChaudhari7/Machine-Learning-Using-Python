# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../Datasets/HeightVsAge.csv')
ind = dataset.iloc[:,0].values #independent variable
ind = ind.reshape(len(ind),1)  #reshaping the independent variables to match the dimensions for fitting
dep = dataset.iloc[:, 1].values #dependent variable

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
ind_train, ind_test, dep_train, dep_test = train_test_split(ind, dep, test_size = 0.2, random_state = 0)

# Training the Random Forest Regression model on the training set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)# n_estimators is the number of trees we want in the forest
regressor.fit(ind_train, dep_train)


# Visualising the Random Forest Regression results for training set

X_grid = np.arange(min(ind), max(ind), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(ind_train, dep_train, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Training Set Results')
plt.xlabel('Age')
plt.ylabel('Height')
plt.show()


# Visualising the Random Forest Regression results for test set

X_grid = np.arange(min(ind), max(ind), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(ind_test, dep_test, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Test Set Results')
plt.xlabel('Age')
plt.ylabel('Height')
plt.show()