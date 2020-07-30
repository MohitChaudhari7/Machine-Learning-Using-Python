# Polynomial Regression code for github

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

# Fitting Polynomial Regression to the dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)  # you can change the degree of polynomial here
ind_poly = poly_reg.fit_transform(ind_train)
poly_reg.fit(ind_poly, dep)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(ind_poly, dep_train)


# Visualising the Polynomial Regression Training set results
#Here we do not use the method that we used previously,because here we have a curve and is we use the previous methond,we will not get a smooth curve
X_grid = np.arange(min(ind), max(ind), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(ind_train, dep_train, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Training Set Results')
plt.xlabel('Age')
plt.ylabel('Height')
plt.show()

# Visualising the Polynomial Regression Test Set results
X_grid = np.arange(min(ind), max(ind), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(ind_test, dep_test, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Test Set Results')
plt.xlabel('Age')
plt.ylabel('Height')
plt.show()
