# Support Vector Regression (SVR) code for github

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../Datasets/HeightVsAge.csv')
ind = dataset.iloc[:,0].values #independent variable
ind = ind.reshape(len(ind),1)  #reshaping the independent variables,this is required for feature scaling
dep = dataset.iloc[:, 1].values #dependent variable
dep = dep.reshape(len(dep),1)  #reshaping the dependent variable,this is required for feature scaling

# Feature Scaling,this is required, because the previous classes did the feature scaling for us
#This one doesn't do it for us, so we do it using different library i.e standard scalar for sklearn
from sklearn.preprocessing import StandardScaler
sc_ind = StandardScaler()   #making objects of the standard scalar class
sc_dep = StandardScaler()
ind = sc_ind.fit_transform(ind) # this is where the feature scaling is done
dep = sc_dep.fit_transform(dep)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
ind_train, ind_test, dep_train, dep_test = train_test_split(ind, dep, test_size = 0.2, random_state = 0)

# Training the SVR model on the training data
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') #here we are using rbf kernel which is non linear
regressor.fit(ind_train, dep_train)


# Visualising the SVR results for training set
# For visualing the data,before plotting , we take the inverse transform of the independent and dependent variables
#By this we can interpert the data more efficiently as we have the previous scaled versions of the variables
X_grid = np.arange(min(sc_ind.inverse_transform(ind)), max(sc_ind.inverse_transform(ind)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_ind.inverse_transform(ind_train), sc_dep.inverse_transform(dep_train), color = 'red')
plt.plot(X_grid, sc_dep.inverse_transform(regressor.predict(sc_ind.transform(X_grid))), color = 'blue')
plt.title('Training set results')
plt.xlabel('Age')
plt.ylabel('Height')
plt.show()


# Visualising the SVR results for test set
X_grid = np.arange(min(sc_ind.inverse_transform(ind)), max(sc_ind.inverse_transform(ind)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_ind.inverse_transform(ind_test), sc_dep.inverse_transform(dep_test), color = 'red')
plt.plot(X_grid, sc_dep.inverse_transform(regressor.predict(sc_ind.transform(X_grid))), color = 'blue')
plt.title('Test set results')
plt.xlabel('Age')
plt.ylabel('Height')
plt.show()