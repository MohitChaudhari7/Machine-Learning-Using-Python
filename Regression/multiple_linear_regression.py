#Code for multiple linear regression github

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d

# Importing the dataset
dataset = pd.read_csv('../Datasets/Admission_Predict.csv')
ind = dataset.iloc[:,[1,2]].values  #independent variable
dep = dataset.iloc[:, -1].values     #dependent variable


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
ind_train, ind_test, dep_train, dep_test = train_test_split(ind, dep, test_size = 0.2, random_state = 0)


#Training the linear regression model with multiple independent variables
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(ind_train, dep_train)


#Visualising the training set results
ax = plt.axes(projection='3d')
zline = regressor.predict(np.vstack((np.linspace(min(ind[:,0]), 340, 3000),np.linspace(min(ind[:,1]), 120, 3000))).T)
xline = np.linspace(min(ind[:,0]), 340, 3000)
yline = np.linspace(min(ind[:,1]), 120, 3000)
ax.set_title('Training Set Result')
ax.set_xlabel('GRE Score')
ax.set_ylabel('TOFEL Score')
ax.set_zlabel('Chance of getting selected')
ax.plot3D(xline, yline, zline, 'black')
xpoints = ind_train[:,0]
ypoints = ind_train[:,1]
zpoints = regressor.predict(ind_train)
ax.scatter3D(xpoints, ypoints, zpoints, c=zpoints, cmap='ocean_r')

# Predicting the Test set results
dep_pred = regressor.predict(ind_test)

#Visualising the Test set results
bx = plt.axes(projection='3d')
zline = regressor.predict(np.vstack((np.linspace(min(ind[:,0]), 340, 3000),np.linspace(min(ind[:,1]), 120, 3000))).T)
xline = np.linspace(min(ind[:,0]), 340, 3000)
yline = np.linspace(min(ind[:,1]), 120, 3000)
bx.set_title('Test Set Results');
bx.set_xlabel('GRE Score')
bx.set_ylabel('TOFEL Score')
bx.set_zlabel('Chance of getting selected')
bx.plot3D(xline, yline, zline, 'black')
xpoints = ind_test[:,0]
ypoints = ind_test[:,1]
zpoints = regressor.predict(ind_test)
bx.scatter3D(xpoints, ypoints, zpoints, c=zpoints, cmap='ocean_r')