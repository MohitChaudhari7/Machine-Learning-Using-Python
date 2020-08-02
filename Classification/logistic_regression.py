# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../Datasets/advertising.csv')
ind = dataset.iloc[:, [0, 2]].values  #independent variables(daily time spent on the site and income)
dep = dataset.iloc[:, -1].values      #dependent variables

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
ind_train, ind_test, dep_train, dep_test = train_test_split(ind, dep, test_size = 0.2, random_state = 0)

# Feature Scaling ,we do not scale the dep variable as it gives only 1 or 0
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
ind_train = sc.fit_transform(ind_train) #we fit the data to training set and not the test set
ind_test = sc.transform(ind_test)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(ind_train, dep_train) #we train the classifier

dep_pred = classifier.predict(ind_test) #we predict the test set results

# read about plotting of contours here "https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/contour_image.html#sphx-glr-gallery-images-contours-and-fields-contour-image-py"
# Plotting the Training set results
from matplotlib.colors import ListedColormap
x, y = ind_train, dep_train
X, Y = np.meshgrid(np.arange(start = x[:, 0].min() - 0.5, stop = x[:, 0].max() + 0.5, step = 0.01),
                     np.arange(start = x[:, 1].min() - 0.5, stop = x[:, 1].max() + 0.5, step = 0.01))
plt.contourf(X, Y, classifier.predict(np.array([X.ravel(), Y.ravel()]).T).reshape(X.shape),
             alpha = 0.5, cmap = ListedColormap(('red', 'blue')))
plt.xlim(X.min(), X.max())
plt.ylim(Y.min(), Y.max())
un_y =np.unique(y)
for i, j in enumerate(un_y):
    plt.scatter(x[y == j, 0], x[y == j, 1],c = ListedColormap(('red', 'blue'))(i), label = j)
plt.title('Training Set Results')
plt.xlabel('Daily time spent on the site')
plt.ylabel('Income')
plt.legend()
plt.show()

# Plotting the Test set results
x, y = ind_test, dep_test
X, Y = np.meshgrid(np.arange(start = x[:, 0].min() - 0.5, stop = x[:, 0].max() + 0.5, step = 0.01),
                     np.arange(start = x[:, 1].min() - 0.5, stop = x[:, 1].max() + 0.5, step = 0.01))
plt.contourf(X, Y, classifier.predict(np.array([X.ravel(), Y.ravel()]).T).reshape(X.shape),
             alpha = 0.5, cmap = ListedColormap(('red', 'blue')))
plt.xlim(X.min(), X.max())
plt.ylim(Y.min(), Y.max())
un_y =np.unique(y)
for i, j in enumerate(un_y):
    plt.scatter(x[y == j, 0], x[y == j, 1],c = ListedColormap(('red', 'blue'))(i), label = j)
plt.title('Test Set results')
plt.xlabel('Daily time spent on the site')
plt.ylabel('Income')
plt.legend()
plt.show()

# Confusion Matrix(this matrix contains the amount of datapoints those are in correct region and those are in incorrect region)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(dep_test, dep_pred))
