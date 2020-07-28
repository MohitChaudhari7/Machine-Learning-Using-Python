# Machine-Learning-Using-Python
This repository contains codes for machine learning algorithms with examples which are presented visually using graphs 

## Libraries Used
1) matplotlib
2) numpy
3) pandas
4) sklearn

## Usage

Install the required libraries and then run the code using any IDE(I have used spyder).

**Note:** Don't forget to set the working directory to the folder that contains the python code as the datasets are imported using relative paths.

## Regression(work in progress)
In regression, we train different models to predict the dependent variable according to the independent variables.For achieving this goal, we divide our dataset into two sets, a training set and a test set.We will train the model using the training set and then see how good is the model on the basis of test set.

**Note:** in these examples,we will visualise the test set and training set results rather than judging them mathmatically.

### Simple Linear Regression

Code for simple linear regression is in the file "linear_regression.py".In simple linear regression we make a linear model.Therefore, the equation for the model is:
**y=b<sub>0</sub>+b<sub>1</sub>x**<br/>here x = independent variable,y = dependent variables,b<sub>0</sub>&b<sub>1</sub> are constants obtained by training the model.
</a> 
<br/>![linear_regression_image](Images/linear_regression_train.png) &nbsp; &nbsp; &nbsp;
![linear_regression_image](Images/linear_regression_test.png)
<a>

 In the above images, the blue line is the model that we trained and the red dots are the data points obtained from the dataset.
</a> 
