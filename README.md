# Machine-Learning-Using-Python
This repository contains codes for machine learning algorithms with examples which are presented visually using graphs 

# Table Of Contents 
  * [Libraries Used](#libraries-used)
  * [Usage](#usage)
  * [Regression](#regression)
    * [Simple Linear Regression](#simple-linear-regression)
    * [Multiple Linear Regression](#multiple-linear-regression)
    * [Polynomial Regression](#polynomial-regression)
    * [Support Vector Regression(SVR)](#support-vector-regression)
  * [License](#license)
    
# Libraries Used
1) matplotlib(version 3.1.3)
2) numpy(version 1.18.1)
3) pandas(version 1.0.1)
4) sklearn(version 0.22.1)

# Usage

Install the required libraries and then run the code using any IDE(I have used spyder).

**Note:** Don't forget to set the working directory to the folder that contains the python code as the datasets are imported using relative paths.

# Regression
In regression, we train different models to predict the dependent variable according to the independent variables.For achieving this goal, we divide our dataset into two sets, a training set and a test set.We will train the model using the training set and then see how good is the model on the basis of test set.

**Note:** in these examples,we will visualise the test set and training set results rather than judging them mathmatically.

## Simple Linear Regression

Code for simple linear regression is in the file "linear_regression.py".In simple linear regression we make a linear model.Therefore, the equation for the model is:
**y=b<sub>0</sub>+b<sub>1</sub>x**<br/>here x = independent variable,y = dependent variable,b<sub>0</sub>&b<sub>1</sub> are constants obtained by training the model.
<br/>Here we will be using the "Admission_Predict.csv" dataset.Which contains performance of students in various exams and the chances of the student getting selected in the college(0 being 0% chance and 1 being 100% chance).As this is a example of simple linear regression,we will take only one independent variable(GRE marks) and one dependent variable(chance of getting selected).
<br/>![linear_regression_image](Images/linear_regression_train.png) &nbsp; &nbsp; &nbsp;
![linear_regression_image](Images/linear_regression_test.png)
<a>

 In the above images, the blue line is the model that we trained and the red dots are the data points obtained from the dataset.
</a> 

## Multiple Linear Regression

Code for multiple linear regression is in the file "multiple_linear_regression.py".In multiple linear regression the equation for the model is:
**y=b<sub>0</sub>+b<sub>1</sub>x<sub>1</sub>+b<sub>2</sub>x<sub>2</sub>........+b<sub>n</sub>x<sub>n</sub>**<br/>here x<sub>i</sub> = independent variables,y = dependent variable,b<sub>i</sub> are constants obtained by training the model...where 'i' is a positive integer
<br/>Here we will be using the "Admission_Predict.csv" dataset.Which contains performance of students in various exams and the chances of the student getting selected in the college(0 being 0% chance and 1 being 100% chance).As this is a example of simple linear regression,we will take only two independent variables(GRE score & TOFEL score)and one dependent variable(chance of getting selected).
<br/>![multiple_linear_regression_image](Images/multiple_linear_regression_train.png) &nbsp; &nbsp; &nbsp;
![multiple_linear_regression_image](Images/multiple_linear_regression_test.png)
<a>
 
 In the above images, the black line is the model that we trained and the colourful dots are the data points obtained from the dataset.
 <br/>**Note:** The colour and transparency of the dots doesn't signify anything,it is just used to get a proper 3D view.We can also train the model with more than two variables,but we will not be able to visualise it on graphs.
</a> 

## Polynomial Regression

Code for polynomial regression is in the file "polynomianl_regression.py".In polynomial regression the equation for the model is:
**y=b<sub>0</sub>+b<sub>1</sub>x<sup>1</sup>+b<sub>2</sub>x<sup>2</sup>........+b<sub>n</sub>x<sup>n</sup>**<br/>here n = degree of the polynomial,x = independent variable,y = dependent variable,b<sub>i</sub> are constants obtained by training the model...where 'i' is a positive integer.
<br/>Here we will be using "HeightVsAge.csv" dataset which contains the average height at different ages. 
<br/>![polynomial_regression_image](Images/polynomial_regression_train.png) &nbsp; &nbsp; &nbsp;
![polynomial_regression_image](Images/polynomial_regression_test.png)
<a>

 In the above images, the blue curve is the model that we trained and the red dots are the data points obtained from the dataset.
<br/>**Note:** The model in above images is trained using a polynomial of degree 3,you can change the degree and find the optimal fit for a given data.
</a> 

## Support Vector Regression

Code for support vector regression(svr) is in the file "support_vector_regression.py". 
<br/>Here we will be using "HeightVsAge.csv" dataset which contains the average height at different ages.In support vector regression , we can use different kernels to train the model according to the distribution of the data points.In this case , we have used the 'Gaussian RBF Kernel'. For learning more about kernels and other terminologies click [<u>here</u>](/Explanations/svm_svr.md)

<br/>![SVR](Images/svr_train.png) &nbsp; &nbsp; &nbsp;
![SVR](Images/svr_test.png)
<a>

 In the above images, the blue curve is the model that we trained and the red dots are the data points obtained from the dataset.
 
 
# License
Licensed under the [MIT License](LICENSE).
