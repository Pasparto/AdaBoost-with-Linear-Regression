# Repository Name
Adaboost with linear regression

## Description
###### AdaBoost
AdaBoost is a machine learning library written in Python. It is designed to help users build models that can accurately predict outcomes based on input data. It provides a variety of different algorithms and utilities that can be used to construct powerful models. AdaBoost also includes tools to help with data pre-processing and post-processing, such as feature selection and hyperparameter optimization. It is well-suited for both supervised and unsupervised learning tasks.

###### AdaBoost with linear regression
AdaBoost with linear regression is a type of machine learning algorithm that combines the linear regression model with the AdaBoost algorithm. The idea behind this approach is to use the linear regression model to make predictions, then use the AdaBoost algorithm to improve the model's accuracy. The AdaBoost algorithm works by training multiple weak learners, which are linear regressions that each focus on a different part of the data. By combining the predictions of these weak learners, the overall accuracy is improved.

###### Linear regression
Linear regression is a type of statistical model used to analyze the relationship between one or more independent variables (also known as predictors) and one or more dependent variables (also known as the response variable). It is used to predict the value of the response variable based on the values of the predictor variables. The linear regression model assumes that the relationship between the predictor and response variables is linear, meaning that the response can be predicted by a linear combination of the predictors.

## Table of Contents
- [Dependencies](#Dependencies)
- [Installation](#installation)
- [Tests](#tests)
- [Resources](#Resources)


## Dependencies
- Python 3.6+
- Numpy
- Scikit-learn
- Matplotlib
- Seaborn
- Pandas
- Jupyter Notebook

## Installation
To install the AdaBoost repository, you will need to have Python 3.6 or higher installed on your system. Then, you can clone the repository to your local machine using the following command:

```
git clone https://github.com/Pasparto/AdaBoost.git
```

Once the repository has been cloned, you can install the required dependencies using the following command:

```
pip install -r requirements.txt
```

Once all of the dependencies have been installed, you should be able to run the code in the repository.


## Tests
checkout test_model.py \
To test the AdaBoost repository, you can use unit tests to check that the code is running as expected. You can also use the test datasets included in the repository to test the models. Additionally, you can use cross-validation to evaluate the performance of the models.

## Resources
My first step was to understand and translate this paper to hebrew (Section 2.1 - 2.3) including all the mathematical proofs and theorems:
https://www.lri.fr/~kegl/mlcourse/book.pdf

Second, I reproduced AdaBoost algorithm including all the graphs + benchmarks, following this blog:
https://geoffruddock.com/adaboost-from-scratch-in-python/

My final step is to change that Adaboost implementation and replace the Decision tree learner (AKA weak learner) with Logistic Regression learner.


## Plot
![plot](https://github.com/Pasparto/AdaBoost-with-Linear-Regression/blob/master/data/plot-weak-and-strong-learner.jpg?raw=true)
