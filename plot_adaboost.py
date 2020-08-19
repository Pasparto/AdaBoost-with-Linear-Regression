from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pandas import read_csv

from sklearn.datasets import make_gaussian_quantiles
# from sklearn.model_selection import train_test_split

# This was to compare the result with sklearn AdaBoostClassifier calculation
# from sklearn.ensemble import AdaBoostClassifier

# For desision tree classifier
from sklearn.tree import DecisionTreeClassifier
# ￿For logistic regression classifier
from scipy.optimize import fmin_tnc

# from sklearn.linear_model import LogisticRegression


# We’re going to use the function below to visualize our data points,
# and optionally overlay the decision boundary of a fitted AdaBoost model.
def plot_adaboost(X: np.ndarray,
                  y: np.ndarray,
                  clf=None,
                  sample_weights: Optional[np.ndarray] = None,
                  stump_weight=None,
                  error=None,
                  annotate: bool = False,
                  ax: Optional[mpl.axes.Axes] = None) -> None:
    """ Plot ± samples in 2D, optionally with decision boundary if model is provided. """

    assert set(y) == {-1, 1}, 'Expecting response labels to be ±1'

    if not ax:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        fig.set_facecolor('white')

    pad = 1
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad

    if sample_weights is not None:
        sizes = np.array(sample_weights) * X.shape[0] * 100
    else:
        sizes = np.ones(shape=X.shape[0]) * 100

    # False positive get bigger
    X_pos = X[y == 1]
    sizes_pos = sizes[y == 1]
    ax.scatter(*X_pos.T, s=sizes_pos, marker='+', color='red')

    # False Negative get bigger
    X_neg = X[y == -1]
    sizes_neg = sizes[y == -1]
    ax.scatter(*X_neg.T, s=sizes_neg, marker='.', c='blue')

    if clf:
        plot_step = 0.01
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # If all predictions are positive class, adjust color map acordingly
        if list(np.unique(Z)) == [1]:
            fill_colors = ['r']
        else:
            fill_colors = ['b', 'r']

        ax.contourf(xx, yy, Z, colors=fill_colors, alpha=0.2)

    if annotate:
        for i, (x, y) in enumerate(X):
            offset = 0.05
            print("point {}: x={}\t y={}".format(i+1, x + offset, y - offset))
            ax.annotate(f'$x_{i + 1}$', (x + offset, y - offset))

    ax.set_xlim(x_min + 0.5, x_max - 0.5)
    ax.set_ylim(y_min + 0.5, y_max - 0.5)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.text(x_min, y_min, 'Alpha = {:.5f}\nEpsilon = {:.5f}'.format(stump_weight, error),
            style='italic',
            bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})


# We will generate a toy dataset.
# The key here is that we want to have two classes which are not linearly separable,
# since this is the ideal use-case for AdaBoost.
def make_toy_dataset(n: int = 100, random_seed: int = None) -> (np.ndarray, np.ndarray):
    """ Generate a toy dataset for evaluating AdaBoost classifiers

    Source: https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html

    """

    n_per_class = int(n / 2)

    if random_seed:
        np.random.seed(random_seed)

    X, y = make_gaussian_quantiles(n_samples=n, n_features=2, n_classes=2)

    return X, y * 2 - 1

# Benchmark
# Let’s establish a benchmark for what our model’s output should resemble
# by importing AdaBoostClassifier from scikit-learn and fitting it to our toy dataset.
# This code run without AdaBoost class
# X, y = make_toy_dataset(n=10, random_seed=10)
# bench = AdaBoostClassifier(n_estimators=10, algorithm='SAMME').fit(X, y)
# plot_adaboost(X, y, bench)
#
# train_err = (bench.predict(X) != y).mean()
# print(f'Train error: {train_err:.1%}')
# plt.show()

# ------------------------------------------------------------------------------------------------------------------------

class AdaBoost:
    """ AdaBoost ensemble classifier from scratch """

    def __init__(self):
        self.stumps = None
        self.stump_weights = None
        self.errors = None
        self.sample_weights = None

    def _check_X_y(self, X, y):
        """ Validate assumptions about format of input data"""
        assert set(y) == {-1, 1}, 'Expecting response variable to be formatted as ±1'
        return X, y

# ------------------------------------------------------------------------------------------------------------------------

# Logistic Regression Class definition
class LogisticRegressionGD(object):
    """Logistic Regression Classifier using gradient descent.
    Parameters:
    ------------
    eta : float
    Learning rate (between 0.0 and 1.0)
    n_iter : int
    Passes over the training dataset.
    random_state : int
    Random number generator seed for random weight
    initialization.

    Attributes:
    -----------
    w_ : 1d-array
    Weights after fitting.
    cost_ : list
    Sum-of-squares cost function value in each epoch.
    """

    def __init__(self, eta=0.45, n_iter=1000, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.5, size=1 + X.shape[1])

    def net_input(self, X):
        """Calculate net input:
        Calculating the model prediction by using X as the features and w_ as the equation parameters
        The return is a vector of results [ , X.shape[0]] - number of data set rows
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        """Compute logistic sigmoid activation
        Calculating the sigmoid value for each of our predictions
        """
        return 1. / (1. + np.exp(-z))

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
        # Equal to: return np.where(self.activation(self.net_input(X)) >= 0.5, 1, -1)

    def fit(self, X, y, sample_weights):
        """Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of
        samples and
        n_features is the number of features.
        y : array-like, shape = [n_samples] from {1,0}
        Target values.
        sample_weights :the weights for eacg data point - those weights come from AdaBoost

        Returns
        -------
        self : object
        """
        # Here I convert y values from -1 -> 0 and 1 -> 1
        y = (y+1) / 2

        self.cost_ = []

        for i in range(self.n_iter):
            # Here I make a prediction vector with my thetas
            net_input = self.net_input(X)

            # Here I convert my predictions to sigmoid values (values between 0.0 to 1.0)
            # than separate the points into 2 groups {1,0} by 0.5 delimiter
            output = self.activation(net_input)
            converted_output = np.where(output >= 0.5, 1, 0)

            # Here I calculate the differences between the real values and my predictions
            errors = (y - converted_output) * sample_weights

            # update my thetas according the error (gradient)
            self.w_[1:] += self.eta * X.T.dot(errors)
            # This is the biased unit (I do that instead of add 1 column to X)
            self.w_[0] += self.eta * errors.sum()

            # I compute the logistic weighted `cost` now
            # instead of the sum of squared errors cost
            cost = sample_weights * (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))) * sample_weights

            self.cost_.append(cost)
        return self


def fit(self, X: np.ndarray, y: np.ndarray, iters: int):
    """ Fit the model using training data """

    X, y = self._check_X_y(X, y)
    n = X.shape[0]

    # init numpy arrays
    self.sample_weights = np.zeros(shape=(iters, n))
    self.stumps = np.zeros(shape=iters, dtype=object)
    self.stump_weights = np.zeros(shape=iters)
    self.errors = np.zeros(shape=iters)

    # initialize weights uniformly
    self.sample_weights[0] = np.ones(shape=n) / n

    for t in range(iters):
        print("--------- Iteration number {} ------------".format(t))
        # fit  weak learner
        curr_sample_weights = self.sample_weights[t]
        print("Current weights: ", curr_sample_weights)
        # stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
        stump = LogisticRegressionGD()
        # stump = LogisticRegressionUsingGD()
        print("\t--------- Here is the fit function output of LogisticRegression ------------")
        # LR implemantaion of the book
        stump = stump.fit(X, y, sample_weights=curr_sample_weights)

        # Another implemantaion of LR
        # rgen = np.random.RandomState(1)
        # theta = rgen.normal(loc=0.0, scale=0.1, size=X.shape[1])
        # stump = stump.fit(X, y, theta, sample_weights=curr_sample_weights)

        # The original DeicsionTree
        # stump = stump.fit(X, y, sample_weight=curr_sample_weights)
        print("\t--------- Here is the end of fit function output ------------")

        # calculate error and stump weight from weak learner prediction
        stump_pred = stump.predict(X)
        print("My LR prediction is: ", stump_pred)
        print("The real values are: ", y)
        # print("curr_sample_weights[(stump_pred != y)] = ", curr_sample_weights[(stump_pred != y)])
        err = curr_sample_weights[(stump_pred != y)].sum()  # / n
        stump_weight = np.log((1 - err) / err) / 2

        # update sample weights
        new_sample_weights = curr_sample_weights * np.exp(-stump_weight * y * stump_pred)
        new_sample_weights /= new_sample_weights.sum()

        # If not final iteration, update sample weights for t+1
        if t + 1 < iters:
            self.sample_weights[t + 1] = new_sample_weights

        print("This is stump_weight: ", stump_weight)
        print("This is err: ", err)
        print("\n")

        # save results of iteration
        self.stumps[t] = stump
        self.stump_weights[t] = stump_weight
        self.errors[t] = err

        # if (curr_sample_weights == new_sample_weights).all():
        #     break

    return self


def predict(self, X):
    """ Make predictions using already fitted model """
    stump_preds = np.array([stump.predict(X) for stump in self.stumps])
    return np.sign(np.dot(self.stump_weights, stump_preds))


def truncate_adaboost(clf, t: int):
    """ Truncate a fitted AdaBoost up to (and including) a particular iteration """
    assert t > 0, 't must be a positive integer'
    from copy import deepcopy
    new_clf = deepcopy(clf)
    new_clf.stumps = clf.stumps[:t]
    new_clf.stump_weights = clf.stump_weights[:t]
    return new_clf


def plot_staged_adaboost(X, y, clf, iters=17):
    """ Plot weak learner and cumulaive strong learner at each iteration. """

    # larger grid
    fig, axes = plt.subplots(figsize=(8, iters * 3),
                             nrows=iters,
                             ncols=2,
                             sharex=True,
                             dpi=100)

    fig.set_facecolor('white')

    # _ = fig.suptitle('Decision boundaries by iteration')
    for i in range(iters):
        ax1, ax2 = axes[i]

        # Plot weak learner
        _ = ax1.set_title(f'Weak learner at t={i + 1}')
        plot_adaboost(X, y, clf.stumps[i], sample_weights=clf.sample_weights[i], stump_weight=clf.stump_weights[i],
                      error=clf.errors[i], annotate=False, ax=ax1)

        # Plot strong learner
        trunc_clf = truncate_adaboost(clf, t=i + 1)
        _ = ax2.set_title(f'Strong learner at t={i + 1}')
        plot_adaboost(X, y, trunc_clf, sample_weights=clf.sample_weights[i], stump_weight=clf.stump_weights[i],
                      error=clf.errors[i], annotate=False, ax=ax2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()


# assign our individually defined functions as methods of our classifier
AdaBoost.fit = fit
AdaBoost.predict = predict
X, y = make_toy_dataset(n=10, random_seed=10)
new_X = np.c_[X, np.ones((X.shape[0], 1))]
clf = AdaBoost().fit(X, y, iters=17)
plot_staged_adaboost(X, y, clf)

train_err = (clf.predict(X) != y).mean()
print(f'Train error: {train_err:.1%}')


# ----------------------------- This is the Logistic Regression testing part ----------------------

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
# def load_data(path, header):
#     marks_df = pd.read_csv(path, header=header)
#     return marks_df
#
#
# if __name__ == "__main__":
#     # load the data from the file
#     data = load_data("data/marks.txt", None)
#
#     # X = feature values, all the columns except the last column
#     X = data.iloc[:, :-1]
#
#     # y = target values, last column of the data frame
#     y = data.iloc[:, -1]
#
#     # filter out the applicants that got admitted
#     admitted = data.loc[y == 1]
#
#     # filter out the applicants that din't get admission
#     not_admitted = data.loc[y == 0]
#
#     # plots
#     model = LogisticRegressionUsingGD()
#     X = np.c_[np.ones((X.shape[0], 1)), X]
#     y = y[:, np.newaxis]
#     theta = np.zeros((X.shape[1], 1))
#     weights = np.zeros((1, X.shape[0]))
#     weights[0][32] = 0.5
#     weights[0][98] = 0.5
#     model.fit(X, y, theta)
#     parameters = model.w_
#
#     print(parameters)
#
#     x_values = [np.min(X[:, 1] - 5), np.max(X[:, 2] + 5)]
#     y_values = - (parameters[0] + np.dot(parameters[1], x_values)) / parameters[2]
#     plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
#     plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
#     plt.plot(x_values, y_values, label='Decision Boundary')
#     plt.xlabel('Marks in 1st Exam')
#     plt.ylabel('Marks in 2nd Exam')
#     plt.legend()
#     plt.show()

