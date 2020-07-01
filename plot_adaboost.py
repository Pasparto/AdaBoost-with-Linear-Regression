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
                  stump_weight = None,
                  error = None,
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


class AdaBoost:
    """ AdaBoost enemble classifier from scratch """
    def __init__(self):
        self.stumps = None
        self.stump_weights = None
        self.errors = None
        self.sample_weights = None

    def _check_X_y(self, X, y):
        """ Validate assumptions about format of input data"""
        assert set(y) == {-1, 1}, 'Expecting response variable to be formatted as ±1'
        return X, y


# Logistic Regression Class definition
class LogisticRegressionUsingGD:
    @staticmethod
    def sigmoid(x):
        # Activation function used to map any real value between 0 and 1
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def net_input(theta, x):
        # Computes the weighted sum of inputs Similar to Linear Regression

        return np.dot(x, theta)

    def probability(self, theta, x):
        # Calculates the probability that an instance belongs to a particular class

        # print("Hello from probability this is before the sigma: {}".format(self.sigmoid(self.net_input(theta, x))))
        # print("This is x: {}\n\n".format(x))
        # print("This is theta: {}\n\n".format(theta))
        # print("Hello from probability this is net_input(dot between theta and x): {}\n\n".format(self.net_input(theta, x)))
        return self.sigmoid(self.net_input(theta, x))

    def cost_function(self, theta, x, y):
        # Computes the cost function for all the training samples
        # print("Hello from cost_function, this is sample_wieghts: ",sample_weights)
        # m = x.shape[0]

        weighted_cost = (y * np.log(self.probability(theta, x)) + (1 - y) * np.log(1 - self.probability(theta, x)))
        weights = np.zeros((1, weighted_cost.shape[0]))
        weights[0][16] = 1
        weights[0][17] = 1
        total_cost = -1 * np.sum(weighted_cost * weights)
        # print("Hello from cost_function, this is (1 - y) * np.log(1 - self.probability(theta, x)): {}".format(
        #     (1 - y) * np.log(1 - self.probability(theta, x))))
        # print("Im from cost_function and this is the vector before the sum: {}".format(weighted_cost))
        # print("Hello from cost_function, this is y * np.log(self.probability(theta, x)): {}".format(y * np.log(self.probability(theta, x))))
        # print("Im from cost_function and this is y: {}".format(y))
        # print("Im from cost_function and this is x: {}".format(x))
        # print("Im from cost_function and this is theta: {}".format(theta))
        # print("Im from cost_function and this is the sum: {}".format(np.sum(y * np.log(self.probability(theta, x)) + (1 - y) * np.log(1 - self.probability(theta, x)))))
        # print("Im from cost_function and this is the vector before the sum: {}".format(y * np.log(self.probability(theta, x)) + (1 - y) * np.log(1 - self.probability(theta, x))))
        # print("Im from cost_function and this is weighted_cost: {}".format(weighted_cost))
        # print("Im from cost_function and this is total cost: {}".format(total_cost))
        return total_cost

    def gradient(self, theta, x, y):
        # Computes the gradient of the cost function at the point theta
        # print("Hello from gradient function the this is m:\n{}".format(m))
        # print("Hello from gradient this is net input output: {}".format(self.net_input(theta, x)))
        # print("Hello from gradient this is sigmoid output: {}".format(self.sigmoid(self.net_input(theta, x))))
        # print("Hello from gradient this is dot output: {}".format(np.dot(x.T, self.sigmoid(self.net_input(theta, x)) - y)))
        # print("This is sigmoid_output: ", sigmoid_output)
        # print("This is weighted_sigmoid: ", weighted_sigmoid)
        return np.dot(x.T, self.sigmoid(self.net_input(theta, x)) - y)

    def fit(self, x, y, theta):
        """trains the model from the training data
        Uses the fmin_tnc function that is used to find the minimum for any function
        It takes arguments as
            1) func : function to minimize
            2) x0 : initial values for the parameters
            3) fprime: gradient for the function defined by 'func'
            4) args: arguments passed to the function
        Parameters
        ----------
        x: array-like, shape = [n_samples, n_features]
            Training samples
        y: array-like, shape = [n_samples, n_target_values]
            Target classes
        theta: initial weights
        sample_weights: the weights of dataset points
        Returns
        -------
        self: An instance of self
        """
        # print("Hello from Logistic regression fit, this is y: {}".format(y))
        # print("Hello from Logistic regression fit, this is type(y): {}".format(type(y)))
        # print("Hello from Logistic regression fit, this is y.flatten: {}".format(y.flatten()))
        # opt_weights = fmin_tnc(func=self.cost_function, x0=theta, fprime=self.gradient,
        #                        args=(x, y.flatten(), sample_weights))
        # self.w_ = opt_weights[0]
        # return self
        weights = np.zeros((1, x.shape[0]))
        weights[0][16] = 1
        weights[0][17] = 1

        lr = 0.01
        for i in range(1000000):
            z = np.dot(x, theta)
            h = self.sigmoid(z)
            gradient = np.dot(x.T * weights, (h - y)) / y.size
            theta -= lr * gradient
        theta = theta.T.reshape((3,))
        self.w_ = theta
        return self

    def predict(self, x, probab_threshold=0.5):
        """ Predicts the class labels
        Parameters and returns ndarray from {1,-1} values
        ----------
        x: array-like, shape = [n_samples, n_features]
            Test samples
        Returns
        probab_threshold: number from 0 to 1 - for classification
        -------
        predicted class labels
        """
        theta = self.w_[:, np.newaxis]
        logistic_prob = self.probability(theta, x)
        logistic_regression_pred = (logistic_prob >= probab_threshold).astype(int)
        logistic_regression_pred = (logistic_regression_pred * 2) - 1
        logistic_regression_pred = logistic_regression_pred.flatten()
        return logistic_regression_pred
        # print("Hello from predict logistic_prob = ",logistic_prob)
        # print(type(logistic_regression_pred))
        # print(logistic_regression_pred)
        # return self.probability(theta, x)

    def accuracy(self, x, actual_classes, probab_threshold=0.5):
        """Computes the accuracy of the classifier
        Parameters
        ----------
        x: array-like, shape = [n_samples, n_features]
            Training samples
        actual_classes : class labels from the training data set
        probab_threshold: threshold/cutoff to categorize the samples into different classes
        Returns
        -------
        accuracy: accuracy of the model
        """
        predicted_classes = (self.predict(x) >= probab_threshold).astype(int)
        predicted_classes = predicted_classes.flatten()
        accuracy = np.mean(predicted_classes == actual_classes)
        return accuracy * 100


def fit(self, X: np.ndarray, y: np.ndarray, iters: int):
    """ Fit the model using training data """

    X, y = self._check_X_y(X, y)
    n = X.shape[0]
    theta = np.zeros((X.shape[1], 1))

    # init numpy arrays
    self.sample_weights = np.zeros(shape=(iters, n))
    self.stumps = np.zeros(shape=iters, dtype=object)
    self.stump_weights = np.zeros(shape=iters)
    self.errors = np.zeros(shape=iters)

    # initialize weights uniformly
    self.sample_weights[0] = np.ones(shape=n) / n

    for t in range(iters):
        # fit  weak learner
        curr_sample_weights = self.sample_weights[t]
        # stump = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)
        # stump = stump.fit(X, y, sample_weight=curr_sample_weights)
        stump = LogisticRegressionUsingGD()
        y = (y + 1) / 2
        stump = stump.fit(X, y, theta=theta, sample_weights=curr_sample_weights)

        # calculate error and stump weight from weak learner prediction
        stump_pred = stump.predict(X)
        # print("Hello from AdaBoost fit function, logistic regression pred is: ", stump_pred)
        y = (y * 2) - 1
        err = curr_sample_weights[(stump_pred != y)].sum()  # / n
        stump_weight = np.log((1 - err) / err) / 2

        # update sample weights
        new_sample_weights = curr_sample_weights * np.exp(-stump_weight * y * stump_pred)
        new_sample_weights /= new_sample_weights.sum()

        # If not final iteration, update sample weights for t+1
        if t + 1 < iters:
            self.sample_weights[t + 1] = new_sample_weights

        # save results of iteration
        self.stumps[t] = stump
        self.stump_weights[t] = stump_weight
        self.errors[t] = err

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


def plot_staged_adaboost(X, y, clf, iters=10):
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
        plot_adaboost(X, y, clf.stumps[i], sample_weights=clf.sample_weights[i],stump_weight=clf.stump_weights[i], error=clf.errors[i], annotate=False, ax=ax1)

        # Plot strong learner
        trunc_clf = truncate_adaboost(clf, t=i + 1)
        _ = ax2.set_title(f'Strong learner at t={i + 1}')
        plot_adaboost(X, y, trunc_clf, sample_weights=clf.sample_weights[i],stump_weight=clf.stump_weights[i], error=clf.errors[i], annotate=False, ax=ax2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()


# assign our individually defined functions as methods of our classifier
# AdaBoost.fit = fit
# AdaBoost.predict = predict
# X, y = make_toy_dataset(n=10, random_seed=10)
# # X = np.c_[np.ones((X.shape[0], 1)), X]
# clf = AdaBoost().fit(X, y, iters=10)
# plot_staged_adaboost(X, y, clf)
# Those were the last commands


# print(X)
# print(X[:,[0]]) # This is x
# print(X[:,[1]]) # This is y

# sample_weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ,0.1, 0.1]
# X, y = make_toy_dataset(n=10, random_seed=10)
# X = np.c_[np.ones((X.shape[0], 1)), X]
# y = (y+1) / 2
# theta = np.zeros((X.shape[1], 1))
# model = LogisticRegressionUsingGD()
# model.fit(X, y, theta)
# # pred = LR.predict(X)
# accuracy = model.accuracy(X, y.flatten())
# parameters = model.w_
# print("\nThe is from Main:")
# print("The accuracy of the model is {}".format(accuracy))
# print("The model parameters using Gradient descent")
# print("The parameters are (model.w_): ",parameters)
# print("\n\nThe is from Main:")
# print(X)
# print(y)
# print("The prediction of logic regression is: {}".format(pred))
# print("The real data set is:{}".format(y))

def load_data(path, header):
    marks_df = read_csv(path, header=header)
    return marks_df


data = load_data("data/marks.txt", None)

# X = feature values, all the columns except the last column
X = data.iloc[:, :-1]

# y = target values, last column of the data frame
y = data.iloc[:, -1]

# filter out the applicants that got admitted
admitted = data.loc[y == 1]

# filter out the applicants that din't get admission
not_admitted = data.loc[y == 0]

# plots
plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10,
            label='Not Admitted')

# preparing the data for building the model
X = np.c_[np.ones((X.shape[0], 1)), X]
y = y[:, np.newaxis]
theta = np.zeros((X.shape[1], 1))

# Logistic Regression from scratch using Gradient Descent
model = LogisticRegressionUsingGD()
model.fit(X, y, theta)
accuracy = model.accuracy(X, y.flatten())
parameters = model.w_
print("The accuracy of the model is {}".format(accuracy))
print("The model parameters using Gradient descent")
print("\n")
print(parameters)


# --------------------------- FROM HERE IS PLOTTING AND BANCHMARKS ---------------------------


x_values = [np.min(X[:, 1] - 2), np.max(X[:, 2] + 2)]
y_values = - (parameters[0] + np.dot(parameters[1], x_values)) / parameters[2]

plt.plot(x_values, y_values, label='Decision Boundary')
plt.xlabel('Marks in 1st Exam')
plt.ylabel('Marks in 2nd Exam')
plt.legend()
plt.show()