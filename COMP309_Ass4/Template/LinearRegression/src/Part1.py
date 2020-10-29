
import pandas as pd
import numpy as np
import datetime
import random
# Visualisation
# import missingno as msno
import matplotlib.pyplot as plt

# Regressors
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neural_network import MLPRegressor

from utilities.losses import compute_loss
from utilities.optimizers import gradient_descent, pso, mini_batch_gradient_descent
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.preprocessing import  StandardScaler
from pandas import Series

# General settings
from utilities.visualization import visualize_train, visualize_test

seed = 309
# Freeze the random seed
random.seed(seed)
np.random.seed(seed)
train_test_split_test_size = 0.3

# Training settings
alpha = 0.1  # step size
max_iters = 50  # max iterations


def load_data():
    """
    Load Data from CSV
    :return: df    a panda data frame
    """
    df = pd.read_csv("/Users/petadouglas/Documents/Uni/COMP309/309A4/COMP309_Ass4/data/Part1regression/diamonds.csv")
    return df


def data_preprocess(data):
    """
    Data preprocess:
        1. Split the entire dataset into train and test
        2. Split outputs and inputs
        3. Standardize train and test
        4. Add intercept dummy for computation convenience
    :param data: the given dataset (format: panda DataFrame)
    :return: train_data       train data contains only inputs
             train_labels     train data contains only labels
             test_data        test data contains only inputs
             test_labels      test data contains only labels
             train_data_full       train data (full) contains both inputs and labels
             test_data_full       test data (full) contains both inputs and labels
    """

    # look for missing values.
    print(data.isnull().sum())
    # remove first column of instance numbers
    data = data.drop(data.columns[0], axis = 1)
    # remove where volume is 0
    data = data[(data[['x', 'y', 'z']] != 0).all(axis = 1)]
    # change categorical atts to numerical
    data = data.replace(['Fair','Good','Very Good','Premium','Ideal'],[1,2,3,4,5]);
    data = data.replace(['D','E','F','G','H','I','J'],[1,2,3,4,5,6,7]);
    data = data.replace(['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'],[1,2,3,4,5,6,7,8]);

    # Split the data into train and test
    train_data, test_data = train_test_split(data, test_size = train_test_split_test_size)

    # Pre-process data (both train and test)
    train_data_full = train_data.copy()
    train_data = train_data.drop(["price"], axis = 1)
    train_labels = train_data_full["price"]

    test_data_full = test_data.copy()
    test_data = test_data.drop(["price"], axis = 1)
    test_labels = test_data_full["price"]

    # Standardize the inputs
    train_mean = train_data.mean()
    train_std = train_data.std()
    train_data = (train_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std

    return train_data, train_labels, test_data, test_labels, train_data_full, test_data_full

def printResults(regressor, name, test_data, test_labels, train_data, train_labels):
    start_time = datetime.datetime.now()  # Track learning starting time

    baseline = regressor

    baseline.fit(train_data,train_labels)

    #Prediction
    y_prediction = baseline.predict(test_data)
    #print(y_prediction)

    end_time = datetime.datetime.now()  # Track learning ending time
    exection_time = (end_time - start_time).total_seconds()  # Track execution time

    # Step 4: Results presentation
    print("Regressor: " + name)
    print("Learn: execution time = {t:.3f} seconds".format(t = exection_time))
    print("R2: {:.2f}".format(baseline.score(test_data,test_labels)))  # R2 should be maximize
    mean_sq_err = mean_squared_error(test_labels, y_prediction)
    print("MSE: {:.2f}".format(mean_sq_err))
    print("RMSE: {:.2f}".format(np.sqrt(mean_sq_err)))
    print("MAE: {:.2f}".format(mean_absolute_error(test_labels, y_prediction)))
    print("---------------")

if __name__ == '__main__':
    # Settings
    metric_type = "MSE"  # MSE, RMSE, MAE, R2
    optimizer_type = "BGD"  # PSO, BGD

    # Step 1: Load Data
    data = load_data()

    # Step 2: Preprocess the data
    train_data, train_labels, test_data, test_labels, train_data_full, test_data_full = data_preprocess(data)

    printResults(LinearRegression(), "Linear Regression", test_data, test_labels, train_data, train_labels)
    printResults(KNeighborsRegressor(n_neighbors=7), "KNN", test_data, test_labels, train_data, train_labels)
    printResults(Ridge(), "Ridge", test_data, test_labels, train_data, train_labels)
    printResults(DecisionTreeRegressor(max_depth=11), "Decision Tree", test_data, test_labels, train_data, train_labels)
    printResults(RandomForestRegressor(max_depth=15, n_estimators=400), "Random Forest", test_data, test_labels, train_data, train_labels)
    printResults(GradientBoostingRegressor(max_depth=9, n_estimators=500, loss='lad'), "Gradient Boosting", test_data, test_labels, train_data, train_labels)
    printResults(SGDRegressor(), "SGD", test_data, test_labels, train_data, train_labels)
    printResults(SVR(kernel='linear', C=500), "SVR", test_data, test_labels, train_data, train_labels)
    printResults(LinearSVR(), "Linear SVR", test_data, test_labels, train_data, train_labels)
    printResults(MLPRegressor(learning_rate_init=0.15), "Multilayer Perceptron", test_data, test_labels, train_data, train_labels)
