
import pandas as pd
import numpy as np
import datetime
import random
# Visualisation
# import missingno as msno
import matplotlib.pyplot as plt

#Classification Algorithm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score

# Classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from utilities.losses import compute_loss
from utilities.optimizers import gradient_descent, pso, mini_batch_gradient_descent
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.preprocessing import  StandardScaler
from pandas import Series
from collections import Counter

# General settings
from utilities.visualization import visualize_train, visualize_test

seed = 309
# Freeze the random seed
random.seed(seed)
np.random.seed(seed)

# Training settings
alpha = 0.1  # step size
max_iters = 50  # max iterations


def load_data():
    """
    Load Data from CSV
    :return: df    a panda data frame
    """
    att_names = ['age','workclass','fnlgwt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss',
                'hours-per-week','native-country','class']
    train = pd.read_csv("/Users/petadouglas/Documents/Uni/COMP309/309A4/COMP309_Ass4/data/Part2classification/adult.data", names = att_names)
    test = pd.read_csv("/Users/petadouglas/Documents/Uni/COMP309/309A4/COMP309_Ass4/data/Part2classification/adult.test", names = att_names)

    #get rid of unnecessary first line
    test = test.iloc[1:]
    # print(train)
    # print(test)

    return train, test


def data_preprocess(train, test):
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
    # print(train_data.isnull().sum())
    # print(test_data.isnull().sum())
    test = test.replace(' ?',np.nan)
    test = test.fillna(method='ffill')
    train = train.replace(' ?',np.nan)
    train = train.fillna(method='ffill')
    # print(test_data.isin(["?"]).sum())
    # print(train_data.isin(["?"]).sum())

    #remove full stops for consistency
    test = test.replace(' <=50K.', ' <=50K')
    test = test.replace(' >50K.', ' >50K')
    print(test.isin([' <=50K.']).sum())
    print(test.isin([' >50K.']).sum())
    # print(test)

    le = LabelEncoder()
    train["workclass"] = le.fit_transform(train["workclass"])
    test["workclass"] = le.fit_transform(test["workclass"])
    train["education"] = le.fit_transform(train["education"])
    test["education"] = le.fit_transform(test["education"])
    train["marital-status"] = le.fit_transform(train["marital-status"])
    test["marital-status"] = le.fit_transform(test["marital-status"])
    train["occupation"] = le.fit_transform(train["occupation"])
    test["occupation"] = le.fit_transform(test["occupation"])
    train["relationship"] = le.fit_transform(train["relationship"])
    test["relationship"] = le.fit_transform(test["relationship"])
    train["race"] = le.fit_transform(train["race"])
    test["race"] = le.fit_transform(test["race"])
    train["sex"] = le.fit_transform(train["sex"])
    test["sex"] = le.fit_transform(test["sex"])
    train["native-country"] = le.fit_transform(train["native-country"])
    test["native-country"] = le.fit_transform(test["native-country"])
    train["class"] = le.fit_transform(train["class"])
    test["class"] = le.fit_transform(test["class"])
    # print(test)

    train = train.drop(['education'], axis = 1)
    test = test.drop(['education'], axis = 1)
    train = train.drop('fnlgwt', axis = 1)
    test = test.drop('fnlgwt', axis = 1)
    # Pre-process data (both train and test)
    train_data_full = train.copy()
    train_data = train.drop(["class"], axis = 1)
    train_labels = train_data_full["class"]

    test_data_full = test.copy()
    test_data = test.drop(["class"], axis = 1)
    test_labels = test_data_full["class"]

    # Standardize the inputs
    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    return train_data, test_data, train_labels, test_labels, train_data_full, test_data_full

def printResults(classifier, name, test_data, test_labels, train_data, train_labels):
    start_time = datetime.datetime.now()  # Track learning starting time

    baseline = classifier

    baseline.fit(train_data,train_labels)

    #Prediction
    y_prediction = baseline.predict(test_data)
    #print(y_prediction)

    end_time = datetime.datetime.now()  # Track learning ending time
    exection_time = (end_time - start_time).total_seconds()  # Track execution time

    accuracy = accuracy_score(test_labels, y_prediction)
    precision = precision_score(test_labels, y_prediction)
    recall = recall_score(test_labels, y_prediction)
    f1 = f1_score(test_labels, y_prediction)
    auc = roc_auc_score(test_labels, y_prediction)

    # Step 4: Results presentation
    print('Classifier: ' + name)
    print('Learn: execution time = {t:.3f} seconds'.format(t = exection_time))
    print('Accuracy : %0.2f ' % accuracy)
    print('Precision: %0.2f ' % precision)
    print('Recall: %0.2f ' % recall)
    print('F1: %0.2f ' % f1)
    print('AUC: %0.2f ' % auc)
    print('---------------')

if __name__ == '__main__':
    # Settings
    # metric_type = "MSE"  # MSE, RMSE, MAE, R2
    # optimizer_type = "BGD"  # PSO, BGD

    # Step 1: Load Data
    train, test = load_data()

    target = train.values[:,-1]
    counter = Counter(target)
    for k,v in counter.items():
	       per = v / len(target) * 100
	       print('Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))

    # Step 2: Preprocess the data
    train_data, test_data, train_labels, test_labels, train_data_full, test_data_full = data_preprocess(train, test)

    printResults(KNeighborsClassifier(n_neighbors=15, weights = 'distance'), "KNN", test_data, test_labels, train_data, train_labels)
    printResults(GaussianNB(), "Naive Bayes", test_data, test_labels, train_data, train_labels)
    printResults(SVC(kernel='poly'), "SVM", test_data, test_labels, train_data, train_labels)
    printResults(DecisionTreeClassifier(max_depth = 8, criterion = 'entropy'), "Decision Tree", test_data, test_labels, train_data, train_labels)
    printResults(RandomForestClassifier(max_depth=11), "Random Forest", test_data, test_labels, train_data, train_labels)
    printResults(AdaBoostClassifier(n_estimators = 400, learning_rate = 1.5), "Ada Boost", test_data, test_labels, train_data, train_labels)
    printResults(GradientBoostingClassifier(), "Gradient Boosting", test_data, test_labels, train_data, train_labels)
    printResults(LinearDiscriminantAnalysis(), "Linear Discriminant", test_data, test_labels, train_data, train_labels)
    printResults(MLPClassifier(hidden_layer_sizes=(5, 2)), "Multilayer Perceptron", test_data, test_labels, train_data, train_labels)
    printResults(LogisticRegression(), "Logistic Regression", test_data, test_labels, train_data, train_labels)
