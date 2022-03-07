from concurrent.futures.process import _chain_from_iterable_of_lists
#from signal import pause
#from statistics import correlation
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer # define custom scorers
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score, roc_curve

#import plotly.graph_objects as go


import warnings

def cor_selector(X, y,num_feats):
    cor_list = []
    pval_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        xdata = X[i]
        ydata = y
        xdatanonan = xdata[~np.isnan(ydata) & ~np.isnan(xdata) ]
        ydatanonan = ydata[~np.isnan(ydata) & ~np.isnan(xdata) ]
        rho, pval = stats.spearmanr(xdatanonan, ydatanonan)
        #cor = np.corrcoef(xdatanonan, ydatanonan)[0, 1]
        cor_list.append(rho)
        pval_list.append(pval)
    cor_list_np = np.array(cor_list)
    pval_list_np = np.array(pval_list)
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list_np))[-1:-(num_feats+1):-1]].columns.tolist()
    cor_values = cor_list_np[np.argsort(np.abs(cor_list_np))[-1:-(num_feats+1):-1]]
    pval_values = pval_list_np[np.argsort(np.abs(cor_list_np))[-1:-(num_feats+1):-1]]
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature, cor_values, pval_values

def cor_calculator(X, y):
    xdata = X
    ydata = y
    xdatanonan = xdata[~np.isnan(ydata) & ~np.isnan(xdata) ]
    ydatanonan = ydata[~np.isnan(ydata) & ~np.isnan(xdata) ]
    rho, pval = stats.spearmanr(xdatanonan, ydatanonan)
    return rho, pval

def score_cor_calculator(X, y):
    xdata = X
    ydata = y
    xdatanonan = xdata[~np.isnan(ydata) & ~np.isnan(xdata) ]
    ydatanonan = ydata[~np.isnan(ydata) & ~np.isnan(xdata) ]
    rho, _ = stats.spearmanr(xdatanonan, ydatanonan)
    return rho

def selectDesiredData(df, DATA_SELECTOR):
    if DATA_SELECTOR == 1: # Amnamesis
        df = pd.concat([df["S.NO "], df.filter(regex='Comorbidities'), df.filter(regex='Complaints'), df.filter(regex='AGE'), df.filter(regex='SEX')], axis=1)
    elif DATA_SELECTOR == 2: # Ultrasound
        df = pd.concat([df["S.NO "], df.filter(regex='US_')], axis = 1)
    elif DATA_SELECTOR == 3: # X-ray
        df = pd.concat([df["S.NO "], df.filter(regex='XRAY')], axis = 1)
    elif DATA_SELECTOR == 4: # CT
        df = pd.concat([df["S.NO "],df.filter(regex='LUNG_CT')], axis = 1)
    elif DATA_SELECTOR == 5: # Amnamesis + Ultrasound
        df = pd.concat([df["S.NO "], df.filter(regex='US_'), df.filter(regex='Comorbidities'), df.filter(regex='Complaints'), df.filter(regex='AGE'), df.filter(regex='SEX')], axis=1)
    elif DATA_SELECTOR == 6: # Amnamesis + X-ray
        df = pd.concat([df["S.NO "], df.filter(regex='XRAY'), df.filter(regex='Comorbidities'), df.filter(regex='Complaints'), df.filter(regex='AGE'), df.filter(regex='SEX')], axis=1)
    elif DATA_SELECTOR == 7: # Amanamesis + CT
        df = pd.concat([df["S.NO "], df.filter(regex='LUNG_CT'), df.filter(regex='Comorbidities'), df.filter(regex='Complaints'), df.filter(regex='AGE'), df.filter(regex='SEX')], axis=1)
    else:
        df = df
    return df

#LIST OF CLASSIFIERS WE WANT TO TEST

# Reduced subset for debugging

names = [   
    "Nearest Neighbors_01",
    "Nearest Neighbors_02",
    "Nearest Neighbors_03",
    "Nearest Neighbors_04",
    "Nearest Neighbors_05",
    ]

'''
names = [
    "Nearest Neighbors_01",
    "Nearest Neighbors_02",
    "Nearest Neighbors_03",
    "Nearest Neighbors_04",
    "Nearest Neighbors_05",
    "Nearest Neighbors_06",
    "Nearest Neighbors_07",
    "Nearest Neighbors_08",
    "SVM_01",
    "SVM_02",
    "SVM_03",
    "SVM_04",
    "SVM_05",    
    "SVM_06",
    "SVM_07",
    "SVM_08",
    "SVM_09",
    "SVM_10",
    "SVM_11",
    "SVM_12",
    "SVM_13",
    "SVM_14",
    "SVM_15",
    "SVM_16",
    "SVM_17",
    "SVM_18",
    "SVM_19",
    "SVM_20",
    "SVM_21",
    "SVM_22",
    "SVM_23",
    "SVM_24",
    "SVM_25",
    "SVM_26",
    "SVM_27",
    "SVM_28",
    "SVM_29",
    "SVM_30", 
    "Gaussian Process_01",
    "Gaussian Process_02",
    "Gaussian Process_03",
    "Gaussian Process_04",
    "Gaussian Process_05",
    "Decision Tree_01",
    "Decision Tree_02",
    "Decision Tree_03",
    "Decision Tree_04",
    "Decision Tree_05",
    "Decision Tree_06",
    "Decision Tree_07",
    "Decision Tree_08",
    "Random Forest_01",
    "Random_Forest_02",
    "Random_Forest_03",
    "Random_Forest_04",
    "Random_Forest_05",
    "Random_Forest_06",
    "Random_Forest_07",
    "Random_Forest_08",
    "Random Forest_09",
    "Random_Forest_10",    
    "Random Forest_11",
    "Random_Forest_12",
    "Random_Forest_13",
    "Random_Forest_14",
    "Random_Forest_15",
    "Random_Forest_16",
    "Random_Forest_17",
    "Random_Forest_18",
    "Random Forest_19",
    "Random_Forest_20",    
    "Neural Net_01",
    "Neural Net_02",
    "Neural Net_03",
    "Neural Net_04",
    "Neural Net_05",
    "Neural Net_06",
    "Neural Net_07",
    "Neural Net_08",
    "Neural Net_09",
    "Neural Net_10",
    "Neural Net_11",
    "Neural Net_12",
    "Neural Net_13",
    "Neural Net_14",
    "Neural Net_15",
    "Neural Net_16",
    "Neural Net_17",
    "Neural Net_18",
    "Neural Net_19",
    "Neural Net_20",
    "Neural Net_21",
    "Neural Net_22",
    "Neural Net_23",
    "Neural Net_24",   
    "AdaBoost_01",
    "AdaBoost_02",
    "AdaBoost_03",
    "AdaBoost_04",
    "AdaBoost_05",
    "AdaBoost_06",
    "AdaBoost_07",
    "AdaBoost_08",
    "AdaBoost_09",
    "AdaBoost_10",
    "AdaBoost_11",
    "AdaBoost_12",
    "AdaBoost_13",
    "AdaBoost_14",
    "AdaBoost_15",
    "AdaBoost_16",
    "AdaBoost_17",
    "AdaBoost_18",
    "Naive Bayes_01",
    "Naive Bayes_02",
    "Naive Bayes_03",
    "Naive Bayes_04",
    "QDA_01",
    "Logistic_01",
    "Logistic_02",
    "Logistic_03",
    "Logistic_04",    
    "Logistic_05",
    "Logistic_06",    
    "Logistic_07",
    "Logistic_08",
    "Logistic_09",
    "Ridge_01",
    "Ridge_02",
    "Ridge_03",
    "Ridge_04",
    "Ridge_05",
    "Ridge_06",    
    "Bagging_01",
    "Bagging_02",
    "Bagging_03",
    "Bagging_04",
    "Gradient_01",
    "Gradient_02",
    "Gradient_03",
    "Gradient_04",
    "Gradient_05",
    "Gradient_06",
    "Gradient_07",
    "Gradient_08",
    "Gradient_09",
    "Gradient_10",
    "Gradient_11",
    "Gradient_12",
    "Gradient_13",
    "Gradient_14",
    "Gradient_15",
    "Gradient_16",
    "Gradient_17",
    "Gradient_18",
    "Gradient_19",
    "Gradient_20",
    "Gradient_21",
    "Gradient_22",
    "Gradient_23",
    "Gradient_24",
    "Gradient_25",
    "Gradient_26",
    "Gradient_27",
    "Gradient_28",
    "Gradient_29",
    "Gradient_30",
    "Gradient_31",
    "Gradient_32",
    "Gradient_33",
    "Gradient_34",
    "Gradient_35",
    "Gradient_36",
    "Gradient_37",
    "Gradient_38",
    "Gradient_39",
    "Gradient_40",
    "Gradient_41",
    "Gradient_42",
    "Gradient_43",
    "Gradient_44",
    "Gradient_45",
    "Gradient_46",
    "Gradient_47",
    "Gradient_48",
    "Gradient_49",
    "Gradient_50", 
    "Gradient_51",
    "Gradient_52",
    "Gradient_53",
    "Gradient_54",     
]
'''
#mscore = make_scorer(score_cor_calculator, greater_is_better = True)

classifiers = [
    KNeighborsClassifier(3, weights = 'uniform'),
    KNeighborsClassifier(6, weights = 'uniform'),
    KNeighborsClassifier(12, weights = 'uniform'),
    KNeighborsClassifier(24, weights = 'uniform'),
    KNeighborsClassifier(3, weights = 'distance'),
    ]


'''
#https://medium.com/swlh/the-hyperparameter-cheat-sheet-770f1fed32ff
classifiers = [
    KNeighborsClassifier(3, weights = 'uniform'),
    KNeighborsClassifier(6, weights = 'uniform'),
    KNeighborsClassifier(12, weights = 'uniform'),
    KNeighborsClassifier(24, weights = 'uniform'),
    KNeighborsClassifier(3, weights = 'distance'),
    KNeighborsClassifier(6, weights = 'distance'),
    KNeighborsClassifier(12, weights = 'distance'),
    KNeighborsClassifier(24, weights = 'distance'),
    SVC(kernel="linear", C=0.1),
    SVC(kernel="linear", C=1),
    SVC(kernel="linear", C=10),
    SVC(kernel="linear", C=100),
    SVC(kernel="linear", C=1000),    
    SVC(kernel="poly", C=0.1),
    SVC(kernel="poly", C=1),
    SVC(kernel="poly", C=10),
    SVC(kernel="poly", C=100),
    SVC(kernel="poly", C=1000),
    SVC(kernel="sigmoid", C=0.1),
    SVC(kernel="sigmoid", C=1),
    SVC(kernel="sigmoid", C=10),
    SVC(kernel="sigmoid", C=100),
    SVC(kernel="sigmoid", C=1000),
    SVC(kernel="rbf", gamma=2, C=0.01), # they all use adam optimizer    
    SVC(kernel="rbf", gamma=2, C=0.1), # they all use adam optimizer    
    SVC(kernel="rbf", gamma=2, C=1), # they all use adam optimizer    
    SVC(kernel="rbf", gamma=2, C=10), # they all use adam optimizer    
    SVC(kernel="rbf", gamma=2, C=100), # they all use adam optimizer    
    SVC(kernel="rbf", gamma=2, C=1000), # they all use adam optimizer    
    SVC(kernel="rbf", gamma=.5, C=0.1), # they all use adam optimizer    
    SVC(kernel="rbf", gamma=.5, C=1), # they all use adam optimizer    
    SVC(kernel="rbf", gamma=.5, C=10), # they all use adam optimizer    
    SVC(kernel="rbf", gamma=.5, C=100), # they all use adam optimizer    
    SVC(kernel="rbf", gamma=.5, C=1000), # they all use adam optimizer    
    SVC(kernel="rbf", C=0.1), # they all use adam optimizer  
    SVC(kernel="rbf", C=1), # they all use adam optimizer    
    SVC(kernel="rbf", C=10), # they all use adam optimizer    
    SVC(kernel="rbf", C=100), # they all use adam optimizer    
    SVC(kernel="rbf", C=1000), # they all use adam optimizer    
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    GaussianProcessClassifier(1.0 * DotProduct()),
    GaussianProcessClassifier(1.0 * Matern()),
    GaussianProcessClassifier(1.0 * RationalQuadratic()),
    GaussianProcessClassifier(1.0 * WhiteKernel()),
    DecisionTreeClassifier(max_depth=5,criterion="gini"),
    DecisionTreeClassifier(max_depth=10,criterion="gini"),
    DecisionTreeClassifier(max_depth=20,criterion="gini"),
    DecisionTreeClassifier(max_depth=40,criterion="gini"),
    DecisionTreeClassifier(max_depth=5,criterion="entropy"),
    DecisionTreeClassifier(max_depth=10,criterion="entropy"),
    DecisionTreeClassifier(max_depth=20,criterion="entropy"),
    DecisionTreeClassifier(max_depth=40,criterion="entropy"),
    RandomForestClassifier(max_depth=None, n_estimators=10),
    RandomForestClassifier(max_depth=None, n_estimators=20),
    RandomForestClassifier(max_depth=None, n_estimators=40),
    RandomForestClassifier(max_depth=None, n_estimators=80),
    RandomForestClassifier(max_depth=5, n_estimators=10),
    RandomForestClassifier(max_depth=5, n_estimators=20),
    RandomForestClassifier(max_depth=5, n_estimators=40),
    RandomForestClassifier(max_depth=5, n_estimators=80),
    RandomForestClassifier(max_depth=10, n_estimators=10),
    RandomForestClassifier(max_depth=10, n_estimators=20),
    RandomForestClassifier(max_depth=10, n_estimators=40),
    RandomForestClassifier(max_depth=10, n_estimators=80),
    RandomForestClassifier(max_depth=20, n_estimators=10),
    RandomForestClassifier(max_depth=20, n_estimators=20),
    RandomForestClassifier(max_depth=20, n_estimators=40),
    RandomForestClassifier(max_depth=20, n_estimators=80),    
    RandomForestClassifier(max_depth=40, n_estimators=10),
    RandomForestClassifier(max_depth=40, n_estimators=20),
    RandomForestClassifier(max_depth=40, n_estimators=40),
    RandomForestClassifier(max_depth=40, n_estimators=80),  
    MLPClassifier(hidden_layer_sizes = (5,), alpha=0.001, max_iter=1000),
    MLPClassifier(hidden_layer_sizes = (5,), alpha=0.01, max_iter=1000),
    MLPClassifier(hidden_layer_sizes = (5,), alpha=0.1, max_iter=1000),
    MLPClassifier(hidden_layer_sizes = (5,), alpha=1, max_iter=1000),
    MLPClassifier(hidden_layer_sizes = (10,), alpha=0.001, max_iter=1000),
    MLPClassifier(hidden_layer_sizes = (10,), alpha=0.01, max_iter=1000),
    MLPClassifier(hidden_layer_sizes = (10,), alpha=0.1, max_iter=1000),
    MLPClassifier(hidden_layer_sizes = (10,), alpha=1, max_iter=1000),    
    MLPClassifier(hidden_layer_sizes = (20,), alpha=0.001, max_iter=1000),
    MLPClassifier(hidden_layer_sizes = (20,), alpha=0.01, max_iter=1000),
    MLPClassifier(hidden_layer_sizes = (20,), alpha=0.1, max_iter=1000),
    MLPClassifier(hidden_layer_sizes = (20,), alpha=1, max_iter=1000), # they all use adam optimizer
    MLPClassifier(hidden_layer_sizes = (5,2), alpha=0.001, max_iter=1000),
    MLPClassifier(hidden_layer_sizes = (5,2), alpha=0.01, max_iter=1000),
    MLPClassifier(hidden_layer_sizes = (5,2), alpha=0.1, max_iter=1000),
    MLPClassifier(hidden_layer_sizes = (5,2), alpha=1, max_iter=1000),
    MLPClassifier(hidden_layer_sizes = (10,5,2), alpha=0.001, max_iter=1000),
    MLPClassifier(hidden_layer_sizes = (10,5,2), alpha=0.01, max_iter=1000),
    MLPClassifier(hidden_layer_sizes = (10,5,2), alpha=0.1, max_iter=1000),
    MLPClassifier(hidden_layer_sizes = (10,5,2), alpha=1, max_iter=1000),    
    MLPClassifier(hidden_layer_sizes = (20,5,2), alpha=0.001, max_iter=1000),
    MLPClassifier(hidden_layer_sizes = (20,10,5,2), alpha=0.01, max_iter=1000),
    MLPClassifier(hidden_layer_sizes = (20,10,5,2), alpha=0.1, max_iter=1000),
    MLPClassifier(hidden_layer_sizes = (20,10,5,2), alpha=1, max_iter=1000),    
    AdaBoostClassifier(learning_rate = 0.01, n_estimators = 20),
    AdaBoostClassifier(learning_rate = 0.02, n_estimators = 20),
    AdaBoostClassifier(learning_rate = 0.05, n_estimators = 20),
    AdaBoostClassifier(learning_rate = 0.1, n_estimators = 20),
    AdaBoostClassifier(learning_rate = 0.2, n_estimators = 20),
    AdaBoostClassifier(learning_rate = 0.5, n_estimators = 20),
    AdaBoostClassifier(learning_rate = 0.01, n_estimators = 50),
    AdaBoostClassifier(learning_rate = 0.02, n_estimators = 50),
    AdaBoostClassifier(learning_rate = 0.05, n_estimators = 50),
    AdaBoostClassifier(learning_rate = 0.1, n_estimators = 50),
    AdaBoostClassifier(learning_rate = 0.2, n_estimators = 50),
    AdaBoostClassifier(learning_rate = 0.5, n_estimators = 50),    
    AdaBoostClassifier(learning_rate = 0.01, n_estimators = 100),
    AdaBoostClassifier(learning_rate = 0.02, n_estimators = 100),
    AdaBoostClassifier(learning_rate = 0.05, n_estimators = 100),
    AdaBoostClassifier(learning_rate = 0.1, n_estimators = 100),
    AdaBoostClassifier(learning_rate = 0.2, n_estimators = 100),
    AdaBoostClassifier(learning_rate = 0.5, n_estimators = 100),
    GaussianNB(),
    MultinomialNB(),
    ComplementNB(),
    BernoulliNB(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(solver = 'newton-cg', C = 0.1),
    LogisticRegression(solver = 'newton-cg', C = 1),
    LogisticRegression(solver = 'newton-cg', C = 10),
    LogisticRegression(solver = 'lbfgs', C = 0.1),
    LogisticRegression(solver = 'lbfgs', C = 1),
    LogisticRegression(solver = 'lbfgs', C = 10),
    LogisticRegression(solver = 'liblinear', C = 0.1),
    LogisticRegression(solver = 'liblinear', C = 1),
    LogisticRegression(solver = 'liblinear', C = 10),
    RidgeClassifier(alpha = 0.1),
    RidgeClassifier(alpha = 1),
    RidgeClassifier(alpha = 10),
    RidgeClassifier(alpha = 20),
    RidgeClassifier(alpha = 50),
    RidgeClassifier(alpha = 100),
    BaggingClassifier(n_estimators = 10),
    BaggingClassifier(n_estimators = 20),
    BaggingClassifier(n_estimators = 40),
    BaggingClassifier(n_estimators = 80),
    GradientBoostingClassifier(n_estimators = 10, learning_rate = 0.01,  max_depth = 5, subsample=1),
    GradientBoostingClassifier(n_estimators = 10, learning_rate = 0.1,  max_depth = 5, subsample=1),
    GradientBoostingClassifier(n_estimators = 10, learning_rate = 1,  max_depth = 5, subsample=1),
    GradientBoostingClassifier(n_estimators = 30, learning_rate = 0.01,  max_depth = 5, subsample=1),
    GradientBoostingClassifier(n_estimators = 30, learning_rate = 0.1,  max_depth = 5, subsample=1),
    GradientBoostingClassifier(n_estimators = 30, learning_rate = 1,  max_depth = 5, subsample=1),
    GradientBoostingClassifier(n_estimators = 80, learning_rate = 0.01,  max_depth = 5, subsample=1),
    GradientBoostingClassifier(n_estimators = 80, learning_rate = 0.1,  max_depth = 5, subsample=1),
    GradientBoostingClassifier(n_estimators = 80, learning_rate = 1,  max_depth = 5, subsample=1),
    GradientBoostingClassifier(n_estimators = 10, learning_rate = 0.01,  max_depth = 15, subsample=1),
    GradientBoostingClassifier(n_estimators = 10, learning_rate = 0.1,  max_depth = 15, subsample=1),
    GradientBoostingClassifier(n_estimators = 10, learning_rate = 1,  max_depth = 15, subsample=1),
    GradientBoostingClassifier(n_estimators = 30, learning_rate = 0.01,  max_depth = 15, subsample=1),
    GradientBoostingClassifier(n_estimators = 30, learning_rate = 0.1,  max_depth = 15, subsample=1),
    GradientBoostingClassifier(n_estimators = 30, learning_rate = 1,  max_depth = 15, subsample=1),
    GradientBoostingClassifier(n_estimators = 80, learning_rate = 0.01,  max_depth = 15, subsample=1),
    GradientBoostingClassifier(n_estimators = 80, learning_rate = 0.1,  max_depth = 15, subsample=1),
    GradientBoostingClassifier(n_estimators = 80, learning_rate = 1,  max_depth = 15, subsample=1),
    GradientBoostingClassifier(n_estimators = 10, learning_rate = 0.01,  max_depth = 30, subsample=1),
    GradientBoostingClassifier(n_estimators = 10, learning_rate = 0.1,  max_depth = 30, subsample=1),
    GradientBoostingClassifier(n_estimators = 10, learning_rate = 1,  max_depth = 30, subsample=1),
    GradientBoostingClassifier(n_estimators = 30, learning_rate = 0.01,  max_depth = 30, subsample=1),
    GradientBoostingClassifier(n_estimators = 30, learning_rate = 0.1,  max_depth = 30, subsample=1),
    GradientBoostingClassifier(n_estimators = 30, learning_rate = 1,  max_depth = 30, subsample=1),
    GradientBoostingClassifier(n_estimators = 80, learning_rate = 0.01,  max_depth = 30, subsample=1),
    GradientBoostingClassifier(n_estimators = 80, learning_rate = 0.1,  max_depth = 30, subsample=1),
    GradientBoostingClassifier(n_estimators = 80, learning_rate = 1,  max_depth = 30, subsample=1),    
    GradientBoostingClassifier(n_estimators = 10, learning_rate = 0.01,  max_depth = 5, subsample=0.5),
    GradientBoostingClassifier(n_estimators = 10, learning_rate = 0.1,  max_depth = 5, subsample=0.5),
    GradientBoostingClassifier(n_estimators = 10, learning_rate = 1,  max_depth = 5, subsample=0.5),
    GradientBoostingClassifier(n_estimators = 30, learning_rate = 0.01,  max_depth = 5, subsample=0.5),
    GradientBoostingClassifier(n_estimators = 30, learning_rate = 0.1,  max_depth = 5, subsample=0.5),
    GradientBoostingClassifier(n_estimators = 30, learning_rate = 1,  max_depth = 5, subsample=0.5),
    GradientBoostingClassifier(n_estimators = 80, learning_rate = 0.01,  max_depth = 5, subsample=0.5),
    GradientBoostingClassifier(n_estimators = 80, learning_rate = 0.1,  max_depth = 5, subsample=0.5),
    GradientBoostingClassifier(n_estimators = 80, learning_rate = 1,  max_depth = 5, subsample=0.5),
    GradientBoostingClassifier(n_estimators = 10, learning_rate = 0.01,  max_depth = 15, subsample=0.5),
    GradientBoostingClassifier(n_estimators = 10, learning_rate = 0.1,  max_depth = 15, subsample=0.5),
    GradientBoostingClassifier(n_estimators = 10, learning_rate = 1,  max_depth = 15, subsample=0.5),
    GradientBoostingClassifier(n_estimators = 30, learning_rate = 0.01,  max_depth = 15, subsample=0.5),
    GradientBoostingClassifier(n_estimators = 30, learning_rate = 0.1,  max_depth = 15, subsample=0.5),
    GradientBoostingClassifier(n_estimators = 30, learning_rate = 1,  max_depth = 15, subsample=0.5),
    GradientBoostingClassifier(n_estimators = 80, learning_rate = 0.01,  max_depth = 15, subsample=0.5),
    GradientBoostingClassifier(n_estimators = 80, learning_rate = 0.1,  max_depth = 15, subsample=0.5),
    GradientBoostingClassifier(n_estimators = 80, learning_rate = 1,  max_depth = 15, subsample=0.5),
    GradientBoostingClassifier(n_estimators = 10, learning_rate = 0.01,  max_depth = 30, subsample=0.5),
    GradientBoostingClassifier(n_estimators = 10, learning_rate = 0.1,  max_depth = 30, subsample=0.5),
    GradientBoostingClassifier(n_estimators = 10, learning_rate = 1,  max_depth = 30, subsample=0.5),
    GradientBoostingClassifier(n_estimators = 30, learning_rate = 0.01,  max_depth = 30, subsample=0.5),
    GradientBoostingClassifier(n_estimators = 30, learning_rate = 0.1,  max_depth = 30, subsample=0.5),
    GradientBoostingClassifier(n_estimators = 30, learning_rate = 1,  max_depth = 30, subsample=0.5),
    GradientBoostingClassifier(n_estimators = 80, learning_rate = 0.01,  max_depth = 30, subsample=0.5),
    GradientBoostingClassifier(n_estimators = 80, learning_rate = 0.1,  max_depth = 30, subsample=0.5),
    GradientBoostingClassifier(n_estimators = 80, learning_rate = 1,  max_depth = 30, subsample=0.5), 
]
'''

outcomes = ["OUTCOME", "Deceased", "RT_PCR_POSITIVE_NEGATIVE"]
random_seeds = np.arange(0,5)


### Getting main code started
warnings.filterwarnings("ignore")


WHERE_IS_MY_DATA = "C:/Users/inapi/source/reposCode/datasetHigadoParamsV2.csv" #"C:/Users/inapi/source/reposCode/datasetHigadoParamsV2.csv" #"C:/Users/inapi/Downloads/LUS_data_sanse_ScikitLearn.csv"
df = pd.read_csv(WHERE_IS_MY_DATA)
num_features = 30
# Find most predictive attributes
df1 = df.copy()
df1= df1.drop(columns=["Complaints_None", "Complaints_Vomiting", "Deceased", "OUTCOME", "RT_PCR_LATER", "RT_PCR_POSITIVE_NEGATIVE",
"LEFT_LUNG_US_01_NORMAL","RIGHT_LUNG_US_01_NORMAL", "LEFT_LUNG_US_02_BLINE", "RIGHT_LUNG_US_02_BLINE",
"RIGHT_LUNG_US_03_EFUSSION","LEFT_LUNG_US_03_EFUSSION","LEFT_LUNG_CT_01_NORMAL", "RIGHT_LUNG_CT_01_NORMAL",
"RIGHT_LUNG_CT_05_PLEURALEFFUSION","LEFT_LUNG_CT_02_GGO", "LEFT_LUNG_CT_07_ATELACTASIS",
"RIGHT_LUNG_CT_07_ATELACTASIS", "RT_PCR_BEFORE", "LEFT_LUNG_CT_08_CALCIFIED_OLDHEALED", "LEFT_LUNG_CT_04_GGOplusCONSOLIDATION",
"RIGHT_LUNG_CT_04_GGOplusCONSOLIDATION"])
cor_support1, cor_feature1, cor_values1, p_values1 = cor_selector(df1, df["OUTCOME"], 17)
cor_support2, cor_feature2, cor_values2, p_values2 = cor_selector(df1, df["Deceased"],9)
cor_support3, cor_feature3, cor_values3, p_values3 = cor_selector(df1, df["RT_PCR_POSITIVE_NEGATIVE"],5)

# Prepare data frame for ML models 
df1 = df.copy()
df1 = df1.drop(columns=["Deceased", "OUTCOME", "RT_PCR_LATER", "RT_PCR_POSITIVE_NEGATIVE", "RT_PCR_BEFORE"])
valuesAge = np.array(df1["AGE "])
valuesAge = (valuesAge - np.nanmean(valuesAge))/(3*np.nanstd(valuesAge))
valuesAge = 0.5*(valuesAge + 1)
valuesAge[valuesAge < 0] = 0
valuesAge[valuesAge >= 1] = 1
df1["AGE "] = valuesAge #normalization
df1["SEX"]  = df1["SEX"] - 1


## LIST OF OUTCOMES WE WANT TO VERIFY
data ={'Classifier': names}

columns_classifieroutputs = ["S.NO "]
for outcome in outcomes:
    for indexclassifier in range(len(classifiers)):
        columns_classifieroutputs .append(outcome[0:4] + "_" + str(indexclassifier))
        columns_classifieroutputs .append(outcome[0:4] + "_p_" + str(indexclassifier))

dfclassifieroutputs = pd.DataFrame(np.zeros([len(df1["S.NO "]), len(columns_classifieroutputs)])*np.nan, 
    columns = columns_classifieroutputs)
dfclassifieroutputs["S.NO "] = df1["S.NO "]

for seed in random_seeds:
    for desireddata in range(1,8):
        #print("Data is " + str(desireddata))
        dfsummary = pd.DataFrame(data)    # dfsummary gives us the quality statistics of a specific classifier, for all classification outcomes, for a single input data type
        mdfclassifieroutputs = dfclassifieroutputs.copy()
        # somehow we need to store the classifier outputs for each patient in an organized way (rows = patients, columns = classifiers for each outcome OUTCOME_1, OUTCOME_2 .... and so on)                                    
        
        for outcome in outcomes: # iterate in outcomes
            #print("OUTCOME")
            cor_valuesm_list = []
            mcc_valuesm_list = []
            p_valuesm_list = []
            df2 = df1.copy() #[cor_feature1] ## most correlated variable
            # Select desired data
            df2 = selectDesiredData(df2, desireddata) #1 - Amnamesis, 2 - Ultrasound, 3 - X-ray, 4- CT, 5- Amnamesis + LUS, 6 - Amnamesis + X-ray, 7 - Amnamesis + CT 
            df2.insert(0, outcome, df[outcome], True)
            df2 = df2.dropna() # drop empty rows  - 79 rows in total
            y = df2[outcome]
            df2 = df2.drop(columns = outcome)
            ### Now we can do a random forest classifier
            X = df2.copy()
            ## Strafied cross validation
            skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed*42)
            skf.get_n_splits(X,y)
            index_classifier = 0
            for name, clf in zip(names, classifiers): # iterate in classifiers
                counter = 0
                #print("CLASSIFIER:")
                for train_index, test_index in skf.split(X, y):  # iterate in cross-validation folds
                    #print("TEST:", test_index)
                    #print("TRAIN:", train_index, "TEST:", test_index)
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    X_train = X_train.drop(columns = "S.NO ") 
                    SNO_test = X_test["S.NO "] # patient identifier
                    X_test = X_test.drop(columns = "S.NO ")
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    # Train the model using the training sets
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    try:
                        y_proba = clf.decision_function(X_test)
                    except:
                        try:
                            y_proba = clf.predict_proba(X_test)[:,1]
                        except:
                            y_proba = np.zeros(y_pred.shape)*np.nan
                    mdfclassifieroutputs.loc[SNO_test - 1, outcome[0:4] + "_" + str(index_classifier)] = y_pred
                    mdfclassifieroutputs.loc[SNO_test - 1, outcome[0:4] + "_p_" + str(index_classifier)] = y_proba
                    X_test.insert(0,"OUTCOME_PRED", y_pred)
                    X_test.insert(0,"OUTCOME", y_test)
                    if counter == 0:
                        dfout = pd.DataFrame(X_test)
                    else:
                        dfout = dfout.append(X_test)
                    counter += 1
                yout = dfout["OUTCOME"]
                dfout = dfout.drop(columns = "OUTCOME")
                #matrix = confusion_matrix(yout, dfout["OUTCOME_PRED"])
                #report = classification_report(yout, dfout["OUTCOME_PRED"])
                cor_valuesm, p_valuesm = cor_calculator(dfout["OUTCOME_PRED"], yout)
                ypred = np.array(mdfclassifieroutputs[outcome[0:4] + "_" + str(index_classifier)])
                ytrue = np.array(df[outcome])
                # Get rid of NaNs
                yprednonan = ypred[~np.isnan(ypred) & ~np.isnan(ytrue) ]
                ytruenonan = ytrue[~np.isnan(ypred) & ~np.isnan(ytrue) ]                
                mcc = matthews_corrcoef(ytruenonan, yprednonan)
                mcc_valuesm_list.append(mcc)
                cor_valuesm_list.append(cor_valuesm)
                p_valuesm_list.append(p_valuesm) #p_valuesm
                index_classifier += 1
            dfsummary['Corr: ' + outcome[0:4]] = cor_valuesm_list
            dfsummary['p-v: ' + outcome[0:4]] = p_valuesm_list
            dfsummary['MCC: ' + outcome[0:4]] = mcc_valuesm_list
            mdfclassifieroutputs[outcome[0:4]] = df[outcome]
        print(dfsummary)
        dfsummary.to_csv("dfsummary_" + str(desireddata) + "_seed" + str(seed) + ".csv")

        mdfclassifieroutputs.to_csv("dfclassifieroutputs_" + str(desireddata) + "_seed" + str(seed) + ".csv")

print("PHASE 1 COMPLETED")


#for seed in random_seeds:
    ### Once we reach here, we start analyzing the data for conclusions on best classifier parameters. For these, we compute more detailed metrics

order_according = "Corr: " # "MCC: "
for seed in random_seeds:
    list_dfsummary = []
    for desireddata in range(1,8):
        dfsummary = pd.read_csv("dfsummary_" + str(desireddata)+ "_seed" + str(seed)  + ".csv")
        mdfclassifieroutputs = pd.read_csv("dfclassifieroutputs_" + str(desireddata) + "_seed" + str(seed) + ".csv") #+ "_seed" + str(seed) + 
        ## We get the maximum classifier score for each classifier type
        listClassifiers = dfsummary["Classifier"].str[:-3].drop_duplicates() # find unique list of methods
        listDataFramesOutcomes = []
        for outcome in outcomes:
            columns_outcome = ["Classifier " + outcome[0:4], "MCC: " + outcome[0:4], "Corr: " + outcome[0:4], "p-v: " + outcome[0:4], "Index: "+ outcome[0:4]]
            dfsummary_classifier = pd.DataFrame(np.zeros([len(listClassifiers), len(columns_outcome)])*np.nan,
            columns = columns_outcome)
            index_classifier = 0
            for classifier in listClassifiers:
                dfsummary_classifier["Classifier " + outcome[0:4]].iloc[index_classifier] = classifier
                ## select rows corresponding to this classifier
                contain_rows = dfsummary[dfsummary["Classifier"].str.contains(classifier)]
                bestClassifierSelection = contain_rows[contain_rows[order_according + outcome[0:4]] ==  contain_rows[order_according + outcome[0:4]].max()]
                if len(bestClassifierSelection ) == 0: # no available data
                    index_classifier += 1
                    continue
                bestClassifier = bestClassifierSelection["Classifier"].iloc[0]
                dfsummary_classifier["Classifier " + outcome[0:4]].iloc[index_classifier] = bestClassifier
                indexBestClassifier = names.index(bestClassifier)
                #ypred = np.array(mdfclassifieroutputs[outcome[0:4] + "_" + str(indexBestClassifier)])
                #yref = np.array(mdfclassifieroutputs[outcome[0:4]])
                ### Now we can start computing metrics such as accuracy and so on... (for the summary just r and p-v)
                dfsummary_classifier["Corr: " + outcome[0:4]].iloc[index_classifier] = bestClassifierSelection["Corr: " + outcome[0:4]].iloc[0]
                dfsummary_classifier["MCC: " + outcome[0:4]].iloc[index_classifier] = bestClassifierSelection["MCC: " + outcome[0:4]].iloc[0]
                dfsummary_classifier["p-v: " + outcome[0:4]].iloc[index_classifier] = bestClassifierSelection["p-v: " + outcome[0:4]].iloc[0]
                dfsummary_classifier["Index: " + outcome[0:4]].iloc[index_classifier] = indexBestClassifier
                index_classifier +=1
            # Here we can do some sorting of the table in decreasing order of MCC
            dfsummary_classifier = dfsummary_classifier.sort_values(by=[order_according + outcome[0:4]], ascending = False).reset_index(drop=True)
            listDataFramesOutcomes.append(dfsummary_classifier)
        # Here we can append all this dataframes in one and add a column for desiredata
        dfsummary_classifier_outcomes = pd.concat(listDataFramesOutcomes, axis=1)
        dfsummary_classifier_outcomes["Representation"] = desireddata
        list_dfsummary.append(dfsummary_classifier_outcomes)
    dfsummary_representations = pd.concat(list_dfsummary, axis = 0)
    dfsummary_representations.to_csv("summary_of_representations" + "_seed" + str(seed) + ".csv") #+ "_seed" + str(seed) 
print("PHASE 2 COMPLETED")

for seed in random_seeds:
    ### Visualize confusion matrices and ROC curves for top classifiers
    dfsummary_representations = pd.read_csv("summary_of_representations" + "_seed" + str(seed) + ".csv")
    dfsummary_representations_best = dfsummary_representations.iloc[0::len(listClassifiers)]
    ### Find each model and compute confusion matrix and ROC_CURVE
    listDataFramesOutcomes = []
    for outcome in outcomes:
        columns_outcome = ["Representation","AUROC: " + outcome[0:4], "MCC: " + outcome[0:4], "Bal.Acc: " + outcome[0:4], "p-v: " + outcome[0:4], "Corr: " + outcome[0:4],
         "CM[0,0]: " + outcome[0:4], "CM[0,1]: " + outcome[0:4], "CM[1,0]: " + outcome[0:4], "CM[1,1]: "+ outcome[0:4]]
        dfconfmatrix_representations = pd.DataFrame(np.zeros([7, len(columns_outcome)])*np.nan,
        columns = columns_outcome)
        for desireddata in range(1,8):
            mdfclassifieroutputs = pd.read_csv("dfclassifieroutputs_" + str(desireddata) + "_seed" + str(seed) +  ".csv")
            indexBestClassifier = dfsummary_representations_best["Index: "+ outcome[0:4]].iloc[desireddata-1]
            # Load ypred and ytest ... 
            ypred = np.array(mdfclassifieroutputs[outcome[0:4] + "_" + str(int(indexBestClassifier))])
            ypredproba = np.array(mdfclassifieroutputs[outcome[0:4] + "_p_" + str(int(indexBestClassifier))])
            ytrue = np.array(mdfclassifieroutputs[outcome[0:4]])
            # Get rid of NaNs
            yprednonan = ypred[~np.isnan(ypred) & ~np.isnan(ytrue) & ~np.isnan(ypredproba)]
            ypredprobanonan = ypredproba[~np.isnan(ypred) & ~np.isnan(ytrue) & ~np.isnan(ypredproba)]
            ytruenonan = ytrue[~np.isnan(ypred) & ~np.isnan(ytrue) & ~np.isnan(ypredproba)]
            confmatrix = confusion_matrix(ytruenonan, yprednonan)
            balancedaccuracy = balanced_accuracy_score(ytruenonan, yprednonan)
            mcc = matthews_corrcoef(ytruenonan, yprednonan)
            auroc = roc_auc_score(ytruenonan,ypredprobanonan) 
            corr, p_corr = cor_calculator(yprednonan, ytruenonan)
            dfconfmatrix_representations["CM[0,0]: " + outcome[0:4]].iloc[desireddata-1] = confmatrix[0,0]
            dfconfmatrix_representations["CM[0,1]: " + outcome[0:4]].iloc[desireddata-1] = confmatrix[0,1]
            dfconfmatrix_representations["CM[1,0]: " + outcome[0:4]].iloc[desireddata-1] = confmatrix[1,0]
            dfconfmatrix_representations["CM[1,1]: " + outcome[0:4]].iloc[desireddata-1] = confmatrix[1,1]
            dfconfmatrix_representations["Representation"].iloc[desireddata-1] = desireddata
            dfconfmatrix_representations["Bal.Acc: " + outcome[0:4]].iloc[desireddata-1] = balancedaccuracy
            dfconfmatrix_representations["Corr: " + outcome[0:4]].iloc[desireddata-1] = corr
            dfconfmatrix_representations["p-v: " + outcome[0:4]].iloc[desireddata-1] = p_corr
            dfconfmatrix_representations["MCC: " + outcome[0:4]].iloc[desireddata-1] = mcc
            dfconfmatrix_representations["AUROC: " + outcome[0:4]].iloc[desireddata-1] = auroc
        listDataFramesOutcomes.append(dfconfmatrix_representations)
    dfconfmatrix_representations_outcomes = pd.concat(listDataFramesOutcomes, axis=1)    
    dfconfmatrix_representations_outcomes.to_csv("confusionmatrix_representations" + "_seed" + str(seed) + ".csv")

### Build summary per seeds to identify best performing iterations and average statistics
list_dfconfmatrix_representations_outcomes_seeds = []
#pd.DataFrame(np.zeros([7*len(random_seeds), len(columns_outcome)])*np.nan
for desireddata in range(1,8):
    for seed in random_seeds:
        dfconfmatrix_representations_outcomes = pd.read_csv("confusionmatrix_representations" + "_seed" + str(seed) + ".csv")
        list_dfconfmatrix_representations_outcomes_seeds.append(dfconfmatrix_representations_outcomes[:].iloc[desireddata-1])
dfconfmatrix_representations_outcomes_seeds = pd.concat(list_dfconfmatrix_representations_outcomes_seeds, axis = 1).transpose()
dfconfmatrix_representations_outcomes_seeds.to_csv("confusionmatrix_representations_allseeds.csv")
print("PHASE 3 COMPLETED")

### Build ROC curves with confidence intervals w.r.t. 5-fold cross validation
### Amnamesis, X-ray, Amnamesis + X-ray for deceased
# https://towardsdatascience.com/pooled-roc-with-xgboost-and-plotly-553a8169680c

#c_fill      = 'rgba(52, 152, 219, 0.2)'
#c_line      = 'rgba(52, 152, 219, 0.5)'
#c_line_main = 'rgba(41, 128, 185, 1.0)'
c_grid      = 'rgba(189, 195, 199, 0.5)'
#c_annot     = 'rgba(149, 165, 166, 0.5)'
#c_highlight = 'rgba(192, 57, 43, 1.0)'

c_line_main_list = ['black', 'cornflowerblue', 'darkseagreen', 'salmon', 'navy', 'darkgreen', 'darkmagenta']
labels_desireddata = ['Amnamesis', 'LUS', 'X-ray', 'CT', 'Amn. + LUS', 'Amn. + X-ray', 'Amn. + CT']

for outcome in outcomes:
    fig = go.Figure()
    for desireddata in range(1,8):
        fpr_list = []
        tpr_list = []
        thresholds_list = []
        auroc_list = []
        for seed in random_seeds:
            ### Visualize ROC curves for top classifiers
            dfsummary_representations = pd.read_csv("summary_of_representations" + "_seed" + str(seed) + ".csv")
            dfsummary_representations_best = dfsummary_representations.iloc[0::len(listClassifiers)]
            ### Find each model and compute confusion matrix and ROC_CURVE
            mdfclassifieroutputs = pd.read_csv("dfclassifieroutputs_" + str(desireddata) + "_seed" + str(seed) +  ".csv")
            indexBestClassifier = dfsummary_representations_best["Index: "+ outcome[0:4]].iloc[desireddata-1]
            # Load ypred and ytest ... 
            ypred = np.array(mdfclassifieroutputs[outcome[0:4] + "_" + str(int(indexBestClassifier))])
            ypredproba = np.array(mdfclassifieroutputs[outcome[0:4] + "_p_" + str(int(indexBestClassifier))])
            ytrue = np.array(mdfclassifieroutputs[outcome[0:4]])
            # Get rid of NaNs
            yprednonan = ypred[~np.isnan(ypred) & ~np.isnan(ytrue) & ~np.isnan(ypredproba)]
            ypredprobanonan = ypredproba[~np.isnan(ypred) & ~np.isnan(ytrue) & ~np.isnan(ypredproba)]
            ytruenonan = ytrue[~np.isnan(ypred) & ~np.isnan(ytrue) & ~np.isnan(ypredproba)]
            fpr, tpr, thresholds = roc_curve(ytruenonan, ypredprobanonan)
            auroc = roc_auc_score(ytruenonan,ypredprobanonan) 
            auroc_list.append(auroc)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            thresholds_list.append(thresholds)
        ## Plot ROC curve here and hold on for next iteration
        # Precondition fpr
        fpr_mean = np.linspace(0,1,100)
        interp_tprs = []
        for i in range(len(fpr_list)):
            fpr           = fpr_list[i]
            tpr           = tpr_list[i]
            interp_tpr    = np.interp(fpr_mean, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)
        tpr_mean     = np.mean(interp_tprs, axis=0)
        auroc_mean = np.mean(auroc_list)
        tpr_mean[-1] = 1.0
        tpr_std      = 2*np.std(interp_tprs, axis=0)
        tpr_upper    = np.clip(tpr_mean+tpr_std, 0, 1)
        tpr_lower    = tpr_mean-tpr_std


        '''
        fig.add_trace(go.Scatter(
            x          = fpr_mean,
            y          = tpr_upper,
            line       = dict(color=c_line, width=1),
            hoverinfo  = "skip",
            showlegend = False,
            name       = 'upper'))
        fig.add_trace(go.Scatter(
            x          = fpr_mean,
            y          = tpr_lower,
            fill       = 'tonexty',
            fillcolor  = c_fill,
            line       = dict(color=c_line, width=1),
            hoverinfo  = "skip",
            showlegend = False,
            name       = 'lower'))
        '''
        fig.add_trace(go.Scatter(
            x          = fpr_mean,
            y          = tpr_mean,
            line       = dict(color=c_line_main_list[desireddata-1], width=2),
            hoverinfo  = "skip",
            showlegend = True,
            name       = labels_desireddata[desireddata-1] + f', AUC: {auroc_mean:.3f}'))
        fig.add_shape(
            type ='line', 
            line =dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        fig.update_layout(
            template    = 'plotly_white', 
            title_x     = 0.5,
            xaxis_title = "1 - Specificity",
            yaxis_title = "Sensitivity",
            width       = 800,
            height      = 800,
            legend      = dict(
                yanchor="bottom", 
                xanchor="right", 
                x=0.95,
                y=0.01,
            )
        )
        fig.update_yaxes(
            range       = [0, 1],
            gridcolor   = c_grid,
            scaleanchor = "x", 
            scaleratio  = 1,
            linecolor   = 'black')
        fig.update_xaxes(
            range       = [0, 1],
            gridcolor   = c_grid,
            constrain   = 'domain',
            linecolor   = 'black')
    fig.show()
print("PHASE 4 COMPLETED")



# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# mcnemar test
from scipy.stats import chi2
from scipy.stats import binom
from scipy.special import comb

def mprint (what_to_print, file):
    import sys
    print(what_to_print)  # print to screen
    original_stdout = sys.stdout
    sys.stdout = file
    print(what_to_print) # print to file
    sys.stdout = original_stdout
    return

from statsmodels.stats.contingency_tables import mcnemar
def performMcNemarTest (labels1, predictions1, labels2, predictions2, destination_results):
# def performMcNemarTest (labels1, predictions1, labels2, predictions2)
# performs McNemar test to compare merits of two methods
# can be applied to frames or patient cases
    f = open(destination_results + ".txt", "w")

    contingency_Table = np.array([[0,0],[0,0]])
    classifier1_correct = labels1 == predictions1
    classifier2_correct = labels2 == predictions2
    contingency_Table[0, 0] = np.count_nonzero ((classifier1_correct == True) & (classifier2_correct == True))
    contingency_Table[0, 1] = np.count_nonzero ((classifier1_correct == True) & (classifier2_correct == False))
    contingency_Table[1, 0] = np.count_nonzero ((classifier1_correct == False) & (classifier2_correct == True))
    contingency_Table[1, 1] = np.count_nonzero ((classifier1_correct == False) & (classifier2_correct == False))
    #contingency_Table = np.round(contingency_Table/5) # Due to 5-repeated k-cross validations

    mprint("             2 Correct,       2 Incorre", f)
    mprint("1 Correct    %10d       %10d" % (contingency_Table[0,0], contingency_Table[0,1]), f)
    mprint("1 Incorre    %10d       %10d" % (contingency_Table[1,0], contingency_Table[1,1]), f)
   
    mprint(contingency_Table, f)
    total_num_cases = contingency_Table[0,0] + contingency_Table[0,1]  + contingency_Table[1,0] + contingency_Table[1,1]
    mprint("No elements: %d" % total_num_cases, f)
    contingency_Table_ratios = np.array(contingency_Table)/total_num_cases
    mprint("Contigency table in proportions (%)", f)
    mprint("             2 Correct,       2 Incorre", f)
    mprint("1 Correct    %10.3f       %10.3f" % (contingency_Table_ratios[0,0], contingency_Table_ratios[0,1]), f)
    mprint("1 Incorre    %10.3f       %10.3f" % (contingency_Table_ratios[1,0], contingency_Table_ratios[1,1]), f)
   

    odds_ratio_1_2 = contingency_Table[0,1]/contingency_Table[1,0]
    mprint("Odd ratios OR = 1Correct_2Incorre/1Incorre_1Correct", f) 
    mprint("OR = %10.3f" % (odds_ratio_1_2), f)


    if (np.min(contingency_Table) <= 25):
        isexact = True
    else:
        isexact = False

    mprint(mcnemar(contingency_Table, exact=isexact),f) # we use standard function to avoid issues
    '''
    if exact == False:
        correction = True
        if correction:
            mcnemar_statistic = (np.abs(contingency_Table[0,1] - contingency_Table[1,0]) - 1) ** 2/(contingency_Table[0,1] + contingency_Table[1,0])
        else:
            mcnemar_statistic = (np.abs(contingency_Table[0,1] - contingency_Table[1,0])) ** 2/(contingency_Table[0,1] + contingency_Table[1,0])
        mprint("Mc Nemar statistic %10.3f" % mcnemar_statistic, f)
        pvalue_mcnemar = chi2.sf(mcnemar_statistic, df=1)
    else:
        #pvalue_mcnemar = 2*binom.sf(contingency_Table[0,1] - 1, contingency_Table[0,1] + contingency_Table[1,0], 0.5)
        n = contingency_Table[0,1] + contingency_Table[1,0]
        i = contingency_Table[0,1]
        i_n = np.arange(i+1, n+1)
        pvalue_mcnemar = 1 - np.sum(comb(n, i_n) * 0.5 ** i_n * (1 - 0.5) ** (n - i_n)) 
        pvalue_mcnemar *= 2
        #mid_p_value = pvalue_mcnemar - binom.pmf(i, n, 0.5)
        #pvalue_mcnemar = mid_p_value


    mprint ("p-value = %10.8f" % pvalue_mcnemar, f)
    '''


######### PERFORM McNemarTest - you can group all repetitions together and average by 5 to obtain a global score
for outcome in outcomes:
    ypred_desireddata_list = []
    ytrue_desireddata_list = []
    for desireddata in range(1,8):
        ypred_list = []
        ytrue_list = []
        for seed in random_seeds:
            ### Visualize ROC curves for top classifiers
            dfsummary_representations = pd.read_csv("summary_of_representations" + "_seed" + str(seed) + ".csv")
            dfsummary_representations_best = dfsummary_representations.iloc[0::len(listClassifiers)]
            ### Find each model and compute confusion matrix and ROC_CURVE
            mdfclassifieroutputs = pd.read_csv("dfclassifieroutputs_" + str(desireddata) + "_seed" + str(seed) +  ".csv")
            indexBestClassifier = dfsummary_representations_best["Index: "+ outcome[0:4]].iloc[desireddata-1]
            # Load ypred and ytest ... 
            ypred = np.array(mdfclassifieroutputs[outcome[0:4] + "_" + str(int(indexBestClassifier))])
            ypredproba = np.array(mdfclassifieroutputs[outcome[0:4] + "_p_" + str(int(indexBestClassifier))])
            ytrue = np.array(mdfclassifieroutputs[outcome[0:4]])
            # Group repetitions into one
            ypred_list.append(ypred)
            ytrue_list.append(ytrue)
        ypred_seed = np.concatenate(ypred_list).ravel()
        ytrue_seed = np.concatenate(ytrue_list).ravel()
        ypred_desireddata_list.append(ypred_seed)
        ytrue_desireddata_list.append(ytrue_seed)
    ## Here we start McNemar test ### 
    for desireddata in range(2,8): # We perform all tests with respect to amnamesis
        labels2 = ytrue_desireddata_list[0]
        labels1 = ytrue_desireddata_list[desireddata - 1]
        predictions2 = ypred_desireddata_list[0]
        predictions1 = ypred_desireddata_list[desireddata - 1]
        labels2nonan = labels2[~np.isnan(labels1) & ~np.isnan(labels2) & ~np.isnan(predictions1) & ~np.isnan(predictions2)]
        labels1nonan = labels1[~np.isnan(labels1) & ~np.isnan(labels2) & ~np.isnan(predictions1) & ~np.isnan(predictions2)]
        predictions2nonan = predictions2[~np.isnan(labels1) & ~np.isnan(labels2) & ~np.isnan(predictions1) & ~np.isnan(predictions2)]
        predictions1nonan = predictions1[~np.isnan(labels1) & ~np.isnan(labels2) & ~np.isnan(predictions1) & ~np.isnan(predictions2)]
        performMcNemarTest (labels1nonan, predictions1nonan, labels2nonan, predictions2nonan, "McNemar_" + str(desireddata) + "_" + outcome + '.txt')

    ## Build contigency matrices
    ## Calculate odds ratios
    ## Calculate p-values
    # Get rid of NaNs
    #yprednonan = ypred[~np.isnan(ypred) & ~np.isnan(ytrue) & ~np.isnan(ypredproba)]
    #ypredprobanonan = ypredproba[~np.isnan(ypred) & ~np.isnan(ytrue) & ~np.isnan(ypredproba)]
    #ytruenonan = ytrue[~np.isnan(ypred) & ~np.isnan(ytrue) & ~np.isnan(ypredproba)]
print("PHASE 5 COMPLETED")



'''
columns_classifieroutputs = ["S.NO "]
for outcome in outcomes:
    for indexclassifier in range(len(classifiers)):
        columns_classifieroutputs .append(outcome[0:4] + "_" + str(indexclassifier))
dfclassifieroutputs = pd.DataFrame(np.zeros([len(df1["S.NO "]), len(columns_classifieroutputs)])*np.nan, 
    columns = columns_classifieroutputs)
print("Just finished!")

            ### We could define a DataFrame, which contains Data representation, Classifier, Metrics .... 
            ### Then we could order this from top to bottom according to metric score
            ### Then we could concatenate horizontally for all data representations
            ### With this we would have the results we need
            ### On the meantime, we could store this ypreds in a new summarized table 

'''


### One we reach here, we have selected the best classifier for each data representation. We perform McNemar test to identify statistical diferences
### With this we are done




#### outcomes predictions with multiparametric models
## Amnamesis
## Ultrasound
## X-ray
## CT
## Amnamesis + Ultrasound
## Amnamesis + X-ray
## Amnamesis + CT
## Selected significant features



print(str(len(cor_feature)), 'selected features')
print("hello world")
