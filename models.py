from enum import Enum

import numpy as np
from lssvr import LSSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost.sklearn import XGBRegressor

# Random seed
RANDOM_STATE = 0

# Defining models names
class ModelsEnum(Enum):
    LINEAR = 'LinearRegression'
    ELASTIC_NET = 'ElasticNet'
    DECISION_TREE = 'DecisionTreeRegressor'
    MLP = 'MLPRegressor'
    KNN = 'KNeighborsRegressor'
    SVM_POLY = 'SVR_POLY'
    SVM_SIGMOID = 'SVR_SIGMOID'
    LSSVM_LINEAR = 'LSSVR_LINEAR'
    LSSVM_RBF = 'LSSVR_RBF'
    RANDOM_FOREST = 'RandomForestRegressor'
    XGBOOST = 'XGBoost'

# Definition of the space search for Grid Search
alpha = np.logspace(-6, 6, 13).tolist() 
l1_ratio = [0, .01, .1, .5, .9, .99, 1] 

n_neighbors = [1, 2, 5, 10, 25, 50, 75, 100]
weights = ['distance']

hidden_layer_sizes_1 = [(3**x,) for x in range(4)]
hidden_layer_sizes_2 = [(3**x, 3**y) for x in range(4) for y in range(4)]
hidden_layer_sizes = hidden_layer_sizes_1 + hidden_layer_sizes_2
activation = ["logistic", "tanh", "relu"] 
solver = ["lbfgs"] 

C = np.logspace(-2, 6, 9).tolist()
gamma = np.logspace(-6, 2, 9).tolist()
degree = [2, 3, 4, 5]

criterion = ["squared_error", "friedman_mse", "absolute_error"]
min_samples_leaf = [0.01, 0.025, 0.05, 0.075, 0.1, 0.25]
max_depth = [1, 2, 4, 6, 8, None]

n_estimators = [10, 50, 100, 500]
max_leaves = [2, 4, 16] 

# Defining models to be used and their respective initial params
MODELS = {
    ModelsEnum.LINEAR.value: LinearRegression(),
    ModelsEnum.ELASTIC_NET.value: ElasticNet(random_state=RANDOM_STATE, max_iter=20000),
    ModelsEnum.DECISION_TREE.value: DecisionTreeRegressor(random_state=RANDOM_STATE),
    ModelsEnum.MLP.value: MLPRegressor(learning_rate='adaptive', random_state=RANDOM_STATE, max_iter=20000),
    ModelsEnum.KNN.value: KNeighborsRegressor(algorithm='brute'),
    ModelsEnum.SVM_POLY.value: SVR(kernel='poly', max_iter=10000),
    ModelsEnum.SVM_SIGMOID.value: SVR(kernel='sigmoid', max_iter=10000),
    ModelsEnum.LSSVM_LINEAR.value: LSSVR(kernel='linear'),
    ModelsEnum.LSSVM_RBF.value: LSSVR(kernel='rbf'),    
    ModelsEnum.RANDOM_FOREST.value: RandomForestRegressor(random_state=RANDOM_STATE),
    ModelsEnum.XGBOOST.value: XGBRegressor(random_state=RANDOM_STATE),
}

# Defining the dictionary of params to be passed to the Grid Search
MODELS_PARAMS = {
    ModelsEnum.LINEAR.value: {

    },
    ModelsEnum.ELASTIC_NET.value: {
        'alpha': alpha,
        'l1_ratio': l1_ratio,
    },
    ModelsEnum.DECISION_TREE.value: {
        'criterion': criterion,
        'max_depth': max_depth,
        'min_samples_leaf': min_samples_leaf,

    },
    ModelsEnum.MLP.value: {
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': activation,
        'solver': solver,
    },
    ModelsEnum.KNN.value: {
        'n_neighbors': n_neighbors,
        'weights': weights,

    },
    ModelsEnum.SVM_POLY.value: {
        'C': C,
        'gamma': gamma,
        'degree': degree,
    },
    ModelsEnum.SVM_SIGMOID.value: {
        'C': C,
        'gamma': gamma,
    },
    ModelsEnum.LSSVM_LINEAR.value: {
        'C': C,
    },
    ModelsEnum.LSSVM_RBF.value: {
        'C': C,
        'gamma': gamma,

    },    
    ModelsEnum.RANDOM_FOREST.value: {
        'n_estimators': n_estimators,
        'criterion': criterion,
        'max_depth': max_depth,
        'min_samples_leaf': min_samples_leaf,

    }, 
    ModelsEnum.XGBOOST.value: {
        'max_depth': max_depth,
        'n_estimators': n_estimators,
        'max_leaves': max_leaves,
    }
}
