import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint

class Tuning:
    
    def forest_tun(self):
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)] #10
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt'] #2
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 100, num = 10)] + [None] #1
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        """
        The tuning of the parameters is carried out through the RandomizedSearchCV. In the notebook all parameters shown come from
        the results from RandomizedSearchCV.
                
        rf = RandomForestRegressor()
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions=random_grid,
                                       n_iter = 75, scoring='neg_mean_absolute_error', 
                                       cv = 2, verbose = 2, random_state = 1234, n_jobs = -1)

        rf_random.fit(train_features, train_labels)
        rf_model = rf_random.best_estimator_
        print (rf_random.best_score_, rf_random.best_params_)
        """
 
        # Create the parameter grid based on the results of random search 
        param_grid = {'bootstrap': [False],
                      'max_depth': [None],
                      'max_features': ["sqrt", "auto", "log2"] + [],
                      'min_samples_leaf': [1],
                      'min_samples_split': [6],
                      'n_estimators': [800, 1200] }

        """
        rf = RandomForestRegressor()
        grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                                   scoring = 'neg_mean_absolute_error', cv = 3, 
                                   n_jobs = -1, verbose = 2)

       """
        
        return random_grid
    
    
    