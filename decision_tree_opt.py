import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import operator
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

class Parameters:
    
    def evaluate(self, true, predict):
        self.model = true
        self.features = predict
        
        print('Model Performance')
        print("MAE:", round(mean_absolute_error(true, predict), 5) )
        print("MSE:", round(mean_squared_error(true, predict), 5) )
        print("Variance:", round(np.var(predict), 5) )
        print("Accuracy:", round(100 - np.mean((abs(predict - true)/true)*100), 2), "%" )
        print()
        return None
    
    def dtr_importance(self, df, model):
        self.df = df
        self.model = model
        
        imp = dict(zip(df.drop("appliance_wh", axis = 1).columns, list(model.feature_importances_)))
        imp = pd.DataFrame(imp, index = range(len(list(model.feature_importances_))))
        print(imp.loc[25:].T.sort_values(25, ascending = False).rename(columns = {25:"values"}).head() )
        print()
        return None
        
    def dict_tuning(self, train):
        self.train = train
        ccp_alpha = list(range(100))        # 1
        criterion = ["mse", "friedman_mse", "mae"]        # "mse"
        max_depth = list(range(1,50)) # 2
        max_features = ["auto", "sqrt", "log2"] + list(range(1,train.shape[1]))        # 1
        max_leaf_nodes = list(range(2,50)) # 3
        min_impurity_decrease = np.arange(0.0,1.0, 0.001)        # 0.011
        min_impurity_split = np.arange(0.0,1.0, 0.001)        # 0.485
        min_samples_leaf = np.arange(0,1,0.1)        # 0.2
        min_samples_split = list(range(2,50))        # 49
        min_weight_fraction_leaf = np.arange(0,1,0.1)        # 0.2
        random_state = list(range(0,1000))        # 529
        splitter = ["best", "random"]        # "random"

        random_grid = { "ccp_alpha": ccp_alpha,
                        "criterion": criterion,
                        "max_depth": max_depth,
                        "max_features": max_features,
                        "max_leaf_nodes": max_leaf_nodes,
                        "min_impurity_decrease": min_impurity_decrease,
                        "min_impurity_split": min_impurity_split,
                        "min_samples_leaf": min_samples_leaf,
                        "min_samples_split": min_samples_split,
                        "min_weight_fraction_leaf": min_weight_fraction_leaf,
                        "random_state": random_state,
                        "splitter": splitter }
        """
        The tuning of the parameters is carried out through the GridSearchCV. In the notebook all parameters shown come from
        the results from GridSearchCV.
        
        tree_random = GridSearchCV(decisiontree, grid, n_jobs = -1, cv = 2)
        tree_random.fit(X_train, y_train_tree)
        tree_model = tree_random.best_estimator_
        print (tree_random.best_score_, tree_random.best_params_)
        """
        return random_grid