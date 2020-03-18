import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import operator
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class Parameters:
    
    def bias_var(self, X, y):
        self.X = X
        self.y = y
        
        t_size = int(len(X)*0.66)
        X_train = X[:t_size]
        y_train = y[:t_size]
        X_test = X[t_size:]
        y_test = y[t_size:]
    
        preds = []
        # The loop is iterated over a couple of equivalent parameters
        # to reduce the execution time, at the expense of reduction of results 
        for leave, samp in enumerate(range(2, 101)):
            dt = DecisionTreeClassifier(max_leaf_nodes = leave+2, min_samples_split = samp)
            model = dt.fit(X, y)
            preds += [ list(model.predict(X)) ] # every element is an y_pred
        
        stats = []    
        for x in range(len(preds)):       
            dt_bias = (y - np.mean(preds[x]))**2
            dt_variance = np.var(preds[x])
            dt_error = (preds[x] - y)**2
            acc = accuracy_score(y_true = y, y_pred = preds[x])
            stats += [ (dt_bias.mean(), dt_variance, dt_error.mean(), round(acc, 6)) ]
        
        stats = [ list(x) for x in stats ]
        df = pd.DataFrame(stats, columns = ["error", "bias", "variance", "accuracy"])
        df["bias_plus_var"] = df.bias + df.variance
    
        return df