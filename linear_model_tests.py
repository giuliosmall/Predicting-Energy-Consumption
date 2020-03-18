import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
import operator
import warnings
warnings.filterwarnings(action = "ignore")

class Test: 
    
    def lasso(self, df, X, y):
        self.df = df
        self.X = X
        self.y = y
        
        X_std = StandardScaler().fit_transform(X)
        size = int(len(X)*0.66)
        X_tr = X_std[:size]
        y_tr = y[:size]
        X_tes = X_std[size:]
        y_tes  = y[size:]
        
        regression = Lasso(alpha = 0.99, fit_intercept = True)
        model = regression.fit(X_tr, y_tr)
        y_pred = model.predict(X_std)
        
        coeffs = dict(zip(df.drop("appliance_wh", axis = 1).columns, list(model.coef_)))
        coeffs = pd.DataFrame( sorted(coeffs.items(), key = operator.itemgetter(1))[::-1] ) # sort by value
        coeffs = coeffs.rename(columns = {0: "col", 1:"val"})
        coeffs.drop(coeffs[coeffs.val == 0].index, inplace = True)
        
        X_las = df.drop([x for x in df.columns if x not in list(coeffs.col)], axis = 1).values
        X_las = X_las.reshape( (len(X_las), len(X_las[0])))
        y_las = np.log1p(df.appliance_wh)
        y_las = y_las.values.reshape( len(y_las), 1)
        X_train_las = X_las[:size]
        y_train_las = y_las[:size]
        X_test_las = X_las[size:]
        y_test_las = y_las[size:]
        
        model_2 = LinearRegression(fit_intercept = True)
        model_2.fit(X_train_las, y_train_las)
        y_pred_2 = model_2.predict(X_las)

        print("Reducing features with LASSO", "\n\t ## Remaining features:", len(coeffs),
              "\n\t ## R^2 after reduction:", round(r2_score(y_las, y_pred_2), 5) )
        
        return model_2
    
    def recurs_elimin(self, df, X, y):
        self.df = df
        self.X = X
        self.y = y
        
        size = int(len(X)*0.66)
        X_tr = X[:size]
        y_tr = y[:size]
        X_tes = X[size:]
        y_tes = y[size:]
        
        model = LinearRegression(fit_intercept = True)
        rfecv = RFECV(estimator = model, step = 1, scoring = "neg_mean_squared_error", cv = 10)
        rfecv.fit(X_tr, y_tr)
        rfecv.transform(X_tr)

        feats = dict(zip(df.drop("appliance_wh", axis = 1).columns, rfecv.support_))
        feats = pd.DataFrame( sorted(feats.items(), key = operator.itemgetter(1))[::-1] ) # sort by value
        feats = feats.rename(columns={0: "col", 1:"val"})
        feats.drop(feats[feats.val == False].index, inplace = True)

        data_rfe = df.drop([x for x in df.columns if x not in list(feats.col)], axis = 1)
        X_rfe = data_rfe.values
        y_rfe = np.log1p(df.appliance_wh)
        y_rfe = y_rfe.values.reshape( len(y_rfe), 1)
  
        X_tr_rfe = X_rfe[:size]
        y_tr_rfe = y_rfe[:size]
        X_tes_rfe = X_rfe[size:]
        y_tes_rfe = y_rfe[size:]

        model_rfe = LinearRegression(fit_intercept = True)
        model_rfe.fit(X_tr_rfe, y_tr_rfe)
        y_pred_rfe = model_rfe.predict(X_rfe)

        print("Reducing features with Recursive Feature Elimination", "\n\t ## Remaining features:", rfecv.n_features_,
              "\n\t ## R^2 after reduction:", round(r2_score(y_rfe, y_pred_rfe), 5) )
        
        return model_rfe