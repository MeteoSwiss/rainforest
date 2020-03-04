#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class declarations and reading functions
required to unpickle trained RandomForest models

Daniel Wolfensberger
MeteoSwiss/EPFL
daniel.wolfensberger@epfl.ch
December 2019
"""

# Global imports
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os

current_file = os.path.abspath(__file__)
current_folder = os.path.dirname(current_file)

FOLDER_RF = current_folder + '/rf_models/'


    
##################
# Add here all classes/functions used in the construction of the pickled 
# instances
##################
    
def _polyfit_no_inter(x,y, degree):
    """linear regression with zero intercept"""
    X = []
    for i in range(1,degree+1):
        X.append(x**i)
    X = np.array(X).T
    p, _, _, _ = np.linalg.lstsq(X, y[:,None])
    p = np.insert(p,0,0) # Add zero intercept at beginning for compatibility with polyval
    return p[::-1] # Reverse because that's how it is in polyfit (high degree first)
    

class RandomForestRegressorBC(RandomForestRegressor):
    def __init__(self, 
                 model, 
                 vw, 
                 degree = 1, 
                 regtype = 'cdf', 
                 n_estimators=100,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super().__init__(n_estimators, criterion, max_depth, min_samples_split,
                 min_samples_leaf, min_weight_fraction_leaf, max_features,
                 max_leaf_nodes, min_impurity_decrease, min_impurity_split,
                 bootstrap, oob_score, n_jobs, random_state, verbose, warm_start)
        
        self.degree = degree
        self.regtype = regtype
        self.model = model
        self.vw = vw

    def fit(self, X,y, percentiles = 'auto', sample_weight = None):
        super().fit(X,y,sample_weight)
        y_pred = super().predict(X)
        if self.regtype in ['cdf','raw']:
            if self.regtype == 'cdf':
                x_ = np.sort(y_pred)
                y_ = np.sort(y)
            elif self.regtype == 'raw':
                x_ = y_pred
                y_ = y
            self.p = _polyfit_no_inter(x_,y_,self.degree)
        else:
            self.p = 1
            
        return 
    def predict(self,X, round_func = None, bc = True):
        if round_func == None:
            round_func = lambda x: x
        if bc:
            func = lambda x : np.polyval(self.p,x)
        else:
            func = lambda x: x
        pred = super().predict(X)
        return round_func(func(pred))
    
##################
        
class MyCustomUnpickler(pickle.Unpickler):
    import __main__
    __main__.RandomForestRegressorBC = RandomForestRegressorBC
    def find_class(self, module, name):
        print(module,name)
        return super().find_class(module, name)
    
    
def read_rf(rf_name):
    """
    Reads a randomForest model from the repository using pickle. All custom
    classes and functions used in the construction of these pickled models
    must be defined below
    
    Parameters
    ----------
    rf_name : str
        Name of 
 
        
    Returns
    -------
    A trained sklearn randomForest instance that has the predict() method, 
    that allows to predict precipitation intensities for new points
    """
    
    if rf_name[-2:] != '.p':
        rf_name += '.p'
    
    if os.path.dirname(rf_name) == '':
        rf_name = FOLDER_RF + rf_name
    
    unpickler = MyCustomUnpickler(open(rf_name, 'rb'))
    if not os.path.exists(rf_name):
        raise IOError('RF model {:s} does not exist!'.format(rf_name))
    else:
        return unpickler.load()
      