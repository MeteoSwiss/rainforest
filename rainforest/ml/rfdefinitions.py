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
import gzip
import tempfile
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import os
from scipy.interpolate import UnivariateSpline
from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from contextlib import nullcontext
import logging
logging.getLogger().setLevel(logging.INFO)
import mlflow

# Local imports
from ..common.object_storage import ObjectStorage
ObjStorage = ObjectStorage()
FOLDER_RF = Path(os.environ['RAINFOREST_DATAPATH'], 'rf_models')


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

from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import tempfile
import os
import gzip
import pickle
from contextlib import nullcontext

class RandomForestRegressorBC(RandomForestRegressor):
    """
    Extended RandomForestRegressor with optional bias correction,
    on-the-fly rounding, metadata, and optional cross-validation.
    """
    def __init__(self,
                 variables,
                 beta,
                 visib_weighting,
                 degree=1,
                 bctype='cdf',
                 metadata={},
                 n_estimators=100,
                 criterion="squared_error",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="sqrt",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super().__init__(n_estimators=n_estimators,
                         criterion=criterion,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                         max_features=max_features,
                         max_leaf_nodes=max_leaf_nodes,
                         min_impurity_decrease=min_impurity_decrease,
                         bootstrap=bootstrap,
                         oob_score=oob_score,
                         n_jobs=n_jobs,
                         random_state=random_state,
                         verbose=verbose,
                         warm_start=warm_start)

        self.degree = degree
        self.bctype = bctype
        self.variables = variables
        self.beta = beta
        self.visib_weighting = visib_weighting
        self.metadata = metadata
        self.p = None

    def fit(self, X, y, sample_weight=None, logmlflow='none', cv=0):
        """
        Fit both estimator and a-posteriori bias correction with optional cross-validation.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            The input samples.
        y : array-like, shape=(n_samples,)
            The target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        logmlflow : str, default='none'
            Whether to log training metrics to MLFlow. Can be 'none' to not log anything, 'metrics' to 
            only log metrics, or 'all' to log metrics and the trained model.
        cv : int, default=0
            Number of folds for cross-validation. If set to 0, will not perform
            cross-validation (i.e. no test error)
        Returns
        -------
        self : object
        """
        if cv >= 1:
            cross_validate = True
        else:
            cross_validate = False
        
        if logmlflow != 'none':
            mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
            mlflow.set_experiment(experiment_name='rainforest')

        X.columns = [str(col) for col in X.columns]
        
        
        
        run_context = mlflow.start_run() if logmlflow != 'none' else nullcontext()
        with run_context:
            if logmlflow != 'none':
                features_dic = {'features': X.columns.to_list()}
                mlflow.log_dict(features_dic, 'features.json')
                params_dic = self.get_params()
                mlflow.log_params(params_dic)
            
            logging.info(f"Fitting model to train data")
            
            
            if cross_validate:
                # Perform cross-validation
                kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
                y_pred_test = np.zeros(len(y))
                y_pred_test_bc = np.zeros(len(y))
                y_pred_ref = np.zeros(len(y))
                
                i = 0
                for train_index, test_index in kf.split(X):
                    logging.info(f"Running CV iteration {i}")
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    
                    self.set_params(warm_start=False)  # Ensure a fresh model
                    super().fit(X_train, y_train)
                    
                    y_pred_test[test_index] = super().predict(X_test)
                    
                    self.fit_bias_correction(y=y_test, y_pred=y_pred_test[test_index])
                    
                    y_pred_test_bc[test_index] = self.predict(X=X_test, bc=True)
                    y_pred_ref[test_index] = y_test
                    
                    i+=1
            
            # After CV, train a whole model from scratch
            self.set_params(warm_start=False) 
            super().fit(X, y, sample_weight)
            y_pred = super().predict(X)
            
            self.fit_bias_correction(y=y, y_pred=y_pred)
            
            y_pred_bc = self.predict(X=X, bc=True)

            if logmlflow != 'none':
                logging.info(f"Logging train performance to mlflow")
                self._log_metrics(y, y_pred, metrics_prefix='train')
                self._log_metrics(y, y_pred_bc, metrics_prefix='train_bc')
                if cross_validate:
                    logging.info(f"Logging test performance to mlflow")
                    self._log_metrics(y=y_pred_ref, y_pred=y_pred_test, metrics_prefix='test')
                    self._log_metrics(y=y_pred_ref, y_pred=y_pred_test_bc, metrics_prefix='test_bc')
                
                logging.info(f"Logged metrics to mlflow")
                if logmlflow == 'all':
                    logging.info(f"Upload fitted model to mlflow")
                    # Log the trained model and signature
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        temp_file_path = os.path.join(tmp_dir, "rf.pkl.gz")
                        with gzip.open(temp_file_path, 'wb') as f:
                            pickle.dump(self, f)
                        mlflow.log_artifact(temp_file_path, "rf_gzipped_pickle_model")

                    inpt_exp = X[:1]
                    sign = infer_signature(X[:10], y[:10])
                    mlflow.sklearn.log_model(sk_model=None,
                                            artifact_path='rf_signature_no_model',
                                            input_example=inpt_exp,
                                            signature=sign)
        return self

    def _log_metrics(self, y, y_pred, metrics_prefix):
        """
        Logs metrics to MLFlow.
        """
        mlflow.log_metric(f'{metrics_prefix}_R2', r2_score(y_true=y, y_pred=y_pred))
        mlflow.log_metric(f'{metrics_prefix}_MAE', mean_absolute_error(y_true=y, y_pred=y_pred))
        mlflow.log_metric(f'{metrics_prefix}_RMSE', root_mean_squared_error(y_true=y, y_pred=y_pred))
        
        
    def fit_bias_correction(self, y, y_pred):
        x_ = np.sort(y_pred)
        y_ = np.sort(y)
        
        if self.bctype in ['cdf', 'raw']:
            self.p = _polyfit_no_inter(x_, y_, self.degree)
        elif self.bctype == 'spline':
            _, idx = np.unique(x_, return_index=True)
            self.p = UnivariateSpline(x_[idx], y_[idx])
        else:
            self.p = 1


    def predict(self, X, round_func = None, bc = True):
        """
        Predict regression target for X.
        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.
        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        round_func : lambda function
            Optional function to apply to outputs (for example to discretize them
            using MCH lookup tables). If not provided f(x) = x will be applied
            (i.e. no function)
        bc : bool
            if True the bias correction function will be applied

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """
        
        pred = super().predict(X)

        if round_func == None:
            round_func = lambda x: x

        func = lambda x: x
        if bc and self.p is not None:
            if self.bctype in ['cdf','raw']:
                func = lambda x : np.polyval(self.p,x)
            elif self.bctype == 'spline':
                func = lambda x : self.p(x)
        out = func(pred)
        out[out < 0] = 0
        return round_func(out)

##################

class MyCustomUnpickler(pickle.Unpickler):
    """
    This is an extension of the pickle Unpickler that handles the
    bookeeeping references to the RandomForestRegressorBC class
    """
    import __main__
    __main__.RandomForestRegressorBC = RandomForestRegressorBC
    def find_class(self, module, name):
        return super().find_class(module, name)


def read_rf(rf_name='', filepath='', mlflow_runid=None):
    """
    Reads a randomForest model from the RF models folder using pickle. All custom
    classes and functions used in the construction of these pickled models
    must be defined in the script ml/rf_definitions.py

    Parameters
    ----------
    rf_name : str
        Name of the randomForest model, it must be stored in the folder
        /ml/rf_models and computed with the rf:RFTraining.fit_model function

    filepath: str
        Path to the model files, if not in default folder
        
    mlflow_runid: str
        If the model needs to be downloaded from mlflow, this variable 
        indicates the run ID that contains the model to use. If this value 
        is not None, rf_name and filepath are ignored. The env variable 
        MLFLOW_TRACKING_URI needs to be set.  

    Returns
    -------
    A trained sklearn randomForest instance that has the predict() method,
    that allows to predict precipitation intensities for new points
    """

    is_compressed = False
    if mlflow_runid is not None:
        artifact_path = 'rf_gzipped_pickle_model/rf.pkl.gz'
        logging.info(f'Downloading model from MLFlow at {os.getenv("MLFLOW_TRACKING_URI")}')
        logging.info(f'Run id: {mlflow_runid}, artifact: {artifact_path}')
        is_compressed = artifact_path.endswith('.gz')
        rf_name = mlflow.artifacts.download_artifacts(run_id=mlflow_runid, 
                                                      artifact_path=artifact_path,
                                                      dst_path='./') 
        logging.info(f'Model stored in {rf_name}.')
    else:
        
        if rf_name.endswith('.gz'):
            is_compressed = True
        else:
            if not rf_name[-2:].endswith('.p'):
                rf_name += '.p'

        if filepath == '':
            if os.path.dirname(rf_name) == '':
                rf_name = str(Path(FOLDER_RF, rf_name))
        else:
            rf_name = str(Path(filepath, rf_name))

        # Get model from cloud if needed
        rf_name = ObjStorage.check_file(rf_name)
    
    if not os.path.exists(rf_name):
        raise IOError('RF model {:s} does not exist!'.format(rf_name))
        
    if is_compressed:
        with gzip.open(open(rf_name, 'rb')) as f: 
            return MyCustomUnpickler(f).load()
    else:
        with open(rf_name, 'rb') as f: 
            return MyCustomUnpickler(f).load()

