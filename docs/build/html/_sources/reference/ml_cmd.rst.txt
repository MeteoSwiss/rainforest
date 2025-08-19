Machine-learning command-line tool
==========================================

The *ml* submodule has only a command-line tool to update the input data for the RF QPE algorithm and for training new RF models. For more sophisticated procedures please use :ref:`ml_module`.

.. _rf_train:

*rf_train*
-----------------

Updates any of the three tables of the database *gauge*, *radar* and *reference* with new data. 

**rf_train [options]**


Options:
  -h, --help            show this help message and exit
  -o OUTPUT, --outputfolder=OUTPUT
                        Path of the output folder, default is the ml/rf_models
                        folder in the current library
  -d DBFOLDER, --dbfolder=DBFOLDER
                        Path of the database main folder, default is
                        /store/msrad/radar/radar_database/
  -i DBFOLDER, --inputfolder=DBFOLDER
                        Path where the homogeneized input files for the RF
                        algorithm are stored, default is the subfolder
                        'rf_input_data' within the database folder
  -s START, --start=START
                        Specify the start time in the format YYYYddmmHHMM, if
                        not provided the first timestamp in the database will
                        be used
  -e END, --end=END     Specify the end time in the format YYYYddmmHHMM, if
                        not provided the last timestamp in the database will
                        be used
  -c CONFIG, --config=CONFIG
                        Path of the config file, the default will be
                        default_config.yml in the database module
  -m MODELS, --models=MODELS
                        Specify which models you want to use in the form of a
                        json line of a dict, the keys are names you give to
                        the models, the values the input features they
                        require, for example '{"RF_dualpol": ["RADAR",
                        "zh_visib_mean",
                        "zv_visib_mean","KDP_mean","RHOHV_mean","T",
                        "HEIGHT","VISIB_mean"]}', please note the double and
                        single quotes, which are requiredIMPORTANT: if no
                        model is provided only the ml input data will be
                        recomputed from the database, but no model will be
                        computedTo simplify three aliases are proposed:
                        "dualpol_default" = '{"RF_dualpol": ["RADAR",
                        "zh_visib_mean",
                        "zv_visib_mean","KDP_mean","RHOHV_mean","T",
                        "HEIGHT","VISIB_mean"]}'"vpol_default" = '{"RF_vpol":
                        ["RADAR", "zv_visib_mean","T",
                        "HEIGHT","VISIB_mean"]}'"hpol_default" = '{"RF_hpol":
                        ["RADAR", "zh_visib_mean","T",
                        "HEIGHT","VISIB_mean"]}'You can combine them for
                        example "vpol_default, hpol_default, dualpol_default,
                        will compute all three"
  -g MODELS, --generate_inputs=MODELS
                        If set to 1 (default), the input parquet files
                        (homogeneized tables) for the ml routines will be
                        recomputed from the current database rowsThis takes a
                        bit of time but is needed if you updated the database
                        and want to use the new data in the training
                        
                        

The configuration file must be written in `YAML <https://fr.wikipedia.org/wiki/YAML/>`_, the default file has the following structure:

.. code-block:: yaml

    FILTERING: # conditions to remove some observations
        STA_TO_REMOVE : ['TIT','GSB','GRH','PIL','SAE','AUB']
        CONSTRAINT_MIN_ZH : [0.5,20] # min 20 dBZ if R > 0.5 mm/h
        CONSTRAINT_MAX_ZH : [0,20] # max 20 dBZ if R = 0 mm/h
    RANDOMFORESTREGRESSOR_PARAMS: # parameters to sklearn's class
        max_depth : 20
        n_estimators : 10
    VERTAGG_PARAMS:
        BETA : -0.5 # weighting factor to use in the exponential weighting
        VISIB_WEIGHTING : 1 # whether to weigh or not observations by their visib
    BIAS_CORR : 'raw' # type of bias correction 'raw', 'cdf' or 'spline'


The parameters are the following

-   **FILTERING** : a set of parameters used to filter the input data on which the algorithm is trained

    -   **STA_TO_REMOVE** : list of problematic stations to remove
    -   **CONSTRAINT_MIN_ZH** : constraint on minimum reflectivity, the first value if the precip. intensity, the second the minimum value required value of ZH. For example for [0.5,20] all rows where ZH < 20 dBZ if R >= 0.5 mm/h will be removed. This is to reduce the effect of large spatial and temporal offset between radar and gauge.
    -   **CONSTRAINT_MAX_ZH** : constraint on maximum reflectivity, the first value if the precip. intensity, the second the minimum value required value of ZH. 
-   **RANDOMFORESTREGRESSOR_PARAMS** : set of parameters for the `sklearn random forest regressor <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html>`_ . You can add as many as you want, as long as they are valid parameters for this class

    -   **max_depth** : max depth of the threes
    -   **n_estimators** : number of trees
-   **VERTAGG_PARAMS** : set of parameters for the vertical aggregation of radar data to the ground

    -   **BETA** : the parameter used in the exponential weighting :math:`\exp(-\beta \cdot h)`, where *h* is the height of every observation. *BETA* should be negative, since lower observation should have a larger weight.
    -   **VISIB_WEIGHTING** : if set to 1, the observations will also be weighted proportionally to their visibility
-   **BIAS_CORR** : type of bias-correction to be applied *a-posteriori*. It can be either 'raw' in which case a simple linear regression of prediction vs observation is used, 'cdf' in which a simple linear regression on *sorted* prediction vs *sorted* observation is used and 'spline' which is the same as 'cdf' except that a 1D spline is used instead.



