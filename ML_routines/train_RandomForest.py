
import numpy  as np
import pandas as pd

from xgboost                  import XGBRegressor,          XGBClassifier
from sklearn.ensemble         import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection  import train_test_split, KFold, GridSearchCV


def train_RandomForest(X, y,
                       sw           = None, 
                       regression   = True,                       
                       boost        = False,
                       cv           = None,
                       param_grid   = None,
                       verbose      = False,
                       full_ret     = False,
                       ):
    """
    Train a random forest or boosted forest including k-fold cross-validation.

    :param X:           The input features (rows are datapoints, columns are features).
    :param y:           The targets.
    :param sw:          Sample weight.  
    :param regression:  If True, run a regressor, else a classifier.
    :param boost:       It True, run XGboost rather than a Random Forest.
    :param cv:          Cross-validation instance (use 5-fold if set to None).
    :param verbose:     If True, print results. Else, return only as log-file
    :param full_ret:    If True, return more than just the trained NN (cf. return arguments)

    :return RF:         trained and cross-validated RF instance
    :return grid:       GridSearchCV instance
    :return log:        log with results
    """

    # check input
    ####################################################################################################################
    assert isinstance(X, pd.DataFrame), 'X must be DataFrame'
    assert isinstance(y, pd.Series),    'y must be Series'
    assert X.shape[0]==len(y),          'X and y must be of same length '

    # initialize the training instances
    ################################################################################################################
    if not boost: # use a random forest

        RF = RandomForestRegressor if regression else RandomForestClassifier

        RF = RF(n_estimators             =  800 ,
                max_features             =  'sqrt',
                max_depth                =  14,
                min_weight_fraction_leaf =  0.05,
                n_jobs                   =  -1
                )

        if param_grid is None:  param_grid =    {
                                                'max_features':             [ 'sqrt',  0.5     ], 
                                                'max_depth':                [  2,  4,   16     ],
                                                'min_weight_fraction_leaf': [ 0.05,     0.1    ],
                                                }

        fit_args = { 'sample_weight': sw} 

    else:

        RF =  XGBRegressor if regression else XGBClassifier

        RF = RF(booster             = 'gbtree',             # what method to use
                objective           = 'reg:squarederror',   # typical regression metric
                n_estimators        =  500,                 # chose large but use early stopping
                max_depth           =  14,                  # for regression, can be deep
                learning_rate       =  0.1,                 # ignored for boosting_type='rf'
                gamma               =  0.0,                 # regularization parameter
                colsample_bytree    =  0.3,                 # features to select
                subsample           =  0.5,                 # LGBM doesn't sample with replacement
                n_jobs              =  -1,                  # use all available ressources
                verbosity           =   0,
                silent              =   True,
                )

        if param_grid is None: param_grid =  {
                                              'max_depth':          [  6, 15             ],
                                              'learning_rate':      [  0.01, 0.05        ],
                                              'gamma':              [  0.0,  0.1         ],
                                             }

        # Rather than fixing n_estimators (i.e. n_boosting_rounds) to a decently small value, we instead stop the
        # training based on an early stopping criteria.
        # Unfortunately, the GridSearchCV class cannot handle early stopping as measured on the k-folds. Therefore,
        # we need to reserve some extra data just for the early stopping.  There is, however, some code which would
        # allow us to adjust for this issue, see for instance [1,2]. But that code is not compatible with scikit-
        # learns CV-class. I thus keep things simple, and reserve a bit of data for the stopping criteria.
        # [1] https://www.kaggle.com/yantiz/xgboost-gridsearchcv-with-early-stopping-supported
        # [2] https://discuss.xgboost.ai/t/how-to-do-early-stopping-with-scikit-learns-gridsearchcv/151
        ################################################################################################################
        X, X_stop, y, y_stop = train_test_split( X, y, test_size=0.1 )

        fit_args = {"early_stopping_rounds":    6,
                    "eval_set":                 [[X_stop, y_stop]],
                    "verbose":                  False,
                    "sample_weight":            sw if sw is None else sw.reindex( X.index ), 
                    }


    # cross-validation
    ####################################################################################################################
    scoring  = 'neg_mean_squared_error' if regression else 'f1'
    cv       = KFold( n_splits=5, shuffle=True ) if cv is None else cv
    grid     = GridSearchCV(RF,
                            param_grid   =  param_grid,
                            scoring      =  scoring,
                            cv           =  cv,
                            refit        =  True,
                            n_jobs       =  -1, 
                            verbose      =  verbose,
                            error_score  = 'raise',
                            )

    _         =  grid.fit( X, y, **fit_args  )
    RF        =  grid.best_estimator_

    # print status
    ####################################################################################################################
    log       = f'selected parameters:\n{grid.best_params_}'
    scores    =  grid.cv_results_['mean_test_score']
    args      =  np.argsort(scores)
    params    =  np.array(grid.cv_results_['params'])[args]
    means     =  np.array(grid.cv_results_['mean_test_score'])[args]
    stds      =  np.array(grid.cv_results_['std_test_score'])[args]

    log      += '\n\nscore per combination:\n'

    for mean, std, params in zip(means, stds, params): log += "\n%0.3f (+/-%0.03f) for %r" % (mean, std, params)

    if verbose: print(log)

    if full_ret: return RF, grid, log
    else:        return RF

