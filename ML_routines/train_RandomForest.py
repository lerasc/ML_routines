
import numpy  as np
import pandas as pd

from multiprocessing          import cpu_count
from sklearn.ensemble         import RandomForestRegressor,     RandomForestClassifier
from sklearn.model_selection  import KFold, GridSearchCV
from xgboost                  import XGBRegressor, XGBClassifier

def train_RandomForest(X, y,
                       sw           = None, 
                       regression   = True,                       
                       boost        = False,
                       cv           = None,
                       param_grid   = None,
                       min_data     = None,
                       verbose      = False,
                       full_ret     = False,
                       **kwargs, 
                       ):
    """
    Train a random forest or boosted forest including k-fold cross-validation.

    :param X:           The input features (rows are datapoints, columns are features).
    :param y:           The targets (must have same index as X)
    :param sw:          Sample weight.  
    :param regression:  If True, run a regressor, else a classifier.
    :param boost:       It True, run XGboost rather than a Random Forest.
    :param cv:          Cross-validation instance (use 5-fold if set to None).
    :param param_grid:  Parameter combinations to test (use default ones if None).
    :param min_data:    Minimum number of data points in any terminal leaf. Useful for smart min_weight_fraction_leaf
                        (see inside code for details). If not None, it overwrite the param_grid values.
    :param verbose:     If True, print results. Else, return only as log-file
    :param full_ret:    If True, return more than just the trained NN (cf. return arguments)
    :param kwargs:      Additional arguments for the RF.

    :return RF:         trained and cross-validated RF instance
    :return grid:       GridSearchCV instance
    :return log:        log with results
    """

    # check input
    ####################################################################################################################
    assert isinstance(X, pd.DataFrame), 'X must be DataFrame'
    assert isinstance(y, pd.Series),    'y must be Series'
    assert X.index.identical(y.index),  'X and y must have the same index'

    # initialize the training instances
    ################################################################################################################
    if not boost:                                                                   # use a random forest

        # Set hyper-parameters
        grid = param_grid if param_grid is not None else {} # initialize as empty

        if min_data is not None: grid['min_weight_fraction_leaf'] =  dynamic_min_weight_frac( X, min_data )
        elif param_grid is None: grid['min_weight_fraction_leaf'] = [ 0.10, 0.05, 0.01, 0.001, 0.0001 ]

        ML = RandomForestRegressor if regression else RandomForestClassifier
        ML = ML(
                 n_estimators             =  1000,                                 # just pick it large enough
                 max_features             = 'sqrt',
                 min_weight_fraction_leaf =  0.01,
                 n_jobs                   =  -1, 
                 **kwargs, 
                 )

    else:                                                                           # use a booster

        ML = XGBRegressor if regression else XGBClassifier
        ML = ML( booster          = 'gbtree',
                 objective        = 'reg:squarederror' if regression else 'binary:logistic',
                 slient           =  0,
                 nthread          =  cpu_count()-2,
                 subsample        =  0.75,
                 colsample_bytree =  0.5,
                 **kwargs,
                 )

        if param_grid is None:

            grid = {  "learning_rate":     [  0.01, 0.05, 0.10   ],
                      "max_depth":         [  2,    4,    6      ],
                      "n_estimators":      [  50,   100,  200    ],
                       }

    # cross-validation: For regression, we use correlation as a metric of success
    ####################################################################################################################
    scoring  = 'neg_mean_absolute_error' if regression else 'f1'
    cv       =  KFold( n_splits=5, shuffle=True ) if cv is None else cv

    grid     =  GridSearchCV(ML,
                             param_grid   =  grid,
                             scoring      =  scoring,
                             cv           =  cv,
                             refit        =  True,
                             n_jobs       =  -1, 
                             verbose      =  verbose,
                             error_score  = 'raise',
                             )

    _         =  grid.fit( X, y, sample_weight=sw  )
    ML        =  grid.best_estimator_

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

    if full_ret: return ML, grid, log
    else:        return ML


def dynamic_min_weight_frac( X, min_data=100  ):
    """
    In many regards, min_weight_fraction_leaf is one of the most important parameters, as it allows us to control the
    tree-depth in a very interpretable manner. However, often it is easier to interpret in absolute, rather than in
    relative numbers. Here we thus translate the absolute minimum number of data points in any terminal leaf into a
    fraction and then span a reasonable grid of values, and rerun a log-spaced grid of frac-values.

    :param X:           Feature DataFrame with training data
    :param min_data:    Minimum number of data-points to be held in any terminal leaf.
    """

    min_frac = min_data / len(X)                          # minimum fraction of data
    max_frac = 0.2                                        # maximum fraction of data to test for
    fracs    = np.logspace( np.log10(min_frac),           # log-spaced grid of values
                            np.log10(max_frac),
                            num  = 5,
                            base = 10,
                            )
    fracs    = np.round( fracs, 5 )                       # just for more visual appeal

    return fracs


def analyze_RF_overfit( X_IS, y_IS, X_OS, y_OS, **kwargs ):
    """
    Given training data (X_IS) and targets (y_IS), as well as OS data (X_OS, y_OS), train a random forst regressor and
    analyze the dependence of tree complexity on the IS vs. OS error.
    """

    # set parameters
    ####################################################################################################################
    fracs = dynamic_min_weight_frac( X_IS, min_data=1 )
    errs  = []

    # For each tree depth, train the random forest. We use k-fold so that we can access the CV error, even though we do
    # not actually tune the parameters with k-fold.
    ####################################################################################################################
    for frac in fracs:

        ML, grid, log = train_RandomForest(X            = X_IS,
                                           y            = y_IS,
                                           param_grid   = {'min_weight_fraction_leaf':[frac]}, # just one parameter
                                           regression   = True, 
                                           verbose      = False,
                                           full_ret     = True,
                                           **kwargs,
                                           )

        IS_pred   = pd.Series( ML.predict( X_IS ), index=X_IS.index ) 
        OS_pred   = pd.Series( ML.predict( X_OS ), index=X_OS.index )
        IS_err    = 100 * y_IS.corr( IS_pred )
        OS_err    = 100 * y_OS.corr( OS_pred )
        CV_err    = grid.cv_results_['mean_test_score'][0]

        errs     += [ (IS_err, CV_err, OS_err) ]

    errs = pd.DataFrame( errs, index=fracs, columns=['IS','CV','OS'] )

    return errs