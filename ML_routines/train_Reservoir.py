
import pandas      as pd
import reservoirpy as rpy

from reservoirpy.nodes              import NVAR, Ridge
from sklearn.model_selection        import train_test_split

from ML_routines.ML_routines        import subsample_score
from ML_routines.execution_routines import form_all_combinations

def train_Reservoir(X, y,
                    frac         = 0.1,
                    param_grid   = None,
                    verbose      = False,
                    full_ret     = False,
                    ):
    """
    Train a reservoir echo state network. But rather than actually training an echo state, we train a Non-linear Vector
    AutoRegressive machine (NVAR) as presented in [1]. As argued in [1], NVAR work better than traditional reservoir
    computers. But moreover, it requires many less hyper-parameters to tune.

    :param X:           The input features (rows are datapoints, columns are features).
    :param y:           The targets.
    :param frac:        Fraction of test data.
    :param param_grid:  Parameter combinations to test (use default ones if None).
    :param verbose:     If True, print results. Else, return only as log-file
    :param full_ret:    If True, return not just the instance, but also

    :return ESN:        trained echo state network, with predict method
    :return esn_model:  original reservoirpy instance without predict method (if full_ret=True)
    :return res:        DataFrame with results for each parameter combination (if full_ret=True)

    references:
    ----------
    [1] Gauthier, D. J., Bollt, E., Griffith, A., & Barbosa, W. A. S. (2021). 
        Next generation reservoir computing. Nature Communications, 12(1), 5564.
    """

    # check input
    ####################################################################################################################
    assert isinstance(X, pd.DataFrame), 'X must be DataFrame'
    assert isinstance(y, pd.Series),    'y must be Series'
    assert X.shape[0]==len(y),          'X and y must be of same length '
    assert frac > 0 and frac < 1,       'frac must be between 0 and 1'

    # implement reservoir routine
    ####################################################################################################################
    def initialize_reservoir( delay=2, ridge=1e-2  ):
        """
        Initialize NVAR with main tunable parameters.
        """

        nvar      = NVAR( delay=delay, order=2, strides=1 )
        readout   = Ridge( output_dim=1, ridge=ridge )
        esn_model = nvar >>  readout

        return esn_model

    # form all parameter combinations and train-test split
    ####################################################################################################################
    if param_grid is None:param_grid = {
                                        'delay' :   [ 2,    4,    6     ],
                                        'ridge' :   [ 1e-5, 1e-3, 1e-1  ],
                                        }

    param_grid                       = form_all_combinations(param_grid)
    X_train, X_test, y_train, y_test = train_test_split( X, y,  test_size=frac, shuffle=False )

    # iterate each combination and determine score
    ####################################################################################################################
    _   = rpy.verbosity(0)                              # else prints training status
    res = []                                            # store results per  combination

    for i, params in enumerate(param_grid):

        if verbose: print(f'training combination {i+1} out of {len(param_grid)}', end='\r')

        esn_model = initialize_reservoir( delay     = params['delay'],
                                          ridge     = params['ridge'],
                                          )

        esn_model = esn_model.fit( X      = X_train.values,
                                   Y      = y_train.values.reshape(-1,1),
                                   warmup = 200,
                                  )

        IS_pred    = esn_model.run( X_train.values )
        IS_pred    = pd.Series( IS_pred.squeeze(), index=X_train.index )
        IS_m, IS_e = subsample_score( y_train, IS_pred, score='rmse' )

        OS_pred    = esn_model.run(X_test.values, )
        OS_pred    = pd.Series( OS_pred.squeeze(), index=X_test.index )
        OS_m, OS_e = subsample_score( y_test, OS_pred, score='rmse' )

        res       += [ (IS_m, IS_e, OS_m, OS_e) ]

    # form into a DataFrame and determine the best score
    ####################################################################################################################
    cols   = ['IS_rmse', 'IS_std', 'OS_rmse', 'OS_std']         # column names
    res    = pd.DataFrame(res, columns=cols)                    # merge into a Frame
    params = pd.DataFrame( param_grid )                         # parameters as frame
    res    = pd.concat([params, res], axis=1)                   # merge together
    best   = res['OS_rmse'].idxmin()                            # index of best parameter combination
    res    = res.sort_values( by='OS_rmse', ascending=True )    # lowest OS rmse first

    # retrain on best parameter combination
    ####################################################################################################################
    best_params = param_grid[best] # works because res DataFrame has same index as .iloc

    esn_model   = initialize_reservoir(delay     = best_params['delay'],
                                       ridge     = best_params['ridge'],
                                       )

    esn_model   = esn_model.fit( X      = X.values,
                                 Y      = y.values.reshape(-1,1),
                                 warmup = 10,
                                 )

    # return desired output
    ####################################################################################################################
    class esn_wrapper:
        """
        Simple wrapper around esn model to add .predict-method for consisten usage with other scikit-learn methods.
        """
        def __init__(self,  esn_model ):
            self.__esn = esn_model

        def predict( self, X ):
            """
            Predict data for feature values X.
            """
            pred = self.__esn.run( X.values )
            pred = pred.squeeze()

            return pred

    ESN = esn_wrapper( esn_model )

    if full_ret: return ESN, esn_model, res
    else:        return ESN