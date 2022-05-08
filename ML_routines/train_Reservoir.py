
import pandas      as pd
import reservoirpy as rpy

from reservoirpy.nodes              import Reservoir, Ridge
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
    Train a reservoir echo state network.

    :param X:           The input features (rows are datapoints, columns are features).
    :param y:           The targets.
    :param frac:        Fraction of test data.
    :param param_grid:  Parameter combinations to test (use default ones if None).
    :param verbose:     If True, print results. Else, return only as log-file

    :return ESN:        trained echo state network, with predict method.
    :return esn_model:  original reservoirpy instance without predict method
    :return res:        DataFrame with results for each parameter combination
    """

    # check input
    ####################################################################################################################
    assert isinstance(X, pd.DataFrame), 'X must be DataFrame'
    assert isinstance(y, pd.Series),    'y must be Series'
    assert X.shape[0]==len(y),          'X and y must be of same length '
    assert frac > 0 and frac < 1,       'frac must be between 0 and 1'

    # implement reservoir routine
    ####################################################################################################################
    def initialize_reservoir( units=50, lr=0.1, sr=0.5, ridge=1e-2, feedback=True ):

        reservoir = Reservoir( units=params['units'], lr=params['lr'], sr=params['sr'] )
        readout   = Ridge( output_dim=1, ridge=params['ridge'] )

        if feedback: reservoir <<= readout # feedback connection

        esn_model = reservoir >>  readout

        return esn_model

    # form all parameter combinations and train-test split
    ####################################################################################################################
    if param_grid is None:

        param_grid = {
                    'units' :   [ 50, 100, 200, 400 ],
                    'lr'    :   [ 0.1, 0.3, 0.5     ],
                    'sr'    :   [ 0.25, 0.9, 1.1, 2 ],
                    'ridge' :   [ 1e-3, 1e-2, 1e-1  ],
                    'feedback': [True, False]
                    }

        param_grid = form_all_combinations(param_grid)

    X_train, X_test, y_train, y_test = train_test_split( X, y,  test_size=frac, shuffle=False )

    # iterate each combination and determine score
    ####################################################################################################################
    _   = rpy.verbosity(0)                              # else prints training status
    res = []                                            # store results per  combination

    for i, params in enumerate(param_grid):

        if verbose: print(f'training combination {i+1} out of {len(param_grid)}', end='\r')

        esn_model = initialize_reservoir( units     = params['units'],
                                          lr        = params['lr'],
                                          sr        = params['sr'],
                                          ridge     = params['ridge'],
                                          feedback  = params['feedback']
                                          )

        esn_model = esn_model.fit( X      = X_train.values,
                                   Y      = y_train.values.reshape(-1,1),
                                   warmup = 10,
                                  )

        IS_pred    = esn_model.run(X_train.values, )
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
    best   = res['OS_rmse'].idxmax()                            # index of best parameter combination
    res    = res.sort_values( by='OS_rmse', ascending=True )    # lowest OS rmse first

    # retrain on best parameter combination
    ####################################################################################################################
    best_params = param_grid[best] # works because res DataFrame has same index as .iloc

    esn_model   = initialize_reservoir( units   = best_params['units'],
                                      lr        = best_params['lr'],
                                      sr        = best_params['sr'],
                                      ridge     = best_params['ridge'],
                                      feedback  = best_params['feedback']
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
            pred = esn_model.run( X.values )
            pred = pred.squeeze()

            return pred

    ESN = esn_wrapper( esn_model )

    if full_ret: return ESN, esn_model, res
    else:        return ESN