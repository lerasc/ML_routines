
import os
import logging

import numpy  as np
import pandas as pd

from keras                       import layers
from keras.models                import Sequential
from keras.callbacks             import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

from sklearn.model_selection     import KFold, GridSearchCV, train_test_split

import tensorflow as tf
from tensorflow.python.util import deprecation

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)

# remove the extensive pytorch warnings
logging.getLogger("pytorch_transformers.tokenization_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)


def tensorflow_shutup( ):
    """
    Make Tensorflow (which underlies keras) less verbose. Code is obtained from [1].
    [1] https://www.codegrepper.com/code-examples/python/tensorflow+disable+warnings
    """

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # noinspection PyPackageRequirements

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Monkey patching deprecation utils to shut it up! Maybe good idea to disable this once after upgrade
    # noinspection PyUnusedLocal
    def deprecated(date, instructions, warn_once=True):  # pylint: disable=unused-argument
        def deprecated_wrapper(func):
            return func
        return deprecated_wrapper

    deprecation.deprecated = deprecated


def train_NeuralNet( X, y,
                     architecture = 'FFNN',
                     regression   =  True,
                     cv           =  None,
                     param_grid   =  None,
                     verbose      =  True,
                     full_ret     =  False
                     ):
    """
    Generate, train and return a basic neural net. Here, 'basic' is meant in two senses. First, the models are basic
    because they rely on mostly default parameters and simple architectures. Second, the models are basic because the
    input features X are expected to be one dimensional.

    :param X:               The input features (rows are datapoints, columns are features).
    :param y:               The targets.
    :param architecture:    What neural net architecture to use:
                            - 'LSTM':   train a basic LSTM network
                            - 'CONV':   train a basic convolutional layer
                            - 'FFNN':   train a basic feed forward neural net
    :param regression:      If True, train a regression, else train a binary classifier.
    :param cv:              Cross-validation instance (use 5-fold if set to None).
    :param param_grid:      Parameter combinations to test (use default ones if None).
    :param verbose:         If True, print status progress.
    :param full_ret:        If True, return more than just the trained NN (cf. return arguments)

    :return NN:             trained and cross-validated NN instance
    :return history:        NN training history
    :return grid:           GridSearchCV instance
    :return log:            string with info about training
    """


    # check input
    ####################################################################################################################
    assert isinstance(X, pd.DataFrame), 'X must be DataFrame'
    assert isinstance(y, pd.Series),    'y must be Series'
    assert X.shape[0]==len(y),          'X and y must be of same length '

    tensorflow_shutup()

    # check input data
    ####################################################################################################################
    assert X.shape[0]==len(y),                       'shape of X and y not matching'

    nfeat = X.shape[1] # number of features

    # adjust network topology to architecture specifics
    ####################################################################################################################
    def build_NN( units=100, rate=0.2 ):
        """
        Build the neural net.

        :param units:   number of units (exact interpretation depends on the architecture, see there)
        :param rate:    dropout rate
        """

        model  = Sequential() # initialize the model

        # adjust network topology to architecture specifics
        ################################################################################################################
        if architecture=='LSTM':

            model.add( layers.LSTM(units        =  units,
                                   dropout      =  rate,
                                   input_shape  = (nfeat, nfeat),
                                   ))

        elif architecture== 'CONV':

            model.add( layers.Conv1D( filters      =  units,
                                      kernel_size  =  4,
                                      padding      = 'same',
                                      activation   = 'relu',
                                      input_shape  =  (nfeat, nfeat),
                                      ))

            model.add( layers.MaxPooling1D(pool_size=3) )
            model.add( layers.Dropout(0.1) )

            model.add( layers.Conv1D( filters      =  units//2,
                                      kernel_size  =  4,
                                      padding      = 'same',
                                      activation   = 'relu',
                                      input_shape  =  (nfeat, nfeat),
                                      ))

            model.add( layers.MaxPooling1D(pool_size=3) )
            model.add( layers.Dropout(rate) )

            model.add( layers.Flatten() )
            model.add( layers.Dropout(rate) )

        elif architecture== 'FFNN':

            model.add( layers.BatchNormalization()  )
            model.add( layers.Dense(   units=units,    activation='relu', input_shape=(nfeat,) ))            
            model.add( layers.BatchNormalization()  )
            model.add( layers.Dropout( rate=rate  ) )

            model.add( layers.Dense(   units=units//2, activation='relu', ))
            model.add( layers.BatchNormalization()  )
            model.add( layers.Dropout( rate=rate  ) )

            model.add( layers.Dense(   units=units//4, activation='relu', ))
            model.add( layers.BatchNormalization() ) 

        else:

            raise ValueError("invalid architecture argument: %s" % architecture)

        activation = 'linear' if regression else 'sigmoid'
        model.add( layers.Dense(1, activation=activation)) # final contraction layer

        # add details on objective function and fitting method
        #########################################################################
        model.compile(
                      optimizer=   'adam',
                      loss     =   'mse'      if regression else 'binary_crossentropy',
                      metrics  =  ['mse']     if regression else ['accuracy'],
                     )

        return model


    # We stop training once we don't improve on cross-validation data. Since we already use cross-validation data below
    # to test different architectures, here we reserve some 'stopping data' that we use for early stopping.
    ####################################################################################################################
    X, X_stop, y, y_stop = train_test_split( X, y, test_size=0.1 )

    es  = EarlyStopping(monitor              = 'val_mse' if regression else 'val_loss',
                        patience             =  5,
                        min_delta            =  1e-4,
                        restore_best_weights =  True,
                        verbose              =  True,
                        )

    fit_args =  {
                'epochs'          : 100, # just make it large enough, since we use early stopping
                'batch_size'      : 40,
                'validation_data' : (X_stop, y_stop),
                'callbacks'       : [es],
                }

    # We tune the meta-parameters:
    ####################################################################################################################
    if param_grid is None:

        param_grid = {
                      'units': [ 50,  100,  200  ],
                      'rate':  [ 0.0, 0.1,   0.4 ],
                     }

    wrap  = KerasRegressor if regression else KerasClassifier
    model = wrap(   build_fn   = build_NN,
                    verbose    = False,
                    **fit_args, # call-back
                   )

    grid     = GridSearchCV(   estimator    =  model,
                               param_grid   =  param_grid,
                               scoring      =  'neg_mean_squared_error' if regression else 'f1',
                               cv           =  KFold(n_splits=5) if cv is None else cv,
                               refit        =  False, # we fit again below, to get the history
                               n_jobs       =  -1, 
                               verbose      =  5 if verbose else 0,                               
                               error_score  = 'raise',
                             )

    _     = grid.fit( X, y, **fit_args )
    NN    = build_NN( **grid.best_params_ )

    # print results
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

    # fit again on all data (rather than using refit in Crossvalidation above, since we want the history)
    ####################################################################################################################
    history = NN.fit(   x               =  X,
                        y               =  y,
                        verbose         =  False,
                         **fit_args,
                       )

    if verbose:
        print('model summary:')
        print(NN.summary())


    if full_ret: return NN, history, grid, log 
    else:        return NN

