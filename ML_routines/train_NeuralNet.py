
import os
import logging

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


def train_basic_NN( X, y, architecture='FFNN', regression=True ):
    """
    Generate, train and return a basic neural net. Here, 'basic' is meant in two senses. First, the models are basic
    because they rely on mostly default parameters and simple architectures. Second, the models are basic because the
    input features X are expected to be one dimensional.

    :param X:               DataFrame with training data.

    :param y:               Pandas Series with target data (must have same index as X)

    :param architecture:    What neural net architecture to use:
                            - 'LSTM':   train a basic LSTM network
                            - 'CONV':   train a basic convolutional layer
                            - 'FFNN':   train a basic feed forward neural net

    :param regression:      If True, train a regression, else train a binary classifier.
    """

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

            model.add( layers.BatchNormalization() ) 
            model.add( layers.Dense(   units=units,    activation='relu', input_shape=(nfeat,) ))            
            model.add( layers.BatchNormalization() ) 
            model.add( layers.Dropout( rate=rate  ))            

            model.add( layers.Dense(   units=units//2, activation='relu', ))
            model.add( layers.BatchNormalization() ) 
            model.add( layers.Dropout( rate=rate  ))            

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

    # We tune the meta-parameters:
    ####################################################################################################################
    param_grid = {
                  'units': [ 50,  100,  200  ],
                  'rate':  [ 0.0, 0.1,   0.2 ],
                 }

    wrap  = KerasRegressor if regression else KerasClassifier
    model = wrap(   build_fn   = build_NN,
                    epochs     = 50,
                    batch_size = 40,
                    verbose    = False,
                   )

    grid = GridSearchCV(   estimator    =  model,
                           param_grid   =  param_grid,
                           scoring      = 'neg_root_mean_squared_error',
                           cv           =  KFold(n_splits=5),
                           verbose      =  1,
                           error_score  = 'raise',
                         )

    grid_result   = grid.fit( X, y )

    print('best parameter configuration: %s'%grid_result.best_params_)
    print('best test accuracy: %.2f'%grid_result.best_score_)

    params      = grid_result.best_params_
    best_model  = build_NN( **params )
    print(best_model.summary())

    # set the best parameters and train until we do no longer improve on the cross validation set:
    ####################################################################################################################
    X_train, X_val, y_train, y_val = train_test_split(  X, y,
                                                        stratify=None if regression else y,
                                                        test_size=0.15
                                                     )

    es          = EarlyStopping(monitor              = 'val_mse' if regression else 'val_loss',
                                patience             =  5,
                                min_delta            =  1e-4,
                                restore_best_weights =  True,
                                verbose              =  True,
                                )

    history     = best_model.fit(   x               =  X_train,
                                    y               =  y_train,
                                    validation_data = (X_val, y_val),
                                    epochs          =  120,  # large enough value
                                    batch_size      =  64,
                                    verbose         =  False,
                                    callbacks       = [es],
                                   )

    return best_model, history, grid_result

