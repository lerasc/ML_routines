
import numpy   as np 
import pandas  as pd
import seaborn as sb

from sklearn.metrics  import accuracy_score, f1_score, confusion_matrix, roc_curve, roc_auc_score


def plot_confusion_matrix( y_true, y_pred, ax, labels=None ):
    """
    Plot a binary confusion matrix. 
    This function is an adaptation from [1]. It is currently restricted to binary classification.

    :param y_true:  list of actual labels
    :param y_pred:  list of predicted labels
    :param ax:      axis instance to plot into 
    :param labels:  labels argument is passed on to confusion matrix


    [1] http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-
        examples-model-selection-plot-confusion-matrix-py
    """

    acc          = accuracy_score( y_true, y_pred )                                  # calculate accuracy (for title)
    acc          = int(100*acc)                                                      # rescale
    cm_count     = confusion_matrix(y_true,y_pred,labels=labels, normalize=None)     # count entries 
    cm           = confusion_matrix(y_true,y_pred,labels=labels, normalize='all')    # normalized cm
    cm           = pd.DataFrame(cm, index=labels, columns=labels)                    # adjust labels                
    cm, cm_count = cm.iloc[::-1], cm_count[::-1]                                     # flip to custom form

    annot = np.zeros(cm.shape, dtype=object)                        # stores nice annotation of coeff (p-value)
    for i in range(cm.shape[0]):                                    # iterate each row
        for j in range(cm.shape[1]):                                # iterate each column
            a, p       = cm.values[i,j], cm_count[i,j]              # extract regress coef & associated p-value
            annot[i,j] = f'{a:.3f}\n{int(p)}'                       # format nicely    

    sb.heatmap(data        =  cm,
               cmap        =  'Blues',
               fmt         =   '', # needed for annot below
               annot       =  annot,
               robust      =  True,
               square      =  True,
               linewidths  =  1,
               cbar        =  False,
               ax          =  ax,
              )

    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    ax.set_title('accuracy =%.2f%%'%acc)


def plot_ROC_curve( y_true, y_pred, ax, **kwargs ):
    """
    Plot the ROC curve into axis instance ax (kwargs are for the plotting).
    """

    FP, TP, _ = roc_curve(      y_true, y_pred )
    roc       = roc_auc_score(  y_true, y_pred )
    acc       = accuracy_score( y_true, y_pred )
    f1        = f1_score(       y_true, y_pred )

    roc   = int(100*roc)
    acc   = int(100*acc)
    f1    = int(100*f1)
    score = f'ROC score={roc:.0f}%\naccuracy={acc:.0f}%\nF1={f1:.0f}%'

    ax.plot([0, 1], [0, 1], 'k--')
    ax.plot( FP, TP, **kwargs )
    ax.text( x=0.1, y=0.8, s=score, fontsize=10 )    

    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')    


def balanced_downsample( X, target='target', classification=True, center=0, nr_bins=25  ):
    """
    For many ML tasks, it is important to have a balanced set of targets. For classification, this means the same number
    of data-points in each class. For regression, this generalizes to having the same number of datapoints in each
    positive and negative quantile-bin. This function balances data accordingly.

    Given a feature frame X, downsample all targets to the minority class.

    :param X:               input DataFrame with features

    :param target:          name of the target column

    :param classification:  If True,  the target column is assumed to be an (ordinal) set of targets for classfication.
                            If False, the target column is assumed to be a continous set of targets for regression.

    :param center:          This argument is only needed if classification=False. It specifies the center-point of the
                            distribution around which the data is to be distributed. The function then adds the same
                            number of data-points to the quantile-bins on each side of the center. So for instance,
                            if center=0, there will be the same amount of data in the interval (-1,0) and the interval
                            (0,1). The same amount of data in the interval (-2,-1) as in (1,2), and so forth.

    :param nr_bins:         Specifies the number of quantile bins (only needed if classification=False).
    """

    # check input
    ####################################################################################################################
    assert isinstance(X, pd.DataFrame), 'X must be DataFrame'
    assert target in X.columns, f'target column {target} is missing'
    assert isinstance(classification, bool), 'classification must be bool '
    if not classification: assert not X.index.duplicated().any(), 'for regression, index must be unique'

    # balancing classification labels is straightforward: downsample all classes to the one with least data
    ####################################################################################################################
    if classification:

        counts = X[target].value_counts()               # count all target values
        n      = counts.min()                           # target value with least data
        Xs     = []                                     # stores sub-frames of each target value

        for _, df in X.groupby(target):                 # group by target

            sdf  = df.sample( n=n, replace=False )      # downsample to the minority class
            Xs  += [sdf]                                # collect

        X = pd.concat(Xs, axis='index')                 # concatenate
        X = X.sample( frac=1, replace=False )           # shuffle

        return X

    # Balancing regression targets is slightly more involved. Here, we consider the target as distributed around a
    # center. Then, we make sure that the maximum and minimum value have the same distance from the center by clipping
    # the largest values. Subsequently, we look at quantile bins and make sure that in each bin, there is the same
    # number of data-points with positive and negative sign. We neatly make use of a recursive call to this function.
    ####################################################################################################################
    nX              = X.copy(deep=True)                                 # don't change original DataFrame
    nX[target]      = nX[target] - center                               # remove mean to balance at 0
    Y               = nX[target].copy()                                 # create new Series
    Min, Max        = Y.min(), Y.max()                                  # largest deviations from center point

    assert Min < 0 and Max > 0, 'center must be between largest and smallest data-point'

    clip            = min( abs(Min), Max )                              # where to clip the largest values
    Y               = Y.clip(-clip, clip)                               # now max deviation from center is equal

    Y               = Y.to_frame()                                      # make into DataFrame
    Y['abs_target'] =          Y['target'].abs()                        # take absolute values
    Y['sgn_target'] = np.sign( Y['target'] )                            # sign: deviation from center
    Y['bin']        = pd.cut(  Y['abs_target'], bins=nr_bins  )         # assign to quantile-bins
    new_X           = []                                                # stores the new values

    for _, sY in Y.groupby('bin'):                                      # iterate targets by bin

        nv =  sY['sgn_target'].value_counts()                           # number of values per sign
        assert len(nv) > 0, 'nr_bins must be reduced'                   # bin contains only negative/positive values

        sY = balanced_downsample( X               = sY,                 # recursive call
                                  target         ='sgn_target',         # balance the sign in each bin
                                  classification = True,                # treat as discrete
                                  )

        sX     = nX.reindex(sY.index)                                   # select related X data (index must be unique!)
        new_X += [ sX ]                                                 # append to list

    X = pd.concat(new_X, axis='index')                                  # stack back together

    return X
    