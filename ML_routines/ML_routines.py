
import numpy   as np 
import pandas  as pd
import seaborn as sb

from sklearn.metrics  import accuracy_score, f1_score, confusion_matrix, roc_curve, roc_auc_score


def plot_confusion_matrix( y_true, y_pred, ax, labels=None, **kwargs ):
    """
    Plot a binary confusion matrix. 
    This function is an adaptation from [1]. It is currently restricted to binary classification.

    :param y_true:  list of actual labels
    :param y_pred:  list of predicted labels
    :param ax:      axis instance to plot into 
    :param labels:  labels argument is passed on to confusion matrix
    :param kwargs:  additional arguments for confusion_matrix


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


def balanced_downsample( X, target='target' ):
    """
    Given a feature frame X, downsample all targets to the minority class.

    :param X:           input DataFrame with features
    :param target:      name of the two target column
    """

    assert target in X.columns, f'target column {target} is missing'

    counts = X[target].value_counts()               # count all target values
    n      = counts.min()                           # target value with least data
    Xs     = []                                     # stores sub-frames of each target value

    for _, df in X.groupby(target):                 # group by target

        sdf  = df.sample( n=n, replace=False )      # downsample to the minority class
        Xs  += [sdf]                                # collect

    X = pd.concat(Xs, axis='index')                 # concatenate
    X = X.sample( frac=1, replace=False )           # shuffle

    return X

    