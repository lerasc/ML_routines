
import numpy   as np 
import pandas  as pd
import seaborn as sb

from matplotlib import pyplot as plt

from sklearn.metrics  import accuracy_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, mean_squared_error


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


def subsample_score( y_true, y_pred, score='rmse', ns=100  ):
    """
    Rather than returning one score, return a sub-sampled average score and associated standard deviation.

    :param y_true:  Pandas Series of targets.
    :param y_pred:  Pandas Series of predicted values
    :param score:       What score to use. Default is 'rmse' but can by any score function compatible with scikit-learn.
    :param ns:      Number of times to sub-sample.
    """

    if not callable(score):
        assert score=='rmse', "if score is not a function, it must be 'rmse'"
        score = lambda true, pred: np.sqrt( mean_squared_error(true,pred) )

    true    = y_true.rename('true')
    pred    = y_pred.rename('pred')
    df      = pd.concat([true,pred], axis=1)
    samples = [ df.sample(frac=0.5, replace=True).dropna() for _ in range(ns) ]
    vals    = [ score(sdf['true'], sdf['pred']) for sdf in samples ]
    mean    = np.nanmean(vals)
    std     = np.nanstd(vals)

    return mean, std


def plot_performance_by_bin( y_true, y_pred, score='rmse', bin_by='true', bin_style='quantile', ax=None, **kwargs ):
    """
    Plot the prediction performance sorted by bins.

    :param y_true:      Pandas Series of targets.
    :param y_pred:      Pandas Series of predicted values
    :param score:       What score to use. Default is 'rmse' but can by any score function compatible with scikit-learn.
    :param bin_by:      Whether to bin by the 'true' values or the 'predicted' ones.
    :param bin_style:   Whether to use 'quantile' bins or 'size' bins that are spaced equally around 0.
                        Alternatively, a list of bin boundaries may be provided.
    :param ax:          Plotting instance to plot into
    :param kwargs:      Additional arguments for seaborn's barplot
    """

    # check that input is provided in correct format
    ####################################################################################################################
    assert bin_by in ['true','predicted'], f'invalid bin_by argument {bin_by}'

    if not callable(score):
        assert score=='rmse', "if score is not a function, it must be 'rmse'"
        score = lambda true, pred: np.sqrt( mean_squared_error(true,pred) )

    if ax is None: _, ax = plt.subplots(figsize=(12,6))

    # create DataFrame and calculate total score
    ####################################################################################################################
    data          = pd.concat( [y_true, y_pred], axis=1 )               # merge together
    data.columns  = ['true','predicted']                                # apropriate name
    data          = data.dropna()                                       # remove missing data
    score_func    = lambda x: score( x['true'], x['predicted'] )        # function of one argument
    tot           = score_func( data )                                  # overall score

    # creat the bins
    ####################################################################################################################
    if bin_style=='size': # create same number of equal size bins to the left and right of 0

        left_bins     = np.linspace( data[bin_by].min(),   0,                     6 )
        right_bins    = np.linspace( 0,                       data[bin_by].max(), 6 )
        bins          = list(left_bins) + list(right_bins[1:])
        data['bin']   = pd.cut( data[bin_by], bins=bins )

    elif bin_style=='quantile': # create bins such the same number of data is in each bin

        data['bin']   = pd.qcut( data[bin_by], q=9 )

    else:

        assert isinstance( bin_style, list ), 'bin boundaries must be list'

        data['bin']   = pd.cut( data[bin_by], bins=bin_style )

    # calculate the score on each bin
    ####################################################################################################################
    perf         = data.groupby('bin').apply( score_func ).rename('score')                         # score per bin
    perf.index   = [f'[{np.round(ind.left, 2)}, {np.round(ind.right, 2)}]' for ind in perf.index ] # nice names
    perf         = perf.reset_index().rename({'index':'bin'}, axis=1)                              # rename

    counts       = data.groupby('bin').apply( lambda x: f'{len(x):,}' )       # number of data points per bin
    counts       = counts.values

    _       = sb.barplot(  x          =  'bin',                               # plot the bins
                           y          =  'score',
                           data       =   perf,
                           capsize    =   0.2,
                           fill       =   False,
                           ax         =   ax,
                           **kwargs,
                           )

    _       = ax.bar_label( container   =  ax.containers[-1],                 # annotate nr of data per bin
                            labels      =  counts,
                            label_type  = 'edge',
                            fontsize    =  11,
                            rotation    =  0,
                            )

    _       = ax.set_xticks(range(len(perf)), perf['bin'], rotation=90 )      # annotate bins
    _       = ax.set_ylabel('score', fontsize=12)
    _       = ax.set_xlabel('')
    _       = ax.set_title(f'total score: {tot:.2f}', fontsize=12)


def balanced_downsample( X, target='target', classification=True, center=0, nr_bins=25, upsample=False ):
    """
    For many ML tasks, it is important to have a balanced set of targets. For classification, this means the same number
    of data-points in each class. For regression, this generalizes to having the same number of datapoints in each
    positive and negative quantile-bin. This function balances data accordingly.

    Given a feature frame X, downsample all targets to the minority class.

    :param X:               input DataFrame with features

    :param target:          name of the target column
`
    :param classification:  If True,  the target column is assumed to be an (ordinal) set of targets for classfication.
                            If False, the target column is assumed to be a continous set of targets for regression.

    :param center:          This argument is only needed if classification=False. It specifies the center-point of the
                            distribution around which the data is to be distributed. The function then adds the same
                            number of data-points to the quantile-bins on each side of the center. So for instance,
                            if center=0, there will be the same amount of data in the interval (-1,0) and the interval
                            (0,1). The same amount of data in the interval (-2,-1) as in (1,2), and so forth.

    :param nr_bins:         Specifies the number of quantile bins (only needed if classification=False).

    :param upsample:        If True, upsample instead of downsample within the bins.

    :return new_X:          Same as X but with balanced training data. Attention: index ordering is now typically
                            different from original one. Sorting and similar will be lost.
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

        counts = X[target].value_counts()                                   # count all target values
        n      = counts.max() if upsample else counts.min()                 # target value with most/least data
        Xs     = []                                                         # stores sub-frames of each target value

        for _, df in X.groupby(target):                                     # group by target

            replace  = False if not upsample or len(df)==n else True        # don't replace for majority class
            sdf      = df.sample( n=n, replace=replace )                    # each class should have n values
            Xs      += [sdf]                                                # collect

        X = pd.concat(Xs, axis='index')                                     # concatenate
        X = X.sample( frac=1, replace=False )                               # shuffle

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
    nX[target]      = nX[target].clip(-clip, clip)                      # also need to clip target

    Y               = Y.to_frame()                                      # make into DataFrame
    Y['abs_target'] =          Y['target'].abs()                        # take absolute values
    Y['sgn_target'] = np.sign( Y['target'] )                            # sign: deviation from center
    Y['bin']        = pd.cut(  Y['abs_target'], bins=nr_bins  )         # assign to quantile-bins
    new_X           = []                                                # stores the new values

    for _, sY in Y.groupby('bin'):                                      # iterate targets by bin

        nv = len(sY['sgn_target'].value_counts())                       # number of distinct values
        assert nv==2, 'nv should have 2 values.'                        # should be two 

        sY = balanced_downsample( X              = sY,                  # recursive call
                                  target         ='sgn_target',         # balance the sign in each bin
                                  classification = True,                # treat as discrete
                                  upsample       = upsample,            # keep consistent
                                  )

        sX     = nX.reindex(sY.index)                                   # select related X data (index must be unique!)
        new_X += [ sX ]                                                 # append to list

    new_X = pd.concat(new_X, axis='index')                              # stack back together

    return new_X
    