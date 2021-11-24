
import numpy  as np
import pandas as pd

from sklearn.linear_model import LinearRegression as LR


def bootstrapped_lin_reg( x, y,
						  n			=  100,
						  frac		=  0.8,
						  q 		=  0.25,
						  full_ret	=  False,
						  ax		=  None,
						  color		= 'blue',
						  **kwargs
						  ):
	"""
	Here, we implement a bootstrapped linear regression, that also can plot error bars simliar to how seaborn's regplot
	does it. However, seaborn does not return the estimated coefficients. Statsmodels does have such functionality, but
	it is cumbersome. Here, we thus implement our own bootstrapped linear regression that, at the same time, allows
	for nice plots.

	:param x: 			argument data

	:param y: 			dependent variable data (of same length as x)

	:param n:			number of times to resample with replacement for the bootstrapped estimate

	:param frac:		fraction of data to sample with replacement

	:param q: 			quantile left and right of the median used for error estimate (q=0.25 means interquantile range)

	:param full_ret:	if True, return not only the fitted parameters, but also the raw data

	:param ax: 			axis instance to plot into (no plot is created if set to None)

	:param color:		plotting color

	:param kwargs: 		additional arguments for the plot obj

	:return:			slope, intcp, slope_err, intcp_err (and all fits if full_ret=True)
	"""

	assert len(x)==len(y), 'x and y must have same length '

	df 	      = pd.DataFrame(np.array([x,y]).T, columns=['x','y']).dropna()                     # for conveneience
	dfs 	  = [ df.sample(frac=frac, replace=True) for _ in range(n) ]                        # sample n-times
	fits  	  = np.array([ np.polyfit( df['x'].values, df['y'].values, deg=1) for df in dfs ])  # fit each sample
	fits      = pd.DataFrame(fits, columns=['slope','intcp'])                                   # make nice
	slope     = fits['slope'].mean()
	intcp     = fits['intcp'].mean()
	slope_err = 0.5 * ( fits['slope'].quantile(q=0.5+q) - fits['slope'].quantile(q=0.5-q) )     # half the IQR
	intcp_err = 0.5 * ( fits['intcp'].quantile(q=0.5+q) - fits['intcp'].quantile(q=0.5-q) )     # half the IQR

	if ax is not None: # plot,  while taking lower anßd upper bounds

		xvals       = np.linspace( np.nanmin(x), np.nanmax(x), 500 )
		yvals       = pd.DataFrame([ f['slope']*xvals + f['intcp'] for _,f in fits.iterrows() ]).T
		yvals.index = xvals
		lower       = yvals.quantile(q=0.5-q, axis=1)
		upper       = yvals.quantile(q=0.5+q, axis=1)
		_           = ax.fill_between(xvals, lower, upper , facecolor=color, alpha=.15)
		reg_line    = slope*xvals + intcp
		ax.plot(xvals, reg_line, color=color, **kwargs )

	if full_ret: return slope, intcp, slope_err, intcp_err, fits
	else: 		 return slope, intcp, slope_err, intcp_err