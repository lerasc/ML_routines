

import numpy as np

from itertools import product


def make_nice_string( nr, sig=2 ):
    """
    Given a number nr, turn it into a nice string, and round it to sig-decimals if its a float.  
    """

    if    isinstance(nr,str):   return  nr
    if    isinstance(nr,list):  return  str(nr)
    if    isinstance(nr,set):   return  str(nr)
    elif  np.isnan(nr):         return 'NaN'
    elif  nr is None:           return 'None'
    elif  abs(nr) > 99:         return "{0:,}".format(int(nr)) # ignore decimals, annotate thousands by comma
    elif  isinstance(nr,float): return str(np.round(nr,sig)) 
    elif  isinstance(nr,int):   return str(nr)
    else:                       return str(nr)


def print_nice_dict( d, keys=None ):
    """
    Print a dictionary in aligned format of type 'key = value'. If keys is None, all keys are used.
    """

    assert isinstance(d,dict), "d must be dictionary"                               # check input

    if len(d)==0: return ''                                                         # special case
    if keys is None: keys = list( d.keys() )                                        # print all key

    max_str     = max([len(s) for s in keys])                                       # key that requires most space
    filling     = [max_str + 3 for d in keys ]                                      # add 3 extra spaces 
    
    text = ''                                                                       # intialize the output string
    for i,(key,val) in enumerate(d.items()):                                        # one line per key:
        text += key.ljust(filling[i])+' = '+ make_nice_string(val)+"\n"             # aligned key=value string

    text = text[:-1]                                                                # get rid of last "\n"

    return text     


def form_all_combinations( args ):
    """
    Input is a dictionary with a list of values for each key. Then, all combinations are formed and rearanged into 
    a list of dictionaries, with one combination each. 
    """

    keys        = list( args.keys()    )                                   # list of keys
    lov         = list( args.values()  )                                   # list of values
    combs       = list( product(*lov) )                                    # all value combinations
    lo_args     = [dict(zip(keys, vals)) for vals in combs]                # list of different x dicts

    return lo_args


def execute_one_combination( func, k, fix, vary, names=None, verbose=True ):
    """
    Apply the k-th parameter configuration to the obj 'func', where we first form all parameter configurations of
    the 'vary' dictionary.  In other words, vary is a dictionary where the keys are obj arguments, and the values
    are lists of obj arguments, of which we then form all combinations and select the k-th one. The fix argument
    contains all func-arguments that remain fixed. The obj 'func' is evaluated at the fixed arguments and the
    k-th parameter configuration of the varied arguments.
  
    In most cases, names can be set to None. However, sometimes the vary dictionary contains arguments that are cannot
    be converted (meaningfully) to a string. In case vary contains one or several such arguments, then the names 
    argument should be used. Specifically, the names dictionary is a dictionary that contains the same keys as the 
    vary dictionary, and its values are list of the same length as their counter-part in the vary dictionary. However, 
    the values in those lists are meaningfully convertable to strings. 

    If verbose is set to True, then the selected parameter configuration is written to the output. 

    Return value is the value of func, as well as a string the uniquely labels the k-th parameter combination. This can
    be used for instance to store the obj output automatically. The argument dictionary is returned as well.
    """

    # check that provided input is in correct format
    ####################################################################################################################
    assert(callable(func)),"func must be obj."
    assert(isinstance(k,int)),"k must be integer"
    assert(isinstance(fix,dict)),"fix must be dictionary"
    assert(isinstance(vary,dict)),"vary must be dictionary"
    for key,vals in vary.items(): assert(type(vals) in [list, np.ndarray]),"vary values must be lists"     

    if names is not None: 
        assert(isinstance(names,dict)),"names must be dictionary"
        for key in vary.keys(): assert(key in names.keys()),"vary must have same keys as names dictionary"
        assert(len(vary)==len(names)),"vary must have same keys as names dictionary"
    else:
        names = vary.copy()


    # Form all different parameter configurations, select k-th one and store everything in one dictionary called dd.
    ####################################################################################################################
    keys    	= list( vary.keys()    )                            # list of keys
    lov     	= list( vary.values()  )                            # list of values
    lon 		= list( names.values() ) 		                    # list of name s
    combs   	= list( product(*lov) )                             # all value combinations
    name_combs 	= list( product(*lon) ) 		                    # all name combinations
    comb    	= combs[k]                                          # select the k-th configuration
    name 		= name_combs[k] 				                    # select the k-th name
    vary    	= dict(zip(keys,comb))                              # form an argument dictionary
    name 		= dict(zip(keys,name)) 			                    # name dictionary
    name 		= '_'.join([ k+'='+v for (k,v) in name.items() ]) 	# transform the dictionary into strings
    params    	= fix.copy()                                        # stores all func arguments for this specific input
    params      = {**params, **vary}                                # all input parameters in one dictionary

    # evaluate the obj for this set of parameters
    ####################################################################################################################
    if verbose: 
        print('running the following parameters:')
        print(name.replace('_',', '))

    ret = func( **params )    

    # return the obj value, the parameter configuration and the filename
    ####################################################################################################################
    return ret, params, name