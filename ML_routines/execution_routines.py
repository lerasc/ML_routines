

from itertools import product


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
    