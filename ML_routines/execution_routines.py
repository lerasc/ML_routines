
import os 

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
    

def get_path( subdir=None, ROOT='', create=False, error='raise' ):
    """
    Savely return and/or create a (sub-) path to a git repo. 

    :param subdir:      if not None, a subdirectory of the root path.
    :param ROOT:        path to root folder where git repo is located. Must exist. 
    :param create:      if True, create the subdir if it does not yet exist
    :param error:       'raise' error if an existig folder is created or non-existing one not created, else 'ignore'

    :return folder:     path to directory
    """

    assert error in ['raise','ignore'], f'invalid error argument {error}'

    # initialize folder path as root to directory, and check that it exists
    ####################################################################################################################
    folder = ROOT # initialize 

    if not os.path.exists(folder): raise ValueError(f'invalid ROOT path {ROOT}')

    # add subdir and check if it exists, if not, create it
    ####################################################################################################################
    if subdir is not None: folder += subdir
    
    folder = folder.replace('//','/') # avoid double backslash (one from ROOT, one from subdir)
    exists = os.path.exists(folder)

    if       exists and     create and error=='raise':  raise ValueError(f'folder {folder} already exsits')
    elif     exists and     create and error=='ignore': os.makedirs( folder, exist_ok=True )
    elif not exists and     create:                     os.makedirs( folder, exist_ok=True )
    elif not exists and not create and error=='raise':  raise ValueError(f'folder {folder} does not exist')
    elif not exists and not create and error=='ignore': pass

    return folder        