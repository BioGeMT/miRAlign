# Imports
import pandas as pd
import numpy as np
from .positional_alignment import get_posaware_alignments
from .shared_global_vars import NUCL, NUCL_DICT

# Positional alignment matrices
def get_params_from_model(logit_model, default_match = 0., default_mismatch = -0.):
    """
    Returns a positional scoring matrix and gap vectors
    from the parameters of a logit model.
    Trailing mismatches are the negative of trailing matches
    (they don't matter though because it's never optimal to mismatch the flanks).
    Returns M, G_miR, G_gene.
    G_miR and G_gene are constant.   
    """
    M = np.zeros((4, 4, 22))   # third coordinate = nt index
    for i in range(1, 21):
        try:
            match_score = logit_model.params[str(i)+'_is_match']
        except KeyError:
            match_score = default_match
        try:
            mismatch_score = logit_model.params[str(i)+'_is_mismatch']
        except KeyError:
            mismatch_score = default_mismatch
        M[...,i] = np.eye(4)*(match_score - mismatch_score)
        M[..., i] += mismatch_score
    match_score = logit_model.params['0_is_match']
    M[..., 0] = np.eye(4)*2*match_score - match_score
    match_score = logit_model.params['21_is_match']
    M[..., 21] = np.eye(4)*2*match_score - match_score
    G_miR = np.zeros(21) + logit_model.params['Gap_miR']
    G_gene = np.zeros(22) +  logit_model.params['Gap_gene']
    return(M, G_miR, G_gene)


# Optimizing the substitution matrix
def logreg_optimization_iterative(dset, MAX_ITER):
    """
    pass
    """
    pass
