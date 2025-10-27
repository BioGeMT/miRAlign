# Imports
import pandas as pd
import numpy as np
from .positional_alignment import get_posaware_alignments

# Alignment encoding in matrices

def encode_biopython_3features(alignment_list):
    """
    Encode biopython alignments in a matrix with counts of matches, mismatches and gaps.
    """
    simple_X_train = np.zeros((len(alignment_list), 3))
    for rowid, aln in enumerate(alignment_list):
        counts = aln.counts()
        simple_X_train[rowid, 0] = counts.identities
        simple_X_train[rowid, 1] = counts.mismatches
        simple_X_train[rowid, 2] = counts.gaps
    print('numpy done')
    simple_X_train = pd.DataFrame(simple_X_train, columns=['Match', 'Mismatch', 'Gap'], dtype='float')
    simple_X_train['Intercept'] = 1
    return simple_X_train

def encode_biopython_simple(alignment_list):
    """
    Encode biopython alignments in a binary matrix
    representing matches and mismatches.
    Assumes 22nt miRNAs. 
    """
    positional_logreg_columns =['%i_%s' % (nt, factor) for nt in range(0, 22) for factor in ['is_match', 'is_mismatch']]
    positional_logreg_columns += ['Gap_miR', 'Gap_gene']
    column_mapping = {c: i for i,c in enumerate(positional_logreg_columns)}
    X = np.zeros((len(alignment_list), len(positional_logreg_columns)))
    for rowid, aln in enumerate(alignment_list):
        for aln_crd, mir_crd in enumerate(aln.indices[0]):
            type_of_bind = ''
            if mir_crd != -1:
                # X_train.loc[miR_index, str(mir_crd) + '_' + 'is_in'] = 1
                if aln[0, aln_crd] == aln[1, aln_crd]: 
                    # match
                    colname = str(mir_crd) + '_' + 'is_match'              
                elif aln[1, aln_crd] != '-':
                    colname = str(mir_crd) + '_' + 'is_mismatch'
                colid = column_mapping[colname]
                X[rowid, colid] = 1
        # # Note: if using the nested model (is present and is match), gaps in gene are included in is_in - is_match
        # X_train.loc[miR_index, 'gaps_in_miR'] = sum(x=='-' for x in aln[0])
        # # Overall gaps in alignment:
        # X_train.loc[miR_index, 'Gap'] = aln.counts().gaps
        # # Separating into miR and mRNA:
        # # Notes: Separating gaps into miR and mRNA slightly improves the score but on the level of 0.001.  
        # # Treating them jointly seems just as good at this stage.  
        X[rowid, -2] = sum(c == '-' for c in aln[0])
        X[rowid, -1] = sum(c == '-' for c in aln[1])
    X = pd.DataFrame(X, columns = positional_logreg_columns)
    X['intercept'] = 1
    return X

def encode_posloc_simple(alignment_list):
    """
    Encode alignments from my implementation of positional alignment in a binary matrix
    representing matches and mismatches.
    Assumes 22nt miRNAs.
    """
    positional_logreg_columns =['%i_%s' % (nt, factor) for nt in range(0, 22) for factor in ['is_match', 'is_mismatch']]
    positional_logreg_columns += ['Gap_miR', 'Gap_gene']
    column_mapping = {c: i for i,c in enumerate(positional_logreg_columns)}
    X = np.zeros((len(alignment_list), len(positional_logreg_columns)))
    for rowid, aln in enumerate(alignment_list):
        s, aligned_miR, aligned_mR, crd_miR, crd_mR  = aln
        for aln_crd, mir_crd in enumerate(crd_miR):
            type_of_bind = ''
            if mir_crd != -1:
                # X_train.loc[miR_index, str(mir_crd) + '_' + 'is_in'] = 1
                if aligned_miR[aln_crd] == aligned_mR[aln_crd]:
                    # match
                    colname = str(mir_crd) + '_' + 'is_match'              
                elif aligned_mR[aln_crd] != '-':
                    colname = str(mir_crd) + '_' + 'is_mismatch'
                colid = column_mapping[colname]
                X[rowid, colid] = 1
        # X_train.loc[miR_index, 'gaps_in_miR'] = sum(x=='-' for x in aln[0])
        # # Note: if using nested model, gaps in gene are included in is_in - is_match
        X[rowid, -2] = sum(c == '-' for c in aligned_miR) 
        X[rowid, -1] = sum(c == '-' for c in aligned_mR)
    X = pd.DataFrame(X, columns = positional_logreg_columns)
    X['intercept'] = 1
    return X


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
