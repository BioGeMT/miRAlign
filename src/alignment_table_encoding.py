import pandas as pd
import numpy as np
from .positional_alignment import get_posaware_alignments
from .shared_global_vars import NUCL, NUCL_DICT

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
    Gaps are summarized as overall count. 
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

def encode_biopython_positionalgaps(alignment_list):
    """
    Encode biopython alignments in a binary matrix
    representing matches and mismatches per nucleotide.
    Includes numbers of gaps between miRNA nucleotides and gaps matched with
    miRNA nucleotides. 
    Assumes 22nt miRNAs. 
    """
    positional_logreg_columns =['%i_%s' % (nt, factor) for nt in range(0, 22) for factor in ['is_match', 'is_mismatch', 'matched_gap']]
    positional_logreg_columns += ['gap_after_%i' % nt for nt in range(0, 21)]
    
    column_mapping = {c: i for i,c in enumerate(positional_logreg_columns)}
    X = np.zeros((len(alignment_list), len(positional_logreg_columns)))
    for rowid, aln in enumerate(alignment_list):
        last_mir_crd = -1
        mir_gap_count = 0 
        for aln_crd, mir_crd in enumerate(aln.indices[0]):
            type_of_bind = ''
            if mir_crd != -1:
                if last_mir_crd > -1:
                    colname = 'gap_after_%i' % last_mir_crd
                    colid = column_mapping[colname]
                    X[rowid, colid] = mir_gap_count
                last_mir_crd = mir_crd
                mir_gap_count = 0
                # X_train.loc[miR_index, str(mir_crd) + '_' + 'is_in'] = 1
                if aln[0, aln_crd] == aln[1, aln_crd]: 
                    # match
                    colname = str(mir_crd) + '_' + 'is_match'              
                elif aln[1, aln_crd] != '-':
                    colname = str(mir_crd) + '_' + 'is_mismatch'
                else:
                    colname = str(mir_crd) + '_' + 'matched_gap'
                colid = column_mapping[colname]
                X[rowid, colid] = 1
            else: # mir_crd == -1
                mir_gap_count += 1
    X = pd.DataFrame(X, columns = positional_logreg_columns)
    X['intercept'] = 1
    return X

def encode_posloc_simple(alignment_list):
    """
    Encode alignments from my implementation of positional alignment in a binary matrix
    representing matches and mismatches.
    Assumes 22nt miRNAs but doesn't check.
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

def encode_posloc_fullM(alignment_list, index):
    """
    Returns a DataFrame encoding alignments from the supplied list.
    Alignments are one-hot encoded as position and nucleotide pair.
    Assumes 22nt miRNAs but doesn't check.
    """
    positional_logreg_columns =['%i_%s%s' % (ntpos, nt1, nt2) for ntpos in range(0, 22) for nt1 in NUCL for nt2 in NUCL]
    positional_logreg_columns += ['Gap_miR', 'Gap_gene']
    column_mapping = {c: i for i,c in enumerate(positional_logreg_columns)}
    X = np.zeros((len(alignment_list), len(positional_logreg_columns)))
    for rowid, aln in enumerate(alignment_list):
        s, aligned_miR, aligned_mR, crd_miR, crd_mR  = aln
        for aln_crd, mir_crd in enumerate(crd_miR):
            type_of_bind = ''
            if mir_crd != -1 and aligned_mR[aln_crd] != '-':
                colname = '%i_%s%s' % (mir_crd, aligned_miR[aln_crd], aligned_mR[aln_crd])
                colid = column_mapping[colname]
                X[rowid, colid] = 1
        # X_train.loc[miR_index, 'gaps_in_miR'] = sum(x=='-' for x in aln[0])
        # # Note: if using nested model, gaps in gene are included in is_in - is_match
        X[rowid, -2] = sum(c == '-' for c in aligned_miR) 
        X[rowid, -1] = sum(c == '-' for c in aligned_mR)
    X = pd.DataFrame(X, columns = positional_logreg_columns, index=index)
    return X

def encode_posloc_MGG(posloc_alignments):
    """
    Returns a tuple of Numpy arrays M_counts, G_miR_counts, G_gene_counts.
    The arrays encode alignments from the list.
    The first dimention of each array corresponds to the index in the
    alignment_list.

    Memory-wise it's inefficient. 
    """
    M_counts = np.zeros((len(posloc_alignments), 4, 4, 22))
    G_miR_counts = np.zeros((len(posloc_alignments), 21))
    G_gene_counts = np.zeros((len(posloc_alignments), 22))
    for rowid, aln in enumerate(posloc_alignments):
        s, aligned_miR, aligned_mR, crd_miR, crd_mR  = aln
        last_mir_crd = -1 # last seen crd before gapped region in miR
        mir_gap_count = 0 # length of the current gapped region in miR
        for aln_crd, mir_crd in enumerate(crd_miR):
            if mir_crd != -1: # not a gap in miR
                if last_mir_crd > -1:  # we're inside the alignment
                    G_miR_counts[rowid, last_mir_crd] = mir_gap_count
                last_mir_crd = mir_crd
                mir_gap_count = 0
                nt1 = aligned_miR[aln_crd]
                nt2 = aligned_mR[aln_crd]
                assert nt1 != '-'
                if nt2 == '-':
                    G_gene_counts[rowid, mir_crd] += 1
                else:
                    nt1id = NUCL_DICT[nt1]
                    nt2id = NUCL_DICT[nt2]
                    M_counts[rowid, nt1id, nt2id, mir_crd] += 1
            else:
                mir_gap_count += 1
    return (M_counts,  G_miR_counts, G_gene_counts)


def encode_posloc_positionalgaps(alignment_list):
    pass
