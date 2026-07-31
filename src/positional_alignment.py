# Imports
import numpy as np
from sklearn.metrics import average_precision_score
from Bio.Seq import Seq
from numba import njit
from .shared_global_vars import NUCL, NUCL_DICT


# Positional-aware alignment implementation
### WORKHORSES
@njit
def pos_aware_align_local_numba(miR_int, mR_int, M, G_miR, G_gene, backtrack=False):
    """
    Returns a score and an alignment for a local positionally-aware alignment
    G_miR = vector of gap penalties for miRNA, len(G_miR) == len(miR) - 1
    G_gene = vector of gap penalties of mRNA, len(G_gene) == len(miR)
    G_gene is the penalty for matching a miRNA nucleotide with a gap, 
    G_miR[i] is the penalty of inserting a gap between miRNA nucleotides i and i+1
    M = a matrix of positionally-aware substitution scores, of shape 4x4x20. The last dimention corresponds
    to the coordinate at miR sequence. 
    """
    n, m = len(miR_int), len(mR_int)
    assert n == 22
    score_matrix = np.zeros((n + 1, m + 1), dtype=np.float32)
    best_score = 0.
    best_score_i = 0
    best_score_j = 0
    if backtrack:
        backtrack_matrix = np.zeros((n + 1, m + 1), dtype=np.uint8) 
    for i in range(1, n+1):
        # gmir uses a dummy gap score outside of the sequence;
        # this works as long as the dummy is negative 
        gmir = -1 if i == n else G_miR[i-1]
        ggene = G_gene[i-1]
        for j in range(1, m+1):
            match_score = M[miR_int[i-1], mR_int[j-1], i-1]
            # partial scores
            ps_match   = score_matrix[i-1, j-1] + match_score
            ps_gap_miR = score_matrix[i, j-1] + gmir
            ps_gap_mR  = score_matrix[i-1, j] + ggene
            s, d = 0., 0
            if ps_match > s:
                d = 1
                s = ps_match
            if ps_gap_miR > s:
                d = 2
                s = ps_gap_miR
            if ps_gap_mR > s:
                d = 3
                s = ps_gap_mR
            score_matrix[i, j] = s
            if s > best_score: 
                best_score = s
                best_score_i = i
                best_score_j = j
            if backtrack:
                backtrack_matrix[i, j] = d
    # Get the shortest alignment:
    align_miR = []
    align_mR = []
    sequence_indices_miR = []
    sequence_indices_mR = []
    if backtrack:
        while score_matrix[best_score_i, best_score_j] != 0:
            curr_dir = backtrack_matrix[best_score_i, best_score_j]
            if curr_dir == 1:
                align_miR.append(miR_int[best_score_i-1])
                align_mR.append(mR_int[best_score_j-1])
                sequence_indices_miR.append(best_score_i-1)
                sequence_indices_mR.append(best_score_j-1)
                best_score_i -= 1
                best_score_j -= 1
            elif curr_dir == 2:
                align_miR.append(4)
                align_mR.append(mR_int[best_score_j-1])
                sequence_indices_miR.append(-1)
                sequence_indices_mR.append(best_score_j-1)
                best_score_j -= 1
            elif curr_dir == 3:
                align_miR.append(miR_int[best_score_i-1])
                align_mR.append(4)
                sequence_indices_miR.append(best_score_i-1)
                sequence_indices_mR.append(-1)
                best_score_i -= 1    
        align_miR = align_miR[::-1]
        align_mR = align_mR[::-1]
        sequence_indices_miR = sequence_indices_miR[::-1]
        sequence_indices_mR = sequence_indices_mR[::-1]
    return (best_score, align_miR, align_mR, sequence_indices_miR, sequence_indices_mR)

@njit
def pos_aware_align_glocal_numba(miR_int, mR_int, M, G_miR, G_gene, backtrack=False):
    """
    Global-local (semi-global) alignment:
      - Global over miRNA (length n): must consume all miRNA positions 1..n
      - Local over mRNA (length m): choose any substring j0..j*
    Conventions:
      - G_gene[i] is the penalty for aligning miRNA position i (0-based) to a gap in mRNA.
      - G_miR[i] is the penalty for inserting a gap between miRNA positions i and i+1.
      - M[a, b, i] is the substitution score at miRNA position i (0-based).
    Requirements:
      len(G_gene) == n, len(G_miR) == n-1, M.shape[-1] == n
    """
    n, m = len(miR_int), len(mR_int)

    score = np.empty((n + 1, m + 1), dtype=np.float32)
    score[0, :] = 0.0                      # free start along mRNA
    score[:, 0] = 0.0
    for i in range(1, n + 1):
        score[i, 0] = score[i - 1, 0] + G_gene[i - 1]   # must consume miRNA prefix

    if backtrack:
        BT_NONE, BT_DIAG, BT_LEFT, BT_UP = 0, 1, 2, 3
        bt = np.zeros((n + 1, m + 1), dtype=np.uint8)

    for i in range(1, n + 1):
        a = miR_int[i - 1]
        ggene = G_gene[i - 1]                          # vertical (gap in mRNA)
        for j in range(1, m + 1):
            b = mR_int[j - 1]
            match = score[i - 1, j - 1] + M[a, b, i - 1]
            up    = score[i - 1, j] + ggene            # gap in mRNA
            if i < n:
                left = score[i, j - 1] + G_miR[i - 1]  # gap in miRNA between (i-1,i)
            else:
                left = -np.inf                         # disallow gaps after last miRNA base

            # Choose best without resetting to 0 (this is NOT local)
            if match >= up and match >= left:
                s = match
                d = BT_DIAG if backtrack else 0
            elif left >= up:
                s = left
                d = BT_LEFT if backtrack else 0
            else:
                s = up
                d = BT_UP if backtrack else 0

            score[i, j] = s
            if backtrack:
                bt[i, j] = d

    # Free end along mRNA: choose best in the last row
    j_star = int(np.argmax(score[n, :]))
    best_score = float(score[n, j_star])

    align_miR = []
    align_mR = []
    seq_idx_miR = []
    seq_idx_mR = []

    if backtrack:
        i, j = n, j_star
        while i > 0:
            if j == 0:
                align_miR.append(miR_int[i - 1])
                align_mR.append(4)
                seq_idx_miR.append(i - 1)
                seq_idx_mR.append(-1)
                i -= 1
                continue
            d = bt[i, j]
            if d == 1:  # DIAG
                align_miR.append(miR_int[i - 1])
                align_mR.append(mR_int[j - 1])
                seq_idx_miR.append(i - 1)
                seq_idx_mR.append(j - 1)
                i -= 1; j -= 1
            elif d == 2:  # LEFT: gap in miRNA between (i-1,i)
                align_miR.append(4)
                align_mR.append(mR_int[j - 1])
                seq_idx_miR.append(-1)
                seq_idx_mR.append(j - 1)
                j -= 1
            elif d == 3:  # UP: gap in mRNA at miRNA[i-1]
                align_miR.append(miR_int[i - 1])
                align_mR.append(4)
                seq_idx_miR.append(i - 1)
                seq_idx_mR.append(-1)
                i -= 1
            else:
                raise ValueError()

        align_miR.reverse()
        align_mR.reverse()
        seq_idx_miR.reverse()
        seq_idx_mR.reverse()

    return (best_score, align_miR, align_mR, seq_idx_miR, seq_idx_mR)


### WRAPPERS
def pos_aware_align_local(miR, mR, M, G_miR, G_gene, backtrack=False):
    """
    Returns a score and an alignment for a local positionally-aware alignment
    G_miR = vector of gap penalties for miRNA, len(G_miR) == len(miR) - 1
    G_gene = vector of gap penalties of mRNA, len(G_gene) == len(miR)
    G_gene is the penalty for matching a miRNA nucleotide with a gap, 
    G_miR[i] is the penalty of inserting a gap between miRNA nucleotides i and i+1
    M = a matrix of positionally-aware substitution scores, of shape 4x4x20. The last dimention corresponds
    to the coordinate at miR sequence. 
    """
    NUCL_WGAP= NUCL+['-']
    A_int = np.array([NUCL_DICT[b] for b in miR], np.int8)
    B_int  = np.array([NUCL_DICT[b] for b in mR], np.int8)
    S, align_miR, align_mR, crdmir, crdgene = pos_aware_align_local_numba(A_int, B_int, 
                                                                          M, G_miR, G_gene, 
                                                                          backtrack=backtrack)
    align_mR = [NUCL_WGAP[i] for i in align_mR]
    align_miR = [NUCL_WGAP[i] for i in align_miR]
    return(S, align_miR, align_mR, crdmir, crdgene)

def pos_aware_align_glocal(miR, mR, M, G_miR, G_gene, backtrack=False):
    """
    Returns a score and an alignment for a local positionally-aware alignment
    G_miR = vector of gap penalties for miRNA, len(G_miR) == len(miR) - 1
    G_gene = vector of gap penalties of mRNA, len(G_gene) == len(miR)
    G_gene is the penalty for matching a miRNA nucleotide with a gap, 
    G_miR[i] is the penalty of inserting a gap between miRNA nucleotides i and i+1
    M = a matrix of positionally-aware substitution scores, of shape 4x4x20. The last dimention corresponds
    to the coordinate at miR sequence. 
    """
    NUCL_WGAP= NUCL+['-']
    A_int = np.array([NUCL_DICT[b] for b in miR], np.int8)
    B_int  = np.array([NUCL_DICT[b] for b in mR], np.int8)
    S, align_miR, align_mR, crdmir, crdgene = pos_aware_align_glocal_numba(A_int, B_int, 
                                                                          M, G_miR, G_gene, 
                                                                          backtrack=backtrack)
    align_mR = [NUCL_WGAP[i] for i in align_mR]
    align_miR = [NUCL_WGAP[i] for i in align_miR]
    return(S, align_miR, align_mR, crdmir, crdgene)










            
