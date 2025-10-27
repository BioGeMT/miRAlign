# Imports
import numpy as np
from sklearn.metrics import average_precision_score
from Bio.Seq import Seq
from numba import njit

# Global variables
NUCL = ['A', 'T', 'C', 'G']
NUCL_DICT = {nt: i for i,nt in enumerate(NUCL)}

# Positional-aware alignment implementation
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
    NUCL = ['A', 'T', 'C', 'G', '-']
    NUCL_DICT = {nt: i for i,nt in enumerate(NUCL)}
    A_int = np.array([NUCL_DICT[b] for b in miR], np.int8)
    B_int  = np.array([NUCL_DICT[b] for b in mR], np.int8)
    S, align_miR, align_mR, crdmir, crdgene = pos_aware_align_local_numba(A_int, B_int, 
                                                                          M, G_miR, G_gene, 
                                                                          backtrack=backtrack)
    align_mR = [NUCL[i] for i in align_mR]
    align_miR = [NUCL[i] for i in align_miR]
    return(S, align_miR, align_mR, crdmir, crdgene)

# Optimizing functions - helpers
### Golden section and scoring functions
def gss(f, a, b, tol=0.01):
    gr = (np.sqrt(5) + 1) / 2.
    checked_pars = {}  
    while abs(b - a) > tol:
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        try:
            fc = checked_pars[np.round(c, 4)]
        except KeyError:
            fc = f(c)
            checked_pars[np.round(c, 4)] = fc
        else:
            #print('Received cached f equal', fc, 'for param', c)
            pass
        try:
            fd = checked_pars[np.round(d, 4)]
        except KeyError:
            fd = f(d)
            checked_pars[np.round(d, 4)] = fd
        else:
            #print('Received cached f equal', fd, 'for param', d)
            pass
        if fc > fd:  # f(c) > f(d) to find the maximum
            b = d
        else:
            a = c

    return (b + a) / 2


def get_posaware_alignments(dset, M, G_miR, G_gene):
    alignments = []
    for i, l in dset.iterrows():
        mRNA = Seq(l['gene'])
        miR = Seq(l['noncodingRNA'])
        if len(miR) != 22:
            raise ValueError('Incorrect miRNA length')
        mRNA = mRNA.reverse_complement()
        alignments.append(pos_aware_align_local(miR, mRNA, 
                                                M, G_miR, G_gene, 
                                                backtrack=True))
    return alignments


def get_posaware_aligner_score(M, G_miR, G_gene, dset):
    scores_train_pos = []
    scores_train_neg = []
    for i, l in dset.iterrows():
        mRNA = Seq(l['gene'])
        miR = Seq(l['noncodingRNA'])
        if len(miR) != 22:
            continue
        mRNA = mRNA.reverse_complement()
        s = pos_aware_align_local(miR, mRNA, M, G_miR, G_gene)[0]
        if l[5] == 1:
            scores_train_pos.append(s)
        elif l[5] == 0:
            scores_train_neg.append(s)
        else:
            raise RuntimeError()
    return average_precision_score([1]*len(scores_train_pos) + [0]*len(scores_train_neg),
                                   scores_train_pos + scores_train_neg)


### Optimization targets
def rescale_seed_around_center(internalM, scale):
    """
    Works in situ
    """
    seed_expectations = internalM[..., :8].mean(axis=(0,1))
    internalM[..., :8] -= seed_expectations
    internalM[..., :8] *= scale
    internalM[..., :8] += seed_expectations


def rescale_tail_around_center(internalM, scale):
    """
    Works in situ
    """
    tail_expectations = internalM[..., 8:].mean(axis=(0,1))
    internalM[..., 8:] -= tail_expectations
    internalM[..., 8:] *= scale
    internalM[..., 8:] += tail_expectations

def create_seed_multiplicative_target(M, G1, G2, dset):
    """
    Seed range right-exclusive
    """
    def target(param_value):
        internalM = M.copy()
        rescale_seed_around_center(internalM, param_value)
        score = get_posaware_aligner_score(internalM, G1, G2, dset)
        return score
    return target

def create_nonseed_multiplicative_target(M, G1, G2, dset):
    """
    seed_range right-exclusive
    """
    def target(param_value):
        internalM = M.copy()
        rescale_tail_around_center(internalM, param_value)
        score = get_posaware_aligner_score(internalM, G1, G2, dset)
        return score
    return target

def create_seed_additive_target(M, G1, G2, dset):
    """
    Seed range right-exclusive
    """
    def target(param_value):
        internalM = M.copy()
        internalM[..., :8] += param_value
        score = get_posaware_aligner_score(internalM, G1, G2, dset)
        return score
    return target

def create_nonseed_additive_target(M, G1, G2, dset):
    """
    Seed range right-exclusive
    """
    def target(param_value):
        internalM = M.copy()
        internalM[..., 8:] += param_value
        score = get_posaware_aligner_score(internalM, G1, G2, dset)
        return score
    return target

def create_miR_gap_seed_multiplicative_target(M, G1, G2, dset):
    """
    Doesn't matter if multiplicative or additive expect for parameter values
    """
    def target(param_value):
        internalG = G1.copy()
        internalG[:8] *= param_value
        score = get_posaware_aligner_score(M, internalG, G2, dset)
        return score
    return target

def create_miR_gap_nonseed_multiplicative_target(M, G1, G2, dset):
    """
    Doesn't matter if multiplicative or additive expect for parameter values
    """
    def target(param_value):
        internalG = G1.copy()
        internalG[8:] *= param_value
        score = get_posaware_aligner_score(M, internalG, G2, dset)
        return score
    return target


def create_gene_gap_seed_multiplicative_target(M, G1, G2, dset):
    """
    For the whole sequence, not just seed
    Doesn't matter if multiplicative or additive expect for parameter values
    """
    def target(param_value):
        internalG = G2.copy()
        internalG[:8] *= param_value
        score = get_posaware_aligner_score(M, G1, internalG, dset)
        return score
    return target

def create_gene_gap_nonseed_multiplicative_target(M, G1, G2, dset):
    """
    For the whole sequence, not just seed
    Doesn't matter if multiplicative or additive expect for parameter values
    """
    def target(param_value):
        internalG = G2.copy()
        internalG[8:] *= param_value
        score = get_posaware_aligner_score(M, G1, internalG, dset)
        return score
    return target





# Optimizing functions - mains
def gss_optim_seed_vs_nonseed(train, test, M, G_miR, G_gene, MAX_ITER=10, tol=0.001):
    """
    Affine optimization of seed vs non-seed matching matrix and gap penalties,
    for both miR and gene gap penalties
    """
    M = M.copy()
    G_miR = G_miR.copy()
    G_gene = G_gene.copy()
    print('Initial score', get_posaware_aligner_score(M, G_miR, G_gene, train))
    for iter_nb in range(MAX_ITER):
        target = create_seed_multiplicative_target(M, G_miR, G_gene, train)
        best_weight = gss(target, 0.5, 2, tol)
        rescale_seed_around_center(M, best_weight)
        print('Found multiplicative weight for seed', best_weight, 'train score', get_posaware_aligner_score(M, G_miR, G_gene, train))
        
        target = create_seed_additive_target(M, G_miR, G_gene, train)
        best_weight = gss(target, -2, 2, tol)
        M[..., :8] += best_weight
        print('Found additive weight for seed', best_weight, 'train score', get_posaware_aligner_score(M, G_miR, G_gene, train)) 
        
        target = create_nonseed_multiplicative_target(M, G_miR, G_gene, train)
        best_weight = gss(target, 0.5, 2, tol)
        rescale_tail_around_center(M, best_weight)
        print('Found multiplicative weight for non-seed', best_weight, 'train score', get_posaware_aligner_score(M, G_miR, G_gene, train)) 
        
        target = create_nonseed_additive_target(M, G_miR, G_gene, train)
        best_weight = gss(target, -2, 2, tol)
        M[..., 8:] += best_weight
        print('Found additive weight for non-seed', best_weight, 'train score', get_posaware_aligner_score(M, G_miR, G_gene, train)) 
        
        target = create_miR_gap_seed_multiplicative_target(M, G_miR, G_gene, train)
        best_weight = gss(target, 0.2, 5, tol)
        G_miR[:8] *= best_weight
        print('Found multiplicative miR gap weight for seed', best_weight, 'train score', get_posaware_aligner_score(M, G_miR, G_gene, train)) 

        target = create_miR_gap_nonseed_multiplicative_target(M, G_miR, G_gene, train)
        best_weight = gss(target, 0.2, 5, tol)
        G_miR[8:] *= best_weight
        print('Found multiplicative miR gap weight for non-seed', best_weight, 'train score', get_posaware_aligner_score(M, G_miR, G_gene, train)) 

        target = create_gene_gap_seed_multiplicative_target(M, G_miR, G_gene, train)
        best_weight = gss(target, 0.2, 5, tol)
        G_gene[:8] *= best_weight
        print('Found multiplicative gene gap weight for seed', best_weight, 'train score', get_posaware_aligner_score(M, G_miR, G_gene, train)) 

        target = create_gene_gap_nonseed_multiplicative_target(M, G_miR, G_gene, train)
        best_weight = gss(target, 0.2, 5, tol)
        G_gene[8:] *= best_weight
        print('Found multiplicative gene gap weight for non-seed', best_weight, 'train score', get_posaware_aligner_score(M, G_miR, G_gene, train)) 

        print('Test score after ITER%i' % iter_nb, get_posaware_aligner_score(M, G_miR, G_gene, test))
    return(M, G_miR, G_gene)
        

def gss_optim_full(dset, valset, M, G_miR, G_gene, MAX_ITER=10, tol=0.1):
    """
    Optimization of each entry of the positional matrix and gap vectors
    """
    M = M.copy()
    G_miR = G_miR.copy()
    G_gene = G_gene.copy()
    for iter_nb in range(MAX_ITER):
        for miR_index in range(22):
            # substitution matrix entries
            for row in range(4):
                for column in range(4):
                    def target(param_value):
                        M[row, column, miR_index] = param_value  # will modify M globally
                        score = get_posaware_aligner_score(M, G_miR, G_gene, dset)
                        return score
                    if row == column:
                        best_value = gss(target, 0, 10, tol)
                        M[row, column, miR_index] = best_value  # to use (b+a)/2 from gss
                    else:
                        best_value = gss(target, -10, 0, tol)
                        M[row, column, miR_index] = best_value  # to use (b+a)/2 from gss
                    print('Found best value for M[%i, %i, %i]:' % (row, column, miR_index), best_value)
            # gene gap
            def target(param_value):
                G_gene[miR_index] = param_value
                score = get_posaware_aligner_score(M, G_miR, G_gene, dset)
                return score
            best_value = gss(target, -10, 0, tol)
            G_gene[miR_index] = best_value
            print('Found best value for G_gene[%i]:' % (miR_index), best_value)
            # miR gap
            if miR_index >= 21:
                print('Skipping G_miR[%i]' % (miR_index))
                continue
            def target(param_value):
                G_miR[miR_index] = param_value
                score = get_posaware_aligner_score(M, G_miR, G_gene, dset)
                return score
            best_value = gss(target, -10, 0, tol)
            G_miR[miR_index] = best_value
            print('Found best value for G_miR[%i]:' % (miR_index), best_value)
        print('***')
        print('Train score after ITER%i' % iter_nb, get_posaware_aligner_score(M, G_miR, G_gene, dset))
        print('Test score after ITER%i' % iter_nb, get_posaware_aligner_score(M, G_miR, G_gene, valset))

        print('***')
    return(M, G_miR, G_gene)












            
