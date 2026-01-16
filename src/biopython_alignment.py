# Imports
import numpy as np
from sklearn.metrics import average_precision_score
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import PairwiseAligner


# Global variables
NUCL = ['A', 'T', 'C', 'G', '-']
NUCL_DICT = {nt: i for i,nt in enumerate(NUCL)}



# Optimization technical functions
def gss(f, a, b):
    gr = (np.sqrt(5) + 1) / 2.
    checked_pars = {}  
    while abs(b - a) > 0.1:
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        try:
            fc = checked_pars[np.round(c, 4)]
        except KeyError:
            fc = f(c)
            checked_pars[np.round(c, 4)] = fc
        else:
            pass
            #print('Received cached f equal', fc, 'for param', c)
        try:
            fd = checked_pars[np.round(d, 4)]
        except KeyError:
            fd = f(d)
            checked_pars[np.round(d, 4)] = fd
        else:
            pass
            # print('Received cached f equal', fd, 'for param', d)
        if fc > fd:  # f(c) > f(d) to find the maximum
            b = d
        else:
            a = c

    return (b + a) / 2

def score_mtis(aligner, dset):
    scores_pos = []
    scores_neg = []
    for i, l in dset.iterrows():
#         if not i%100000:
#             print(i)
        G = Seq(l['gene'])
        mR = Seq(l['noncodingRNA'])
        G = G.reverse_complement()
        s = aligner.score(G, mR)
        if l[5] == 1:
            scores_pos.append(s)
        elif l[5] == 0:
            scores_neg.append(s)
        else:
            raise RuntimeError()
    return ([1]*len(scores_pos) + [0]*len(scores_neg), scores_pos + scores_neg)

def get_alignments(aligner, dset, revcomp_gene=True,
                   check_uniqueness = False):
    alignments = []
    numbers_of_optimals = []
    for i, l in dset.iterrows():
        G = SeqRecord(Seq(l['gene']), id='gene', name='gene', description='gene')
        mR = SeqRecord(Seq(l['noncodingRNA']), id='miRNA', name='miRNA')
        if revcomp_gene: G = G.reverse_complement()
        G.name = G.id = 'gene'
        aln = aligner.align(mR, G)
        if check_uniqueness:
            numbers_of_optimals.append(len(aln))
        alignments.append(next(aln))
    if check_uniqueness:
        return (alignments, numbers_of_optimals)
    else:
        return alignments

def create_fresh_aligner(match=5, mismatch=-4, gapopen=-6, gapextend=None):
    aligner = PairwiseAligner()
    aligner.mode = 'local'
    aligner.match_score = match
    aligner.mismatch_score = mismatch
    aligner.open_gap_score = gapopen
    if gapextend is None:
        aligner.extend_gap_score = gapopen
    else:
        aligner.extend_gap_score = gapextend
    return aligner

def score_aligner(aligner, dset):
    labels, scores = score_mtis(aligner, dset)
    return average_precision_score(labels, scores)


# Optimization targets
def create_match_target(aligner, dset):
    def match_target(param_val):
        aligner.match_score = param_val 
        score = score_aligner(aligner, dset)
        return score
    return match_target

def create_mismatch_target(aligner, dset):
    def mismatch_target(param_val):
        aligner.mismatch_score = param_val 
        score = score_aligner(aligner, dset)
        #print('Evaluated for mismatch score: aligner score', score, 'at', param_val)
        return score
    return mismatch_target


def create_lineargap_target(aligner, dset):
    def gapopen_target(param_val):
        aligner.open_gap_score = param_val
        aligner.extend_gap_score = param_val 
        score = score_aligner(aligner, dset)
        #print('Evaluated for gap open score: aligner score', score, 'at', param_val)
        return score
    return gapopen_target


def create_gapopen_target(aligner, dset):
    def gapopen_target(param_val):
        aligner.open_gap_score = param_val 
        score = score_aligner(aligner, dset)
        #print('Evaluated for gap open score: aligner score', score, 'at', param_val)
        return score
    return gapopen_target

def create_gapextend_target(aligner, dset):
    def gapextend_target(param_val):
        aligner.extend_gap_score = param_val 
        score = score_aligner(aligner, dset)
        #print('Evaluated for gap extend score: aligner score', score, 'at', param_val)
        return score
    return gapextend_target


# Optimizers
def optimize_linear_gap(train, test, MAX_ITER=10, verbose=True):
    """
    Optimize match, mismatch and linear gap penalties.
    Test data set only to monitor scoring during fitting.  
    """
    aligner = create_fresh_aligner()
    test_scores = []
    train_scores = []
    for iter_nb in range(MAX_ITER):
        change_in_params = False
        curr_score = score_aligner(aligner, train)
        curr_test_score = score_aligner(aligner, test)
        train_scores.append(curr_score)
        test_scores.append(curr_test_score)

        current_match = aligner.match_score
        current_mismatch = aligner.mismatch_score
        current_gap = aligner.open_gap_score
        if verbose:
            print('Current train score:', curr_score)
            print('Current test score:', curr_test_score)
        
        match_target = create_match_target(aligner, train)
        new_match = gss(match_target, 0.001, 10)
        if abs(new_match - current_match) > 0.001:
            change_in_params = True
            if verbose: print('***Found new match score', new_match)
            aligner.match_score = new_match

        mismatch_target = create_mismatch_target(aligner, train)
        new_mismatch = gss(mismatch_target, -10, 0.001)
        if abs(new_mismatch - current_mismatch) > 0.001:
            change_in_params = True
            if verbose: print('***Found new mismatch score', new_mismatch)
            aligner.mismatch_score = new_mismatch

        gap_target = create_lineargap_target(aligner, train)
        new_gap = gss(gap_target, -10, 0.001)
        if abs(new_gap - current_gap) > 0.001:
            change_in_params = True
            if verbose: print('***Found new gap open score', new_gap)
            aligner.open_gap_score = new_gap
            aligner.extend_gap_score = new_gap

        if not change_in_params:
            if verbose: print('Optimum reached')
            break
    curr_score = score_aligner(aligner, train)
    curr_test_score = score_aligner(aligner, test)
    train_scores.append(curr_score)
    test_scores.append(curr_test_score)
    if verbose:
        print('Final train score:', curr_score)
        print('Final test score:', curr_test_score)
        
    return(aligner, train_scores, test_scores)    


# Data analysis functions
def aligner_prc(aligner, dset, title):
    from sklearn.metrics import PrecisionRecallDisplay
    baseline_labels, baseline_scores = score_mtis(aligner, dset)
    display = PrecisionRecallDisplay.from_predictions(
        baseline_labels,
        baseline_scores, 
        plot_chance_level=True, pos_label=1
    )
    _ = display.ax_.set_title(title)
    print('Average precision score:', score_aligner(aligner, dset))

def count_substitutions_per_miR_site(alignment_list, weights=None):
    global NUCL_DICT
    positional_substitutions = np.zeros((4, 5, 22))
    if weights is None:
        weights = [1]*len(alignment_list)
    for a, w in zip(alignment_list, weights):        
        for aln_crd, mr_crd in enumerate(a.indices[0]):
            if mr_crd != -1:
                rowid = NUCL_DICT[a[0,aln_crd]]
                colid = NUCL_DICT[a[1,aln_crd]]
                positional_substitutions[rowid, colid, mr_crd] += w
    return positional_substitutions

def count_substitutions_per_gene_site(alignment_list):
    global NUCL_DICT
    positional_substitutions = np.zeros((4, 5, 50))
    for a in alignment_list:        
        for aln_crd, gene_crd in enumerate(a.indices[1]):
            if gene_crd != -1:
                rowid = NUCL_DICT[a[1,aln_crd]]
                colid = NUCL_DICT[a[0,aln_crd]]
                positional_substitutions[rowid, colid, gene_crd] += 1
    return positional_substitutions

def get_nt_count_matrix(alignment_list):
    global NUCL_DICT
    nt_counts = np.zeros((50, 22))
    for a in alignment_list:        
        for aln_crd in range(a.shape[1]):
            miR_crd = a.indices[0,aln_crd]
            gene_crd = a.indices[1,aln_crd]
            if miR_crd != -1 and gene_crd != -1:
                nt_counts[gene_crd, miR_crd] += 1
    return nt_counts
            
def get_nucleotide_correlations(alignment_list):
    from scipy.stats import pearsonr
    presence_matrix = np.zeros((len(alignment_list), 22))
    for aln_id, aln in enumerate(alignment_list):
        ind = aln.indices[0]
        for i in ind:
            if i != -1:
                presence_matrix[aln_id, i] += 1
    corr_matrix = np.corrcoef(presence_matrix.T)
    return corr_matrix    

def get_miR_gap_counts(alignment_list):
    """
    Returns a vector counting the number of gaps in miRNAs
    after each nucleotide
    """  
    gaps_in_miR = np.zeros(22)
    for a in alignment_list:
        # miRNA gaps
        in_gap = False
        last_crd = -1
        for aln_crd, miR_crd in enumerate(a.indices[0]):
            if miR_crd == -1:
                if not in_gap:
                    gaps_in_miR[last_crd] += 1
                in_gap = True
            else:
                in_gap = False
                last_crd = miR_crd
    return gaps_in_miR



















