"""
Scripts for fitting position-specific Smith-Waterman alignments
through a logistic link.
"""
import numpy as np
from Bio.Seq import Seq
from sklearn.metrics import average_precision_score
from scipy.optimize import minimize
from .shared_global_vars import NUCL, NUCL_DICT
from .alignment_table_encoding import encode_posloc_MGG
from numba import njit


### Likelihood and (sub)gradients 
def logit_logl(scores_pos, scores_neg, alpha,
               label_observation_parameters = None,
               label_observation_probs = None,
               M = None,
               G_miR = None,
               G_gene = None,
               M_prior = None,
               G_miR_prior = None,
               G_gene_prior = None,
               lbd = 0):
    """
    Returns the log likelihood up to a proportionality constant
    """
    l = 0
    # Parameter priors (gaussian regularization), up to a constant: 
    if M_prior is not None and M is not None:
        l -= lbd*np.sum((M - M_prior)**2)
    if G_miR_prior is not None and G_miR is not None:
        l -= lbd*np.sum((G_miR - G_miR_prior)**2)
    if G_gene_prior is not None and G_gene is not None:
        l -= lbd*np.sum((G_gene - G_gene_prior)**2)

    # Label prop priors, up to a constant
    if label_observation_parameters is not None:
        l += np.sum(label_observation_parameters*np.log(label_observation_probs))

    # Likelihood
    if label_observation_parameters is not None:
        pos_part = label_observation_probs[1,1]/(1+np.exp(-alpha-scores_pos))
        pos_part += label_observation_probs[0,1]/(1+np.exp(alpha+scores_pos))
        pos_part = np.log(pos_part)
        neg_part = label_observation_probs[1,0]/(1+np.exp(-alpha-scores_neg))
        neg_part += label_observation_probs[0,0]/(1+np.exp(alpha+scores_neg))
        neg_part = np.log(neg_part)
        l += np.sum(pos_part) + np.sum(neg_part)
    else:
        pos_part = -np.log(1 + np.exp(-alpha - scores_pos)) # minus takes care of reversing the sigmoid
        neg_part = -np.log(1 + np.exp(alpha + scores_neg))
        l += np.sum(pos_part)  
        l += np.sum(neg_part)
    return l

def logit_lhood_vect(scores, alpha):
    return 1/(1 + np.exp(-alpha - scores))
            
def logit_derivative_alpha(scores_pos, scores_neg, alpha,
                           label_observation_probs = None):
    if label_observation_probs is None:
        pos_part = 1/(1 + np.exp(alpha + scores_pos))
        neg_part = 1/(1 + np.exp(-alpha - scores_neg))
        alpha_prime = np.sum(pos_part) - np.sum(neg_part) # note: subtraction
    else:
        pos_factor = label_observation_probs[1, 1] - label_observation_probs[0, 1]
        neg_factor = label_observation_probs[1, 0] - label_observation_probs[0, 0]
        pos_sum_inv = label_observation_probs[0, 1]*(1+np.exp(-alpha-scores_pos))
        pos_sum_inv += label_observation_probs[1, 1]*(1+np.exp(alpha+scores_pos))
        pos_sum = sum(1/pos_sum_inv)
        pos_sum *= pos_factor
        neg_sum_inv = label_observation_probs[0, 0]*(1+np.exp(-alpha-scores_neg))
        neg_sum_inv += label_observation_probs[1, 0]*(1+np.exp(alpha+scores_neg))
        neg_sum = sum(1/neg_sum_inv)
        neg_sum *= neg_factor 
        alpha_prime = pos_sum + neg_sum  # note: addition
    return alpha_prime

def logit_hess_alpha(scores_pos, scores_neg, alpha):
    pos_part = np.exp(alpha + scores_pos)/(1 + np.exp(alpha + scores_pos))**2
    neg_part = np.exp(-alpha - scores_neg)/(1 + np.exp(-alpha - scores_neg))**2
    return -np.sum(pos_part) - np.sum(neg_part)

def logit_derivative_label_props(scores_pos, scores_neg,
                           alpha,
                           label_observation_parameters,
                           label_observation_probs):
    """
    Returns a derivative of the logit score function
    with respect to the probabilities of observing
    correct labels (eta_00 and eta_11).
    """
    d_00_prior = label_observation_parameters[0,0]/label_observation_probs[0,0]
    d_00_prior -= label_observation_parameters[0,1]/label_observation_probs[0,1]
    d_00_neg = 1/(1+np.exp(alpha+scores_neg))
    d_00_neg /= (label_observation_probs[0,0]/(1+np.exp(-alpha-scores_neg)) + label_observation_probs[1,0]/(1+np.exp(alpha + scores_neg)))
    d_00_pos = -1/(1+np.exp(alpha+scores_pos))
    d_00_pos /= (label_observation_probs[0,1]*(1+np.exp(-alpha-scores_pos)) + label_observation_probs[1,1]*(1+np.exp(alpha + scores_pos)))
    d_00 = d_00_prior + sum(d_00_neg) + sum(d_00_pos)

    d_11_prior = label_observation_parameters[1,1]/label_observation_probs[1,1]
    d_11_prior -= label_observation_parameters[1,0]/label_observation_probs[1,0]
    d_11_neg = -1/(1+np.exp(-alpha-scores_neg))
    d_11_neg /= (label_observation_probs[1,0]/(1+np.exp(-alpha-scores_neg)) + label_observation_probs[0,0]/(1+np.exp(alpha + scores_neg)))
    d_11_pos = 1/(1+np.exp(-alpha-scores_pos))
    d_11_pos /= (label_observation_probs[1,1]/(1+np.exp(-alpha-scores_pos)) + label_observation_probs[0,1]/(1+np.exp(alpha + scores_pos)))
    d_11 = d_11_prior + sum(d_11_neg) + sum(d_11_pos)
    return np.array([d_00, d_11])


def logit_partial_subderivative_theta(theta_value,
                              scores_pos, scores_neg,
                              counts_pos, counts_neg,
                              alpha,
                              prior_value = 0,
                              lbd = 0):
    pos_part = counts_pos/(1 + np.exp(alpha + scores_pos))
    neg_part = counts_neg/(1 + np.exp(-alpha - scores_neg))
    return -2*lbd*(theta_value - prior_value) + np.sum(pos_part) - np.sum(neg_part)


def logit_subderivative_theta(alignments, labels, alpha,
                              label_observation_probs = None):
    """
    Return a subderivative of the logistic link function
    with respect to all parameters of the positional alignment:
    M, G_miR and G_gene.
    mislab_prob: 4x4 array with rows summable to 1 indicating
    probabilities of observed labels; mislab_prob[i, j] is the probability
    of true label i observed as j
    """
    M_subd = np.zeros((4, 4, 22))
    G_miR_subd = np.zeros(21)
    G_gene_subd = np.zeros(22)
    for aln, lab in zip(alignments, labels):
        s, aligned_miR, aligned_mR, crd_miR, crd_mR  = aln
        if lab == 1:
            if label_observation_probs is None:
                scaling_factor = 1/(1+np.exp(alpha+s))
            else:
                # note: denominator calculated first then reversed
                scaling_factor = label_observation_probs[0,1]*(1+np.exp(-alpha-s))
                scaling_factor += label_observation_probs[1, 1]*(1+np.exp(alpha+s))
                scaling_factor = (label_observation_probs[1,1] - label_observation_probs[0,1])/scaling_factor
        elif lab == 0:
            if label_observation_probs is None:
                scaling_factor = -1/(1+np.exp(-alpha - s))
            else:
                scaling_factor = label_observation_probs[0,0]*(1+np.exp(-alpha-s))
                scaling_factor += label_observation_probs[1, 0]*(1+np.exp(alpha+s))
                scaling_factor = (label_observation_probs[1,0] - label_observation_probs[0,0])/scaling_factor
        else:
            raise ValueError('Incorrect label - needs to be 0 or 1')
        last_mir_crd = -1 # last seen crd before gapped region in miR
        mir_gap_count = 0 # length of the current gapped region in miR
        for aln_crd, mir_crd in enumerate(crd_miR):
            if mir_crd != -1: # not a gap in miR
                if last_mir_crd > -1:  # we're inside the alignment
                    G_miR_subd[last_mir_crd] += mir_gap_count*scaling_factor
                last_mir_crd = mir_crd
                mir_gap_count = 0
                nt1 = aligned_miR[aln_crd]
                nt2 = aligned_mR[aln_crd]
                assert nt1 != '-'
                if nt2 == '-':
                    G_gene_subd[mir_crd] += scaling_factor
                else:
                    nt1id = NUCL_DICT[nt1]
                    nt2id = NUCL_DICT[nt2]
                    M_subd[nt1id, nt2id, mir_crd] += scaling_factor
            else:
                mir_gap_count += 1
    return(M_subd, G_miR_subd, G_gene_subd)

@njit
def logit_subderivative_theta_numba(alignments, labels, alpha):
    """
    Return a subderivative of the logistic link function
    with respect to all parameters of the positional alignment:
    M, G_miR and G_gene.
    """
    global NUCL
    M_subd = np.zeros((4, 4, 22))
    G_miR_subd = np.zeros(21)
    G_gene_subd = np.zeros(22)
    for aln, lab in zip(alignments, labels):
        s, aligned_miR, aligned_mR, crd_miR, crd_mR  = aln
        if lab == 1:
            scaling_factor = 1/(1+np.exp(alpha+s))
        elif lab == 0:
            scaling_factor = -1/(1+np.exp(-alpha - s))
        else:
            raise ValueError('Incorrect label - needs to be 0 or 1')
        last_mir_crd = -1 # last seen crd before gapped region in miR
        mir_gap_count = 0 # length of the current gapped region in miR
        for aln_crd, mir_crd in enumerate(crd_miR):
            if mir_crd != -1: # not a gap in miR
                if last_mir_crd > -1:  # we're inside the alignment
                    G_miR_subd[last_mir_crd] += mir_gap_count*scaling_factor
                last_mir_crd = mir_crd
                mir_gap_count = 0
                nt1 = aligned_miR[aln_crd]
                nt2 = aligned_mR[aln_crd]
                assert nt1 != '-'
                if nt2 == '-':
                    G_gene_subd[mir_crd] += scaling_factor
                else:
                    nt1id = NUCL.index(nt1)
                    nt2id = NUCL.index(nt2)
                    M_subd[nt1id, nt2id, mir_crd] += scaling_factor
            else:
                mir_gap_count += 1
    return(M_subd, G_miR_subd, G_gene_subd)

### 
def logit_subhessian_theta(theta1_value,
                           theta2_value,
                           scores_pos, scores_neg,
                           x1_counts_pos, x1_counts_neg,
                           x2_counts_pos, x2_counts_neg,
                           alpha,
                           lbd = 0):
    pos_part = x1_counts_pos*x2_counts_pos*np.exp(alpha + scores_pos)/(1 + np.exp(alpha + scores_pos))**2
    neg_part = x1_counts_neg*x2_counts_neg*np.exp(-alpha - scores_neg)/(1 + np.exp(-alpha - scores_neg))**2
    return -2*lbd - np.sum(pos_part) - np.sum(neg_part)


### Functions for calculating steps in optimization
def _subgradient_descent_step_fullM(step_factor, step_power,
                                    power_offset,  # iter nb for power to kick in
                                    iter_nb,
                                    scores_pos, scores_neg, 
                                    X_pos, X_neg, 
                                    alpha,
                                    verbose):
    """
    A subgradient step to optimize the model with full positional
    substitution matrix and constant gap penalties
    (rather than e.g. match/mismatch model)
    """
    subgradient_dict = {c: logit_subderivative_theta(0, scores_pos, 
                                             scores_neg, 
                                             np.array(X_pos[c]), np.array(X_neg[c]), 
                                             alpha) for c in X_pos.columns}
    if iter_nb > power_offset:
        stepsize = step_factor/(iter_nb-power_offset)**step_power
    else:
        stepsize = step_factor
    G_miR_step = stepsize*subgradient_dict['Gap_miR']
    G_gene_step = stepsize*subgradient_dict['Gap_gene']
    M_step = np.zeros((4, 4, 22))
    for parname in subgradient_dict:
        if parname == 'Gap_miR' or parname == 'Gap_gene':
            continue
        else:
            miR_nt, pair = parname.split('_')
            miR_nt = int(miR_nt)
            pair = tuple(pair)
            pair = [NUCL_DICT[p] for p in pair]
            M_step[pair[0], pair[1], miR_nt] = stepsize*subgradient_dict[parname]
    return {'G_miR': G_miR_step, 'G_gene': G_gene_step, 'M': M_step}

def create_subgradient_step_fullM(step_factor, step_power, power_offset):
    def stepfunction(iter_nb,
                     scores_pos, scores_neg, 
                     X_pos, X_neg,
                     alpha,
                     verbose):
        return _subgradient_descent_step_fullM(step_factor, step_power,
                                               power_offset,
                                               iter_nb,
                                                scores_pos, scores_neg, 
                                                X_pos, X_neg,
                                                alpha,
                                                verbose)
    return stepfunction

def _subgradient_descent_step(
    step_factor, step_power,
    power_offset,  # iter nb for power to kick in
    iter_nb,
    posloc_alignments,
    labels,
    alpha,
    #M_var_groups = None,
    #G_miR_groups = None,
    #G_gene_groups = None,
    current_M = None,  # for regularization
    current_G_miR = None,
    current_G_gene = None,
    M_prior = None,
    G_miR_prior = None,
    G_gene_prior = None,
    lbd = 0,
    label_observation_probs = None,
    verbose=False):
    """
    A subgradient step to optimize the model with customizable
    groups of variables to encode constraints,
    such as equal mismatch penalties or constant gap penalties. 
    """
##    # Move this outside: 
##    if M_var_groups is not None:
##        M_group_ids = np.unique(M_var_groups)
##    else:
##        M_group_ids = np.array(list(range(4*4*22)))
##        M_var_groups = np.arange(4*4*22, dtype='int').reshape((4, 4, 22))
##        
##    if G_miR_groups is not None:
##        G_miR_group_ids = np.unique(G_miR_groups)
##    else:
##        G_miR_group_ids = np.arange(21, dtype='int')
##        G_miR_groups = np.arange(21, dtype='int')
##
##    if G_gene_groups is not None:
##        G_gene_group_ids = np.unique(G_gene_groups)
##    else:
##        G_gene_group_ids = np.arange(22, dtype='int')
##        G_gene_groups = np.arange(22, dtype='int')
##
    M_subdiv, G_miR_subdiv, G_gene_subdiv = logit_subderivative_theta(posloc_alignments,
                                                                      labels,
                                                                      alpha,
                                                                      label_observation_probs)

    # Add priors on alignment parameters
    if current_M is not None and M_prior is not None:
        M_subdiv -= 2*lbd*(current_M - M_prior)
    if current_G_miR is not None and G_miR_prior is not None:
        G_miR_subdiv -= 2*lbd*(current_G_miR - G_miR_prior)
    if current_G_gene is not None and G_gene_prior is not None:
        G_gene_subdiv -= 2*lbd*(current_G_gene - G_gene_prior)

    if iter_nb > power_offset:
        stepsize = step_factor/(iter_nb-power_offset)**step_power
    else:
        stepsize = step_factor

    M_step = M_subdiv*stepsize
    G_miR_step = G_miR_subdiv*stepsize
    G_gene_step = G_gene_subdiv*stepsize
    
    return {'G_miR_step': G_miR_step,
            'G_gene_step': G_gene_step,
            'M_step': M_step,
            'G_miR_subderivative': G_miR_subdiv,
            'G_gene_subderivative': G_gene_subdiv,
            'M_subderivative': M_subdiv}
    
def create_subgradient_step(step_factor, step_power, power_offset):
    def stepfunction(iter_nb,
                     posloc_alignments, labels,
                     alpha,
                     current_M = None,  # for regularization
                     current_G_miR = None,
                     current_G_gene = None,
                     M_prior = None,
                     G_miR_prior = None,
                     G_gene_prior = None,
                     lbd = 0,
                     label_observation_probs = None,
                     verbose = False):
        return _subgradient_descent_step(step_factor = step_factor,
                                         step_power = step_power,
                                         power_offset = power_offset,
                                         iter_nb = iter_nb,
                                         posloc_alignments = posloc_alignments,
                                         labels = labels,
                                         alpha = alpha,
                                         current_M = current_M,  # for regularization
                                         current_G_miR = current_G_miR,
                                         current_G_gene = current_G_gene,
                                         M_prior = M_prior,
                                         G_miR_prior = G_miR_prior,
                                         G_gene_prior = G_gene_prior,
                                         lbd = lbd,
                                         verbose = verbose,
                                         label_observation_probs = label_observation_probs)
    return stepfunction
            
def _newton_rhapson_step_fullM(scores_pos, scores_neg, 
                                    X_pos, X_neg, 
                                    alpha,
                                    verbose):
    column_map = {c: i for i, c in enumerate(X_pos.columns)}
    if verbose:
        print('Calculating subgradient')
    theta_subdiv = [logit_subderivative_theta(scores_pos, 
                                              scores_neg, 
                                              X_pos, X_neg, 
                                              alpha, c) for c in column_map]
    if verbose:
        print('Calculating subhessian')
    theta_subhess = [[logit_subhessian_theta(scores_pos, 
                                             scores_neg, 
                                             X_pos, X_neg, 
                                             alpha, c1, c2) for c1 in column_map] for c2 in column_map]
    theta_subhess = np.array(theta_subhess)
    if verbose:
        print('Subhessian shape', theta_subhess.shape)
    ##  Moore–Penrose pseudoinverse
    step_vector, *_ = np.linalg.lstsq(theta_subhess, theta_subdiv, rcond=None)
    step_vector *= -1
    if verbose:
        print('Step vector norm', np.linalg.norm(step_vector))
    mirgap_col_id = column_map['Gap_miR']
    genegap_col_id = column_map['Gap_gene']
    G_miR_step = step_vector[mirgap_col_id]
    G_gene_step = step_vector[genegap_col_id]
    M_step = np.zeros((4, 4, 22))
    for parname in column_map:
        if parname == 'Gap_miR' or parname == 'Gap_gene':
            continue
        else:
            match_col_id = column_map[parname]
            step_value = step_vector[match_col_id]
            miR_nt, pair = parname.split('_')
            miR_nt = int(miR_nt)
            pair = tuple(pair)
            pair = [NUCL_DICT[p] for p in pair]
            M_step[pair[0], pair[1], miR_nt] = step_value
    return {'G_miR': G_miR_step, 'G_gene': G_gene_step, 'M': M_step}

def create_newton_step_fullM():
    def stepfunction(iter_nb,
                     scores_pos, scores_neg, 
                     X_pos, X_neg,
                     alpha,
                     verbose): 
        return _newton_rhapson_step_fullM(scores_pos, scores_neg, 
                                                X_pos, X_neg,
                                                alpha,
                                                verbose)
    return stepfunction



### Wrapper
def miRAlign(train, test,
             posaware_aligner, 
             step_function,
             miRNA_length = 22,  # not fully implemented yet
             M_prior = None,
             G_miR_prior = None,
             G_gene_prior = None,
             lbd = 0,
             label_observation_parameters = None,
             MAX_ITER=100, tol=1e-3,
             verbose=False):
    """
    A function to fit a position-aware aligner to miRNA-target interaction data.

    ----
    Parameters:
    train: pd.DataFrame, needs to have a format like in miRBench.
    test: None or pd.DataFrame to monitor test score to detect overfitting. 
    posaware_aligner: An alignment function from positional_alignment.py.
    alignment_encoder: A function to encode alignments as data frames. 
    step_function: A function to calculate optimization steps.
    miRNA_length: miRNAs with other lengths are ignored.
    
    ----
    Returns:
    Substution matrix and gap penalties.
    """
    # Extracting data
    train_miR = []
    train_gene = []
    for i, l in train.iterrows():
        mRNA = Seq(l['gene'])
        miR = Seq(l['noncodingRNA'])
        if len(miR) != miRNA_length:
            continue
        mRNA = mRNA.reverse_complement()
        train_miR.append(miR)
        train_gene.append(mRNA)

    if test is not None:
        test_miR = []
        test_gene = []
        for i, l in test.iterrows():
            mRNA = Seq(l['gene'])
            miR = Seq(l['noncodingRNA'])
            if len(miR) != miRNA_length:
                continue
            mRNA = mRNA.reverse_complement()
            test_miR.append(miR)
            test_gene.append(mRNA)

    # Baseline scores estimated elsewhere
    baseline_match_score = 0.724709
    baseline_mismatch_score = -0.647892
    baseline_gap_score = -0.901264
    baseline_alpha = -5.22

    if label_observation_parameters is not None:
        label_observation_probs = label_observation_parameters/np.sum(label_observation_parameters, axis=1)
    else:
        label_observation_probs = None

    # Initial parameters for PosAlign functions:
    M = np.zeros((4, 4, 22))   # third coordinate = nt index
    for i in range(22):
        M[...,i] = np.eye(4)*(baseline_match_score - baseline_mismatch_score)
        M[...,i] += baseline_mismatch_score
    G_miR = np.zeros(21) + baseline_gap_score
    G_gene = np.zeros(22) + baseline_gap_score
    alpha = baseline_alpha
 
    # To track parameter changes:
    previous_M = M.copy()
    previous_G_miR = G_miR.copy()
    previous_G_gene = G_gene.copy()
    previous_alpha = alpha

    # To track scores during optimization:
    train_scores_aligner = []
    test_scores_aligner = []
    train_loglik_values = []
    test_loglik_values = []
    subgradient_norm_values = []
    
    # Optimizing:
    for iter_nb in range(MAX_ITER):
        if verbose: print('Iter', iter_nb+1)

        alignments_train = []
        for miR, gene in zip(train_miR, train_gene):   
            aln = posaware_aligner(
                miR, gene,
                M, G_miR, G_gene,
                backtrack=True
                )
            alignments_train.append(aln)
        
        #X_train = alignment_encoder(alignments_train, train.index)
        
        #X_pos = X_train[train['label']==1]
        #X_neg = X_train[train['label']==0]
        
        scores = np.array([x[0] for x in alignments_train])
        scores_pos = scores[train['label']==1]
        scores_neg = scores[train['label']==0]
        train_scores_aligner.append(average_precision_score(train['label'],
                                                            scores))
        curr_logl = logit_logl(scores_pos, scores_neg, alpha,
               label_observation_parameters = label_observation_parameters,
               label_observation_probs = label_observation_probs,
               M = M,
               G_miR = G_miR,
               G_gene = G_gene,
               M_prior = M_prior,
               G_miR_prior = G_miR_prior,
               G_gene_prior = G_gene_prior,
               lbd = lbd)
        train_loglik_values.append(curr_logl)
        if verbose:
            print('Current loglik:', curr_logl)
        
        # for test score monitoring - optional
        alignments_test = []
        if test is not None:
            for miR, gene in zip(test_miR, test_gene):   
                aln = posaware_aligner(
                    miR, gene,
                    M, G_miR, G_gene,
                    backtrack=True
                    )
                alignments_test.append(aln)
            scores_test = np.array([x[0] for x in alignments_test])
            test_scores_aligner.append(average_precision_score(test['label'],
                                                               scores_test))
            test_scores_pos = scores_test[test['label']==1]
            test_scores_neg = scores_test[test['label']==0]
            test_logl = logit_logl(test_scores_pos, test_scores_neg, alpha,
                               label_observation_parameters = label_observation_parameters,
                               label_observation_probs = label_observation_probs,
                               M = M,
                               G_miR = G_miR,
                               G_gene = G_gene,
                               M_prior = M_prior,
                               G_miR_prior = G_miR_prior,
                               G_gene_prior = G_gene_prior,
                               lbd = lbd)
            test_loglik_values.append(test_logl)

        
        # Step 1: optimizing alpha - finding the exact value
        def alpha_target(x):
            return -logit_logl(scores_pos, scores_neg,
                               x,
                               label_observation_parameters = label_observation_parameters,
                               label_observation_probs = label_observation_probs,
                               M = M,
                               G_miR = G_miR,
                               G_gene = G_gene,
                               M_prior = M_prior,
                               G_miR_prior = G_miR_prior,
                               G_gene_prior = G_gene_prior,
                               lbd = lbd)

        def alpha_fprime(x):
            return  -logit_derivative_alpha(scores_pos, scores_neg,
                                            x,
                                            label_observation_probs)
        alpha = minimize(alpha_target,
                         alpha,
                         jac=alpha_fprime,
                         # hess=alpha_fsec,
                         #method='Newton-CG'
                         )['x'][0]
        if verbose:
            print("Updated alpha:", alpha)
        if label_observation_parameters is not None:
            def eta_target(z):
                """
                Target for label tranisition prob optimization;
                x = 2d array of probs of correct labeling
                """
                x = 1/(1+np.exp(-z))
                prob_array = np.array([[x[0], 1-x[0]], [1-x[1], x[1]]])
                return -logit_logl(scores_pos, scores_neg,
                                   alpha,
                                   label_observation_parameters = label_observation_parameters,
                                   label_observation_probs = prob_array,
                                   M = M,
                                   G_miR = G_miR,
                                   G_gene = G_gene,
                                   M_prior = M_prior,
                                   G_miR_prior = G_miR_prior,
                                   G_gene_prior = G_gene_prior,
                                   lbd = lbd)

            def eta_fprime(z):
                """
                Jacobian for label tranisition prob optimization;
                x = 2d array of probs of correct labeling
                """
                x = 1/(1+np.exp(-z))
                prob_array = np.array([[x[0], 1-x[0]], [1-x[1], x[1]]])
                dL_dx = -logit_derivative_label_props(scores_pos, scores_neg,
                               alpha = alpha,
                               label_observation_parameters = label_observation_parameters,
                               label_observation_probs = prob_array)
                dx_dz = x*(1-x)
                return dL_dx * dx_dz

            z0 = np.diag(label_observation_probs)
            z0 = np.log(z0/(1-z0))
            z_star = minimize(eta_target,
                             z0,
                             jac=eta_fprime
                             # hess=alpha_fsec,
                             #method='Newton-CG'
                             )['x']
            correct_label_probs = 1/(1+np.exp(-z_star))
            if verbose:
                print("Updated correct label probs:", correct_label_probs)
            label_observation_probs = np.array([[correct_label_probs[0], 1-correct_label_probs[0]],
                                                [1-correct_label_probs[1], correct_label_probs[1]]])
            
        # Step 2: optimizing matrices - step change
##        theta_subdiv = {c: logit_partial_subderivative_theta(0,
##                                             scores_pos, 
##                                             scores_neg, 
##                                             np.array(X_pos[c]), np.array(X_neg[c]),
##                                             alpha) for c in X_train.columns}
##        subgradient_norm = np.linalg.norm([theta_subdiv[c] for c in theta_subdiv])
##        subgradient_norm_values.append(subgradient_norm)
        
        step_theta = step_function(iter_nb = iter_nb,
                             posloc_alignments = alignments_train, 
                             labels = train['label'],
                             alpha = alpha,
                             verbose = verbose,
                             current_M = M,
                             current_G_miR = G_miR,
                             current_G_gene = G_gene,
                             M_prior=M_prior,
                             G_miR_prior=G_miR_prior,
                             G_gene_prior=G_gene_prior,
                             lbd=lbd,
                             label_observation_probs = label_observation_probs
                             )
        G_miR += step_theta['G_miR_step']
        G_gene += step_theta['G_gene_step']
        M += step_theta['M_step']
            
        M_subgradient_norm = np.linalg.norm(step_theta['M_subderivative'])
        G_miR_subgradient_norm = np.linalg.norm(step_theta['G_miR_subderivative'])
        G_gene_subgradient_norm = np.linalg.norm(step_theta['G_gene_subderivative'])
        subgradient_norm_values.append(np.sqrt(M_subgradient_norm**2 + G_miR_subgradient_norm**2 + G_gene_subgradient_norm**2))
        if verbose:
            print('M subgradient norm:', M_subgradient_norm)
            print('G_miR subgradient norm:', G_miR_subgradient_norm)
            print('G_gene subgradient norm:', G_gene_subgradient_norm)
            
    return {'G_miR': G_miR, 'G_gene': G_gene, 'M': M, 'alpha': alpha,
            'train_scores': train_scores_aligner,
            'test_scores': test_scores_aligner,
            'train_LogLik_trajectory': train_loglik_values,
            'test_LogLik_trajectory': test_loglik_values,
            'LogLik_final': train_loglik_values[-1],
            'subgradient_norm_values': subgradient_norm_values,
            'Final_alignments_train': alignments_train,
            'Final_alignments_test': alignments_test,
            'label_observation_probs': label_observation_probs}


# Posterior analysis
def get_label_posteriors(scores, labels, alpha,
                           label_observation_probs):
    posterior_0 = []
    posterior_1 = []
    for s, l in zip(scores, labels):
        if l == 0:
            p0 = label_observation_probs[0, 0]/(1+np.exp(alpha+s))
            p1 = label_observation_probs[1, 0]/(1+np.exp(-alpha-s))
            posterior_0.append(p0)
            posterior_1.append(p1)
        else:
            p0 = label_observation_probs[0, 1]/(1+np.exp(alpha+s))
            p1 = label_observation_probs[1, 1]/(1+np.exp(-alpha-s))
            posterior_0.append(p0)
            posterior_1.append(p1)
    posterior = np.array([posterior_0, posterior_1])
    posterior /= np.sum(posterior, axis=0)
    posterior = posterior.T
    return posterior
    
