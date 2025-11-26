"""
Scripts for fitting position-specific Smith-Waterman alignments
through a logistic link.
"""
import numpy as np
from Bio.Seq import Seq
from sklearn.metrics import average_precision_score
from scipy.optimize import minimize
from .shared_global_vars import NUCL, NUCL_DICT

### Likelihood and (sub)gradients 
def logit_logl(scores_pos, scores_neg, alpha):
    scores_pos = np.array(scores_pos)
    scores_neg = np.array(scores_neg)
    pos_part = np.log(1 + np.exp(-alpha - scores_pos))
    neg_part = np.log(1 + np.exp(alpha + scores_neg))
    return -np.sum(pos_part) - np.sum(neg_part)

def logit_lhood_vect(scores, alpha):
    return 1/(1 + np.exp(-alpha - scores))

def logit_derivative_alpha(scores_pos, scores_neg, alpha):
    pos_part = 1/(1 + np.exp(alpha + scores_pos))
    neg_part = 1/(1 + np.exp(-alpha - scores_neg))
    return np.sum(pos_part) - np.sum(neg_part)

def logit_hess_alpha(scores_pos, scores_neg, alpha):
    pos_part = np.exp(alpha + scores_pos)/(1 + np.exp(alpha + scores_pos))**2
    neg_part = np.exp(-alpha - scores_neg)/(1 + np.exp(-alpha - scores_neg))**2
    return -np.sum(pos_part) - np.sum(neg_part)

def logit_subderivative_theta(scores_pos, scores_neg, X_pos, X_neg, alpha, theta_name):
    pos_part = X_pos[theta_name]/(1 + np.exp(alpha + scores_pos))
    neg_part = X_neg[theta_name]/(1 + np.exp(-alpha - scores_neg))
    return np.sum(pos_part) - np.sum(neg_part)

def logit_subhessian_theta(scores_pos, scores_neg, X_pos, X_neg, alpha, theta1_name, theta2_name):
    pos_part = X_pos[theta1_name]*X_pos[theta2_name]*np.exp(alpha + scores_pos)/(1 + np.exp(alpha + scores_pos))**2
    neg_part = X_neg[theta1_name]*X_neg[theta2_name]*np.exp(-alpha - scores_neg)/(1 + np.exp(-alpha - scores_neg))**2
    return -np.sum(pos_part) - np.sum(neg_part)


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
    subgradient_dict = {c: logit_subderivative_theta(scores_pos, 
                                             scores_neg, 
                                             X_pos, X_neg, 
                                             alpha, c) for c in X_pos.columns}
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
             alignment_encoder,
             step_function,
             miRNA_length = 22,
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
        
        X_train = alignment_encoder(alignments_train, train.index)
        
        X_pos = X_train[train['label']==1]
        X_neg = X_train[train['label']==0]
        
        scores = np.array([x[0] for x in alignments_train])
        scores_pos = scores[train['label']==1]
        scores_neg = scores[train['label']==0]
        train_scores_aligner.append(average_precision_score(train['label'],
                                                            scores))
        curr_logl = logit_logl(scores_pos, scores_neg, alpha)
        train_loglik_values.append(curr_logl)
        if verbose:
            print('Current loglik:', curr_logl)
        
        # for test score monitoring - optional
        if test is not None:
            alignments_test = []
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
            test_logl = logit_logl(test_scores_pos, test_scores_neg, alpha)
            test_loglik_values.append(test_logl)

        
        # Step 1: optimizing alpha - finding the exact value
        alpha_target = lambda x: -logit_logl(scores_pos, scores_neg, x)
        alpha_fprime = lambda x: -logit_derivative_alpha(scores_pos,
                                                         scores_neg, x)
        alpha_fsec   = lambda x: -logit_hess_alpha(scores_pos, scores_neg, x)
        alpha = minimize(alpha_target,
                         alpha,
                         jac=alpha_fprime,
                         hess=alpha_fsec,
                         method='Newton-CG')['x'][0]
        if verbose:
            print("Updated alpha:", alpha)
        
        # Step 2: optimizing matrices - step change
        theta_subdiv = {c: logit_subderivative_theta(scores_pos, 
                                             scores_neg, 
                                             X_pos, X_neg, 
                                             alpha, c) for c in X_train.columns}
        subgradient_norm = np.linalg.norm([theta_subdiv[c] for c in theta_subdiv])
        subgradient_norm_values.append(subgradient_norm)
        if verbose:
            print('Gradient norm:', subgradient_norm)
        step = step_function(iter_nb,
                             scores_pos, scores_neg, 
                             X_pos, X_neg, 
                             alpha,
                             verbose)
        G_miR += step['G_miR']
        G_gene += step['G_gene']
        M += step['M']
    return {'G_miR': G_miR, 'G_gene': G_gene, 'M': M, 'alpha': alpha,
            'train_scores': train_scores_aligner,
            'test_scores': test_scores_aligner,
            'train_LogLik_trajectory': train_loglik_values,
            'test_LogLik_trajectory': test_loglik_values,
            'LogLik_final': train_loglik_values[-1],
            'subgradient_norm_values': subgradient_norm_values,
            'Final_alignments_train': alignments_train,
            'Final_alignments_test': alignments_test}

    
