"""
Scripts for fitting position-specific Smith-Waterman alignments
through a logistic link.
"""
import numpy as np
from Bio.Seq import Seq
from sklearn.metrics import average_precision_score
from scipy.optimize import minimize
from .shared_global_vars import NUCL, NUCL_DICT
from .likelihood_and_subgradients import logit_logl, logit_derivative_alpha, logit_derivative_label_probs
from .optimization_functions import logreg_starting_point
from math import ceil

def _align_pair_chunk(pair_chunk, aligner, M, G_miR, G_gene):
    """Align a chunk of sequence pairs in one joblib task.

    Dispatching one pair per task creates substantial scheduling overhead when
    this function is called at every optimization iteration. Chunking keeps the
    returned alignment order stable while reducing per-iteration joblib overhead.
    """
    alignments = []
    for miR, gene in pair_chunk:
        alignments.append(
            aligner(
                    miR, gene,
                    M, G_miR, G_gene,
                    backtrack=True
                    )
                          )
    return alignments


def _pair_chunks(mirna_list, gene_list, chunk_size):
    pair_count = len(mirna_list)
    for start in range(0, pair_count, chunk_size):
        stop = min(start + chunk_size, pair_count)
        yield zip(mirna_list[start:stop], gene_list[start:stop])


def miRAlign(mirna_list, gene_list, label_list,
             aligner, step_function,
             M_prior = None,
             G_miR_prior = None,
             G_gene_prior = None,
             prior_precision = 0,
             label_prior = None,
             MAX_ITER=100, tol=1e-3,
             num_threads=1,
             verbose=False):
    """
    A function to fit a position-aware aligner to miRNA-target interaction data.

    ----
    Parameters:
    mirna_list: list of miRNA sequences.
        All sequences need to have the same length equal n > 0.
    gene_list: list of target sequences
    label_list: list of binary labels (0/1 or False/True) 
    aligner: An alignment function from positional_alignment.py.
    step_function: A function to calculate optimization step lengths.
    M_prior: prior on the substitution matrix, with shape nx16x16, n=length of miRNA
    G_miR_prior: prior on gap penalties between miRNA nucleotides, length n-1
    G_gene_prior: prior on miRNA bulges (gaps in gene), length n
    prior_precision: inverse of the standard deviation of prior distribution,
        often referred to as "lambda"
    label_prior: 2x2 matrix of prior parameters for mislabeling probabilities.
        label_prior[i, j] is the number of "previously observed" instances with
        true label i and observed label j.  
    
    ----
    Returns:
    Substution matrix and gap penalties.
    """
    miRNA_length = set(len(mirna) for mirna in mirna_list)
    assert len(miRNA_length) == 1, 'miRNA sequences need to have the same length'
    miRNA_length = miRNA_length.pop()
    assert len(mirna_list) == len(gene_list)
    assert len(mirna_list) == len(label_list)
    pair_count = len(mirna_list)

    label_list = np.array(label_list)

    if label_prior is not None:
        label_probs = label_prior/np.sum(label_prior, axis=1, keepdims=True)
    else:
        label_probs = None

        
    # Starting parameters. We use up to 10 000 randomly selected
    # data points, because this doesn't need to be very accurate
    starting_point= logreg_starting_point(mirna_list, gene_list, label_list,
                                          match_weight = 5,
                                          mismatch_weight = -4,
                                          gap_weight = -6,
                                          num_threads=num_threads,
                                          max_sample_size=10000)
    M = starting_point['M']
    G_miR = starting_point['G_miR']
    G_gene = starting_point['G_gene']
    alpha = starting_point['alpha']
    if verbose:
        print('Initial alpha:', alpha)

    # Initialize multiprocessing
    if num_threads > 1:
        from joblib import Parallel, delayed
        n_jobs = min(int(num_threads), pair_count)
        # Aim for a few chunks per worker so the work is balanced, without creating
        # one joblib task per sequence pair at every fitting iteration.
        chunk_size = max(1, ceil(pair_count / (n_jobs * 4)))
        if verbose:
            print('Chunk size:', chunk_size)
        parallel = Parallel(n_jobs=n_jobs, prefer="processes", return_as="list")
        # Force numba JIT compile
        _ = aligner(
            mirna_list[0],
            gene_list[0],
            M,
            G_miR,
            G_gene,
            backtrack=True
        )

    # Lists to track scores during optimization:
    auprc_trajectory = []
    loglik_trajectory = []
    subgradient_norm_trajectory = []
    
    # Optimizing:
    for iter_nb in range(MAX_ITER):
        if verbose: print('Iter', iter_nb+1)

        # Align the sequences
        if num_threads > 1:
            chunked_alignments = parallel(
                delayed(_align_pair_chunk)(pair_chunk, aligner, M, G_miR, G_gene)
                for pair_chunk in _pair_chunks(mirna_list, gene_list, chunk_size)
            )
            alignments = [aln for chunk in chunked_alignments for aln in chunk]
        else:
            alignments = []
            for miR, gene in zip(mirna_list, gene_list):   
                aln = aligner(
                    miR, gene,
                    M, G_miR, G_gene,
                    backtrack=True
                    )
                alignments.append(aln)

        # Calculate the current AUPRC and loglik
        scores = np.array([x[0] for x in alignments])
        scores_pos = scores[label_list==1]
        scores_neg = scores[label_list==0]
        auprc_trajectory.append(average_precision_score(label_list, scores))
        curr_logl = logit_logl(scores_pos, scores_neg, alpha,
               label_observation_parameters = label_prior,
               label_observation_probs = label_probs,
               M = M,
               G_miR = G_miR,
               G_gene = G_gene,
               M_prior = M_prior,
               G_miR_prior = G_miR_prior,
               G_gene_prior = G_gene_prior,
               lbd = prior_precision)
        loglik_trajectory.append(curr_logl)
        if verbose:
            print('Current loglik:', curr_logl)

        # Optimization steps        
        # Step 1: optimizing alpha 
        def alpha_target(x):
            """
            Target function for optimization of the intercept parameter alpha.
            x = float, alpha value to inspect
            """
            return -logit_logl(scores_pos, scores_neg,
                               x,
                               label_observation_parameters = label_prior,
                               label_observation_probs = label_probs,
                               M = M,
                               G_miR = G_miR,
                               G_gene = G_gene,
                               M_prior = M_prior,
                               G_miR_prior = G_miR_prior,
                               G_gene_prior = G_gene_prior,
                               lbd = prior_precision)

        def alpha_fprime(x):
            return  -logit_derivative_alpha(scores_pos, scores_neg,
                                            x,
                                            label_probs)
        alpha = minimize(alpha_target,
                         alpha,
                         jac=alpha_fprime,
                         tol=1e-3)
        if alpha.success is False:
            raise RuntimeError('Estimation of the intercept failed. Try decreasing the step size. If the problem persists, let me know about this.')
        else:
            alpha = alpha['x'][0]
        if verbose:
            print("Updated alpha:", alpha)

        # Step 2: optimizing the probabilities of observing correct labels
        if label_prior is not None:
            # Step 2.1: Get the probabilities of observing correct labels
            # and do a logit transformation to use unconstrained optimization methods
            z0 = np.diag(label_probs)
            z0 = np.log(z0/(1-z0))
            
            # Step 2.2: Define target functions
            def eta_target(z):
                """
                Target for optimization of label observation probabilities;
                z = vector of logit-transformed probabilities of correct labels
                """
                x = 1/(1+np.exp(-z))
                prob_array = np.array([[x[0], 1-x[0]], [1-x[1], x[1]]])
                return -logit_logl(scores_pos=scores_pos,
                                   scores_neg=scores_neg,
                                   alpha=alpha,
                                   label_observation_parameters = label_prior,
                                   label_observation_probs = prob_array,
                                   M = M,
                                   G_miR = G_miR,
                                   G_gene = G_gene,
                                   M_prior = M_prior,
                                   G_miR_prior = G_miR_prior,
                                   G_gene_prior = G_gene_prior,
                                   lbd = prior_precision)

            def eta_fprime(z):
                """
                Jacobian for optimization of label observation probabilities;
                z = vector of logit-transformed probabilities of correct labels
                """
                x = 1/(1+np.exp(-z))
                prob_array = np.array([[x[0], 1-x[0]], [1-x[1], x[1]]])
                dL_dx = -logit_derivative_label_probs(scores_pos=scores_pos,
                                   scores_neg=scores_neg,
                                   alpha=alpha,
                                   label_observation_parameters = label_prior,
                                   label_observation_probs = prob_array)
                dx_dz = x*(1-x)
                return dL_dx * dx_dz
            
            # Step 2.3: optimize the logit-transformed probabilities
            z_star = minimize(eta_target,
                             z0,
                             jac=eta_fprime,
                             tol=1e-3)
            if z_star.success is False:
                raise RuntimeError('Estimation of label probabilities failed. Try decreasing the step size. If the problem persists, let me know about this.')
            else:
                z_star = z_star['x']
            # Step 2.4: Transform back to a vector of probabilities of
            # observing correct labels, fill the label_prob array
            correct_label_probs = 1/(1+np.exp(-z_star))
            label_probs = np.array([[correct_label_probs[0], 1-correct_label_probs[0]],
                                    [1-correct_label_probs[1], correct_label_probs[1]]])
            if verbose:
                print("Updated probs of correct labels:", correct_label_probs)
                
        # Step 3: optimizing matrices - subgradient step        
        step_theta = step_function(iter_nb = iter_nb,
                             posloc_alignments = alignments, 
                             labels = label_list,
                             alpha = alpha,
                             verbose = verbose,
                             current_M = M,
                             current_G_miR = G_miR,
                             current_G_gene = G_gene,
                             M_prior=M_prior,
                             G_miR_prior=G_miR_prior,
                             G_gene_prior=G_gene_prior,
                             lbd=prior_precision,
                             label_observation_probs = label_probs
                             )
        G_miR += step_theta['G_miR_step']
        G_gene += step_theta['G_gene_step']
        M += step_theta['M_step']
            
        M_subgradient_norm = np.linalg.norm(step_theta['M_subgradient'])
        G_miR_subgradient_norm = np.linalg.norm(step_theta['G_miR_subgradient'])
        G_gene_subgradient_norm = np.linalg.norm(step_theta['G_gene_subgradient'])
        subgradient_norm_trajectory.append(np.sqrt(M_subgradient_norm**2 + G_miR_subgradient_norm**2 + G_gene_subgradient_norm**2))
        if verbose:
            print('M subgradient norm:', M_subgradient_norm)
            print('G_miR subgradient norm:', G_miR_subgradient_norm)
            print('G_gene subgradient norm:', G_gene_subgradient_norm)
            
    return {'G_miR': G_miR, 'G_gene': G_gene, 'M': M, 'alpha': alpha,
            'auprc_trajectory': auprc_trajectory,
            'loglik_trajectory': loglik_trajectory,
            'final_loglik': loglik_trajectory[-1],
            'subgradient_norm_trajectory': subgradient_norm_trajectory,
            'final_alignments': alignments,
            'label_observation_probs': label_probs}


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


