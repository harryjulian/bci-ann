import numpy as np
import numpy.matlib as ml
import numba
from itertools import product
from scipy.stats import multinomial
from skopt.optimizer import gp_minimize
from skopt.space import Real

# Utility Functions

def get_conditions(possible_locations, variance_conditions):
    return list(product(possible_locations, possible_locations, variance_conditions))

def binner(arr, bins):
    bin_centers = (bins[:-1] + bins[1:])/2 
    idx = np.digitize(arr, bin_centers)
    result = bins[idx]
    return result

def counter(arr, possible_locations):
    l = arr.tolist()
    return np.array([l.count(i) for i in possible_locations])

def clip(arr, roundup = 0.001): 
    arr = np.clip(arr, 0.001, np.inf)
    return arr

def multinomial_nll():

    pass

# Model Internals

#@numba.njit
def get_samples(N, vloc, aloc, pCommon, sigV, varV, sigA, varA, sigP, varP):
    return sigV * np.random.randn(N,1) + vloc, sigA * np.random.randn(N,1) + aloc

#@numba.njit
def calculate_likelihood_c1(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP):
    # likelihood P(Xv, Xa|C =1)
    
    firstDenom = 2*np.pi*np.sqrt(varV*varA + varV*varP +varA*varP)
    firstTerm = 1/firstDenom 
    secondNum = (Xv - Xa)**2 * varP + (Xv -0)**2 * varA + (Xa - 0)**2* varV 
    secondDenom = (varV * varA) + (varV * varP) + (varA * varP)
    secondTerm = np.exp((-0.5*(secondNum/secondDenom)))
    likelihood_com = firstTerm*secondTerm

    return likelihood_com

#@numba.njit
def calculate_likelihood_c2(Xv,Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP):
    # likelihood P(Xv, Xa|C =2)
    
    firstTerm = 2*np.pi*np.sqrt((varV + varP)*(varA+varP))
    secondTerm1 = (Xv - 0)**2/(varV + varP)
    secondTerm2 = (Xa - 0)**2 / (varA + varP)
    secondTermFull = np.exp((-0.5*(secondTerm1+secondTerm2)) )
    likelihood_ind = secondTermFull/firstTerm

    return likelihood_ind

#@numba.njit
def calculate_posterior(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP):
    # p(C = 1|Xv,Xa) posterior
    
    likelihood_common = calculate_likelihood_c1(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)
    likelihood_ind = calculate_likelihood_c2(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)
    post_common = likelihood_common * pCommon 
    post_indep = likelihood_ind * (1-pCommon)
    posterior = post_common/(post_common +post_indep)

    return posterior

def opt_position_conditionalised_C1(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP):
    # Get optimal location given C = 1
    
    cues = Xv/varV + Xa/varA + ml.repmat(pCommon,N,1)/varP
    inverseVar = 1/varV + 1/varA + 1/varP
    sHatC1 = cues/inverseVar

    return sHatC1

def opt_position_conditionalised_C2(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP):
        # Get optimal locationS given C = 2
        
    visualCue = Xv/varV +ml.repmat(pCommon,N,1)/varP
    visualInvVar = 1/varV + 1/ varP
    sHatVC2 = visualCue/visualInvVar
    audCue = Xa/varA + ml.repmat(pCommon,N,1)/varP
    audInvVar = 1/varA + 1/ varP
    sHatAC2 = audCue/audInvVar

    return sHatVC2, sHatAC2

def optimal_visual_location(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP):
    # Use Model Averaging to compute final visual est
    
    posterior_1C = calculate_posterior(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)
    sHatVC1 = opt_position_conditionalised_C1(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)
    sHatVC2 = opt_position_conditionalised_C2(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)[0]
    sHatV = posterior_1C*sHatVC1 + (1-posterior_1C)*sHatVC2 #model averaging

    return sHatV

def optimal_aud_location(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP):
    # Use Model Averaging to compute final auditory est
    
    posterior_1C = calculate_posterior(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)
    sHatAC1 = opt_position_conditionalised_C1(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)
    sHatAC2 = opt_position_conditionalised_C2(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)[1]
    sHatA = posterior_1C*sHatAC1 + (1-posterior_1C)*sHatAC2 #model averaging
    return sHatA

# Write full Model Wrapper

def bciobjective(pars):

    """
        Parameters --> np.array(dtype=object) of the following:
                       pCommon, sigV, sigA, sigP,
                       data, conditions, possible_locations N (in this order).
    """

    pCommon, sigV, sigA, sigP, data, conditions, possible_locations, N = pars #unpack array
    varV, varA, varP = sigV**2, sigA**2, sigP**2
    nLL = 0

    for cond in conditions:
        d, vloc, aloc, variance = data[cond], cond[0], cond[1], cond[2]
        Xv, Xa = get_samples(N, vloc, aloc, pCommon, sigV, varV, sigA, varA, sigP, varP)
        sHatA = optimal_aud_location(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)
        sHatAbin = clip(binner(sHatA, possible_locations))
        sHatAcount = counter(sHatAbin, possible_locations)
        ll = multinomial.logpmf(d, np.sum(d), sHatAcount)
        nLL += ll

    return nLL
    
def bcifit(data, conditions):

    # Setup paramater space
    searchspace = [Real(0.1, 0.7), Real(1, 20), Real(1, 20), Real(1, 20)]

    # Run Optimizer
    res = gp_minimize(bciobjective, dimensions=searchspace, n_initial_points=100, 
                    initial_point_generator='lhs', noise='gaussian')

    return res

def bcirecompute():

    pass