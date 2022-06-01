from typing import Dict
import numpy as np
import numpy.matlib as ml
from itertools import product
from functools import partial
from scipy.stats import multinomial
from skopt.optimizer import gp_minimize
from skopt.space import Real
from skopt.plots import plot_objective

"""Model Internals largely copied from github user benjybarnett!
   Just added fitting functions, recomputation functions and an OOP implementation."""

# Fitting Utility Functions

def binner(arr, bins):
    bin_centers = (bins[:-1] + bins[1:])/2 
    idx = np.digitize(arr, bin_centers)
    result = bins[idx]
    return result

def counter(arr, possible_locations):
    l = arr.tolist()
    return np.array([l.count(i) for i in possible_locations])

def getprobs(arr):
    return arr / np.sum(arr)

def clip(arr, roundup = 0.01): 
    a = np.clip(arr, roundup, np.inf)
    b = np.ndarray.sum(a, axis = 0)
    norm = a/b
    return norm

# Model Internals

def get_samples(N, vloc, aloc, pCommon, sigV, varV, sigA, varA, sigP, varP):
    return sigV * np.random.randn(N,1) + vloc, sigA * np.random.randn(N,1) + aloc

def calculate_likelihood_c1(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP):
    # likelihood P(Xv, Xa|C =1)
    
    firstDenom = 2*np.pi*np.sqrt(varV*varA + varV*varP +varA*varP)
    firstTerm = 1/firstDenom 
    secondNum = (Xv - Xa)**2 * varP + (Xv -0)**2 * varA + (Xa - 0)**2* varV 
    secondDenom = (varV * varA) + (varV * varP) + (varA * varP)
    secondTerm = np.exp((-0.5*(secondNum/secondDenom)))
    likelihood_com = firstTerm*secondTerm

    return likelihood_com

def calculate_likelihood_c2(Xv,Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP):
    # likelihood P(Xv, Xa|C =2)
    
    firstTerm = 2*np.pi*np.sqrt((varV + varP)*(varA+varP))
    secondTerm1 = (Xv - 0)**2/(varV + varP)
    secondTerm2 = (Xa - 0)**2 / (varA + varP)
    secondTermFull = np.exp((-0.5*(secondTerm1+secondTerm2)) )
    likelihood_ind = secondTermFull/firstTerm

    return likelihood_ind

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

def bciobjective(data, conditions, possible_locations, N, pars):

    pCommon, sigV, sigA, sigP = pars #unpack array
    varV, varA, varP = sigV**2, sigA**2, sigP**2

    nLL = 0

    for cond in conditions:
        d, vloc, aloc, variance = data[cond], cond[0], cond[1], cond[2]
        Xv, Xa = get_samples(N, vloc, aloc, pCommon, sigV, varV, sigA, varA, sigP, varP)
        sHatA = optimal_aud_location(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)
        sHatAbin = binner(sHatA, possible_locations)
        sHatAcount = counter(sHatAbin, possible_locations)
        sHatAprobs = clip(getprobs(sHatAcount))
        ll = multinomial.logpmf(d[1], np.sum(d[1]), sHatAprobs)
        nLL += ll

    return nLL
    
def bcifit(data : Dict[tuple : np.array], conditions : list, 
           possible_locations : np.array, N = 20000):

    """
       Fits the bci model to a given data dict, using Bayesian Optimization
       based on Gaussian Processes. 
       
       Args:
        data -> Dict[tuple of vloc, aloc, variance_cond : np.array representing multinomial dist of responses]
        conditions -> list of tuples
        possible_locations -> pre-specified 5x1 np.array, 
        N -> nunber of samples for the monte carlo simulations, set at 20000 by default
       
       Returns:
        res -> OptimizeResult object
    """

    # Get Partial function to feed into Opt
    objective = partial(bciobjective, data, conditions, possible_locations, N)

    # Setup paramater space
    pars = [Real(0.1, 0.6, name = 'pCommon'), 
            Real(1, 15, name = 'sigV'), 
            Real(1, 15, name = 'sigA'), 
            Real(1, 15, name = 'sigP')]

    # Run Optimizer
    res = gp_minimize(objective, dimensions=pars, n_initial_points=100, 
                    initial_point_generator='lhs', noise='gaussian', n_jobs = 4)

    return res

def bcirecompute(data, conditions, possible_locations, N, pars):

    pCommon, sigV, sigA, sigP = pars #unpack array
    varV, varA, varP = sigV**2, sigA**2, sigP**2

    # Define new data structure to store results, depending on the kind of matrix we're after here

    # run the model

    for cond in conditions:
        d, vloc, aloc, variance = data[cond], cond[0], cond[1], cond[2]
        Xv, Xa = get_samples(N, vloc, aloc, pCommon, sigV, varV, sigA, varA, sigP, varP)
        sHatV = optimal_visual_location(N, vloc, aloc, pCommon, sigV, varV, sigA, varA, sigP, varP)
        sHatA = optimal_aud_location(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)
        sHatVbin, sHatAbin = binner(sHatV, possible_locations), binner(sHatA, possible_locations)
        sHatVcount, sHatAcount = counter(sHatVbin, possible_locations), counter(sHatAbin, possible_locations)
        sHatVprobs, sHatAprobs = clip(getprobs(sHatVcount)), clip(getprobs(sHatAcount))

    pass # return data structures

# OOP Wrapper

class BCIModel:

    def __init__(self, data, conditions, possible_locations):
        self.data = data
        self.conditions = conditions
        self.possible_locations = possible_locations

    def fit(self):

        """
            Calls the bcifit function on the classes params.
            Returns OptimizeResult object.
        """
        
        # Run Bayesian Optimization
        res = bcifit(self.data, self.conditions, self.possible_locations)
        
        # Save Params in class
        self.res = res
        self.pCommon = res['x'][0]
        self.sigV = res['x'][1]
        self.sigA = res['x'][2]
        self.sigP = res['x'][3]

        # Return if wanted
        return res

    def plotopt(self):

        # plot dependencies of params
        plot_objective(self.res)

    def output(self):

        """
            Given the fitted parameters, recompute the BCI model 
            with a large amount of Monte Carlo simulations for
            construction of the RDMs.
        """

        pass # should poss be two dicts of condition tuples : np.arrays of 5x1 multinomials OR 5x1 counts, equivalent anyway