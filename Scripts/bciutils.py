import numpy as np
import numpy.matlib as ml
import numba
import pickle as pkl
from itertools import product
from scipy.stats import multinomial
from scipy.optimize import minimize, brute

# utility for no nasty divide by zeros

@numba.jit
def clip(l, roundup = 0.001): # i bet the first part of this could be 2 lines...
        l2 = []
        for i in l:
            if i == 0:
                i += roundup
                l2.append(i)
            else:
                l2.append(i)
        l2 = [float(i)/sum(l2) for i in l2]
        return l2

# Model outline - single condition

def bci_montecarlo(vloc, aloc, N, params):

    """
        Given the true conditional location of the visual and auditory stimuli,
        N samples and the parameters, compute the BCI estimate of sHatA.

        Args:
            vloc - visual location
            aloc - auditory locatin
            N - samples of the model
            pCommon - prior
            sigV - SD of vis
            sigA - SD of aud
            sigP - SD of central bias
        Returns:
            sHatA - array of auditory estimates
    """
    @jit
    def calculate_likelihood_c1(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP):
        # likelihood P(Xv, Xa|C =1)
        
        firstDenom = 2*np.pi*np.sqrt(varV*varA + varV*varP +varA*varP)
        firstTerm = 1/firstDenom 
        secondNum = (Xv - Xa)**2 * varP + (Xv -0)**2 * varA + (Xa - 0)**2* varV 
        secondDenom = (varV * varA) + (varV * varP) + (varA * varP)
        secondTerm = np.exp((-0.5*(secondNum/secondDenom)))
        likelihood_com = firstTerm*secondTerm
        return likelihood_com

    @jit
    def calculate_likelihood_c2(Xv,Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP):
        # likelihood P(Xv, Xa|C =2)
        
        firstTerm = 2*np.pi*np.sqrt((varV + varP)*(varA+varP))
        secondTerm1 = (Xv - 0)**2/(varV + varP)
        secondTerm2 = (Xa - 0)**2 / (varA + varP)
        secondTermFull = np.exp((-0.5*(secondTerm1+secondTerm2)) )
        likelihood_ind = secondTermFull/firstTerm
        return likelihood_ind

    def
    
    likelihood_c2 = np.exp((-0.5*((Xv - 0)**2/(varV + varP)+(Xa - 0)**2/(varA + varP))))/2*np.pi*np.sqrt((varV + varP)*(varA+varP))

    @jit
    def calculate_posterior(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP):
        # p(C = 1|Xv,Xa) posterior
        
        likelihood_common = calculate_likelihood_c1(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)
        likelihood_ind = calculate_likelihood_c2(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)
        post_common = likelihood_common * pCommon 
        post_indep = likelihood_ind * (1-pCommon)
        posterior = post_common/(post_common +post_indep)
        return posterior

    @jit
    def opt_position_conditionalised_C1(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP):
        # Get optimal location given C = 1
        
        cues = Xv/varV + Xa/varA + ml.repmat(pCommon,N,1)/varP
        inverseVar = 1/varV + 1/varA + 1/varP
        sHatC1 = cues/inverseVar
        return sHatC1

    @jit
    def opt_position_conditionalised_C2(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP):
            # Get optimal locationS given C = 2
            
        visualCue = Xv/varV +ml.repmat(pCommon,N,1)/varP
        visualInvVar = 1/varV + 1/ varP
        sHatVC2 = visualCue/visualInvVar
        audCue = Xa/varA + ml.repmat(pCommon,N,1)/varP
        audInvVar = 1/varA + 1/ varP
        sHatAC2 = audCue/audInvVar
        return sHatVC2, sHatAC2

    @jit
    def optimal_visual_location(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP):
        # Use Model Averaging to compute final visual est
        
        posterior_1C = calculate_posterior(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)
        sHatVC1 = opt_position_conditionalised_C1(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)
        sHatVC2 = opt_position_conditionalised_C2(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)[0]
        sHatV = posterior_1C*sHatVC1 + (1-posterior_1C)*sHatVC2 #model averaging
        return sHatV

    @jit
    def optimal_aud_location(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP):
        # Use Model Averaging to compute final auditory est
        
        posterior_1C = calculate_posterior(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)
        sHatAC1 = opt_position_conditionalised_C1(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)
        sHatAC2 = opt_position_conditionalised_C2(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)[1]
        sHatA = posterior_1C*sHatAC1 + (1-posterior_1C)*sHatAC2 #model averaging
        return sHatA

    # Unpack params
    pCommon, sigV, sigA, sigP = params[0], params[1], params[2], params[3]

    # Get Variances
    varV, varA, varP = sigV**2, sigA**2, sigP**2

    # Get N samples
    Xv, Xa = sigV * np.random.randn(N,1) + vloc, sigA * np.random.randn(N,1) + aloc

    # Compute optimal auditory location
    sHatV = optimal_visual_location(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)
    sHatA = optimal_aud_location(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)

    return sHatV, sHatA

# Model outline - all conditions

def bci(params, data, possible_locations, variance_conds, N):

    conditions = list(product(possible_locations, possible_locations, variance_conds))
    nLLV = 0
    nLLA = 0

    for cond in conditions:
        vloc, aloc = cond[0], cond[1]
        sHatV, sHatA = bci_montecarlo(vloc, aloc, N, params)

        # Bin Responses, get counts
        binnedV = [min(possible_locations, key=lambda x:abs(x-bv)) for bv in sHatV]
        countsV = [binnedV.count(bvc) for bvc in possible_locations]
        multinomialV = clip([mv / sum(countsV) for mv in countsV])
        llV = multinomial.logpmf(data[cond][0], np.sum(data[cond][0]), multinomialV)
        nLLV += llV

    return (nLLA + nLLV) * -1

# Model fitting

def fitgrid(params, data, possible_locations, variance_conds, N):

    # Define paramter grid (about 500 points)
    rranges = (slice(0.05, 0.45, 0.05), slice(1, 13, 3), slice(1, 13, 3), slice(1, 13, 3))

    # Run for all points in grid
    x0 = brute(bci, ranges = rranges, args = (data, possible_locations, variance_conds, N))

    return x0

def fitmodel(method, data, possible_locations, variance_conds, N):

    # Create Parameter Variables
    pCommon, sigV, sigA, sigP = float, float, float, float
    params = [pCommon, sigV, sigA, sigP]
    
    # Create bounds and get x0 from grid fitting procedure
    bounds = [(0.01, 0.6), (0.1, 15), (0.01, 15), (0.01, 15)]
    x0 = fitgrid(params, data, possible_locations, variance_conds, N)
    print(x0)

    res = minimize(bci, x0 = x0, bounds = bounds, args = (data, possible_locations, variance_conds, N), method = method)

    return res, res['fun']

def recomputemodel():

    # recompute 20,000 monte carlo samples in each condition
    # then get the rdms
    # save as pkl?
    # in the correct experimental folder.

    pass






