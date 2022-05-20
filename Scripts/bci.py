from gc import callbacks
import numpy as np
import numpy.matlib as ml
import pickle as pkl
from scipy.optimize import minimize, Bounds
from scipy.stats import multinomial

# Model Utility Functions

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

def computeResponseDistribution(N, possible_locations, vloc, aloc, pCommon, sigV, sigA, sigP):

    """Given parameter estimates, compute the response distributions and get
        the multinomial distribution which will be used to compute the loglikelihood."""

    # Get Variances
    varV, varA, varP = sigV**2, sigA**2, sigP**2

    # Calculate Model Internals
    Xv = sigV * np.random.randn(N,1) + vloc
    Xa = sigA * np.random.randn(N,1) + aloc
    #sHatV = optimal_visual_location(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)
    sHatA = optimal_aud_location(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)

    # Get Multinomial Distribution from p(sHatA | Xa, Xv) for model fitting
    binnedA = [min(possible_locations, key=lambda x:abs(x-i)) for i in sHatA] # bin responses
    countsA = [binnedA.count(i) for i in possible_locations] # get counts
    multinomialA = [i / sum(countsA) for i in countsA] # get probabilities

    return multinomialA

def runBCI(params, data, N, possible_locations):

    # Define Metric
    overall_ll = 0

    # Unpack Parameters
    pCommon, sigV, sigA, sigP = params

    # Fit Params to each condition
    for cond in data:
        print(cond)
        vloc, aloc = cond[0], cond[1]
        print(vloc, aloc)
        respdisp = computeResponseDistribution(N, possible_locations, vloc, aloc, pCommon, sigV, sigA, sigP)
        print(respdisp)
        print(data[cond][1])
        print(data[cond][1])
        ll = multinomial.logpmf(data[cond][1], np.sum(data[cond][1]), respdisp)
        overall_ll =+ ll
        print(overall_ll)

    return overall_ll * -1 # returned as nLL

def recomputeBCI(data, N, possible_locations, params):

    def bin(arr, possible_locations):
        return [min(possible_locations, key=lambda x:abs(x-i)) for i in arr]

    def recompute_bci(N, possible_locations, pCommon, sigV, sigA, sigP):

        # Get Variances
        varV, varA, varP = sigV**2, sigA**2, sigP**2

        # Calculate Model Internals
        Xv = sigV * np.random.randn(N,1) + vloc
        Xa = sigA * np.random.randn(N,1) + aloc

        # Compute Posterior for each sample
        posteriorVA = calculate_posterior(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)

        # Compute Responses
        sHatV = optimal_visual_location(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)
        sHatA = optimal_aud_location(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)

        # Get Binned Responses ### Let's not 
        sHatV, sHatA = bin(sHatV, possible_locations), bin(sHatA, possible_locations)

        return sHatV, sHatA, posteriorVA

    pCommon, sigV, sigA, sigP = params

    bcidata = {i:{"sHatV":None, "sHatA":None, "posterior":None} for i in data.keys()}

    for cond in data:
        vloc, aloc = cond[0], cond[1]
        sHatV, sHatA, posteriorVA = recompute_bci(N, possible_locations, pCommon, sigV, sigA, sigP)
        bcidata[cond]['sHatV'], bcidata[cond]['sHatA'], bcidata[cond]['posterior'] = np.mean(sHatV), np.mean(sHatA), np.mean(posteriorVA)

    return bcidata

# OOP Implementation

class BCIModel:

    """
        Class to implement the BCI model in terms of fitting and then recomputing
        response distributions.

        Args:
            nn_data --> dict; behavioural data from a trained NN.
            loc_list --> list; of possible stimulus locations on the azimuth.

        Methods:
            .fit() --> fits the model to the data, parameterizing self.param_values (array)
            .recompute() --> recomputes model internals with fitted params for use in RSA. 
                             Returns a dict in the form {cond:{sHatV:..., sHatA:..., posteriors:, }
    """

    def __init__(self, nn_data, loc_list):
        self.nn_data = nn_data
        self.possible_locations = loc_list
        self.param_values = None
        self.bcidata = None
        self.N = 10000

    def fit(self):

        # Might be important to define more accurate bounds + starting points?

        def getmin(data = self.nn_data, N = self.N, possible_locations = self.possible_locations):

            pCommon, sigV, sigA, sigP = float(), float(), float(), float()
            params = [pCommon, sigV, sigA, sigP]
            bounds = ((0.01, 0.99), (0,30), (0,30), (0,30))
            x0 = [0.5, 10, 10, 10]
            res = minimize(runBCI, x0 = x0, bounds = bounds, args = (data, N, possible_locations))
            print(res['x'])
            return res

        # Run Opt
        res = getmin()

        # Get Fitted Pars
        param_values = {'pCommon': res['x'][0], 'sigV': res['x'][1], 'sigA': res['x'][2], 'sigP': res['x'][3]}
        self.param_values = res['x']

        return param_values

    def recompute(self):

        bcidata = recomputeBCI(self.nn_data, self.N, self.possible_locations, self.param_values)

        return bcidata