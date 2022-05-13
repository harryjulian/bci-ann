import numpy as np
import numpy.matlib as ml
import pickle as pkl
from scipy.optimize import minimize

# Model Utility Functions ~~~~~~~ These Compute the internal aspects of the model

def calculate_likelihood_c1(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP):
    #likelihood P(Xv, Xa|C =1)
    
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

# Utility functions for fitting the model!

def get_multinomial(sHatV, sHatA, vloc, aloc, conditions):

    pass

def get_multinomial_likelihood():

    pass

# Runs Model a single time

def run_bci(pCommon, sigV, varV, sigA, varA, sigP, varP, conditions, N = 10000):

    """
        Function which runs the Bayesian Causal Inference model, given parsed parameters.
        Computes shared likelihood across all conditions, returns predictions.

        Args:
            pCommon
            sigV
            varV
            sigA
            varA
            sigP
            varP
            conditions
            N

        Returns:
            output --> should contain both the ll and the predicted frequency of location guesses
                        based on conditions????
    """

    overall_ll = 0

    for cond in conditions:

        vloc, aloc = cond[0], cond[1]

        Xv = sigV * np.random.randn(N,1) + vloc
        Xa = sigA * np.random.randn(N,1) + aloc

        # Calculate Model Internals
        sHatV = optimal_visual_location(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)
        sHatA = optimal_aud_location(Xv, Xa, N, pCommon, sigV, varV, sigA, varA, sigP, varP)

        # Generate the Multinomial Distributions
        multinomial = get_multinomial(sHatV, sHatA, vloc, aloc, conditions)

        # Calculate Loglikelihood
        ll = get_multinomial_likelihood(multinomial)

        overall_ll += ll
        
    return overall_ll

# After the best parameters have been found, recompute the predicted stimulus distributions for each condition. Export this as some sort of metric, a vector of probabilities perhaps?

def recompute_bci():

    """
        Given the fitted parameters, 
    
    """

    pass

# Class to implement all of this

class bci_model:

    """
        Class to setup and fit the bci model to 'behavioural' data
        from a trained instance of a neural network. The model should be initialised
        with the neural networks output, fitted and then the outputs of each model should be
        gathered with the output function.

        Args:

        output -> dict

    """

    def __init__(self, nn_behav_data, id):
        self.nn_data = nn_behav_data
        self.id = id
        self.pCommon = float()
        self.sigV = float()
        self.varV = float()
        self.sigA = float()
        self.varA = float()
        self.sigP = float()
        self.varP = float()
        self.conditions = None
        self.N = 10000

    def fit(self):

        """
            Finds the optimal parameters for the BCI model, fitted to
            behavioural data from a neural network.

            This should be written to recompute the optimization procedure


            Args:
                None
            Returns:
                output -> dict; of optimal parameter values. 
        """

        def optimize_model(self, model, n_optimisations = 10):

            if model == 'MA':
                self.pCommon = float()
            elif model == 'ff':
                self.pCommon = 1
            elif model == 'ss':
                self.pCommon = 0
            else:
                print("Incorrect model type specified.") 

            internal_dict = {'ll':None,'params':None}

            ### Decide how many times we'll run the optimizer...
            ### Decide how we can determine the starting points...

            for run in range(n_optimisations):

                output = minimize(fit_bci) ####################################################

                if output['ll'] > internal_dict['ll']:
                    internal_dict = output # as they have the same keys, we just want to replace it

            return output

        # Run Optimizations for each Model type
        ma_params = optimize_model(model = 'ma') 
        ff_params = optimize_model(model = 'ff') 
        ss_params = optimize_model(model = 'ss') 
        
        self.param_output = {"MA Params":ma_params, "FF Params":ff_params, "SS Params":ss_params}

    def output():

        pass

    def save():

        """
            Saves fitted params + 
        """

        pass