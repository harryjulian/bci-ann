import numpy as np
import pickle as pkl
import os
from itertools import product
from tensorflow import convert_to_tensor
from sklearn.model_selection import train_test_split
from keract import get_activations

from generatestimuli import stimGenerator
from bcimodel import BCIModel
from nnmodels import init_ffnet, init_recnet

# Util functions

def get_conditions(possible_locations, variance_conditions):
    return list(product(possible_locations, possible_locations, variance_conditions))

def make_expdir(foldername, id):

    """
        Given a unique id for the experiment, create a directory 
        in 'results' to store experimental info. This contains a variety
        of subfolders.
        
        Args:
            self.foldername -> str
            self.id -> str
    """
    parent_path = os.path.join(foldername, id)
    os.mkdir(parent_path)

    for i in ['stimuli', 'rdms', 'nndata', 'bcidata', 'figures', 'metadata']:
        subfolder_path = os.path.join(parent_path, i)
        os.mkdir(subfolder_path)

    return parent_path

# OOP Implementation

class BCIANNExperiment:

    # so we need something to generate stimuli / or select a pre-generated 
    # initialize + parameterize the neural network
    # train the neural network on the stimuli
    # save outputs in the correct place
    # run the bci model for each version of the model
        # plot convergence & obj in figures
    # given all of the saved data...create the rdms - save them as pkl files but ensure they're still in the local env
    # compute stats on the rdms. make sure this function is adaptable, maybe even in notebook format.

    def __init__(self, foldername, id, expparams, stimparams, nnparams, bciparams):
        
        self.id = id # identifier for this exp
        self.foldername = foldername
        self.stimparams = stimparams
        self.nnparams = nnparams
        self.bciparams = bciparams

        self.possible_locations = expparams["possible_locations"]
        self.variance_conditions = expparams["variance_conditions"]
        self.nn = nnparams['nn']
        self.conditions = get_conditions(self.possible_locations, self.variance_conditions)

    def setupdirs(self):
        """Makes a folder to save experimental results given parsed foldername and id.
           Also get combinatronics of conditions. Parse both cond args as arrays.
           Returns self.expdir"""
        self.expdir = make_expdir(self.foldername, self.id)

    def getstimuli(self):
        """Generates stimuli given the parameters in the parsed stimparams dict.
           Returns the dataset as self.stimuli and saves in the relevant stimuli folder."""

        fext = self.expdir + "/stimuli/stim-array-" + self.id

        stimgen = stimGenerator(self.stimparams["array_length"], self.stimparams["spread"], 
                                self.stimparams["n_locations"], self.stimparams["variance_conditions"])

        stimuli = stimgen.generate(size = self.stimparams["size"], id = self.id, 
                                   fext = fext, save = True)

        self.stimuli = stimuli

    # For the Neural Network

    def traintestnn(self):
        """Initializes neural network with parsed params."""

        # Init Model
        model = self.nn()

        # Create behavioural dicts to save results
        results_behavioural = dict.fromkeys([i for i in self.stimuli.keys()]) 
        
        # Create nested behavioural dict to save results
        results_activations = dict.fromkeys([i for i in self.stimuli.keys()])
        nest_dict = dict.fromkeys([i for i in range(len(model.layers) - 2)])
        results_activations = dict.fromkeys(results_activations, nest_dict)

        X = []
        y = []

        for cond in self.conditions:
            X.append(self.stimuli[cond])
            y.append(np.array(([cond] * len(self.stimuli[cond]))))

        # Create Aligned Arrays of all stimuli!
        X, y = np.concatenate(X, axis = 0), np.concatenate(y, axis = 0)

        # Get Split
        X_train, y_train, X_test, y_test = train_test_split(X, y, test_size = self.nnparams['testsize'], 
                                                            random_state=self.nnparams['random_state'])
        
        # Train model
        model.fit(X_train, y_train)

        # Test Model, iterably, on each condition
        for i in results_behavioural(): # select each key of the dict (i.e. each condition)
            stimlocs = np.where() # return locations of stimuli X_test and y_test
            condarray = np.array() # all of the stimuli

            for j in condarray: # EXAMPLE EXAMPLE EXAMPLE
                predicted_label = nn.predict(condarray)
                layer_activations = get_activations(nn, j, auto_compile = True)


        #self.nnoutV
        #self.nnoutA
        #self.nnoutactivations
        #bcioutVCI, bcioutACI, bcioutposteriorCI, # for CI model
        #     bcioutVFF, bcioutAFF, bcioutposteriorFF, # for FF model
        #     bcioutVFS, bcioutAFS, bcioutposteriorFS, # for FS model
        #     nnoutV, nnoutA, # for NN behav data
        #     nnoutactivations

    # for BCI

    def runbci(self):

        """All we need for this are the bci parameters & nnoutV and nnoutA."""
        def reshape_nnouput():
            pass

        ### So get data into correct format here for the bci model to work wiht

        models = ['CI', 'FF', 'FS']

        # run all models, likely to take like 15 mins if you're running it locally.
        # tho unsure how the fixed pCommon models take as they're likely to be a little less complex
        for model in models:
            bci = BCIModel(data, self.conditions, self.possible_locations, modeltype = model)
            bci.fit()
            bci.plotopt()
            bcioutV, bcioutA, bcioutposterior = bci.recompute()

            savepath = self.foldername + "/bcidata"

            for i,j in zip([bcioutV, bcioutA, bcioutposterior], ['bcioutV', 'bcioutA', 'bcioutposterior']):
                with open(savepath + "//" + j + self.id + ".pkl", 'wb') as f:
                    pkl.dump(i, f)

    def metadata():

        pass