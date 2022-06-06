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

def find_index(cond, y_test): # I know this is absurdly slow and nooby, literally couldn't be bothered
    l = []
    arr = y_test.tolist()
    for i in range(len(arr)):
        if arr[i] == list(cond):
            l.append
    return l

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
    """"""

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
        """Initializes neural network with parsed params. Given the generated stimuli, train the 
        network on the training set and then iterably test the network on the test set. Save behavioural 
        responses as well as activations. Returns: results_behavioural, results-activations (two dicts)"""

        # Initialize Model
        model = self.nn()

        # Create behavioural dicts to save results
        results_behavioural = {i:[] for i in self.stimuli.keys()}
        
        # Create nested behavioural dict to save results
        results_activations = {i:{i.name:[] for i in model.layers} for i in results_behavioural.keys}

        X = []
        y = []

        for cond in self.conditions:
            X.append(self.stimuli[cond])
            y.append(np.array(([cond] * len(self.stimuli[cond]))))

        # Create Aligned Arrays of all stimuli!
        X, y = np.concatenate(X, axis = 0), np.concatenate(y, axis = 0)

        # Get Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.nnparams['testsize'], 
                                                            random_state=self.nnparams['random_state'])
        
        # Train model
        model.fit(X_train, y_train)

        # Test Model, iterably, on each condition
        for cond in self.conditions:
            indices = find_index(cond, y_test)
            X_testcond, y_testcond = X_test[indices], y_test[indices]

            # For each stimulus, get behavioural results and append to nested list in dict
            for j in X_testcond:
                predicted_labels = model.predict(j)
                results_behavioural[cond].append(predicted_labels)

                # Get activations into correct list in nested dict
                layer_activations = get_activations(model, j, auto_compile = True)
                for layer, activations in layer_activations.items():
                    results_activations[cond][layer].append(layer_activations[layer])
                
        return results_behavioural, results_activations

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

        pass

    def metadata():

        pass