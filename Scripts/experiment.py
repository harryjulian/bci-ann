import numpy as np
import pickle as pkl
import os

from sklearn.model_selection import train_test_split
from generatestimuli import stimGenerator
from bci import bci_model
from keract import get_activations

### Utility Functions

def load_dataset(save_dir, fname):

    file = open(save_dir + '/' + fname,'r')
    dataset = pkl.load(file)

    return dataset

def setup_dataset(dataset):

    # transfer data from dict to array

    pass

def get_split(dataset, test_size, random_state):

    """
    """

    X_train, X_test, y_train, y_test = train_test_split(dataset, test_size = test_size, random_state = random_state)

    return X_train, X_test, y_train, y_test

### Full function to run experiment

def run_experiment(init_model, array_length, spread, n_locations, variance_conditions, 
                   size, id, dataset_name, datasetsize, savedataset, save_dir, epochs, batch_size, learning_rate, momentum,
                   train_size, test_size, random_state, save_results):

    # Load OR generate synthetic trials depending on whether or not we can find the fpath
    if dataset_name != None:
        try:
            dataset = load_dataset(save_dir, dataset_name)
        except:
            print("Failed to load the dataset {0}.".format(dataset_name))

    elif dataset_name == None:
        dataset = stimGenerator(array_length = array_length, spread = spread, 
                  n_locations = n_locations, variance_conditions = variance_conditions).generate(size = size, 
                                                                                id = id, save = savedataset)

    # Initialize Model
    nnmodel = init_model()
    n_layers = len(nnmodel.layers) - 2 # Minus input and output

    # Create dict for results
    results_behavioural = dict.fromkeys([i for i in dataset.keys()]) 
    results_activations = dict.fromkeys([i for i in dataset.keys()])
    
    nest_dict = dict.fromkeys([i for i in range(n_layers)])
    results_activations = dict.fromkeys(results_activations, nest_dict)

    # Format dataset correctly (dict --> np.array)
    dataset = setup_dataset(dataset)

    # Split into Train & Test
    X_train, X_test, y_train, y_test = get_split(dataset, test_size, random_state)
    
    # Train Model
    nnmodel = nnmodel.fit(X_train, y_train)

    # Test Model
    for i in results_behavioural(): # select each key of the dict (i.e. each condition)
        stimlocs = np.where() # return locations of stimuli X_test and y_test
        condarray = np.array() # all of the stimuli

        for j in condarray:
            predicted_label = nnmodel.predict(condarray)
            layer_activations = keract.get_activations(nnmodel, j, auto_compile = True)

    if save_results == True:

        ### pkl the results and model in the save_dir!

        pass
    
    return results_behavioural, results_activations