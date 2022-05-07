import numpy as np
import pickle as pkl
import os

from sklearn.model_selection import train_test_split
from generatestimuli import stimGenerator
from bci import bci_model
from keract import get_activations

def get_split(dataset):

    """
    

    """

    return train, test

def load_dataset(save_dir + fname):

    file = open(sae_dir + '/' + fname,'r')
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

def run_experiment(init_model, array_length, spread, n_locations, variance_conditions, 
                   size, id, dataset_name, datasetsize, savedataset, save_dir, test_size, epochs, batch_size, learning_rate, momentum,
                   train_size, test_size, random_state):

    # Load or generate synthetic trials
    if dataset_name != None:
        try:
            dataset = load_dataset(save_dir, dataset_name)
        except:
            print("Failed to load the dataset {0}.".format(dataset_name))

    elif dataset_name == None:
        dataset = stimGenerator(array_length = array_length, spread = spread, 
                  n_locations = n_locations, variance_conditions = variance_conditions).generate(size = size, 
                                                                                id = id, save = savedataset)
    
    # format dataset correctly
    dataset = setup_dataset(dataset)

    # then split into train and test set
    X_train, X_test, y_train, y_test = get_split(dataset, test_size, random_state)

    # Train the network
    nnmodel = init_model() 
    nnmodel = nnmodel.fit(X_train, y_train)

    # define results dict
    results_dict = {} # with the same condition labels

    # then for each condition
    # test the network on each stimulus!
    # get behaviouraul result 
    # and activity patterns in hidden layers
    # and create nested dict 

    for i in results_dict():

        stimlocs = np.where() # return locations of stimuli X_test and y_test
        condarray = np.array() # all of the stimuli

        for j in condarray:
            predicted_label = nnmodel.predict(condarray)
            layer_activations = keract.get_activations(nnmodel, j, auto_compile = True)
    
    pass