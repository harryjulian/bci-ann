from typing import List
import numpy as np
import pickle as pkl
from itertools import product
from scipy.stats import truncnorm
from tensorflow import convert_to_tensor

"""
    Python file for generating artificial stimuli, which represent those
    shown to humans in Kording et al., (2007) - Causal Inference in
    Multisensory Perception.

    Last remaining job to do here is to integrate the convert_to_tensor command,
    but this is contingent on getting a working dist of tensorflow on this bloody 
    M1 chip...
"""

def trialgen(array_length, plusminusspread, size, vloc, aloc, vvar, avar):

    """
        Used to generate stimuli for an individual condition, which
        should then be wrapped and placed in a dict.

        Args:
         array_length -> int()
         plusminusspread -> int()
         size -> int()
         vloc -> int()
         aloc -> int()
         vvar -> int()
         avar -> int()

        Returns:
         stimarray -> np.array of shape size * 2 modalities * array_length

    """

    # Create Bounds like this, as truncnorm is defined with regards to the standard normal
    lowerbound, upperbound = (((0 - vloc) / vvar) + (0.5*plusminusspread)), (((array_length - vloc) / vvar) - (0.5*plusminusspread))
    
    # Sample from these stimulus distributions
    Vsamples, Asamples = np.round(truncnorm.rvs(lowerbound, upperbound, loc = vloc, scale = vvar, size = size)), np.round(truncnorm.rvs(lowerbound, upperbound, loc = aloc, scale = avar, size = size))
    Vsamples, Asamples = Vsamples.astype('int32'), Asamples.astype('int32')
    # stimulus array of shape n_stim * 2 modalities * length of each subarray
    stimarray = np.zeros((size, 2, array_length))

    # Iterably add sampled stimuli to the correct location in the subarray
    for i,j,c, in zip(Vsamples, Asamples, range(0, (size-1))): # c is essentially an iteration marker to get it all into the correct place
        vspread, aspread = [(i-plusminusspread), i, (i+plusminusspread)], [(j-plusminusspread), j, (j+plusminusspread)] # So get a list of where to begin and end adding zeros

        for p,q in zip(range(vspread[0], vspread[-1]), range(aspread[0], aspread[-1])):
            p, q = (p-1), (q-1) # Make indexing pythonic
            print("i", i, "j", j, 'c', c, 'p', p, 'q', q, 'vspread', vspread, 'aspread', aspread)
            stimarray[c][0][p], stimarray[c][1][q] = 1, 1

    return stimarray

class stimGenerator:

    """
        Class to create artificial datasets mimicking stimulus statistics
        in conventional multisensory integration paradigms, which can be
        tested on neural networks.

        Args:
         array_length -> length of the proposed 1 * n array.

         spread -> determines how many entries around the "stimulus
                    landing spot" that is changed, i.e. size of the
                    stimulus in the array. 

         n_locations -> determines the amount of equally spaced
                       locations at which stimuli may appear. 

         variance_conditions -> determines n of variance conditions
                                for which stimuli are generated.
    """

    def __init__(self, array_length : int, 
                 spread : int,
                 n_locations : int, 
                 variance_conditions : List):

                self.array_length = array_length
                self.plusminusspread = spread
                self.n_locations = n_locations
                self.variance_conditions = variance_conditions
                print("Loaded stimulus generator.")
    
    def generate(self, size : int, id : str, fext : str, save = False):
        
        """
            Generates dataset of arrays where size = n.

            Args:
             size -> size of the dataset. 
             id -> unique identifier for the datasetsf
             fext -> file extension
             save -> bool
        """

        n_percond = int(((size / len(self.variance_conditions)) / self.n_locations)) # find n to generate per cond
        
        # Find locations at which stimuli can be placed
        bin_center = self.array_length / self.n_locations
        loc_list = [i*bin_center+bin_center for i in range(self.n_locations)]
        Vlist, Alist = loc_list, loc_list

        # Create Dataset Obj
        combinations = list(product(Vlist, Alist, self.variance_conditions))
        dataset = {i:None for i in combinations}

        # For each combination, generate n_percond trials and add to dict 
        for i in combinations:
            trials = convert_to_tensor(trialgen(self.array_length, self.plusminusspread, n_percond, vloc = i[0], aloc = i[1], vvar = i[2], avar = i[2]))
            dataset[i] = trials

        # Save as pkl if True
        fname = fext + '.pkl'
        filetowrite = pkl.open(fname, 'rb')
        pkl.dump(dataset, filetowrite)

        return dataset