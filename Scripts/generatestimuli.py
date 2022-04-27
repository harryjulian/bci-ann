import numpy as np
import pickle as pkl

class stimGenerator:

    """
    
        Class to create artificial datasets mimicking stimulus statistics
        in conventional multisensory integration paradigms, which can be
        tested on neural networks.

        Args:

        array_length -> int(); length of the proposed 1 * n array.

        spread -> int(); determines how many entries around the "stimulus
                    landing spot" that is changed, i.e. size of the
                    stimulus in the array. 

        n_locations -> int(); determines the amount of equally spaced
                       locations at which stimuli may appear. 

        variance_conditions -> list; determines n of variance conditions
                                for which stimuli are generated.

        size -> int(); size of the dataset.

    """

    def __init__(self, array_length, spread, 
                n_locations, variance_conditions, size):

                self.array_length = array_length
                self.spread = spread
                self.n_locations = n_locations
                self.variance_conditions = variance_conditions
                self.size = size

    def generate():

        pass