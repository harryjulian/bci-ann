import numpy as np
import pickle as pkl
import itertools

def trailgen(array_length, spread, size, Vloc, Aloc):

    """
        Used to generate stimuli for an individual condition, which
        should then be wrapped and placed in a dict.
    """



    return trial

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
    """

    def __init__(self, array_length, spread, n_locations, variance_conditions):

                self.array_length = array_length
                self.spread = spread
                self.n_locations = n_locations
                self.variance_conditions = variance_conditions
                print("Loaded generator.")
    
    def generate(self, size):
        
        """Generates dataset of arrays of where n = size
        size -> int(); size of the dataset. 
        """

        # Create data structure to hold the dataset
        dataset = {i:None for i in self.variance_conditions} # use dict to enable types of trials to remain marked?
        n_percond = ((size / len(self.variance_conditions)) / self.n_locations)) # find n to generate per cond
        
        # Find locations at which stimuli can be placed
        bin_center = self.array_length / n_positions
        loc_list = [i*bin_center+bin_center for i in range(self.n_locations)]
        Vlist, Alist = loc_list, loc_list

        conditions = [itertools.product(Vlist, Alist, self.variance_conditions)] 

        return dataset



if __name__ == "__main__":

    generator = stimGenerator(array_length=100, spread=1, n_locations=5, variance_conditions=[0, 1])
    dataset = generator.generate()
    print(dataset)