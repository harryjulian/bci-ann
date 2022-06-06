import numpy as np
import rsatoolbox as rsa
from rsatoolbox.data import Dataset

def get_rdms(bcioutVCI, bcioutACI, bcioutposteriorCI, # for CI model
             bcioutVFF, bcioutAFF, bcioutposteriorFF, # for FF model
             bcioutVFS, bcioutAFS, bcioutposteriorFS, # for FS model
             nnoutV, nnoutA, # for NN behav data
             nnoutactivations, # for NN activation data -- this will change as architectures change, so future proof
             conditions): # conditions to get groundtruth rdm

    # Fix any shapes here.

    # Then create rsa dataset objects
    data_dict = {'bcioutVCI':bcioutVCI,
                 'bcioutACI':bcioutACI,
                 'bcioutposteriorCI':bcioutposteriorCI,
                 'bcioutVFF':bcioutVFF,
                 'bcioutAFF':bcioutAFF,
                 'bcioutposteriorFF':bcioutposteriorFF,
                 'bcioutVFS':bcioutVFS,
                 'bcioutAFS':bcioutAFS,
                 'bcioutposteriorFS':bcioutposteriorFS,
                 'nnoutV':nnoutV,
                 'nnoutA':nnoutA}
    
    # then add the activations stuff

    # then calculate th 
    pass

def plot_rdms():

    pass

def analyze_rdms():

    pass

class RSA:
    """Parse the experimental directory and retrieves all relevant saved data.
       Calculates, plots and runs statistics on the RDMs for all a priori comparisons of interest."""

    def __init__(self, directory):
        self.directory

    def run():
        pass

    def plot():
        pass

    pass