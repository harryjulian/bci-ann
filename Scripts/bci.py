import pickle as pkl
from bciutils import fitmodel, recomputemodel, bci_rdms

# Implement the whole model in a unified class

class BCIModel:

    def __init__(self, data, N = 5000):
        self.data = data
        self.N = N

    def fit():

        # Fit model to data, save parameters within the class

        pass

    def recompute():

        # Recompute the model given best pars, generate the rdms!

        pass
