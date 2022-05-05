import numpy as np
from scipy.optimize import minimize

### To me it probably makes more sense to only fit the causal inference model, and then use the dervied parameters
### to get the results from the other models...easier, computationally too.

def run_bci():

    """Function to compute the model"""

    pass

class bci_model(nn_data: pd.DataFrame):

    """
        Class to setup and fit the bci model to 'behavioural' data
        from a trained instance of a neural network. 

        Args:

        data -> pd.DataFrame(); dataframe of 'behavioural data' from a given trained neural network. Doesn't matter which type coz
                we're taking the finak output here

    """

    def __init__(self, nn_data):
        self.nn_data = nn_data

        print("Initialized the Bayesian Causal Inference Model.")

    def fit():

        pass

