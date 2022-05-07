import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Concatenate, SimpleRNN, LSTM, GRU
from tensorflow.keras.models import Model

### Helper Functions --------------------------------------------------------------------

def plot_convergance(fitted_model, string):

    """
        Plots model accuracy with respect to epochs.
        fitted_model --> fitted
        string --> 
    """

    plt.plot(fitted_model.history[string])
    plt.plot(fitted_model.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

def get_activations():

    pass

### Models ------------------------------------------------------------------------------

def init_ffnet(array_length, v1shape = 8, a1shape = 8, ptshape = 8, ftshape = 8):
    
    """
        Creates and returns ffnet, a simple feedforward network created
        to being testing the benchmarknet paradigm. Created using the Keras functional API.
        Args: array_length --- determines how many features we'll use.
        Returns: ffnet model
    """
    
    ### AV Input --> V1 & A1 --> Combined in PT --> Into Dense FT Layer --> Output
    
    # Input Layers
    vis_input = Input(shape = (array_length,), name = "VIS") # Visual Info
    aud_input = Input(shape = (array_length,), name = "AUD") # Auditory Info

    # V1 - Unisensory Vision
    V1 = Dense(v1shape, activation = "relu", name = "V1")(vis_input)
    V1 = Model(inputs = vis_input, outputs = V1)
    
    # A1 - Unisensory Audition
    A1 = Dense(a1shape, activation = "relu", name = "A1")(aud_input)
    A1 = Model(inputs = aud_input, outputs = A1)

    # Parietal-Temporal - Forced Fusion?
    
    PT = Concatenate()([V1.output, A1.output])
    PT = Dense(ptshape, activation = "relu", name = "PT")(PT)
    
    # Frontal - Causal Inference?
    FT = Dense(ftshape, activation = "relu", name = "FT")(PT)
    
    # Output?
    OUT_VIS = Dense(5, activation = "sigmoid", name = "VOUT")(FT)
    OUT_AUD = Dense(5, activation = "sigmoid", name = "AOUT")(FT)
    
    # BetaNet
    ffnet = Model(inputs = [vis_input, aud_input], outputs = [OUT_VIS, OUT_AUD])
    ffnet.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return ffnet

def init_recnet(array_length):

    """

    """

    vis_input = Input(shape = (array_length,), name = "VIS") # Visual Info
    aud_input = Input(shape = (array_length,), name = "AUD") # Auditory Info

    V1 = GRU(v1shape, activation = "relu", name = "V1")(vis_input)
    V1 = Model(inputs = vis_input, outputs = V1)
    
    # A1 - Unisensory Audition
    A1 = GRU(a1shape, activation = "relu", name = "A1")(aud_input)
    A1 = Model(inputs = aud_input, outputs = A1)

    # Parietal-Temporal - Forced Fusion?
    
    PT = Concatenate()([V1.output, A1.output])
    PT = Dense(ptshape, activation = "relu", name = "PT")(PT)
    
    # Frontal - Causal Inference?
    FT = GRU(ftshape, activation = "relu", name = "FT")(PT)
    
    # Output?
    OUT_VIS = Dense(5, activation = "sigmoid", name = "VOUT")(FT)
    OUT_AUD = Dense(5, activation = "sigmoid", name = "AOUT")(FT)

    recnet = Model(inputs = [vis_input, aud_input], outputs = [OUT_VIS, OUT_AUD])
    recnet.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return recnet