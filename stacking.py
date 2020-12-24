'''
Stacking module for cancer prognosis project - FA2020 CSCS536 final project, Daniel Kaiser, Lauren Benson, Joanna Li

@authors: Daniel Kaiser, Lauren Benson, Joanna Li - Created on: 2020-11-13 15:45:00 EST
'''
# ========================= AUTHORSHIP ==========================
__author__ = "Daniel Kaiser, Lauren Benson, Joanna Li"
__credits__ = ["Daniel Kaiser", "Lauren Benson, Joanna Li"]

__version__ = "0.1"
__maintainer__ = "Daniel Kaiser"
__email__ = "kaiserd@iu.edu"
__status__ = "Development"

# =================== IMPORTS & GLOBALS =========================
# ----------- System imports -----------------
import os 
import sys
import datetime
import time
import argparse

# ------------ Scientific imports ------------
import pandas as pd
import seaborn as sns
import numpy as np

import sklearn as skl
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
import keras
import tensorflow.keras.models as km
from keras.layers import Dense

# ---------------- Custom imports -------------
import config
import neural_network as nn

# ~~~~~~~ debug ~~~~~~~
import logging as log
#log.basicConfig(filename='../logs/{}_neural_network.log'.format(datetime.date.today().__str__()), format='%(asctime)s - %(levelname)s: %(message)s', level=log.DEBUG)
#log.info('Initialize log')

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
# ------------------ Globals -------------------


# ================== FUNCTIONS & CLASSES ========================
def generate_ensemble(activation_functions=['relu'], output_activations=['sigmoid'], 
    num_layers_param=[3], num_neurons_param=[8,20], 
    epochs=10):
    '''
    Generates neural network base models with all possible configurations given the passed parameter space.
    
    Inputs : 
        STUFF

    Returns : 
        STUFF
    '''

    # start network generation process
    models = []

    # looping through parameter hyperspace
    for act_fun in activation_functions:
        for out_fun in output_activations:
            # could add conditional here to weed out
            # function pairs that don't make sense
            for n_layer in num_layers_param:
                for n_neuron in num_neurons_param:
                    # adding model to ensemble
                    models.append(
                        nn.NeuralNetwork(
                            activation_function=act_fun, 
                            output_activation=out_fun, 
                            num_layers=n_layer, 
                            num_neurons=n_neuron, 
                            epochs=epochs
                        )
                    )

    return tuple(models)


class StackedModel:
    def __init__(self, ensemble=[]):
        if not ensemble:
            ensemble = generate_ensemble()
        self.ensemble = ensemble

    def train_ensemble(self, X_train, y_train):
        for model in self.ensemble:
            model.build_model(X_train=X_train, y_train=y_train)
            _, model.pred = model.evaluate_model(X_train, y_train)

        return self.ensemble

    def build_stacked_model(self, X_train, y_train, X_val, Y_val,
        activation_function='relu', output_activation='sigmoid',
        num_layers=1, num_neurons=8,
        epochs=100):

        # initiate empty neural network
        model = Sequential()
        
        # Build input layer
        model.add(
            Dense(
                num_neurons, 
                activation=activation_function, 
                input_shape=(len(self.ensemble),)
            )
        )
        
        # input layer has input shape of number of features. 
        # number of layers specifies number of HIDDEN layers.
        
        #if num_layers > 1: NOT NEEDED? REDUCED TO RANGE 0
        for _ in range(num_layers-1):
            model.add(Dense(num_neurons, activation=activation_function))
        
        #output layer
        model.add(Dense(1, activation=output_activation)) #output has 1 node

        # Add the fancy fluff to make it into a SKLearn estimator
        model.compile(loss='binary_crossentropy',
              optimizer='sgd', #gradient descent
              metrics=['accuracy']) #accuracy of model

        '''
        def fit_stacked_model(model, inputX, inputy):
            # prepare input data
            X = [inputX for _ in range(len(model.input))]
            # encode output data
            inputy_enc = to_categorical(inputy)
            # fit model
            model.fit(X, inputy_enc, epochs=300, verbose=0)
        '''
        # Preparing observation data for stacked fitting
        X_val = ([X_val for _ in range(model.input.shape[1])])
        X_train = ([X_train for _ in range(model.input.shape[1])])

        # Train stacked model
        model.fit(X_train, y_train, 
            validation_data=(X_val, Y_val),
            epochs=epochs, verbose=1
        )
        
        # Store stacked model
        self.stacked_model = model

        return model


    def nonseq_stacked_model(self, X, Y):
        # Declare inputs to be as many submodels as we have
        inputs = keras.Input(shape=(1,))#len(self.ensemble),))

        # Declare a hidden layer
        x = Dense(4, activation=tf.nn.relu)(inputs)

        # Declare output layer
        outputs = Dense(1, activation=tf.nn.softmax)(x)

        # Declare empty model, to fill in layers
        stacked_model = km.Model(inputs=inputs, outputs=outputs)
        
        # Compile for training
        stacked_model.compile()

        X = [x.model.output for x in self.ensemble]
        for c, x in enumerate(X):
            x = tf.reshape(x, (1,))
            X[c] = x

        #try:
        stacked_model.fit(X[0], verbose=0)#, Y[0].reshape(1,10))
        #except:
            #print("YO WHAT THE FUCK")

        #tf.keras.utils.plot_model(stacked_model, "test_stack.png", show_shapes=True)

        self.stacked_model = stacked_model


        


# ============== MAIN =================
if __name__ == '__main__':
    ###### README ########
    '''
    Stuff, I guess. 
    '''


    # Generate empty stack of untrained ensemble
    thing = StackedModel(ensemble=[])

    # Add data from data directory to path
    data = pd.read_csv(os.path.join('data', 'wpbc.data'))
    X_train, X_test, y_train, y_test = nn.prepare_data(data)
    X_train, X_val = X_train[10:], X_train[:10]
    y_train, Y_val = y_train[10:], y_train[:10]

    X_val = ([X_val, X_val])
    Y_val = ([Y_val, Y_val])

    # Train ensemble of submodels
    thing.train_ensemble(X_train=X_train, y_train=y_train)

    # Fill in stacking model, train
    '''
    model = thing.build_stacked_model(X_train=X_train, y_train=y_train, X_val=X_val, Y_val=Y_val,
        activation_function='relu', output_activation='sigmoid',
        num_layers=2, num_neurons=8,
        epochs=1)
    '''

    thing.nonseq_stacked_model(X=X_val, Y=Y_val)

    #print(thing.stacked_model.evaluate(X_test, y_test, verbose=1))