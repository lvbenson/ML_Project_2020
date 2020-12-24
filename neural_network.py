'''Main neural network module for cancer prognosis project - FA2020 CSCS536 final project, Daniel Kaiser, Lauren Benson, Joanna Li

@authors: Daniel Kaiser, Lauren Benson, Joanna Li - Created on: 2020-11-11 15:10:00 EST
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
from keras.models import Sequential
from keras.layers import Dense

# ---------------- Custom imports -------------
import config

# ~~~~~~~ debug ~~~~~~~
import logging as log
'''
log.basicConfig(
    filename=os.path.join(
        '..', 
        'logs', 
        '{}_neural_network.log'.format(datetime.date.today().__str__()) ), 
    format='%(asctime)s - %(levelname)s: %(message)s', 
    level=log.DEBUG)
log.info('Initialize log')
'''

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
# ------------------ Globals -------------------


# ================== FUNCTIONS & CLASSES ========================
def prepare_data(data):
    #convert outcome to numbers (N = 0, R = 1)
    lb_make = LabelEncoder()
    data["outcome_code"] = lb_make.fit_transform(data["outcome"]) 
    
    #remove missing data
    data = data.loc[data['lymph_node_status'] != '?']
    data = data.drop('outcome',axis=1)
    
    # construct feature observation matrix and response vector
    labels = data['outcome_code']
    features = data.iloc[:,:-1]
    X=features
    y=np.ravel(labels)
    
    # split off test/train data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2) 
    
    # scale and return data
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = y_train
    y_test = y_test

    return X_train, X_test, y_train, y_test


class NeuralNetwork():
    def __init__(self, 
                    activation_function, output_activation, 
                    num_layers, num_neurons, 
                    epochs):

        #self.weight_init = weight_init
        self.activation_function = activation_function
        self.output_activation = output_activation
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.epochs = epochs


    def build_model(self, X_train, y_train):
        # initiate empty neural network
        model = Sequential()
        
        # Build input layer
        model.add(
            Dense(
                self.num_neurons, 
                activation=self.activation_function, 
                input_shape=(X_train.shape[1],)
            )
        ) 
        
        # input layer has input shape of number of features. 
        # number of layers specifies number of HIDDEN layers.
        if self.num_layers > 1:
            for _ in range(self.num_layers-1):
                model.add(Dense(self.num_neurons, activation=self.activation_function))
        
        #output layer
        model.add(Dense(1, activation=self.output_activation)) #output has 1 node

        # Add the fancy fluff to make it into a SKLearn estimator
        model.compile(loss='binary_crossentropy',
              optimizer='sgd', #gradient descent
              metrics=['accuracy']) #accuracy of model

        # Train model
        model.fit(X_train, y_train, epochs=self.epochs, batch_size=1, verbose=1)

        # Store trained model
        self.model = model
        return model

    def evaluate_model(self, X_test, y_test):
        # Predict on test data
        y_pred = self.model.predict(X_test) 

        # Score predictions
        score = self.model.evaluate(X_test, y_test, verbose=1)
        
        print("Accuracy: ", score[1])
        return(score[1], y_pred)


# ========================= MAIN ==================
if __name__ == '__main__':
    # Add data from data directory to path
    data = pd.read_csv(os.path.join('..', 'data', 'wpbc.data'))
    X_train, X_test, y_train, y_test = prepare_data(data)

    # Initialize class with some test parameters
    NN = NeuralNetwork(
        activation_function='relu',
        output_activation='sigmoid',
        num_layers=3,
        num_neurons=8,
        epochs=10,
    )

    NN.build_model(X_train=X_train, y_train=y_train)
    NN.evaluate_model(X_test=X_test, y_test=y_test)
    
    print('it got this far')