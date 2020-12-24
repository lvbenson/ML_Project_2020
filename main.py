'''
Main workflow module for cancer prognosis project - FA2020 CSCS536 final project, Daniel Kaiser, Lauren Benson, Joanna Li

@authors: Daniel Kaiser, Lauren Benson, Joanna Li - Created on: 2020-10-30 11:23:00 EST
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

# ---------------- Custom imports -------------
import config

# ~~~~~~~ debug ~~~~~~~
import logging as log
log.basicConfig(filename='../logs/{}_main.log'.format(datetime.date.today().__str__()), format='%(asctime)s - %(levelname)s: %(message)s', level=log.DEBUG)
log.info('Initialize log')

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
# ------------------ Globals -------------------


# ================== FUNCTIONS & CLASSES ========================
class Model:
    def __init__(self, raw_data_location):
        '''
        Initialize 'Model' class to handle data processing and model training

        Inputs : 
            raw_data_location : str
                Path to raw data file. Must be CSV format. 

        Returns : 
            None
        '''
        log.info('Model class initiated')
        self.raw_data_location = raw_data_location
        self.X_train = []
        self.X_test = [] 
        self.y_train = []
        self.y_test = []

    def summarize_model(self):
        '''
        Function to log and print to console a summary of the learned model

        Inputs : 
            None

        Returns : 
            None
        '''
        log.info('''Raw data: {location}\n
        Response variable: {response}\n
        Scaling: {scaling}\n
        Model type: {type}\n
        Training score: {train_score:.3f}\n
        '''.format(
            location=self.raw_data_location,
            response=self.response,
            scaling=self.scaler,
            type=self.model_type,
            train_score=self.train_score
        ))
        print('''Raw data: {location}\n
        Response variable: {response}\n
        Scaling: {scaling}\n
        Model type: {type}\n
        Training score: {train_score:.3f}\n
        '''.format(
            location=self.raw_data_location,
            response=self.response,
            scaling=self.scaler,
            type=self.model_type,
            train_score=self.train_score
        ))
    
    def data_pipeline(self, response='time', scaler=None):
        '''
        Pipeline organization function for processing data - includes separating response from features,
        scaling, and training/test splitting.

        Inputs : 
            file_ : str
                File location of data file. Must be CSV format with comma delimeter.
            response : str
                Name of response variable in CSV data frame
            scaler : str or None
                Type of scaling to apply to data, if any

        Returns : 
            X_train : array 
                Training feature array
            X_test : array
                Testing feature array
            y_train : array
                Training response vector
            y_test : array
                Testing response vector
        '''
        # Declare data processing parameters for logging later
        self.response = response
        self.scaler = scaler

        # read in data
        df = pd.read_csv(self.raw_data_location)
        log.debug('Data frame created')
        
        # attempt response vector assignment
        try:
            y = np.array(df[response])
        # handle nonexistent responses
        except KeyError:
            log.error('Nonexistent response variable! Retrying...')
            print("Nonexistent response variable! Options are: \n", list(df.columns))
            new_response = input('\n\n Type a new response here: ')
            data_pipeline(file_ = file_, response = new_response, scaler = scaler)
        finally:
            log.debug('Response selected')

        # decleare features as everything else
        # drops columns with inconsistent data types
        # also drops columns with missing data
        object_columns = [x for x in df.columns if df[x].dtype == 'object']
        X = np.array(df.drop(columns=[response] + object_columns))
        log.debug('Feature and response sets formed')

        # check desired scaler is supported
        if scaler not in config.scalers:
            log.error('Not a scaler option! Retrying...')
            print("Not a valid scaler option! The options are \n", config.scalers)
            new_scaler = input('\n\n Type a new scaler here: ')
            data_pipeline(file = file_, response = response, scaler = new_scaler)

        # scale data
        if scaler == 'Standard':
            sk_scaler = skl.preprocessing.StandardScaler()
            X_scaled, y_scaled = sk_scaler.fit_transform(X, y)
        elif scaler == 'MinMax':
            sk_scaler = skl.preprocessing.MinMaxScaler()
            X_scaled, y_scaled = sk_scaler.fit_transform(X, y)
        elif not scaler:
            X_scaled, y_scaled = X, y
        log.debug('Data scaled')

        # split into training and test data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled)
        log.debug('Data split into training and test sets')

        return X_train, X_test, y_train, y_test

    def model_pipeline(self, data_tuple, model_type):
        '''
        Pipeline organization function for training model - includes estimator selection,
        grid search cross validation, and fitting to training data.

        Inputs : 
            data_tuple : tuple of array-likes
                Tuple of training feature matrix and training response, in that order
            model_type : str
                String to indicate if classification or regression problem based on response type

        Returns : 
            grid_search_model.best_estimator_ : SKLearn Estimator
                Learned model with best performance on cross validation of grid search
            grid_search_model.cv_results_ : dict
                Dictionary formatted as a data frame, includes all grid search results
        '''
        # Declare model processing parameters for logging later
        self.model_type = model_type

        # Extract feature matrix and response vector
        X = data_tuple[0]
        y = data_tuple[1]

        # Decide which model to use, classifier or regressor
        if model_type == 'classifier':
            model = MLPClassifier()
        else:
            model = MLPRegressor()
        log.debug('Model created of type: {}'.format(model_type))
        
        # Declare grid search parameter grid from config file
        # Presumed not to change
        params = config.params

        # Run grid search cross-validation
        grid_search_model = GridSearchCV(estimator=model, param_grid=params, cv=5)    
        
        # Train model
        grid_search_model.fit(X, y)
        log.debug('Model cross-validated on parameter grid')

        # Declare model best estimator for logging later 
        self.best_estimator = grid_search_model.best_estimator_
        self.train_score = grid_search_model.best_score_

        return grid_search_model.best_estimator_, grid_search_model.cv_results_

def main():
    '''
    Main function. Actually runs experiments of interest to scientific results.

    Inputs : 
        None

    Returns : 

    '''
    
    # Runs experiment one
    log.info('-------- Starting experiment one ------------')
    training_scores = []
    test_scores = []
    responses = []
    scalers = []
    types = []

    for i in range(config.experiment_one['num_iterations']):
        log.info('Iteration: {}'.format(i+1))
        model = Model(config.data_)
        X_train, X_test, y_train, y_test = \
            model.data_pipeline(response=config.experiment_one['response'],
                            scaler=config.experiment_one['scaler'])
        log.info('Data pipeline ran')

        best_estimator, cv_results = \
            model.model_pipeline(data_tuple=(X_train, y_train), 
                            model_type=config.experiment_one['model_type'])
        log.info('Model pipeline ran')

        training_scores.append(model.train_score)
        test_scores.append(best_estimator.score(X_test, y_test))
        log.info('Scores appended')

        model.summarize_model()

    df = pd.DataFrame({
        'training score': training_scores,
        'test score': test_scores,
        'responses': [config.experiment_one['response'] 
                    for _ in range(config.experiment_one['num_iterations'])],
        'type': [config.experiment_one['model_type'] 
                    for _ in range(config.experiment_one['num_iterations'])],
        'scaling': [config.experiment_one['scaler'] 
                    for _ in range(config.experiment_one['num_iterations'])]
    })
    log.debug('Experiment data put into dataframe')

    df.to_csv('2020-11-02_experiment_one.csv')
    log.debug('Experiment data saved to file')
    log.info(' ------------- Finished experiment one ----------------')
        

    return df
    
# ======================== MAIN ===================================
if __name__ == '__main__':    
    '''
    parser = argparse.ArgumentParser(description='Process neural network stacking model parameters')
    parser.add_argument('-d', metavar='del', type=float,  nargs = '?', default = 0.02, help='convergence parameter (default = 0.02). Converges when less than delta proportion of the edges are with wt = 1')
    args = parser.parse_args()
    '''

    # Processes data, builds/trains model, runs experiments
    main()