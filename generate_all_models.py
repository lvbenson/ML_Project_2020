# develop an mlp for blobs dataset
from sklearn.datasets import make_blobs
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np

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
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense


# stacked generalization with neural net meta model on blobs dataset
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from keras.models import load_model
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.merge import concatenate
from numpy import argmax
from sklearn.feature_selection import SelectKBest,f_classif


# FUNCTION TO MAKE ALL POSSIBLE SUBMODELS
def giant_model_pipeline(trainX, trainy, name):
	# name has to be "cancer" or "simulated"


	# setting up param grid
	activations = ['relu','sigmoid','tanh']
	output_activations = ['softplus']
	losses = ['categorical_crossentropy', 'poisson']
	layers = [2, 3, 4, 5, 6, 7, 8, 9, 10]
	layer_size = [5, 10, 15, 20, 25, 30, 35]
	
	tracked_losses = []
	tracked_accuracies = []
	tracked_counts = []

	count = 0
	for a in activations:
		for o_a in output_activations:
			for l in losses:
				for layer in layers:
					for layer_s in layer_size:
						# Building model
						model = Sequential()
						model.add(Dense(layer_s, input_dim=trainX.shape[1], activation=a))
						for _ in range(layer-1):
							model.add(Dense(layer_s, activation=a))
						model.add(Dense(1, activation=o_a))
						model.compile(loss=l, optimizer='adam', metrics=['accuracy'])
						model.fit(trainX, trainy, epochs=10, verbose=0)
						
						# Generate tracking data
						loss, accuracy = model.evaluate(trainX,trainy)
						count += 1
						
						# Store tracking data 
						tracked_accuracies.append(accuracy)
						tracked_losses.append(loss)
						tracked_counts.append(count)

						# Save model to file
						filename = 'model_{count}_{act}_{loss}_{layers}_{layersize}_3cs8f.h5'.format(count=count+1, act=a, loss=l, layers=layer, layersize=layer_s)
						model.save('data/{}/{}_models/{}'.format(name, name, filename))

	dicty = {
		"Model ID": tracked_counts,
		"Model Loss": tracked_losses,
		"Model Accuracy": tracked_accuracies
	}

	df = pd.DataFrame(dicty)
	df.to_csv('data/{name}/submodel_stuff_3c_small_8f.csv'.format(name=name))

	
	return 

# =============================================
if __name__ == '__main__':
	# load data
	X_train = pd.read_csv('data/simulated/simulated_raw/X_train_3c_small_8f.csv')
	X_test = pd.read_csv('data/simulated/simulated_raw/X_test_3c_small_8f.csv')
	y_train = pd.read_csv('data/simulated/simulated_raw/y_train_3c_small_8f.csv')
	y_test = pd.read_csv('data/simulated/simulated_raw/y_test_3c_small_8f.csv')

	# Get all them submodels, yo!
	giant_model_pipeline(X_train, y_train, "simulated")
	
	print("hell yeah, science bitch!")



