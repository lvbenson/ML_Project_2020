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

############## START DATA PREP ##################
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
	#ranking all of our features for the classification task


	import pandas as pd
	import matplotlib.pyplot as plt

	# correlation matrix
	correlation_matrix = pd.DataFrame(X).corr(method = 'spearman').abs()
	sns.set(font_scale = 1)
	f, ax = plt.subplots(figsize=(12, 12))

	# Make heatmap
	sns.heatmap(correlation_matrix,linewidths = 0.05, cmap= 'YlGnBu', square=True, ax = ax)

	# Save figure
	f.savefig('correlation_matrix.png', dpi = 1080)

	# remove highly correlated features
	correlated_features = set()

	for i in range(len(correlation_matrix.columns)):
		for j in range(i):
			if abs(correlation_matrix.iloc[i, j]) > 0.75:
				colname = correlation_matrix.columns[i]
				correlated_features.add(colname)

	print(correlated_features)
	X.drop(labels=correlated_features, axis=1, inplace=True)
#	x_test.drop(labels=correlated_features, axis=1, inplace=True)



	selector = SelectKBest(f_classif, k = 8)
	X_new = selector.fit_transform(X, y)
	X = X_new

	scaler = MinMaxScaler().fit(X)
	X = scaler.transform(X)

	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=37)

	return X_train, X_test, y_train, y_test

# --------------------------------------------------------
if __name__ == '__main__':
	# -------- Cancer data --------------
	# Load and scale and select and all that on cancer data
	trainX, testX, trainy, testy = prepare_data(pd.read_csv('data/cancer/cancer_raw/wpbc.data'))

	# Save cleaned cancer data to file, never need to generate again
	np.savetxt('data/cancer/cancer_raw/X_train.csv', trainX, delimiter=',')
	np.savetxt('data/cancer/cancer_raw/X_test.csv', testX, delimiter=',')
	np.savetxt('data/cancer/cancer_raw/y_train.csv', trainy, delimiter=',')
	np.savetxt('data/cancer/cancer_raw/y_test.csv', testy, delimiter=',')


	# --------- Simulated data -----------
	# 3 centers, large, 2 features
	X, y = make_blobs(n_samples=1200, centers=3, n_features=2, cluster_std=2, random_state=2)
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=37)
	np.savetxt('data/simulated/simulated_raw/X_train_3c_large_2f.csv', X_train, delimiter=',')
	np.savetxt('data/simulated/simulated_raw/X_test_3c_large_2f.csv', X_test, delimiter=',')
	np.savetxt('data/simulated/simulated_raw/y_train_3c_large_2f.csv', y_train, delimiter=',')
	np.savetxt('data/simulated/simulated_raw/y_test_3c_large_2f.csv', y_test, delimiter=',')

	# 3 centers, large, 8 features
	X, y = make_blobs(n_samples=1200, centers=3, n_features=8, cluster_std=2, random_state=2)
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=37)
	np.savetxt('data/simulated/simulated_raw/X_train_3c_large_8f.csv', X_train, delimiter=',')
	np.savetxt('data/simulated/simulated_raw/X_test_3c_large_8f.csv', X_test, delimiter=',')
	np.savetxt('data/simulated/simulated_raw/y_train_3c_large_8f.csv', y_train, delimiter=',')
	np.savetxt('data/simulated/simulated_raw/y_test_3c_large_8f.csv', y_test, delimiter=',')

	# 2 centers, large, 2 features
	X, y = make_blobs(n_samples=1200, centers=2, n_features=2, cluster_std=2, random_state=2)
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=37)
	np.savetxt('data/simulated/simulated_raw/X_train_2c_large_2f.csv', X_train, delimiter=',')
	np.savetxt('data/simulated/simulated_raw/X_test_2c_large_2f.csv', X_test, delimiter=',')
	np.savetxt('data/simulated/simulated_raw/y_train_2c_large_2f.csv', y_train, delimiter=',')
	np.savetxt('data/simulated/simulated_raw/y_test_2c_large_2f.csv', y_test, delimiter=',')
	
	# 2 centers, large, 8 features
	X, y = make_blobs(n_samples=1200, centers=2, n_features=8, cluster_std=2, random_state=2)
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=37)
	np.savetxt('data/simulated/simulated_raw/X_train_2c_large_8f.csv', X_train, delimiter=',')
	np.savetxt('data/simulated/simulated_raw/X_test_2c_large_8f.csv', X_test, delimiter=',')
	np.savetxt('data/simulated/simulated_raw/y_train_2c_large_8f.csv', y_train, delimiter=',')
	np.savetxt('data/simulated/simulated_raw/y_test_2c_large_8f.csv', y_test, delimiter=',')


	# 3 centers, small, 2 features
	X, y = make_blobs(n_samples=200, centers=3, n_features=2, cluster_std=2, random_state=2)
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=37)
	np.savetxt('data/simulated/simulated_raw/X_train_3c_small_2f.csv', X_train, delimiter=',')
	np.savetxt('data/simulated/simulated_raw/X_test_3c_small_2f.csv', X_test, delimiter=',')
	np.savetxt('data/simulated/simulated_raw/y_train_3c_small_2f.csv', y_train, delimiter=',')
	np.savetxt('data/simulated/simulated_raw/y_test_3c_small_2f.csv', y_test, delimiter=',')

	# 3 centers, small, 8 features
	X, y = make_blobs(n_samples=200, centers=3, n_features=8, cluster_std=2, random_state=2)
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=37)
	np.savetxt('data/simulated/simulated_raw/X_train_3c_small_8f.csv', X_train, delimiter=',')
	np.savetxt('data/simulated/simulated_raw/X_test_3c_small_8f.csv', X_test, delimiter=',')
	np.savetxt('data/simulated/simulated_raw/y_train_3c_small_8f.csv', y_train, delimiter=',')
	np.savetxt('data/simulated/simulated_raw/y_test_3c_small_8f.csv', y_test, delimiter=',')

	# 2 c8nters, small, 2 features
	X, y = make_blobs(n_samples=200, centers=2, n_features=2, cluster_std=2, random_state=2)
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=37)
	np.savetxt('data/simulated/simulated_raw/X_train_2c_small_2f.csv', X_train, delimiter=',')
	np.savetxt('data/simulated/simulated_raw/X_test_2c_small_2f.csv', X_test, delimiter=',')
	np.savetxt('data/simulated/simulated_raw/y_train_2c_small_2f.csv', y_train, delimiter=',')
	np.savetxt('data/simulated/simulated_raw/y_test_2c_small_2f.csv', y_test, delimiter=',')

	# 2 centers, small, 8 features
	X, y = make_blobs(n_samples=200, centers=2, n_features=8, cluster_std=2, random_state=2)
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=37)
	np.savetxt('data/simulated/simulated_raw/X_train_2c_small_8f.csv', X_train, delimiter=',')
	np.savetxt('data/simulated/simulated_raw/X_test_2c_small_8f.csv', X_test, delimiter=',')
	np.savetxt('data/simulated/simulated_raw/y_train_2c_small_8f.csv', y_train, delimiter=',')
	np.savetxt('data/simulated/simulated_raw/y_test_2c_small_8f.csv', y_test, delimiter=',')