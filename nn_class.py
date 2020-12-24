import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense

#read data
#data = pd.read_csv('/Users/lvbenson/Research_Projects/final-project-group-7/data/wpbc.data',delimiter=',')

#fix categorical variables



class NeuralNetwork():
    def __init__(self, data, activation_function, output_activation, num_layers, num_neurons,epochs):
        
        #self.weight_init = weight_init
        self.activation_function = activation_function
        self.output_activation = output_activation
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.epochs = epochs
        self.data = data


    def model_(self):
        lb_make = LabelEncoder()
        self.data["outcome_code"] = lb_make.fit_transform(self.data["outcome"]) #convert outcome to numbers (N = 0, R = 1)
        self.data = self.data.drop('lymph_node_status',axis=1) #this feature has missing variables
        self.data = self.data.drop('outcome',axis=1) #objects

        labels = self.data['outcome_code']
        features = self.data.iloc[:,:-1]

        X=features
        y=np.ravel(labels)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2) 
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = Sequential()
        #need at least the input layer
        model.add(Dense(self.num_neurons, activation=self.activation_function, input_shape=(33,))) #HL 1 has 8 nodes
        #input layer has input shape of number of features. number of layers specifies number of HIDDEN layers.
        if self.num_layers > 1:
            for _ in range(self.num_layers-1):
                model.add(Dense(8, activation=self.activation_function)) #HL 2 has 8 nodes
        #output layer
        model.add(Dense(1, activation=self.output_activation)) #output has 1 node

        model.compile(loss='binary_crossentropy',
              optimizer='sgd', #gradient descent
              metrics=['accuracy']) #accuracy of model

        model.fit(X_train, y_train,epochs=self.epochs, batch_size=1, verbose=1)

        _, accuracy = model.evaluate(X_train,y_train) #accuracy of test model
        #print('Accuracy: %.2f' % (accuracy*100))
        y_pred = model.predict(X_test) #stores all of the predictions
        
        score = model.evaluate(X_test, y_test,verbose=1)
        print(score[1])
        return(score[1])



data = pd.read_csv('/Users/lvbenson/Research_Projects/final-project-group-7/data/wpbc.data',delimiter=',')
#NeuralNetwork(data,'relu','sigmoid',2,8,15)

#plotting the accuracies

activation_list = ['relu','sigmoid','tanh','softmax']
output_activation_list = ['relu','sigmoid','tanh','softmax']
scores = []
for act in activation_list:
    for out_act in output_activation_list:
        nn = NeuralNetwork(data,act,out_act,2,8,15)
        val = nn.model_()
        #plt.plot(val,label=act)
        scores.append(val)
plt.plot(scores)
plt.show()








