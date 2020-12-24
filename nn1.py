#neural network class, using backprop, different activation functions, different weight initializations 



#imports

import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import LabelEncoder


#read data
data = pd.read_csv('/Users/lvbenson/Research_Projects/final-project-group-7/data/wpbc.data')
#fix categorical variables
lb_make = LabelEncoder()
data["outcome_code"] = lb_make.fit_transform(data["outcome"]) #convert outcome to numbers (N = 0, R = 1)
data = data.drop('lymph_node_status',axis=1) #this feature has missing variables
data = data.drop('outcome',axis=1) #objects
#preprocessing
X = data.iloc[:,:-1] # Features 
y = data.iloc[:,-1] # Target 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2) 
sc = StandardScaler() 
X_train = sc.fit_transform(X_train) 

#neural network class
#weight initialization methods: Zeros, random, he, xavier
#activation functions: Relu, sigmoid, tanh, leaky relu

class NeuralNetwork():
    def __init__(self, weight_alg = 'random', activation_function = 'sigmoid',num_layers=4,num_HL=2,inputs=2,outputs=4):

        self.params={} #initialized values of weights and biases
        self.params_h = []
        self.num_layers = num_layers #total number of layers (input layer, hidden layers, output layers)
        self.num_HL = num_HL #number of neurons in hidden layers
        self.inputs = inputs #number of neurons in input layer
        self.outputs = outputs #number of neurons in output layer 
        self.hidden_sizes = [inputs, [num_HL for i in num_layers[i]],outputs]
        self.layer_sizes = [item for sublist in self.hidden_sizes for item in sublist]
        self.activation_function = activation_function #select which activation function to use
        self.weight_alg = weight_alg #select which weight initialization


        #weight initializations
        if weight_alg == 'zeros':
            #weights and biases are initialized with zeros. Biases will have no effect in this case.
            for i in range(1,self.num_layers+1):
                self.params["W"+str(i)]=np.zeros((self.layer_sizes[i-1],self.layer_sizes[i])) #iterates through every layer, sets weights and biases to all zeros 
                self.params["B"+str(i)]=np.zeros((1,self.layer_sizes[i]))

        elif weight_alg == "zeros":
            for i in range(1,self.num_layers+1):
                self.params["W"+str(i)] = np.random.randn(self.layer_sizes[i-1],self.layer_sizes[i])
                self.params["B"+str(i)] = np.random.randn(1,self.layer_sizes[i])


        elif weight_alg == "he":
            for i in range(1,self.num_layers+1):
                self.params["W"+str(i)] = np.random.randn(self.layer_sizes[i-1],self.layer_sizes[i])*np.sqrt(2/self.layer_sizes[i-1])
                self.params["B"+str(i)] = np.random.randn(1,self.layer_sizes[i])

        elif weight_alg == "xavier":
            for i in range(1,self.num_layers+1):
                self.params["W"+str(i)]=np.random.randn(self.layer_sizes[i-1],self.layer_sizes[i])*np.sqrt(1/self.layer_sizes[i-1])
                self.params["B"+str(i)]=np.random.randn(1,self.layer_sizes[i])


def input_activation(self, X):     
    if self.activation_function == "sigmoid":      
        return 1.0/(1.0 + np.exp(-X))    
    elif self.activation_function == "tanh":      
        return np.tanh(X)    
    elif self.activation_function == "relu":      
        return np.maximum(0,X)    
    elif self.activation_function == "leaky_relu":      
        return np.maximum(self.leaky_slope*X,X)


def deriv_activation(self, X): #computes derivative
    if self.activation_function == "sigmoid":
      return X*(1-X) 
    elif self.activation_function == "tanh":
      return (1-np.square(X))
    elif self.activation_function == "relu":
      return 1.0*(X>0)
    elif self.activation_function == "leaky_relu":
      d=np.zeros_like(X)
      d[X<=0]=0.01 #setting the leaky slope to be 0.01
      d[X>0]=1
      return d

 def forward_pass(self, X, params = None):
    if params is None:
        params = self.params
    self.A1 = np.matmul(X, params["W1"]) + params["B1"] #dot product between the input & weights w and adds bias b
    self.H1 = self.forward_activation(self.A1) 
    self.A2 = np.matmul(self.H1, params["W2"]) + params["B2"] #applies the activation function
    self.H2 = self.softmax(self.A2) 
    return self.H2

 def grad(self, X, Y, params = None):
    if params is None:
      params = self.params 
      
      self.forward_pass(X, params)
      m = X.shape[0]
      self.gradients["dA2"] = self.H2 - Y # (N, 4) - (N, 4) -> (N, 4)
      self.gradients["dW2"] = np.matmul(self.H1.T, self.gradients["dA2"]) # (2, N) * (N, 4) -> (2, 4)
      self.gradients["dB2"] = np.sum(self.gradients["dA2"], axis=0).reshape(1, -1) # (N, 4) -> (1, 4)
      self.gradients["dH1"] = np.matmul(self.gradients["dA2"], params["W2"].T) # (N, 4) * (4, 2) -> (N, 2)
      self.gradients["dA1"] = np.multiply(self.gradients["dH1"], self.grad_activation(self.H1)) # (N, 2) .* (N, 2) -> (N, 2)
      self.gradients["dW1"] = np.matmul(X.T, self.gradients["dA1"]) # (2, N) * (N, 2) -> (2, 2)
      self.gradients["dB1"] = np.sum(self.gradients["dA1"], axis=0).reshape(1, -1) # (N, 2) -> (1, 2)
    
