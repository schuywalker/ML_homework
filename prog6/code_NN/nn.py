# Place your EWU ID and Name here. 

### Delete every `pass` statement below and add in your own code. 



# Implementation of the forwardfeed neural network using stachastic gradient descent via backpropagation
# Support parallel/batch mode: process every (mini)batch as a whole in one forward-feed/backtracking round trip. 



import numpy as np
import math
from . import math_util as mu
from . import nn_layer


class NeuralNetwork:
    
    def __init__(self):
        self.layers = []     # the list of L+1 layers, including the input layer. 
        self.L = -1          # Number of layers, excluding the input layer. 
                             # Initting it as -1 is to exclude the input layer in L. 
    
    
    def add_layer(self, d = 1, act = 'tanh'):
        ''' The newly added layer is always added AFTER all existing layers.
            The firstly added layer is the input layer.
            The most recently added layer is the output layer. 
            
            d: the number of nodes, excluding the bias node, which will always be added by the program. 
            act: the choice of activation function. The input layer will never use an activation function even if it is given. 
            
            So far, the set of supported activation functions are (new functions can be easily added into `math_util.py`): 
            - 'tanh': the tanh function
            - 'logis': the logistic function
            - 'iden': the identity function
            - 'relu': the ReLU function
        '''
        self.layers.append(nn_layer.NeuralLayer(d, act))
        self.L += 1

    

    def _init_weights(self):
        ''' Initialize every layer's edge weights with random numbers from [-1/sqrt(d),1/sqrt(d)], 
            where d is the number of nonbias node of the layer
        '''
        # W rows are nodes in layer L-1, columns are nodes in current layer L
        # W[i,j] is the weight of the edge from node i in layer L-1 to node j in layer L
        for L in range(1,len(self.layers)):
            d = self.layers[L].d
            # if (L < len(self.layers)-1):
            #     d += 1 # bias node, except on last layer
            d_Lminus1 = self.layers[L-1].d
            # if (L > 1):
            #     d_Lminus1 += 1 # bias node, except on first layer
            L_W = np.random.uniform((-1/math.sqrt(d)),(1/math.sqrt(d)),(d_Lminus1,d)) # uniform distribution between (low, high)
            L_W = np.reshape(L_W, (d_Lminus1, d))
            # if (L != len(self.layers)-1):
            L_W = np.insert(L_W, 0, 1, axis=0)
            self.layers[L].W = L_W
            
            # print(f"layer {L}.W (weights from L{L-1} to L{L})\tshape:{self.layers[L].W.shape}\n")
            print(f"layer {L}.W (weights from L{L-1} to L{L})\tshape:{self.layers[L].W.shape}\n", L_W, "\n")
            
        # bias nodes initialized to 1 ???????
    
    
        
    def fit(self, X, Y, eta = 0.01, iterations = 1000, SGD = True, mini_batch_size = 1):
        ''' Find the fitting weight matrices for every hidden layer and the output layer. Save them in the layers.
          
            X: n x d matrix of samples, where d >= 1 is the number of features in each training sample
            Y: n x k vector of lables, where k >= 1 is the number of classes in the multi-class classification
            eta: the learning rate used in gradient descent
            iterations: the maximum iterations used in gradient descent
            SGD: True - use SGD; False: use batch GD
            mini_batch_size: the size of each mini batch size, if SGD is True.  
        '''
        self._init_weights()  # initialize the edge weights matrices with random numbers.

        
        
        # I will leave you to decide how you want to organize the rest of the code, but below is what I used and recommend. Decompose them into private components/functions. 

        ## prep the data: add bias column; randomly shuffle data training set. 

        ## for every iteration:
        #### get a minibatch and use it for:
        ######### forward feeding
        ######### calculate the error of this batch if you want to track/observe the error trend for viewing purpose.
        ######### back propagation to calculate the gradients of all the weights
        ######### use the gradients to update all the weight matrices. 

        pass
    
    
    
    def predict(self, X):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column.
            
            return: n x 1 matrix, n is the number of samples, every row is the predicted class id.
         '''

        pass
    
    
    def error(self, X, Y):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column. 
               n is the number of samples. 
               d is the number of (non-bias) features of each sample. 
            Y: n x k matrix, the labels of the input n samples. Each row is the label of one sample, 
               where only one entry is 1 and the rest are all 0. 
               Y[i,j]=1 indicates the ith sample belongs to class j.
               k is the number of classes. 
            
            return: the percentage of misclassfied samples
        '''
        
        pass
 
        def forward_feed(self, X):
            '''
            use in predicct and error'''
            pass

        def prep(self, X):
            ''' bias node and ... '''
            pass