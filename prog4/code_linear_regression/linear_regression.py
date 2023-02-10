########## >>>>>> Put your full name and 6-digit EWU ID here. 
# Schuyler Asplin 


# DONT FORGET TO PUT EWU ID!!!!!!!!!!

# Implementation of the linear regression with L2 regularization.
# It supports the closed-form method and the gradient-desecent based method. 



import numpy as np
import math
import sys
sys.path.append("..")

# from misc.utils import MyUtils
# import utils as MyUtils
# misc.utils import MyUtils
from misc.utils import MyUtils

class LinearRegression:
    def __init__(self):
        self.w = None   # The (d+1) x 1 numpy array weight matrix
        self.degree = 1

        # self.MSE only for testing purposes?? just if you want to save the results into here if you want to graph MSEs (should trend down).
        
    '''
    INIT w to zeros!!
    '''

    def fit(self, X, y, CF = True, lam = 0, eta = 0.01, epochs = 1000, degree = 1):
        ''' Find the fitting weight vector and save it in self.w. 
            
            parameters: 
                X: n x d matrix of samples, n samples, each has d features, excluding the bias feature
                y: n x 1 matrix of lables
                CF: True - use the closed-form method. False - use the gradient descent based method
                lam: the ridge regression parameter for regularization
                eta: the learning rate used in gradient descent
                epochs: the maximum epochs used in gradient descent
                degree: the degree of the Z-space
        '''
        self.degree = degree
        X = MyUtils.z_transform(X, degree = self.degree)
        # 

        if CF:
            self._fit_cf(X, y, lam)
        else: 
            self._fit_gd(X, y, lam, eta, epochs)
 


            
    def _fit_cf(self, X, y, lam = 0):
        ''' Compute the weight vector using the clsoed-form method.
            Save the result in self.w
        
            X: n x d matrix, n samples, each has d features, excluding the bias feature
            y: n x 1 matrix of labels. Each element is the label of each sample. 
        '''
        ## Enter your code here that implements the closed-form method for linear regression 
        # Goal: 'create' X and y, then compute w* = ((X.T*X)^-1)X.T*y
        
        n, d = np.shape(X)
        
        # X = MyUtils.z_transform(X,self.degree) SHOULDNT NEED, IN FIT
        # d = number of non-bias features in each sample
        # n = number of non-bias samples in training set
        # y = labels
        X = np.insert(X, 0, 1, axis =1) # add bias feature
        print(f"X after bias feature:\n{X}")
        # self.w = np.random.rand(d+1,1)
        if self.w == None:
            self.w = np.zeros(d+1) # change data type here if needed
        # w = a weight vector of size d+1. w = [w0...wd].T

        print(f"X shape:\n{np.shape(X)}")
        print(f"X.T shape:\n{np.shape(X.T)}")
        print(f"Y shape:\n{np.shape(y)}")

        w_star = np.linalg.pinv((X@X.T)@(X.T@y))        
        self.w = w_star
        print(self.w)
        print(np.shape(self.w))


        '''
        pinv(X.T@X)@(X.T@y) <- can all be one line
                    ^^^^^^^ parens make it slightly more optimal.
        d = X.shape[1]
        
        '''

    
    
    def _fit_gd(self, X, y, lam = 0, eta = 0.01, epochs = 1000):
        ''' Compute the weight vector using the gradient desecent based method.
            Save the result in self.w

            X: n x d matrix, n samples, each has d features, excluding the bias feature
            y: n x 1 matrix of labels. Each element is the label of each sample. 
        '''
        # np.random.seed()

        n, d = X.shape()
        I = np.identity(d+1) # - do red code, then while epochs > 0, w = w - nDelE(w)
        print(f"I:\n{I}")
        XTX = X.T@X
        term1 = (I - ((2*eta)/n)@XTX)@self.w
        term2 = ((2*eta)/n)@(X.T@y)
        # self.w = np.array([[0],]*(d+1))

        self.w = np.random.rand(d+1,1) # random number from [0,1]
        self.w = ((self.w * 2) - 1)/math.sqrt(d) # random numbers from [-1,sqrt(d),1/sqrt(d)]
        
        ## Enter your code here that implements the gradient descent based method
        ## for linear regression 


  

    
    def predict(self, X):
        ''' parameter:
                X: n x d matrix, the n samples, each has d features, excluding the bias feature
            return:
                n x 1 matrix, each matrix element is the regression value of each sample
        '''
        X = MyUtils.z_transform(X, self.degree)

        ## Delete the `pass` statement below.
        
        ## Enter your code here that produces the label vector for the given samples saved
        ## in the matrix X. Make sure your predication is calculated at the same Z
        ## space where you trained your model. Make sure X has been normalized via the same
        ## normalization used by the training process. 

        '''
        allows for multiple samples to be submitted. x= transform, insert, return X@self.w 
        X = np.insert(X, 0, 1, axis =1)
        '''

        pass
        

    
    
    def error(self, X, y):
        ''' parameters:
                X: n x d matrix of future samples
                y: n x 1 matrix of labels
            return: 
                the MSE for this test set (X,y) using the trained model
        '''

        ## Delete the `pass` statement below.
        '''
        gimme predicted price with real price, get MSE (sum of differences squared) / N or something
        if necessary, transform to Z-space first. use vector calculations, dont use for/while loop. itll be too slow.

        temp = X @ w -y
        get sum of all temps (all elements), square them, then divide by ..N?
        '''
        ## Enter your code here that calculates the MSE between your predicted
        ## label vector and the given label vector y, for the sample set saved in matraix X
        ## Make sure your predication is calculated at the same Z space where you trained your model. 
        ## Make sure X has been normalized via the same normalization used by the training process. 

        pass




###
'''
he does add bias feature (in cf2)
in every epoch he does the red text code. self.w = a @ self.w + b
the black lines claculate the read line (slides)
self.MSE

predict, z-thransform. x = np.insert(... axis=1)
then return the red text
'''
###