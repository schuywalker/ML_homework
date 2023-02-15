########## >>>>>> Put your full name and 6-digit EWU ID here. 
# Schuyler Asplin 


# DONT FORGET TO PUT EWU ID!!!!!!!!!!

# Implementation of the linear regression with L2 regularization.
# It supports the closed-form method and the gradient-desecent based method. 



import numpy as np
import math
import sys

# from utils import MyUtils
sys.path.append("..")

# from misc.utils import MyUtils
# import utils as MyUtils
# misc.utils import MyUtils

from misc.utils import MyUtils

class LinearRegression:
    def __init__(self):
        self.w = None   # The (d+1) x 1 numpy array weight matrix
        self.degree = 1
        # self.MSE_CF = []
        # self.MSE_GD = []
        self.MSE = []


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
        
        X = np.insert(X, 0, 1, axis =1) # add bias feature
        d = len(X[0]) - 1 # d = number of non-bias features in each sample. AKA d = X.shape[1]
        # n, d = np.shape(X)
        
        
        if self.w == None:
            self.w = np.zeros(d+1) # change data type here if needed
        # w = a weight vector of size d+1. w = [w0...wd].T

        # # pre regularization (original subproject)
        # more optimal to put second term inside inverse function than outside     
        
        I = np.identity(d+1)
        w_star = np.linalg.pinv(((X.T@X)+(lam*I)))@(X.T@y) # very different results is second term is outside pinv!!
        self.w = w_star
         
        



    def _fit_gd(self, X, y, lam = 0, eta = 0.01, epochs = 1000):
        ''' Compute the weight vector using the gradient desecent based method.
            Save the result in self.w

            X: n x d matrix, n samples, each has d features, excluding the bias feature
            y: n x 1 matrix of labels. Each element is the label of each sample. 
        '''

        X = np.insert(X, 0, 1, axis =1)

        n, d = np.shape(X)
        
        I = np.identity(d) # - do red code, then while epochs > 0, w = w - nDelE(w)
        
        XTX = X.T@X
        
        self.w = np.array([[0],]*(d))

        term1 = (I - ((2*eta)/n)*(XTX+(lam*I))) #only difference from before is +(lam*I)
        term2 = ((2*eta)/n)*(X.T@y)
        
        for e in range(0,epochs): # switch to vector ops for much better performance!!!!
            self.w = (term1@self.w) + term2


    
    def predict(self, X):
        ''' parameter:
                X: n x d matrix, the n samples, each has d features, excluding the bias feature
            return:
                n x 1 matrix, each matrix element is the regression value of each sample
        '''
        ## Enter your code here that produces the label vector for the given samples saved
        ## in the matrix X. Make sure your predication is calculated at the same Z
        ## space where you trained your model. Make sure X has been normalized via the same
        ## normalization used by the training process. 

        '''
        allows for multiple samples to be submitted. x= transform, insert, return X@self.w 
        X = np.insert(X, 0, 1, axis =1)
        '''

        X = MyUtils.z_transform(X, self.degree)
        X = np.insert(X, 0, 1, axis =1)
        return X@self.w #slides say w.T but dimension sizes dont line up. check back here later

    
    def error(self, X, y):
        ''' parameters:
                X: n x d matrix of future samples
                y: n x 1 matrix of labels
            return: 
                the MSE for this test set (X,y) using the trained model
        '''
        X = MyUtils.z_transform(X, self.degree)
        X = np.insert(X, 0, 1, axis =1)

        sum = (X @ self.w) - y # () shouldnt make a difference but check to be sure
        sum = sum**2
        errorTotals = np.sum(sum)

        mse = ((1/len(X))*errorTotals)
        self.MSE.append(mse)
        return mse
        
        ## Enter your code here that calculates the MSE between your predicted
        ## label vector and the given label vector y, for the sample set saved in matraix X
        ## Make sure your predication is calculated at the same Z space where you trained your model. 
        ## Make sure X has been normalized via the same normalization used by the training process. 

