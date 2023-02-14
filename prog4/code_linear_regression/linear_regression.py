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
        # print("\nCLOSED FORM FIT\n")
        X = np.insert(X, 0, 1, axis =1) # add bias feature
        d = len(X[0]) - 1 # d = number of non-bias features in each sample. AKA d = X.shape[1]
        # n, d = np.shape(X)
        
        
        if self.w == None:
            self.w = np.zeros(d+1) # change data type here if needed
        # w = a weight vector of size d+1. w = [w0...wd].T

        # print(f"X shape:\n{np.shape(X)}")
        # print(f"X.T shape:\n{np.shape(X.T)}")
        # print(f"Y shape:\n{np.shape(y)}")

        # # pre regularization (original subproject)
        # more optimal to put second term inside inverse function than outside     
        # self.w = w_star
        
        I = np.identity(d+1)
        w_star = np.linalg.pinv(((X.T@X)+(lam*I)))@(X.T@y) # very different results is second term is outside pinv!!
        self.w = w_star
         
        # print(f"self.w:\n{self.w}")



    def _fit_gd(self, X, y, lam = 0, eta = 0.01, epochs = 1000):
        ''' Compute the weight vector using the gradient desecent based method.
            Save the result in self.w

            X: n x d matrix, n samples, each has d features, excluding the bias feature
            y: n x 1 matrix of labels. Each element is the label of each sample. 
        '''
        # np.random.seed()
        print("\nGRADIENT DESCENT FIT\n")

        X = np.insert(X, 0, 1, axis =1)

        n, d = np.shape(X)-1
        
        I = np.identity(d+1) # - do red code, then while epochs > 0, w = w - nDelE(w)
        # print(f"I:\n{I}")
        # print(f"I shape:\n{np.shape(I)}")
        # X = np.insert(X, 0, 1, axis =1) # add bias feature
        XTX = X.T@X
        # print(f"XTX shape:\n{np.shape(XTX)}")
        
        # if self.w == None:
        #     self.w = np.zeros(d+1)
        
        # how he did w init: 
        # self.w = np.random.rand(d+1,1) # random number from [0,1]
        # self.w = ((self.w * 2) - 1)/math.sqrt(d) # random numbers from [-1,sqrt(d),1/sqrt(d)]        
        # or...
        self.w = np.array([[0],]*(d+1))

        term1 = (I - ((2*eta)/n)*(XTX+(lam*I))) #only difference from before is +(lam*I)
        term2 = ((2*eta)/n)*(X.T@y)
        
        for e in range(0,epochs): # switch to vector ops for much better performance!!!!
            self.w = (term1@self.w) + term2
            # print(f"self.w\n{self.w}\n")


    
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
        # X = MyUtils.normalize_0_1(X)
        X = np.insert(X, 0, 1, axis =1)
        # sum = X @ self.w - y
        print(f"X {np.shape(X)}")
        print(f"w {np.shape(self.w)}")
        print(f"y {np.shape(y)}") 

        sum = (X @ self.w) - y
        sum = sum**2
        # print("sum:\n",sum)
        errorTotals = np.sum(sum)

        # print("\n\nsum**2\n",sum**2)
        # for i in range(len(X)):
        #     sum += (self.w[i+1]-y[i])**2
        mse = ((1/len(X))/errorTotals)
        self.MSE.append(mse)
        return mse
        
        # use .dot(sum)
        # X @ w is y* y hat etc.

        
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