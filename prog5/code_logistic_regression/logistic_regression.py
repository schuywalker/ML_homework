########## >>>>>> Put your full name and 6-digit EWU ID here. 

# Implementation of the logistic regression with L2 regularization and supports stachastic gradient descent




import numpy as np
import math
import sys
sys.path.append("..")

from code_misc.utils import MyUtils



class LogisticRegression:
    def __init__(self):
        self.w = None
        self.degree = 1

        

    def fit(self, X, y, lam = 0, eta = 0.01, iterations = 1000, SGD = False, mini_batch_size = 1, degree = 1):
        ''' Save the passed-in degree of the Z space in `self.degree`. 
            Compute the fitting weight vector and save it `in self.w`. 
         
            Parameters: 
                X: n x d matrix of samples; every sample has d features, excluding the bias feature. 
                y: n x 1 vector of lables. Every label is +1 or -1. 
                lam: the L2 parameter for regularization
                eta: the learning rate used in gradient descent
                iterations: the number of iterations used in GD/SGD. Each iteration is one epoch if batch GD is used. 
                SGD: True - use SGD; False: use batch GD
                mini_batch_size: the size of each mini batch size, if SGD is True.  
                degree: the degree of the Z space
        '''
        self.degree = degree
        # X = MyUtils.z_transform(X, self.degree) 
        X = np.insert(X, 0, 1, axis=0)
        y = np.insert(y, 0, 1, axis=0)
        N, d = X.shape 
        if (self.w == None): 
            self.w = np.array([[0],]*(d))
        
        print(f"X.shape: {X.shape}")
        print(f"self.w.shape: {self.w.shape}")
        print(f"y.shape: {y.shape}")

        if SGD is False:
            V_S = np.vectorize(LogisticRegression._sigmoid)
            s = (X@self.w)
            # s = y * s
            s = s * y
            print(f"s:{s}\n\n")

            print(f"V_S(s): {V_S(s)}\n\n")

            self.w = self.w + ((eta/N)*(X.T@(y*V_S(s))))


    
    def predict(self, X):
        ''' parameters:
                X: n x d matrix; n samples; each has d features, excluding the bias feature. 
            return: 
                n x 1 matrix: each row is the probability of each sample being positive. 
        '''
        # theta(w.T@x) -- change for x to have a +1 label
        return LogisticRegression._sigmoid(self.w.T@X)
    
    
    def error(self, X, y):
        ''' parameters:
                X: n x d matrix; n samples; each has d features, excluding the bias feature. 
                y: n x 1 matrix; each row is a labels of +1 or -1.
            return:
                The number of misclassified samples. 
                Every sample whose sigmoid value > 0.5 is given a +1 label; otherwise, a -1 label.
        '''
        N,d = X.shape
        print("X shape ",X.shape) # 280, 34
        print("w shape ", self.w.shape) # 34, 1
        print("y ",y.shape) # 280, 1
        
        
        weightedSampleDotProduct = X@self.w # slides say w.T but I think w is transposed somehow already. shape = 280, 1
        # print(f"\n\nTEST:\n@ symbol {weightedSampleDotProduct} \n\ndot method: {X.dot(self.w)}")
        
        # print(weightedSampleDotProduct.T.shape) (1,280)
        # print(y)
        negativeY = -1 * y
        crossEntropyExponent = negativeY@weightedSampleDotProduct.T
        print(f"in error method. type: {type(crossEntropyExponent)}, shape: {crossEntropyExponent.shape}")
        
        # # OR ??
        # y_dot_w = (-1 * y)@self.w.T
        # crossEntropyExponent = (y_dot_w@X.T)
        
        
        


        vectorized_e_to_arg = np.vectorize(math.exp)
        
        eToYWX = vectorized_e_to_arg(crossEntropyExponent) 
        OnePlus_e_ToYWX = 1 + eToYWX # applies to every cell in matrix
        
        sumAsVector = vectorized_e_to_arg(OnePlus_e_ToYWX) # calling sum because we used dot product, which summed each element..
        
        print(f"sum: {sumAsVector}")
        sumAsNumber = np.ma(sumAsVector)
        print(f"sumAsNum: {sumAsNumber}")
        return ((1/N)*sumAsNumber)
    
    

    def _v_sigmoid(s):
        '''
            vectorized sigmoid function
            
            s: n x 1 matrix. Each element is real number represents a signal. 
            return: n x 1 matrix. Each element is the sigmoid function value of the corresponding signal. 
        '''
            
        # Hint: use the np.vectorize API

    
    
        
    def _sigmoid(s):
        ''' s: a real number
            return: the sigmoid function value of the input signal s
        '''
        return (1.0 / (1.0 + (math.exp(s * (-1.0)))))