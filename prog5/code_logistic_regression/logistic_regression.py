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
        # N, d
        # if X.ndim == 1:
        #     N = len(X)
        #     d = 1 ## assuming we wont ever get a 1-sample, multi-feature X array
        # else:
        #     N, d = X.shape 
        N, d = X.shape 
        if (self.w == None): 
            self.w = np.array([[0],]*(d))
        
        # print(f"X.shape: {X.shape}")
        # print(f"self.w.shape: {self.w.shape}")
        # print(f"y.shape: {y.shape}")

        if SGD is False:
            s = (X@self.w)
            s = s * y
            self.w = self.w + ((eta/N)*(X.T@(y*LogisticRegression._v_sigmoid(s))))


    
    def predict(self, X):
        ''' parameters:
                X: n x d matrix; n samples; each has d features, excluding the bias feature. 
            return: 
                n x 1 matrix: each row is the probability of each sample being positive. 
        '''
        # theta(w.T@x) -- probability of x to have a +1 label
        print(f"X.shape: {X.shape}")
        # X = np.insert(X, 0, 1, axis=0)
        # print(f"X.shape after insert: {X.shape}")
        print(f"self.w.T.shape: {self.w.T.shape}")
        return LogisticRegression._v_sigmoid(X@self.w.T)
    
    
    def error(self, X, y):
        ''' parameters:
                X: n x d matrix; n samples; each has d features, excluding the bias feature. 
                y: n x 1 matrix; each row is a labels of +1 or -1.
            return:
                The number of misclassified samples. 
                Every sample whose sigmoid value > 0.5 is given a +1 label; otherwise, a -1 label.
        '''
        N,d = X.shape
        # print("X shape ",X.shape) # 280, 34
        # print("w shape ", self.w.shape) # 34, 1
        # print("y ",y.shape) # 280, 1
        
        weightedSampleDotProduct = X@self.w # slides say w.T but I think w is transposed somehow already. shape = 280, 1
        # print("X@w:   ",weightedSampleDotProduct.shape) #(1,280)

        # getting same result for all GD. perhaps .multiply is applying y to every sample? investigate to find a 1-1 multiplier (or write with for loop).

        ywx = np.multiply(y,weightedSampleDotProduct)
        # print(f"ywx shape: {ywx.shape}\nywx: {ywx}")
        results = LogisticRegression._v_sigmoid(ywx)
        # print(results)

        ret = 0
        for x in range(0,len(results)):
            if (results[x][0] < 0.5):
                ret += 1
        return ret
        '''
        negativeY = -1 * y
        crossEntropyExponent = weightedSampleDotProduct.T@negativeY
        # print(f"in error method. type: {type(crossEntropyExponent)}, shape: {crossEntropyExponent.shape}")
        
        vectorized_e_to_arg = np.vectorize(math.exp)
        eToYWX = vectorized_e_to_arg(crossEntropyExponent) 
        OnePlus_e_ToYWX = 1 + eToYWX # applies to every cell in matrix
        
        sumOfErrors = vectorized_e_to_arg(OnePlus_e_ToYWX) # calling sum because we used dot product, which summed each element..
        
        # print(f"sumOfErrors: {sumOfErrors}")
        sumDividedByN = (1/N)*sumOfErrors
        ret = sumDividedByN[0][0]
        print(f"ret: {ret}")
        return (ret)
        '''
    

    def _v_sigmoid(s):
        '''
            vectorized sigmoid function
            
            s: n x 1 matrix. Each element is real number represents a signal. 
            return: n x 1 matrix. Each element is the sigmoid function value of the corresponding signal. 
        '''
            
        # Hint: use the np.vectorize API
        vec_sigmoid = np.vectorize(LogisticRegression._sigmoid)
        return vec_sigmoid(s)
    
    
        
    def _sigmoid(s):
        ''' s: a real number
            return: the sigmoid function value of the input signal s
        '''
        return (1.0 / (1.0 + (math.exp(s * (-1.0)))))