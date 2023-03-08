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
        X = MyUtils.z_transform(X, self.degree) 
        X = np.insert(X, 0, 1, axis=1)
        N, d = X.shape 
        self.w = np.array([[0],]*(d))

        if SGD is False:
            for i in range(iterations):
                s = y*(np.dot(X,self.w))
                
                term1 = (1.0-((2.0*lam*eta)/N))*self.w
                
                term2 = (eta/N)*(np.dot(X.T,(y * LogisticRegression._v_sigmoid((s * -1.0)))))
                self.w = term1 + term2
                
        
        else: # SGD is true
            batches_per_loop = math.ceil(float(len(X) )/ float(mini_batch_size))


            if (len(X) % mini_batch_size != 0):
                remainderRunLength = math.remainder(len(X),mini_batch_size)
                finalInLoopSignal = max(1, (batches_per_loop % (batches_per_loop - 1))) # need max to avoid divide by zero if mbs == len(X)
                
                for i in range(0,math.ceil(iterations/batches_per_loop)):
                    for j in range(0,batches_per_loop):
                        # CHECK FOR EARLY STOP
                        if ((i * batches_per_loop) + j > iterations): # may occur on last i
                            print(f"early stop from iterations at i: {i} j: {j}")
                            break
                        if ((i * batches_per_loop) + j > len(X)): # may occur whenever i * batch_per_loop isn't evenly divisible by by sample size: len(X) 
                            print(f"early stop from batches at i: {i} j: {j}")
                            break

                        if (j == (batches_per_loop-1)): # and remainderRunLength > 0)... handled outside of loop for performance
                            localRunLength = int(remainderRunLength)
                        else:
                            localRunLength = mini_batch_size
                        
                        # N_Prime = localRunLength
                        start = ((i*batches_per_loop) + j) % len(X)
                        # print(f"start: {start}")
                        end = (start + localRunLength) # % len(X) ???
                        assert end <= len(X), "in IF: end is greater than len(X)"
                        sPrime = y[start:end] * (np.dot(X[start:end], self.w))
                        N_Prime = X[start:end].shape[0]
                        
                        term1 = (1.0 - ((2.0*lam*eta)/N_Prime))*self.w
                        term2 = (eta/N_Prime) * np.dot(X[start:end].T,(y[start:end] * LogisticRegression._v_sigmoid(-1.0 * sPrime)))
                        self.w = term1 + term2
            
            else:
                # skips partial runs (i.e. remainder of (10,000/280)), so missing out of a tiny amount of training.
                # should be significant in testing if iterations are sufficiently high but is a wishlist improvement
                
                for i in range(0,iterations): # cast won't truncate because we checked that len(X) % mini_batch_size == 0
                    
                    
                    start = ((i*mini_batch_size)) % len(X)
                    end = (start + mini_batch_size) # % len(X) ???

                    assert end <= len(X), "in ELSE end is greater than len(X)"

                    N_Prime = X[start:end].shape[0]
                    sPrime = y[start:end] * np.dot(X[start:end], self.w)

                    term1 = (1.0 - ((2.0*lam*eta)/N_Prime))*self.w
                    term2 = (eta/N_Prime) * np.dot(X[start:end].T,(y[start:end] * LogisticRegression._v_sigmoid(-1.0 * sPrime)))
                    self.w = term1 + term2
                

    
    def predict(self, X):
        ''' parameters:
                X: n x d matrix; n samples; each has d features, excluding the bias feature. 
            return: 
                n x 1 matrix: each row is the probability of each sample being positive. 
        '''
        X = MyUtils.z_transform(X, self.degree) 
        X = np.insert(X, 0, 1, axis=1)
        return LogisticRegression._v_sigmoid(np.dot(X,self.w))
    
    
    def error(self, X, y):
        ''' parameters:
                X: n x d matrix; n samples; each has d features, excluding the bias feature. 
                y: n x 1 matrix; each row is a labels of +1 or -1.
            return:
                The number of misclassified samples. 
                Every sample whose sigmoid value > 0.5 is given a +1 label; otherwise, a -1 label.
        '''
        N,d = X.shape
        X = MyUtils.z_transform(X, self.degree) 
        X = np.insert(X, 0, 1, axis=1)
        

        predictions = np.dot(X,self.w)
        predictions = np.sign(predictions)
        # print(f"predictions: {predictions[0:5]}")
        predictions = predictions - 0.1 # treating 0s as -1. 0's. 0's can be -0 or +0 and will throw error off
        predictions = np.sign(predictions)
        # print(f"predictions: {predictions[0:5]}")
        number_of_errors = np.sum(predictions != y) # boolean evaluates to 1 if != and 0 if ==

        return number_of_errors
      
    

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