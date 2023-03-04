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
                s = y*(X@self.w)
                
                
                # self.w = ((1-((2*lam*eta)/N))*self.w) + ((eta/N)*(X.T@(y*LogisticRegression._v_sigmoid(s*(-1.0))))) # non-regularization
                term1 = (1.0-((2.0*lam*eta)/N))*self.w
                # print(y_sig_s)
                term2 = (eta/N)*(X.T@(y * LogisticRegression._v_sigmoid((s * -1.0))))
                self.w = term1 + term2
                # print("w: ",self.w[:5])
        
        else: # SGD is true
            # print(f"mini batch {mini_batch_size}")
            # print(f"number of Samples X {len(X)}")
            # print(f"iterations: {iterations}")
            batches_per_loop = math.ceil(float(len(X) )/ float(mini_batch_size))
            # print(batches_per_loop)

            # defaultRunLength = (int)(mini_batch_size) delete
            

            if (len(X) % mini_batch_size != 0):
                
                remainderRunLength = math.remainder(len(X),mini_batch_size)
                finalInLoopSignal = max(1, (batches_per_loop % (batches_per_loop - 1))) # need max to avoid divide by zero if mbs == len(X)
                # print(f"remainderRunLength: {remainderRunLength} finalInLoopSignal: {finalInLoopSignal}")
                
                for i in range(0,math.ceil(iterations/batches_per_loop)):
                    
                    
                    for j in range(0,batches_per_loop):
                        # print(f"\ni: {i} j: {j}")
                        
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
                        
                        N_Prime = localRunLength
                            
                        
                        # start = (mini_batch_size*i)%len(X)
                        start = ((i*batches_per_loop) + j) % len(X)
                        # print(f"start: {start}")
                        end = (start + localRunLength) # % len(X) ???

                        assert end <= len(X), "in IF: end is greater than len(X)"


                        # print(f"end: {end}")
                        # print(f"y: {len(y)}  X: {len(X)}")

                        
                        y_prime = y[start:end] 
                        X_prime = X[start:end] 
                        
                        # y_prime = y[start:end] if (i%len(X) < len(X) and i != 559) else y[(start%len(X)):(end % len(X))] # lol annoying to fix. try simpler design with second for loop.
                        # X_prime = X[start:end] if (i%len(X) < len(X)) else X[(start%len(X)):(end % len(X))]
                        # sPrime = (y[start:end] * (X[start:end]@self.w)) if (i <= len(X)) else (y[(start%len(X)):(end % len(X))] * (X[(start%len(X)):(end % len(X))]@self.w)) 
                        # print(f"y_prime: {y_prime.shape}  X_prime: {X_prime.shape}")
                        sPrime = y_prime * (X_prime@self.w)
                        # print(f"sPrime: {sPrime}\n")
                        
                        
                        term1 = (eta/N_Prime)*(((y[start:end])*LogisticRegression._v_sigmoid(-1.0 * sPrime)).T @ X[start:end]).T
                        term2 = (1 - ((2*lam*eta)/N_Prime))*self.w
                        self.w = term1 + term2
                        # print(f"w: {self.w[:5]}\n")
            
            else:
                # skips partial runs (i.e. remainder of (10,000/280)), so missing out of a tiny amount of training.
                # should be significant in testing if iterations are sufficiently high but is a wishlist improvement
                for i in range(0,int(iterations/mini_batch_size)): # cast won't truncate because we checked that len(X) % mini_batch_size == 0
                    
                    N_Prime = mini_batch_size
                    start = ((i*mini_batch_size)) % len(X)
                    end = (start + mini_batch_size) # % len(X) ???

                    assert end <= len(X), "in ELSE end is greater than len(X)"

                    # print(f"start: {start} end: {end}")
                    # print(f"y: {len(y)}  X: {len(X)}")
                    
                    y_prime = y[start:end] 
                    X_prime = X[start:end] 
                    # print(f"y_prime: {y_prime.shape}  X_prime: {X_prime.shape}")
                    
                    
                    sPrime = y_prime * (X_prime@self.w)
                    # print(f"sPrime: {sPrime}\n")
                    
                    
                    term1 = (eta/N_Prime)*(((y[start:end])*LogisticRegression._v_sigmoid(-1.0 * sPrime)).T @ X[start:end]).T
                    term2 = (1 - ((2*lam*eta)/N_Prime))*self.w
                    self.w = term1 + term2
                    # print(f"w: {self.w[:5]}\n")
                

    
    def predict(self, X):
        ''' parameters:
                X: n x d matrix; n samples; each has d features, excluding the bias feature. 
            return: 
                n x 1 matrix: each row is the probability of each sample being positive. 
        '''
        return LogisticRegression._v_sigmoid(X@self.w)
    
    
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
        # prediction_signs = np.sign(predictions)
        # print("predictions: ",predictions[:10])
        predictions = predictions - 0.1 # treating 0s as -1. 0's and -0's can throw this off
        predictions = np.sign(predictions)
        number_of_errors = np.sum(predictions != y)

        # for i in range(len(predictions)):
        #     # print(f"pred: {np.sign(predictions[i][0])} y: {y[i][0]}")
        #     if (np.sign(predictions[i]) != y[i]):
        #         number_of_errors += 1

        return number_of_errors
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