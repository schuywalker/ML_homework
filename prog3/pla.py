# Implementation of the perceptron learning algorithm. Support the pocket version for linearly unseparatable data. 
# Authro: Bojian Xu, bojianxu@ewu.edu

#Important observation: 
#    - The PLA can increase or decrease $w[0]$ by 1 per update, so if there is a big difference between $w^*[0]$ and the #initial value of $w[0]$, the PLA is likely to take a long time before it halts. However, the theoretical bound $O((L/d)^2)$ #step of course still holds, where $L = \max\{\lVert x\rVert\}$ and $d$ is the margine size.
#    - This can solved by always have feature values within [0,1], because by doing so, the $x_0=1$ becomes relatively larger (or one can also say $x_0$ becomes fairly as important as other feathers), which makes the changes to $w[0]$ much faster. This is partially why nueral network requires all feature value to be [0,1] --- the so-called data normalization process!!!

# Another reason for normalizing the feature into [0,1] is: no matter which Z space the samples are tranformed to, the Z-space sample features will still be in the [0,1] range. 

import numpy as np

#import sys
#sys.path.append("..")

from utils import MyUtils
import pandas as pd



class PLA:
    def __init__(self, degree=1):
        self.w = None
        self.degree = degree
        
    def fit(self, X, y, pocket = True, epochs = 100):
        ''' find the classifer weight vector and save it in self.w
            X: n x d matrix, i.e., the bias feature is not included. 
            It is assumed that X is already normalized be data preprocessing. 
            y: n x 1 vector of {+1, -1}
            degree: the degree of the Z space
            return self.w
        '''
        
        if(self.degree > 1):
            Z = MyUtils.z_transform(X, degree=self.degree)
    
            
        ### BEGIN YOUR SOLUTION
    
        # ADDING BIAS FEATURE
        # for s in range(0, len(X)):
        #     X[s].insert(0, 1)
        # X_DF = pd.DataFrame(X)
        # print("X with bias feature\n",X_DF)


        misclassifiedSamples = True
        self.w = np.zeros((len(X[0]),1))
        # print("w: ",self.w)
        wT = self.w.T

        if (pocket == True):        
            while(misclassifiedSamples and epochs > 0):
                misclassifiedSamples = False
                for s in range(len(X)):
                    assessment = wT.dot(X[s])
                    if (assessment == 0 and y[s] == 0):
                        print("wow")
                    elif ((assessment > 0 and y[s] == -1) or (assessment < 0 and y[s] == 1)):
                        # sample misclassified. move w line closer to x
                        w = w + (y*X[s])
                        misclassifiedSamples = True
                epochs -= 1
       
        elif (pocket == False):
           while(misclassifiedSamples):
                misclassifiedSamples = False
                for s in range(len(X)):
                    assessment = wT.dot(X[s])
                    if (assessment == 0 and y[s] == 0):
                        print("wow")
                    elif ((assessment > 0 and y[s] == -1) or (assessment < 0 and y[s] == 1)):
                        # sample misclassified. move w line closer to x
                        w = w + (y*X[s])
                        misclassifiedSamples = True


        print("\n\nending w:\n",self.w)
    
#         raise NotImplementedError()
        ### END YOUR SOLUTION
            
                          
        return self.w
    
                          


    def predict(self, X):
        ''' x: n x d matrix, i.e., the bias feature is not included.
            return: n x 1 vector, the labels of samples in X
        '''
        if(self.degree > 1):
            X = MyUtils.z_transform(X, degree = self.degree)

    
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

        
        


    def error(self, X, y):
        ''' X: n x d matrix, i.e., the bias feature is not included. 
            y: n x 1 vector
            return the number of misclassifed elements in X using self.w
        '''
        
        if(self.degree > 1):
            X = MyUtils.z_transform(X, degree = self.degree)

        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
            


    