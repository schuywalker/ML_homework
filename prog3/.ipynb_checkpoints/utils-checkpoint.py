# Various tools for data manipulation. 
# Author: Bojian Xu, bojianxu@ewu.edu


import numpy as np
import math

class MyUtils:
    def rand_matrix(nb_rows = 1, nb_cols = 1): 
        ''' return a nb_row x nb_col matrix of random numbers from (-1,1)
        '''
        X = np.random.rand(nb_rows, nb_cols) * np.sign(np.random.rand(nb_rows, nb_cols)-0.5)

        return X
        
        
        
        
    def normalize_0_1(X):
        ''' Normalize the value of every feature into the [0,1] range, using formula: x = (x-x_min)/(x_max - x_min)
            1) First shift all feature values to be non-negative by subtracting the min of each column 
               if that min is negative.
            2) Then divide each feature value by the max of the column if that max is not zero. 
            
            X: n x d matrix of samples, excluding the x_0 = 1 feature. X can have negative numbers.
            return: the n x d matrix of samples where each feature value belongs to [0,1]
        '''

        n, d = X.shape
        X_norm = X.astype('float64') # Have a copy of the data in float

        for i in range(d):
            col_min = min(X_norm[:,i])
            col_max = max(X_norm[:,i])
            gap = col_max - col_min
            if gap:
                X_norm[:,i] = (X_norm[:,i] - col_min) / gap
            else:
                X_norm[:,i] = 0 #X_norm[:,i] - X_norm[:,i]
        
        return X_norm

    def normalize_neg1_pos1(X):
        ''' Normalize the value of every feature into the [-1,+1] range. 
            
            X: n x d matrix of samples, excluding the x_0 = 1 feature. X can have negative numbers.
            return: the n x d matrix of samples where each feature value belongs to [-1,1]
        '''
        # To be implemented
        n, d = X.shape
        X_norm = X.astype('float64') # Have a copy of the data in float

        for i in range(d):
            col_min = min(X_norm[:,i])
            col_max = max(X_norm[:,i])
            col_mid = (col_max + col_min) / 2
            gap = (col_max - col_min) / 2
            if gap:
                X_norm[:,i] = (X_norm[:,i] - col_mid) / gap
            else: 
                X_norm[:,i] = 0 #X_norm[:,i] - X_norm[:,i]

        return X_norm
        
        
        
    def z_transform(X, degree = 2):
        ''' Transforming training samples to the Z space
            X: n x d matrix of samples, excluding the x_0 = 1 bias feature
            degree: the degree of the Z space
            return: the n x d' matrix of samples in the Z space, excluding the z_0 = 1 feature.
            It can be mathematically calculated: d' = \sum_{k=1}^{degree} (k+d-1) \choose (d-1)
        '''
        
        if degree == 1:
            return X
        
        ### BEGIN YOUR SOLUTION
#         degree = degree
        print(degree)
        B = list() # BucketSizes
        numFeatures = len(X[0]) # (d)
        print(numFeatures)
        
        for i in range(1,degree+1):
            if (i <= 1):
                B.append(int((i+numFeatures-1)/(numFeatures-1)))
            else:
                B.append(int(((i+numFeatures-1)/(numFeatures-1))+B[i-2]))

        print(numFeatures)        
        print(B)
        
        Z = X
        
        dPrime = len(B)
        L = list()
        for i in range(numFeatures):
            L.append(i)
        q = 0 # total size of all buckets before PREVIOUS bucket
        p = numFeatures -1# total size of all previous buckets
        g = numFeatures -1# index of the new column
        print(L,q,p,g)
        
        print("L:",L)
        print("Z:",Z)
        print("X:",X) 
        
        for i in range(2,degree): # for each dimension elevation
            print("TOP OF DIMENSION i LOOP RUN:\ni: ", i, "Z: ", Z, "L: ", L)
            for r in Z:
                print(*r)
            for j in range(q,p): # for each element in previous bucket
        #         print("hi2 g=", g)
                for k in range(L[j],numFeatures):
                    print("\n\nK RIGHT HERE =", k)
        #             for each feature, starting at 
        #             the last buckets corresponding largestFeatures list

                    temp = []
                    for ii in range(len(Z)):
                        temp.append(Z[ii][j]*X[ii][k])

                    print("temp: ",temp)
                    temp2 = np.arange(len(Z))
                    for i in range(len(temp)):
        #                 temp2[i] = temp[i]
                        Z[i].append(temp[i])
        #             np.append(Z,temp2.reshape(-1,1),1)


                    print("numFeatuers", numFeatures)
                    L.append(k)
                    L[g] = k
                    g += 1
                    print("i: ",i ," j: ", j," k: ",k, "g: ",g, ", L: ", L, ", L[g]: ", L[g])


            q = p # new total size of all previous buckets
            p = p + B[i]
        
        return Z
            
        
        ### END YOUR SOLUTION
        
        
        
