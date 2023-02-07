##### >>>>>> Please put your name and 6-digit EWUID here


# Various tools for data manipulation. 



import numpy as np
import math

class MyUtils:

    
    def z_transform(X, degree = 2):
        ''' Transforming traing samples to the Z space
            X: n x d matrix of samples, excluding the x_0 = 1 feature
            degree: the degree of the Z space
            return: the n x d' matrix of samples in the Z space, excluding the z_0 = 1 feature.
            It can be mathematically calculated: d' = \sum_{k=1}^{degree} (k+d-1) \choose (d-1)
        '''
    
    
    ######### place here the code that you have submitted for the previous programming assignment
        
        if degree == 1:
            return X
        
        ### BEGIN YOUR SOLUTION
#         degree = degree
        B = list() # BucketSizes
        numFeatures = len(X[0]) # (d)
        
        # GET BUCKET SIZES (gets TOTAL number of features after each dimension elevation)
        # if only 1 feature, each new bucket will be of size 1, 
        # resulting in divide by zero error in the standard algorithm to get B
        if (numFeatures == 1):
            for deg in range(0,degree):
                B.append(deg+1)
        else:
            for i in range(0,degree):
                if (i < 1):
                    B.append(int((i+numFeatures)/(numFeatures-1)))
                else:
                    B.append(int(((i+numFeatures)/(numFeatures-1))+B[i-1]))

        # print("B: ",B)

        # converting array to list. with more time, I'd rewrite my function to work with the input type from tester.py
        if (type(X) != list):
            X = X.tolist()
        

        Z = X
        
        dPrime = len(B)
        L = list()
        for ii in range(numFeatures):
            L.append(ii)
        q = 0 # total size of all buckets before PREVIOUS bucket
        p = numFeatures # total size of all previous buckets
        g = numFeatures # index of the new column
        # print(L,q,p,g)

        

        # print("L:",L)
        # print("Z:",Z)
        # print("X:",X) 
        
        for degree_i in range(1,degree): # for each dimension elevation
            # print("Top of i run, iL ", degree_i)
            
            for j in range(q,p): # for each element in previous bucket
       
                for k in range(L[j],numFeatures):
                    #  for each feature, starting at the last buckets corresponding largestFeatures list
                    temp = []
                    for ii in range(len(Z)):
                        temp.append(Z[ii][j]*X[ii][k])

                    for i in range(len(temp)):
                        Z[i].append(temp[i])
                        # np.append(Z[i],temp[i])
                    if (g > (len(L)-1)):
                        L.append(None)
                    L[g] = k
                    g += 1
                    # print("i: ",i ," j: ", j," k: ",k, "g: ",g, ", L: ", L, ", L[g]: ", L[g])


            q = p # new total size of all previous buckets
            # p = B[degree_i]
            p = len(Z[0])
            # print("Bottom OF DIMENSION i LOOP RUN:\ni: ", degree_i) 
            # Z_DF = pd.DataFrame(Z, columns=L)
            # print(Z_DF)
            # if (len(Z[0]) != B[degree_i]): # B[degree_i] or i ??
            #     print("ERROR: Z length is not equal to B[degree_i]")
            #     print("degree_i: ",degree_i ," j: ", j," k: ",k, "g: ",g, "q: ",q, "p: ",p ) #, ", L[g]: ", L[g])
        
        Z = np.asarray(Z)
        # print("\n\nZ at end:\n",Z)
        return Z





    
    
    ## below are the code that your instructor wrote for feature normalization. You can feel free to use them
    ## but you don't have to, if you want to use your own code or other library functions. 

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
